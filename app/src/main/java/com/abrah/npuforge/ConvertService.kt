package com.abrah.npuforge

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.net.Uri
import android.os.IBinder
import android.os.Build
import android.os.PowerManager
import android.os.SystemClock
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.content.IntentCompat
import androidx.core.net.toUri
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.time.Duration.Companion.minutes

/**
 * Owns one conversion and its foreground notification until native work and cleanup finish.
 */
class ConvertService : Service() {

    sealed interface State {
        data object Idle : State
        data class Running(
            val stage: String,
            val detail: String = "",
            val step: Int = 0,
            val steps: Int = 0,
            val log: List<String> = emptyList(),
            val startedAt: Long = 0,
        ) : State
        data class Done(val dir: String, val seconds: Long, val log: String = "") : State
        data class Failed(val message: String, val log: String = "") : State
        data class ComponentsDone(val message: String) : State
    }

    companion object {
        private const val TAG = "ConvertService"
        private const val CHANNEL = "convert"
        internal const val NOTE_ID = 1
        private const val ACTION_STOP = "com.abrah.npuforge.STOP_CONVERSION"
        private val WAKE_LOCK_LEASE = 10.minutes
        private val PERCENT = Regex("""(\d{1,3})%""")
        const val EXTRA_URI = "uri"
        const val EXTRA_NAME = "name"
        const val EXTRA_LORAS = "loras"
        const val EXTRA_MODEL = "model"
        const val ACTION_BACKUP = "com.abrah.npuforge.BACKUP_COMPONENTS"
        const val ACTION_RESTORE = "com.abrah.npuforge.RESTORE_COMPONENTS"

        private val _state = MutableStateFlow<State>(State.Idle)
        val state: StateFlow<State> = _state.asStateFlow()

        fun start(
            context: Context,
            uri: Uri,
            name: String,
            loras: List<Pair<Uri, Float>> = emptyList(),
            model: CheckpointInfo.Model = CheckpointInfo.Model.SD15,
        ) {
            val i = Intent(context, ConvertService::class.java)
                .putExtra(EXTRA_URI, uri)
                .putExtra(EXTRA_NAME, name)
                .putExtra(EXTRA_MODEL, model.name)
                // "uri|strength" strings rather than a Uri ArrayList: the same
                // path is then drivable from `adb shell am --esa`, so the
                // end-to-end LoRA flow can be tested without tapping through
                // a file picker.
                .putExtra(EXTRA_LORAS, loras.map { "${it.first}|${it.second}" }.toTypedArray())
            context.startForegroundService(i)
        }

        fun reset() {
            if (_state.value !is State.Running) _state.value = State.Idle
        }
    }

    private val job: Job = SupervisorJob()
    private val scope = CoroutineScope(Dispatchers.Main.immediate + job)
    private var conversion: Job? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(
            NotificationChannel(CHANNEL, getString(R.string.channel_convert),
                NotificationManager.IMPORTANCE_LOW)
        )
    }

    private fun note(text: String): Notification =
        NotificationCompat.Builder(this, CHANNEL)
            .setContentTitle(getString(R.string.app_name))
            .setContentText(text)
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .setForegroundServiceBehavior(NotificationCompat.FOREGROUND_SERVICE_IMMEDIATE)
            .setContentIntent(PendingIntent.getActivity(
                this, 0, Intent(this, MainActivity::class.java)
                    .addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP),
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
            ))
            .addAction(0, getString(R.string.stop_conversion), PendingIntent.getService(
                this, 1, Intent(this, ConvertService::class.java).setAction(ACTION_STOP),
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
            ))
            .setProgress(0, 0, true)
            .build()

    private var step = 0
    private var steps = 0
    private var startedAt = 0L
    private val logLines = ArrayDeque<String>()
    private var report: ConversionReport? = null

    private fun post(stage: String, detail: String = "") {
        job.ensureActive()
        if ((_state.value as? State.Running)?.stage != stage) {
            report?.record("stage=$stage")
            logLines.addLast(stage)
        }
        while (logLines.size > 200) logLines.removeFirst()
        _state.value = State.Running(stage, detail, step, steps, logLines.toList(), startedAt)
        getSystemService(NotificationManager::class.java)
            .notify(NOTE_ID, note(if (steps > 0) "$step/$steps  $stage" else stage))
    }

    /** One line of tool output; keeps the last few for the UI. */
    private fun logLine(stage: String, line: String) {
        val t = line.trim()
        if (t.isEmpty()) return
        logLines.addLast(t)
        // The generator draws progress bars; show the percentage, not the bar.
        val pct = PERCENT.findAll(t).lastOrNull()?.groupValues?.get(1)
        if (pct != null) {
            post(stage, "$pct%")
            return
        }
        post(stage, t.take(80))
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            conversion?.cancel()
            if (conversion == null || conversion?.isCompleted == true) {
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf()
            }
            return START_NOT_STICKY
        }
        // onStartCommand is serialized on the main thread. Keep ownership through cancellation cleanup.
        if (conversion?.isCompleted == false) {
            Log.i(TAG, "Ignoring duplicate start; conversion is already running")
            return START_NOT_STICKY
        }
        val uri = intent?.let {
            IntentCompat.getParcelableExtra(it, EXTRA_URI, Uri::class.java)
        }
        val name = intent?.getStringExtra(EXTRA_NAME).orEmpty().ifBlank { "converted" }
        val model = CheckpointInfo.Model.valueOf(
            intent?.getStringExtra(EXTRA_MODEL) ?: CheckpointInfo.Model.SD15.name
        )
        val loraSpecs: List<Pair<Uri, Float>> =
            (intent?.getStringArrayExtra(EXTRA_LORAS) ?: emptyArray()).mapNotNull { spec ->
                val bar = spec.lastIndexOf('|')
                if (bar <= 0) null
                else spec.substring(0, bar).toUri() to
                    (spec.substring(bar + 1).toFloatOrNull() ?: 1f)
            }
        if (uri == null) {
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
            return START_NOT_STICKY
        }
        val firstStage = getString(when (intent.action) {
            ACTION_BACKUP -> R.string.components_backing_up
            ACTION_RESTORE -> R.string.components_restoring
            else -> R.string.stage_import
        })
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(NOTE_ID, note(firstStage),
                ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
        } else {
            startForeground(NOTE_ID, note(firstStage))
        }

        startedAt = SystemClock.elapsedRealtime()
        logLines.clear()
        step = 0
        steps = 0
        post(firstStage)
        conversion = scope.launch {
            var result: State = State.Idle
            val work = File(cacheDir, "work")
            val wakeLock = getSystemService(PowerManager::class.java)
                .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "npuforge:conversion")
                .apply { setReferenceCounted(false) }
            var wakeLockRenewal: Job? = null
            try {
                wakeLock.acquire(WAKE_LOCK_LEASE.inWholeMilliseconds)
                // Renew a bounded lease while this conversion owns the work. A
                // slow download/compile must not lose its lock at a fixed deadline.
                wakeLockRenewal = launch {
                    while (true) {
                        delay(WAKE_LOCK_LEASE / 2)
                        wakeLock.acquire(WAKE_LOCK_LEASE.inWholeMilliseconds)
                    }
                }
                withContext(Dispatchers.IO) {
                    val diagnostic = ConversionReport(this@ConvertService, "$name (${model.name})")
                    report = diagnostic
                    if (intent.action == ACTION_BACKUP || intent.action == ACTION_RESTORE) {
                        val progress: (Long, Long) -> Unit = { bytes, total ->
                            val percent = if (total > 0) " ${bytes * 100 / total}%" else ""
                            post(firstStage, "${bytes / 1_000_000} MB$percent")
                        }
                        if (intent.action == ACTION_BACKUP) {
                            Donor.backup(this@ConvertService, model, uri, progress)
                        } else {
                            Donor.ensure(this@ConvertService, model, uri, progress)
                        }
                        result = State.ComponentsDone(getString(
                            if (intent.action == ACTION_BACKUP) R.string.components_backup_done
                            else R.string.components_restore_done, model.name,
                        ))
                        return@withContext
                    }
                    diagnostic.record("LoRA strengths=${loraSpecs.map { it.second }}")
                    work.deleteRecursively()
                    work.mkdirs()
                    steps = 10 + loraSpecs.size
                    step = 0

                    step++
                    post(getString(R.string.stage_import))
                    val ckpt = Converter.importFile(this@ConvertService, uri, File(work, "ckpt.safetensors")) { bytes ->
                        post(getString(R.string.stage_import), "${bytes / 1_000_000} MB")
                    }

                    step++
                    post(getString(R.string.stage_validate))
                    CheckpointInfo.validate(this@ConvertService, ckpt, model)

                    val loraFiles = loraSpecs.mapIndexed { idx, (u, strength) ->
                        step++
                        post(getString(R.string.stage_lora))
                        Converter.importFile(this@ConvertService, u, File(work, "lora_$idx.safetensors")) { bytes ->
                            post(getString(R.string.stage_lora), "${bytes / 1_000_000} MB")
                        } to strength
                    }

                    val componentFiles = File(work, "components").apply { mkdirs() }

                    step++
                    post(getString(R.string.stage_clip))
                    Converter.stageClip(this@ConvertService, ckpt, work, componentFiles, model, diagnostic) {
                        logLine(getString(R.string.stage_clip), it)
                    }
                    // Separate compiler processes and work directories keep the two VAE
                    // packs out of memory and release temporary disk before the UNet.
                    for ((component, weightsStage, compileStage) in listOf(
                        Triple("vae_encoder", R.string.stage_vae_encoder_weights, R.string.stage_vae_encoder_compile),
                        Triple("vae_decoder", R.string.stage_vae_decoder_weights, R.string.stage_vae_decoder_compile),
                    )) {
                        val componentWork = File(work, component).apply { mkdirs() }
                        step++
                        post(getString(weightsStage))
                        val componentPack = Converter.stageWeights(
                            this@ConvertService, ckpt, componentWork, model, diagnostic,
                            templateDirectory = "${model.componentDirectory}/$component",
                        ) { logLine(getString(weightsStage), it) }
                        step++
                        post(getString(compileStage))
                        val output = Converter.stageCompile(
                            this@ConvertService, componentPack, componentWork, model, diagnostic, component,
                        ) { logLine(getString(compileStage), it) }
                        if (!output.renameTo(File(componentFiles, "$component.bin"))) {
                            throw Converter.Failure("could not retain $component.bin")
                        }
                        componentWork.deleteRecursively()
                    }

                    step++
                    post(getString(R.string.stage_weights))
                    val pack = Converter.stageWeights(this@ConvertService, ckpt, work, model, diagnostic, loraFiles) {
                        logLine(getString(R.string.stage_weights), it)
                    }

                    step++
                    post(getString(R.string.stage_compile))
                    val unet = Converter.stageCompile(this@ConvertService, pack, work, model, diagnostic) {
                        logLine(getString(R.string.stage_compile), it)
                    }
                    pack.delete()
                    ckpt.delete()
                    loraFiles.forEach { (file, _) -> file.delete() }

                    step++
                    post(getString(R.string.stage_assemble))
                    val where = Converter.assemble(this@ConvertService, unet, name, model, componentFiles) { f ->
                        post(getString(R.string.stage_assemble), f)
                    }

                    result = State.Done(
                        where, (SystemClock.elapsedRealtime() - startedAt) / 1000,
                        logLines.joinToString("\n"),
                    )
                }
            } catch (e: CancellationException) {
                report?.record("Conversion cancelled")
                throw e
            } catch (e: Converter.Failure) {
                report?.record(e.stackTraceToString())
                Log.e(TAG, "conversion failed: ${e.message}\n${e.log}")
                result = State.Failed(e.message ?: "failed", e.log)
            } catch (e: Exception) {
                report?.record(e.stackTraceToString())
                Log.e(TAG, "conversion failed", e)
                result = State.Failed(e.message ?: e.javaClass.simpleName, logLines.joinToString("\n"))
            } finally {
                try {
                    withContext(NonCancellable + Dispatchers.IO) {
                        if (!work.deleteRecursively()) Log.w(TAG, "Could not remove conversion work files")
                        report?.record("Conversion finished: ${result.javaClass.simpleName}")
                        report?.close()
                        report = null
                    }
                } finally {
                    // Renewal and cleanup both run on Main: cancellation here
                    // prevents a later reacquire after release.
                    wakeLockRenewal?.cancel()
                    if (wakeLock.isHeld) wakeLock.release()
                    stopForeground(STOP_FOREGROUND_REMOVE)
                    stopSelf()
                    _state.value = result
                }
            }
        }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        scope.cancel()
        stopForeground(STOP_FOREGROUND_REMOVE)
        super.onDestroy()
    }
}

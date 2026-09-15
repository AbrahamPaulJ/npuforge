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
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

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
    }

    companion object {
        private const val TAG = "ConvertService"
        private const val CHANNEL = "convert"
        internal const val NOTE_ID = 1
        private const val ACTION_STOP = "com.abrah.npuforge.STOP_CONVERSION"
        private val PERCENT = Regex("""(\d{1,3})%""")
        const val EXTRA_URI = "uri"
        const val EXTRA_NAME = "name"
        const val EXTRA_LORAS = "loras"
        const val EXTRA_MODEL = "model"
        const val EXTRA_COMPONENTS = "components"

        private val _state = MutableStateFlow<State>(State.Idle)
        val state: StateFlow<State> = _state.asStateFlow()

        fun start(
            context: Context,
            uri: Uri,
            name: String,
            loras: List<Pair<Uri, Float>> = emptyList(),
            model: CheckpointInfo.Model = CheckpointInfo.Model.SD15,
            components: Uri? = null,
        ) {
            val i = Intent(context, ConvertService::class.java)
                .putExtra(EXTRA_URI, uri)
                .putExtra(EXTRA_NAME, name)
                .putExtra(EXTRA_MODEL, model.name)
                .putExtra(EXTRA_COMPONENTS, components)
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

    private fun post(stage: String, detail: String = "") {
        job.ensureActive()
        if ((_state.value as? State.Running)?.stage != stage) {
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
        val components = intent?.let {
            IntentCompat.getParcelableExtra(it, EXTRA_COMPONENTS, Uri::class.java)
        }
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
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(NOTE_ID, note(getString(R.string.stage_import)),
                ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
        } else {
            startForeground(NOTE_ID, note(getString(R.string.stage_import)))
        }

        startedAt = SystemClock.elapsedRealtime()
        logLines.clear()
        step = 0
        steps = 0
        post(getString(R.string.stage_import))
        conversion = scope.launch {
            var result: State = State.Idle
            val work = File(cacheDir, "work")
            val wakeLock = getSystemService(PowerManager::class.java)
                .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "npuforge:conversion")
            wakeLock.acquire()
            try {
                withContext(Dispatchers.IO) {
                    work.deleteRecursively()
                    work.mkdirs()
                    val donorReady = Donor.isReady(this@ConvertService, model)
                    steps = 4 + (if (donorReady) 0 else 1) + loraSpecs.size
                    step = 0
                    // Fetch the model family's shared components before copying the checkpoint.
                    if (!donorReady) {
                        step++
                        post(getString(R.string.stage_donor))
                        Donor.ensure(this@ConvertService, model, components) { seen, total ->
                            val pct = if (total > 0) " ${seen * 100 / total}%" else ""
                            post(getString(R.string.stage_donor), "${seen / 1_000_000} MB$pct")
                        }
                    }

                    step++
                    post(getString(R.string.stage_import))
                    val ckpt = Converter.importFile(this@ConvertService, uri, File(work, "ckpt.safetensors")) { bytes ->
                        post(getString(R.string.stage_import), "${bytes / 1_000_000} MB")
                    }

                    val loraFiles = loraSpecs.mapIndexed { idx, (u, strength) ->
                        step++
                        post(getString(R.string.stage_lora))
                        Converter.importFile(this@ConvertService, u, File(work, "lora_$idx.safetensors")) { bytes ->
                            post(getString(R.string.stage_lora), "${bytes / 1_000_000} MB")
                        } to strength
                    }

                    step++
                    post(getString(R.string.stage_weights))
                    val pack = Converter.stageWeights(this@ConvertService, ckpt, work, model, loraFiles) {
                        logLine(getString(R.string.stage_weights), it)
                    }

                    step++
                    post(getString(R.string.stage_compile))
                    val unet = Converter.stageCompile(this@ConvertService, pack, work, model) {
                        logLine(getString(R.string.stage_compile), it)
                    }

                    step++
                    post(getString(R.string.stage_assemble))
                    val where = Converter.assemble(this@ConvertService, unet, name, model) { f ->
                        post(getString(R.string.stage_assemble), f)
                    }

                    result = State.Done(
                        where, (SystemClock.elapsedRealtime() - startedAt) / 1000,
                        logLines.joinToString("\n"),
                    )
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Converter.Failure) {
                Log.e(TAG, "conversion failed: ${e.message}\n${e.log}")
                result = State.Failed(e.message ?: "failed", e.log)
            } catch (e: Exception) {
                Log.e(TAG, "conversion failed", e)
                result = State.Failed(e.message ?: e.javaClass.simpleName, logLines.joinToString("\n"))
            } finally {
                try {
                    withContext(NonCancellable + Dispatchers.IO) {
                        if (!work.deleteRecursively()) Log.w(TAG, "Could not remove conversion work files")
                    }
                } finally {
                    wakeLock.release()
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

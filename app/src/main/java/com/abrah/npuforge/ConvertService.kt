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
    }

    companion object {
        private const val TAG = "ConvertService"
        private const val CHANNEL = "convert"
        internal const val NOTE_ID = 1
        private const val ACTION_STOP = "com.abrah.npuforge.STOP_CONVERSION"
        const val ACTION_DOWNLOAD_VAE = "com.abrah.npuforge.DOWNLOAD_VAE"
        private val WAKE_LOCK_LEASE = 10.minutes
        private val PERCENT = Regex("""(\d{1,3})%""")
        const val EXTRA_URI = "uri"
        const val EXTRA_NAME = "name"
        const val EXTRA_LORAS = "loras"
        const val EXTRA_MODEL = "model"
        /** Plain 4-channel SD1.5 checkpoint -> inpainting model by add-difference ([InpaintDiff]). */
        const val EXTRA_INPAINT_DIFF = "inpaint_diff"
        /** A user-downloaded copy of the inpainting difference, imported instead of downloading. */
        const val EXTRA_INPAINT_DIFF_URI = "inpaint_diff_uri"
        /** SD1.5 text-encoder clip skip, 1 or 2 ([CheckpointInfo.clipDirectory]). */
        const val EXTRA_CLIP_SKIP = "clip_skip"
        /** Explicit UI/automation override; absent means use trusted checkpoint metadata. */
        const val EXTRA_PREDICTION_TYPE = "prediction_type"
        /**
         * SD1.5 Swap v3: the features to KEEP, comma-separated ([Converter.SWAP_FEATURES]);
         * absent = every feature the template supports. "inp" makes an inpainting model by
         * add-difference. A plain string so `adb shell am --es` can drive it.
         */
        const val EXTRA_SWAP_FEATURES = "swap_features"

        private val _state = MutableStateFlow<State>(State.Idle)
        val state: StateFlow<State> = _state.asStateFlow()

        fun downloadVae(context: Context) {
            val i = Intent(context, ConvertService::class.java).setAction(ACTION_DOWNLOAD_VAE)
            context.startForegroundService(i)
        }

        fun start(
            context: Context,
            uri: Uri,
            name: String,
            loras: List<Pair<Uri, Float>> = emptyList(),
            model: CheckpointInfo.Model = CheckpointInfo.Model.SD15,
            inpaintDiff: Boolean = false,
            inpaintDiffUri: Uri? = null,
            clipSkip: Int = 2,
            swapFeatures: Collection<String>? = null,
            predictionType: CheckpointInfo.PredictionType? = null,
        ) {
            val i = Intent(context, ConvertService::class.java)
                .putExtra(EXTRA_URI, uri)
                .putExtra(EXTRA_NAME, name)
                .putExtra(EXTRA_MODEL, model.name)
                .putExtra(EXTRA_INPAINT_DIFF, inpaintDiff)
                .putExtra(EXTRA_INPAINT_DIFF_URI, inpaintDiffUri)
                .putExtra(EXTRA_CLIP_SKIP, clipSkip)
                // "uri|strength" strings rather than a Uri ArrayList: the same
                // path is then drivable from `adb shell am --esa`, so the
                // end-to-end LoRA flow can be tested without tapping through
                // a file picker.
                .putExtra(EXTRA_LORAS, loras.map { "${it.first}|${it.second}" }.toTypedArray())
            if (swapFeatures != null) i.putExtra(EXTRA_SWAP_FEATURES, swapFeatures.joinToString(","))
            if (predictionType != null) i.putExtra(EXTRA_PREDICTION_TYPE, predictionType.name)
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

    /**
     * Mr.J's SDXL VAE contexts from the Fancy-AI Hugging Face repository.
     * Resumable and mirrored through [HfDownload]; the Utility tab's VAE import
     * is the offline route.
     */
    private suspend fun downloadSdxlVaeFiles(onProgress: (name: String, detail: String) -> Unit) {
        val vaeDir = File(filesDir, "vae_sdxl").apply { mkdirs() }
        for (name in listOf("vae_decoder.bin", "vae_encoder.bin")) {
            val cached = File(vaeDir, name)
            if (cached.isFile && cached.length() > 0L) continue
            onProgress(name, "")
            val part = File(vaeDir, "$name.part")
            var lastPct = -1
            try {
                HfDownload.fetch(
                    this, "Mr-J-369/Fancy-AI/resolve/main/$name", part,
                    onProgress = { done, total ->
                        val pct = (done * 100 / total).toInt()
                        if (pct != lastPct) {
                            lastPct = pct
                            onProgress(name, "$pct%")
                        }
                    },
                    onRetry = { onProgress(name, "retrying ($it)") },
                )
            } catch (e: java.io.IOException) {
                throw Converter.Failure(
                    "SDXL VAE download failed: ${e.message}. Progress is kept; retry, or import " +
                        "a VAE zip from the Utility tab.",
                )
            }
            if (!part.renameTo(cached)) throw Converter.Failure("could not store $name")
        }
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
        if (intent?.action == ACTION_DOWNLOAD_VAE) {
            val stage = getString(R.string.stage_vae_download)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
                startForeground(NOTE_ID, note(stage),
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
            } else {
                startForeground(NOTE_ID, note(stage))
            }
            startedAt = SystemClock.elapsedRealtime()
            logLines.clear()
            step = 0
            steps = 1
            post(stage)
            conversion = scope.launch {
                val wakeLock = getSystemService(PowerManager::class.java)
                    .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "npuforge:vae_download")
                    .apply { setReferenceCounted(false) }
                try {
                    wakeLock.acquire(WAKE_LOCK_LEASE.inWholeMilliseconds)
                    withContext(Dispatchers.IO) {
                        downloadSdxlVaeFiles { name, progress ->
                            post(stage, "$name $progress".trim())
                        }
                    }
                    _state.value = State.Idle
                } catch (e: CancellationException) {
                    _state.value = State.Idle
                    throw e
                } catch (e: Exception) {
                    _state.value = State.Failed(e.message ?: e.javaClass.simpleName, logLines.joinToString("\n"))
                } finally {
                    wakeLock.release()
                    stopForeground(STOP_FOREGROUND_REMOVE)
                    stopSelf()
                }
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
        // Add-difference makes an SD15_INPAINT export from a plain SD15 checkpoint:
        // the checkpoint validates as SD15, everything downstream is SD15_INPAINT.
        // SD15_SWAP is the same shape without the difference.
        val inpaintDiff = intent?.getBooleanExtra(EXTRA_INPAINT_DIFF, false) == true &&
            model == CheckpointInfo.Model.SD15_INPAINT
        val inpaintDiffUri = intent?.let {
            IntentCompat.getParcelableExtra(it, EXTRA_INPAINT_DIFF_URI, Uri::class.java)
        }
        // Swap v3: what this UNet keeps and what its compile leaves out. A template that cannot
        // drop features (Swap v2) keeps everything it has, and gets no features marker.
        val swapSupported = if (model == CheckpointInfo.Model.SD15_SWAP) Converter.swapFeaturesSupported(this)
            else emptyList()
        val swapKept: List<String>? = if (swapSupported.isEmpty()) null else {
            val asked = intent?.getStringExtra(EXTRA_SWAP_FEATURES)?.split(',')?.map { it.trim() }?.toSet()
            swapSupported.filter { asked == null || it in asked }
        }
        val swapDrop = swapKept?.let { kept -> swapSupported.filterNot { it in kept }.joinToString(",") }.orEmpty()
        // A Swap model with the inpaint feature is add-differenced like an SD15_INPAINT export.
        val swapInpaint = swapKept?.contains("inp") == true
        val clipSkip = if (model == CheckpointInfo.Model.SDXL) 2
            else intent?.getIntExtra(EXTRA_CLIP_SKIP, 2)?.takeIf { it == 1 } ?: 2
        val requestedPredictionType = intent?.getStringExtra(EXTRA_PREDICTION_TYPE)?.let { value ->
            runCatching { CheckpointInfo.PredictionType.valueOf(value) }.getOrNull()
        }
        val clipDirectory = CheckpointInfo.clipDirectory(model, clipSkip)
        val templateDir = model.templateDirectory
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
        val firstStage = getString(R.string.stage_import)
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
            // Active inputs, compiler backing files and outputs must survive
            // cache reclamation. This directory is cleaned explicitly below.
            val work = File(noBackupFilesDir, "conversion-work")
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
                    diagnostic.record("LoRA strengths=${loraSpecs.map { it.second }}")
                    diagnostic.record("Inpaint add-difference=$inpaintDiff")
                    diagnostic.record("Clip skip=$clipSkip")
                    if (model == CheckpointInfo.Model.SD15_SWAP) {
                        diagnostic.record("Swap features supported=$swapSupported kept=$swapKept drop=[$swapDrop]")
                    }
                    work.deleteRecursively()
                    work.mkdirs()
                    diagnostic.record("Conversion workspace: ${work.absolutePath}")
                    steps = (if (model == CheckpointInfo.Model.SDXL) 6 else 10) + loraSpecs.size +
                        (if (inpaintDiff || swapInpaint) 1 else 0)
                    step = 0

                    step++
                    post(getString(R.string.stage_import))
                    val ckpt = Converter.importFile(this@ConvertService, uri, File(work, "ckpt.safetensors")) { bytes ->
                        post(getString(R.string.stage_import), "${bytes / 1_000_000} MB")
                    }

                    step++
                    post(getString(R.string.stage_validate))
                    if (model == CheckpointInfo.Model.SDXL && Converter.socTier() == "8gen1") {
                        // The SDXL graphs and VAE contexts target v75; 8 Gen 1 is v69.
                        throw Converter.Failure(
                            "SDXL needs Snapdragon 8 Gen 3 or newer. This phone (8 Gen 1) can convert SD1.5 models.",
                        )
                    }
                    if (model == CheckpointInfo.Model.SD15_SWAP && loraSpecs.isNotEmpty()) {
                        // Baking is the v1 route; Swap takes its LoRAs per render.
                        throw Converter.Failure(
                            "SD1.5 Swap does not bake LoRAs in; choose them per render in the generating app.",
                        )
                    }
                    val checkpointReport = CheckpointInfo.validate(
                        this@ConvertService, ckpt,
                        if (inpaintDiff || model == CheckpointInfo.Model.SD15_SWAP) CheckpointInfo.Model.SD15 else model,
                        clipSkip,
                    )
                    val predictionType = requestedPredictionType ?: checkpointReport.predictionType
                        ?: CheckpointInfo.PredictionType.EPSILON
                    diagnostic.record(
                        "Prediction type=$predictionType " +
                            if (requestedPredictionType != null) "(explicit)"
                            else if (checkpointReport.predictionType != null) "(checkpoint metadata)"
                            else "(default; metadata absent or unrecognised)",
                    )

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
                    Converter.stageClip(
                        this@ConvertService, ckpt, work, componentFiles, model, diagnostic, clipDirectory,
                    ) {
                        logLine(getString(R.string.stage_clip), it)
                    }
                    if (model == CheckpointInfo.Model.SDXL) {
                        downloadSdxlVaeFiles { name, progress ->
                            post(getString(R.string.stage_vae_download), "$name $progress".trim())
                        }
                        val vaeDir = File(filesDir, "vae_sdxl")
                        for (name in listOf("vae_decoder.bin", "vae_encoder.bin")) {
                            File(vaeDir, name).copyTo(File(componentFiles, name), overwrite = true)
                        }
                    } else {
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
                    }

                    val unetArgs = if (inpaintDiff || swapInpaint) {
                        step++
                        val diff = if (inpaintDiffUri != null && !InpaintDiff.isReady(this@ConvertService)) {
                            post(getString(R.string.stage_inpaint_diff_import))
                            InpaintDiff.import(this@ConvertService, inpaintDiffUri) {
                                post(getString(R.string.stage_inpaint_diff_import), it)
                            }
                        } else {
                            post(getString(R.string.stage_inpaint_diff_download))
                            InpaintDiff.ensure(this@ConvertService) {
                                post(getString(R.string.stage_inpaint_diff_download), it)
                            }
                        }
                        diagnostic.record("Inpaint difference: ${diff.name} ${diff.length()} bytes")
                        // Swap v3 splits conv_in: its sample-side conv reads the first 4 of the
                        // 9 add-differenced input channels; the other 5 are a template constant.
                        listOf("--inpaint-diff", diff.absolutePath) +
                            (if (swapInpaint) listOf("--input-channel-prefix") else emptyList())
                    } else emptyList()

                    step++
                    post(getString(R.string.stage_weights))
                    val pack = Converter.stageWeights(
                        this@ConvertService, ckpt, work, model, diagnostic, loraFiles,
                        templateDirectory = templateDir,
                        extraArgs = unetArgs,
                    ) {
                        logLine(getString(R.string.stage_weights), it)
                    }
                    // Nothing reads the checkpoint or adapters after the weight pack. Free
                    // them before the compile, whose storage-backed allocator (SDXL, and
                    // SD1.5 on low-RAM phones) needs several GB of temporary files.
                    ckpt.delete()
                    loraFiles.forEach { (file, _) -> file.delete() }
                    diagnostic.snapshot()

                    step++
                    post(getString(R.string.stage_compile))
                    val unet = Converter.stageCompile(
                        this@ConvertService, pack, work, model, diagnostic, tplDrop = swapDrop,
                    ) {
                        logLine(getString(R.string.stage_compile), it)
                    }
                    pack.delete()

                    step++
                    post(getString(R.string.stage_assemble))
                    val where = Converter.assemble(
                        this@ConvertService, unet, name, model, componentFiles, swapFeatures = swapKept,
                        predictionType = predictionType,
                    ) { f ->
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

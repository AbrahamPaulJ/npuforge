package com.abrah.npuforge

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.net.Uri
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.io.File

/**
 * Runs a conversion as a foreground service.
 *
 * ⚠ Not a coroutine in the Activity: the two stages take ~117 s together and the
 * compile alone peaks around 4.8 GB. A backgrounded app in that state is exactly
 * what Android kills, and losing the work at second 100 is worse than the
 * notification is annoying.
 */
class ConvertService : Service() {

    sealed interface State {
        data object Idle : State
        data class Running(val stage: String, val detail: String = "") : State
        data class Done(val dir: String, val seconds: Long) : State
        data class Failed(val message: String, val log: String = "") : State
    }

    companion object {
        private const val TAG = "ConvertService"
        private const val CHANNEL = "convert"
        private const val NOTE_ID = 1
        const val EXTRA_URI = "uri"
        const val EXTRA_NAME = "name"
        const val EXTRA_LORAS = "loras"

        private val _state = MutableStateFlow<State>(State.Idle)
        val state: StateFlow<State> = _state.asStateFlow()

        fun start(
            context: Context,
            uri: Uri,
            name: String,
            loras: List<Pair<Uri, Float>> = emptyList(),
        ) {
            val i = Intent(context, ConvertService::class.java)
                .putExtra(EXTRA_URI, uri)
                .putExtra(EXTRA_NAME, name)
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
    private val scope = CoroutineScope(Dispatchers.IO + job)

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
            .setProgress(0, 0, true)
            .build()

    private fun post(stage: String, detail: String = "") {
        _state.value = State.Running(stage, detail)
        getSystemService(NotificationManager::class.java).notify(NOTE_ID, note(stage))
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val uri = intent?.let {
            @Suppress("DEPRECATION")
            it.getParcelableExtra<Uri>(EXTRA_URI)
        }
        val name = intent?.getStringExtra(EXTRA_NAME).orEmpty().ifBlank { "converted" }
        val loraSpecs: List<Pair<Uri, Float>> =
            (intent?.getStringArrayExtra(EXTRA_LORAS) ?: emptyArray()).mapNotNull { spec ->
                val bar = spec.lastIndexOf('|')
                if (bar <= 0) null
                else Uri.parse(spec.substring(0, bar)) to
                    (spec.substring(bar + 1).toFloatOrNull() ?: 1f)
            }
        if (uri == null) {
            stopSelf()
            return START_NOT_STICKY
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(NOTE_ID, note(getString(R.string.stage_import)),
                ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        } else {
            startForeground(NOTE_ID, note(getString(R.string.stage_import)))
        }

        scope.launch {
            val began = System.currentTimeMillis()
            val work = File(cacheDir, "work").apply { deleteRecursively(); mkdirs() }
            try {
                // One-time, ~1.0 GB downloaded for 395 MB kept. Done first so a
                // missing network fails before 2 GB has been copied around.
                if (!Donor.isReady(this@ConvertService)) {
                    post(getString(R.string.stage_donor))
                    Donor.ensure(this@ConvertService) { seen, total ->
                        val pct = if (total > 0) " ${seen * 100 / total}%" else ""
                        post(getString(R.string.stage_donor), "${seen / 1_000_000} MB$pct")
                    }
                }

                post(getString(R.string.stage_import))
                val ckpt = Converter.importCheckpoint(this@ConvertService, uri) { bytes ->
                    post(getString(R.string.stage_import), "${bytes / 1_000_000} MB")
                }

                val loraFiles = loraSpecs.mapIndexed { idx, (u, strength) ->
                    post(getString(R.string.stage_lora))
                    Converter.importLora(this@ConvertService, u, idx) to strength
                }

                post(getString(R.string.stage_weights))
                val pack = Converter.stageWeights(this@ConvertService, ckpt, work, loraFiles)

                post(getString(R.string.stage_compile))
                val unet = Converter.stageCompile(this@ConvertService, pack, work)

                post(getString(R.string.stage_assemble))
                val where = Converter.assemble(this@ConvertService, unet, name) { f ->
                    post(getString(R.string.stage_assemble), f)
                }

                // ~3 GB of checkpoint + pack; keeping it would fill the device
                // after two conversions.
                ckpt.delete()
                loraFiles.forEach { it.first.delete() }
                work.deleteRecursively()

                _state.value = State.Done(where, (System.currentTimeMillis() - began) / 1000)
            } catch (e: Converter.Failure) {
                Log.e(TAG, "conversion failed: ${e.message}\n${e.log}")
                _state.value = State.Failed(e.message ?: "failed", e.log)
                work.deleteRecursively()
            } catch (e: Exception) {
                Log.e(TAG, "conversion failed", e)
                _state.value = State.Failed(e.message ?: e.javaClass.simpleName)
                work.deleteRecursively()
            } finally {
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf()
            }
        }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        scope.cancel()
        super.onDestroy()
    }
}

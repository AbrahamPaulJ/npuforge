package com.abrah.npuforge

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.os.IBinder
import android.util.Log
import java.io.File
import kotlin.concurrent.thread

/**
 * Debug-only: can THIS PROCESS create a QNN HTP device at all?
 *
 * The conversion's compile step fails with "Device Creation failure" from the
 * app, while the identical binary, libraries, paths and file modes succeed under
 * `run-as`. `run-as` keeps a different SELinux domain, so the open question is
 * whether an ordinary app can talk to the DSP at all, or only whether it can do
 * graph *preparation*.
 *
 * This answers that by loading a 54 KB prebuilt context binary with
 * `qnn-net-run`. Loading a context binary is pure inference setup -- no graph
 * preparation -- so:
 *
 *   "Device Creation failure"   -> the app cannot reach the DSP at all
 *   anything later (missing inputs, etc.) -> the DSP is reachable and the
 *                                  problem is specific to graph preparation
 *
 * ⚠ Lives in src/debug so it is never part of a release build.
 */
class ProbeService : Service() {

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // Android refuses a background service start; foreground it immediately.
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(
            NotificationChannel(CH, "probe", NotificationManager.IMPORTANCE_LOW)
        )
        startForeground(
            42,
            Notification.Builder(this, CH)
                .setContentTitle("npuforge probe")
                .setSmallIcon(android.R.drawable.stat_notify_sync)
                .build(),
        )
        thread {
            try { probe() } catch (e: Exception) { Log.e(TAG, "probe threw", e) }
            stopSelf()
        }
        return START_NOT_STICKY
    }

    private fun probe() {
        val libs = File(applicationInfo.nativeLibraryDir)
        val dsp = Converter.stageDspLibs(this, libs)
        val work = File(cacheDir, "probe").apply { mkdirs() }

        // Both canaries: the shipped probe reports "inconclusive" if neither is
        // present, and that would look like a pass.
        for (n in listOf("canary_249.bin", "canary_228.bin")) {
            val f = File(work, n)
            assets.open("probe/$n").use { i -> f.outputStream().use { i.copyTo(it) } }
        }

        // Local Dream's backend (CC BY-NC 4.0, never committed -- see NOTICE),
        // purpose-built for this: it creates a QNN device
        // and load-tests the canary pair. Reusing it beats hand-rolling a probe
        // whose failure mode I would have to trust -- the first attempt, with
        // qnn-net-run and an empty input list, bailed during argument parsing
        // and reported a pass without ever creating a device.
        val cmd = listOf(
            File(libs, "libstable_diffusion_core.so").absolutePath,
            "--fp16_probe", work.absolutePath,
            "--lib_dir", libs.absolutePath,
        )
        val pb = ProcessBuilder(cmd).directory(work).redirectErrorStream(true)
        pb.environment()["LD_LIBRARY_PATH"] =
            "${libs.absolutePath}:/system/lib64:/vendor/lib64:/vendor/lib64/egl"
        pb.environment()["ADSP_LIBRARY_PATH"] =
            "${dsp.absolutePath};/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp"
        val p = pb.start()
        val log = p.inputStream.bufferedReader().readText()
        val rc = p.waitFor()

        Log.i(TAG, "=== PROBE rc=$rc ===")
        log.lineSequence().filter { it.isNotBlank() }.forEach { Log.i(TAG, it) }
        // Key off the probe's own JSON, not off the absence of an error string:
        // a run that dies during argument parsing produces no error string
        // either, and that is exactly how the first attempt reported a pass.
        val verdict = when {
            log.contains("\"ok\":true") ->
                "DSP REACHABLE from the app (device created, canary loaded) " +
                    "-- the failure is SPECIFIC TO GRAPH PREPARATION"
            log.contains("Device Creation failure") || log.contains("Failed to load skel") ->
                "APP CANNOT REACH THE DSP AT ALL"
            else ->
                "INCONCLUSIVE -- the probe did not get far enough to say"
        }
        Log.i(TAG, "=== VERDICT: $verdict ===")
    }

    companion object {
        private const val TAG = "NpuProbe"
        private const val CH = "probe"
    }
}

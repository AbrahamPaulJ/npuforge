package com.abrah.npuforge

import android.app.ActivityManager
import android.app.ApplicationExitInfo
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.os.StatFs
import android.os.SystemClock
import java.io.File
import java.io.IOException
import java.time.Instant
import java.util.UUID

/** Owns the durable diagnostic record, independent of conversion cache cleanup. */
class ConversionReport(private val context: Context, model: String) : AutoCloseable {
    val file = File(File(context.filesDir, "conversion-reports").apply { mkdirs() },
        "conversion-${System.currentTimeMillis()}-${UUID.randomUUID()}.txt")
    private val writer = file.bufferedWriter()
    private val started = SystemClock.elapsedRealtime()

    init {
        val app = context.packageManager.getPackageInfo(context.packageName, PackageManager.PackageInfoFlags.of(0))
        record("npuforge ${app.versionName} (${app.longVersionCode}); model=$model")
        record("device=${Build.MANUFACTURER} ${Build.MODEL}; product=${Build.PRODUCT}; SoC=${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL}")
        record("Android=${Build.VERSION.RELEASE}; SDK=${Build.VERSION.SDK_INT}; ABI=${Build.SUPPORTED_ABIS.joinToString()}; fingerprint=${Build.FINGERPRINT}")
        try {
            record("vm.max_map_count=${File("/proc/sys/vm/max_map_count").readText().trim()}")
        } catch (e: IOException) {
            record("vm.max_map_count unavailable: ${e.message}")
        }
        snapshot()
    }

    @Synchronized
    fun record(line: String) {
        writer.append("${Instant.now()} +${SystemClock.elapsedRealtime() - started}ms ")
            .append(line).append('\n')
        writer.flush()
    }

    fun snapshot(pid: Long? = null) {
        val memory = ActivityManager.MemoryInfo()
        context.getSystemService(ActivityManager::class.java).getMemoryInfo(memory)
        val storage = StatFs(context.filesDir.path)
        record("resources RAM_available=${memory.availMem} RAM_total=${memory.totalMem} lowMemory=${memory.lowMemory} storage_available=${storage.availableBytes} storage_total=${storage.totalBytes}")
        if (pid == null) return
        try {
            var mappings = 0
            var scudo = 0
            var backed = 0
            File("/proc/$pid/maps").forEachLine { line ->
                mappings++
                if ("[anon:scudo:secondary]" in line) scudo++
                if (".compiler-heap-" in line) backed++
            }
            record("pid=$pid memory_mappings=$mappings scudo_secondary=$scudo storage_backed=$backed")
        } catch (e: IOException) {
            record("pid=$pid memory_mappings unavailable: ${e.message}")
        }
        for (name in listOf("status", "oom_score", "oom_score_adj")) {
            try {
                val contents = File("/proc/$pid/$name").readText()
                val summary = if (name == "status") contents.lineSequence().filter {
                    it.startsWith("Vm") || it.startsWith("Rss") || it.startsWith("Threads:") ||
                        it.startsWith("State:") || it.startsWith("Cpus_allowed_list:")
                }.joinToString("; ") else contents.trim()
                record("pid=$pid $name: $summary")
            } catch (e: IOException) {
                record("pid=$pid $name unavailable: ${e.message}")
            }
        }
    }

    fun exitDetails(pid: Int) {
        try {
            val exits = context.getSystemService(ActivityManager::class.java)
                .getHistoricalProcessExitReasons(context.packageName, pid, 1)
            if (exits.isEmpty()) record("Android has no exit record available yet for pid=$pid")
            for (exit in exits) {
                record("Android exit pid=${exit.pid} time=${exit.timestamp} reason=${exit.reason} status=${exit.status} importance=${exit.importance} pssKiB=${exit.pss} rssKiB=${exit.rss} description=${exit.description}")
                if (exit.reason == ApplicationExitInfo.REASON_CRASH_NATIVE) {
                    // API 31+ exposes a protobuf tombstone. It may already have
                    // expired from Android's global crash buffer, or be unavailable.
                    val summary = exit.traceInputStream?.use { NativeTombstone.summarize(it, pid) }
                    if (summary == null) record("Android native tombstone unavailable for pid=$pid")
                    else summary.forEach(::record)
                }
            }
        } catch (e: Exception) {
            // Diagnostic access or decoding must never replace the tool's failure.
            runCatching { record("Android exit diagnostics unavailable for pid=$pid: ${e.message}") }
        }
    }

    @Synchronized
    override fun close() = writer.close()
}

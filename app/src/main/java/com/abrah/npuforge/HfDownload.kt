package com.abrah.npuforge

import android.content.Context
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.util.TimeZone

/**
 * Resumable download of one Hugging Face file, starting from the source the
 * user chose ([Source], Utility tab) and alternating with the other one.
 *
 * ⚠ huggingface.co is unreliable or blocked from mainland China: a connection
 * delivers some bytes and then stalls or resets. The SDXL VAE download never
 * resumed or timed out, so it hung or restarted from zero on every attempt
 * (field report, September 2026). hf-mirror.com serves the same `resolve/`
 * paths inside China and redirects to huggingface.co elsewhere, so trying both
 * costs nothing outside China. Until the user picks a source, a phone on a
 * mainland-China timezone starts with the mirror.
 *
 * Bytes accumulate in `<target>.part` across attempts, hosts and app restarts;
 * an attempt that delivered bytes resets the failure count, so only
 * [MAX_STALLS] consecutive attempts that made no progress end the download.
 * The caller checks content (size, hash) before using the file.
 */
object HfDownload {
    enum class Source(val host: String) {
        HUGGING_FACE("https://huggingface.co/"),
        MIRROR("https://hf-mirror.com/"),
    }

    private const val PREFS = "downloads"
    private const val KEY_SOURCE = "source"
    private val CHINA_ZONES = setOf("Asia/Shanghai", "Asia/Chongqing", "Asia/Harbin", "Asia/Urumqi", "PRC")
    private const val MAX_STALLS = 6

    fun source(context: Context): Source {
        val saved = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getString(KEY_SOURCE, null)
        return Source.entries.firstOrNull { it.name == saved }
            ?: if (TimeZone.getDefault().id in CHINA_ZONES) Source.MIRROR else Source.HUGGING_FACE
    }

    fun setSource(context: Context, source: Source) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putString(KEY_SOURCE, source.name).apply()
    }

    /** [path] is `<owner>/<repo>/resolve/<revision>/<file>`. Returns the `.part` file, complete. */
    suspend fun fetch(
        context: Context,
        path: String,
        part: File,
        expectedSize: Long? = null,
        onProgress: (done: Long, total: Long) -> Unit,
        onRetry: (String) -> Unit = {},
    ): File {
        val first = source(context)
        val hosts = listOf(first.host) + Source.entries.filter { it != first }.map { it.host }
        var stalls = 0
        var attempt = 0
        while (true) {
            currentCoroutineContext().ensureActive()
            val before = part.length()
            val host = hosts[attempt++ % hosts.size]
            try {
                val total = fetchOnce(host + path, part, expectedSize, onProgress)
                if (part.length() == total) return part
                if (part.length() > total) {
                    part.delete()
                    throw IOException("partial file is longer than the server copy")
                }
                throw IOException("connection closed at ${part.length()} of $total bytes")
            } catch (e: IOException) {
                if (part.length() > before) stalls = 0 else stalls++
                if (stalls >= MAX_STALLS) throw IOException("${e.message} (after $attempt attempts)", e)
                onRetry("${host.removePrefix("https://").trimEnd('/')}: ${e.message}")
                delay(2000L * stalls.coerceAtLeast(1))
            }
        }
    }

    /**
     * One connection: appends to [part] and returns the file's full size.
     * Nothing is written or deleted until the response is known to be the
     * expected file, so an error page from one host cannot cost the progress.
     */
    private suspend fun fetchOnce(
        start: String,
        part: File,
        expectedSize: Long?,
        onProgress: (done: Long, total: Long) -> Unit,
    ): Long {
        var from = part.length()
        var url = start
        var conn: HttpURLConnection
        var redirects = 0
        while (true) {
            conn = URL(url).openConnection() as HttpURLConnection
            // Followed by hand: HttpURLConnection will not follow a redirect that
            // changes host, and each hop has to carry the Range header.
            conn.instanceFollowRedirects = false
            conn.connectTimeout = 20_000
            conn.readTimeout = 30_000
            if (from > 0) conn.setRequestProperty("Range", "bytes=$from-")
            val code = conn.responseCode
            if (code in 301..308) {
                val next = conn.getHeaderField("Location") ?: throw IOException("redirect without Location")
                conn.disconnect()
                if (++redirects > 8) throw IOException("too many redirects")
                url = URL(URL(url), next).toString()
                continue
            }
            if (code == 416 && from > 0 && expectedSize == null) {
                // Range starts at or past the end of a file of unknown size:
                // the partial file cannot be trusted.
                conn.disconnect()
                part.delete()
                throw IOException("HTTP 416; restarting")
            }
            if (code != 200 && code != 206) {
                conn.disconnect()
                throw IOException("HTTP $code")
            }
            break
        }
        val code = conn.responseCode
        val total = if (code == 206) {
            conn.getHeaderField("Content-Range")?.substringAfterLast('/')?.toLongOrNull()
        } else {
            conn.contentLengthLong.takeIf { it >= 0 }
        }
        if (total == null || (expectedSize != null && total != expectedSize)) {
            conn.disconnect()
            throw IOException(if (total == null) "server did not report the file size"
                else "server offers $total bytes, expected $expectedSize")
        }
        if (from > 0 && code == 200) {
            // Server ignored the Range: start over rather than append a second copy.
            from = 0
            part.delete()
        }
        try {
            conn.inputStream.use { input ->
                FileOutputStream(part, from > 0).use { out ->
                    val buffer = ByteArray(1 shl 16)
                    var done = from
                    while (true) {
                        currentCoroutineContext().ensureActive()
                        val n = input.read(buffer)
                        if (n < 0) break
                        out.write(buffer, 0, n)
                        done += n
                        onProgress(done, total)
                    }
                }
            }
        } finally {
            conn.disconnect()
        }
        return total
    }
}

package com.abrah.npuforge

import android.content.Context
import android.net.Uri
import android.util.Log
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.util.zip.ZipInputStream

/**
 * Shared components for the UNet-only conversion pipeline.
 * Downloads shared components by model family, or imports an explicit local ZIP.
 * Both are cached separately and reused for subsequent conversions.
 */
object Donor {

    private const val TAG = "Donor"

    /** xororz's published DreamShaper 8 build — the template's own checkpoint. */
    private const val URL_ =
        "https://huggingface.co/xororz/sd-qnn/resolve/main/DreamShaperV8_qnn2.28_8gen2.zip"

    fun isReady(context: Context, model: CheckpointInfo.Model): Boolean {
        val d = File(context.filesDir, model.donorDirectory)
        return model.components.all { File(d, it).let { f -> f.isFile && f.length() > 0 } }
    }

    /**
     * Imports the components selected by the service when its cache is incomplete.
     *
     * @param onProgress bytes of the archive consumed so far, and its total
     *   (-1 when the server does not say).
     */
    fun ensure(
        context: Context,
        model: CheckpointInfo.Model,
        archive: Uri?,
        onProgress: (Long, Long) -> Unit,
    ) {
        val out = File(context.filesDir, model.donorDirectory).apply { mkdirs() }
        // A part-written donor is worse than none: it passes isReady() on the
        // files that landed and fails at render time on the ones that did not.
        for (f in out.listFiles().orEmpty()) f.delete()

        val url = when (model) {
            CheckpointInfo.Model.SD15 -> URL_
            CheckpointInfo.Model.SDXL ->
                "https://huggingface.co/Mr-J-369/SDXL-OnDevice-Conversion/resolve/main/" +
                    "sdxl-shared-mnn-clips-qnn250-vae1024-v75.zip"
        }
        val conn = if (archive == null) (URL(url).openConnection() as HttpURLConnection).apply {
            connectTimeout = 30_000
            readTimeout = 60_000
            instanceFollowRedirects = true
        } else null
        try {
            val input = if (conn == null) {
                context.contentResolver.openInputStream(archive!!)!!
            } else {
                if (conn.responseCode !in 200..299) {
                    throw Converter.Failure("download failed: HTTP ${conn.responseCode}")
                }
                conn.inputStream
            }
            val total = conn?.contentLengthLong ?: -1L
            var seen = 0L
            val counting = object : java.io.FilterInputStream(input) {
                override fun read(b: ByteArray, off: Int, len: Int): Int {
                    val n = super.read(b, off, len)
                    if (n > 0) {
                        seen += n
                        onProgress(seen, total)
                    }
                    return n
                }
            }
            var kept = 0
            ZipInputStream(counting.buffered(1 shl 20)).use { zin ->
                while (true) {
                    val e = zin.nextEntry ?: break
                    val base = e.name.substringAfterLast('/')
                    if (e.isDirectory || base !in model.components) {
                        zin.closeEntry()
                        continue
                    }
                    File(out, base).outputStream().use { zin.copyTo(it, 1 shl 20) }
                    zin.closeEntry()
                    kept++
                    Log.i(TAG, "extracted $base")
                }
            }
            if (kept != model.components.size) {
                for (f in out.listFiles().orEmpty()) f.delete()
                throw Converter.Failure("archive had $kept of ${model.components.size} expected files")
            }
        } finally {
            conn?.disconnect()
        }
    }
}

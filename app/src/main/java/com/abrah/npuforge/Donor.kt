package com.abrah.npuforge

import android.content.Context
import android.util.Log
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.util.zip.ZipInputStream

/**
 * The text encoder, tokenizer and VAE that every converted model borrows.
 *
 * ⚠ The recipe covers the **UNet only**. A loadable model directory needs six
 * more files, and they are not produced by conversion — they come from the
 * template checkpoint's own published build. That is a real limitation, not a
 * packaging shortcut: a converted checkpoint keeps its style and subject (the
 * UNet) but inherits DreamShaper's *prompt interpretation* (CLIP) and colour
 * response (VAE).
 *
 * Downloaded once rather than bundled: these are 395 MB and would nearly
 * quadruple the APK, while almost every user converts more than one checkpoint.
 *
 * ⚠ The archive is a **QAIRT 2.28** build while conversion emits **2.49**. Mixing
 * them in one model directory is fine and is already proven — every `sweep_p0`
 * render used exactly this pairing — because the runtime loads older context
 * binaries and the graphs are independent.
 */
object Donor {

    private const val TAG = "Donor"

    /** xororz's published DreamShaper 8 build — the template's own checkpoint. */
    private const val URL_ =
        "https://huggingface.co/xororz/sd-qnn/resolve/main/DreamShaperV8_qnn2.28_8gen2.zip"

    /**
     * Matched against each zip entry's BASENAME, so the archive's internal
     * folder layout does not matter. `unet.bin` is deliberately absent: it is
     * ~880 MB, it is the one file conversion replaces, and streaming past it
     * costs bandwidth but no disk.
     */
    private val KEEP = setOf(
        "clip_v2.mnn", "pos_emb.bin", "token_emb.bin", "tokenizer.json",
        "vae_encoder.bin", "vae_decoder.bin",
    )

    fun dir(context: Context): File = File(context.filesDir, "donor")

    fun isReady(context: Context): Boolean {
        val d = dir(context)
        return KEEP.all { File(d, it).let { f -> f.isFile && f.length() > 0 } }
    }

    fun installedBytes(context: Context): Long =
        dir(context).listFiles().orEmpty().sumOf { it.length() }

    /**
     * Downloads and extracts, skipping the work if it is already there.
     *
     * @param onProgress bytes of the archive consumed so far, and its total
     *   (-1 when the server does not say).
     */
    fun ensure(context: Context, onProgress: (Long, Long) -> Unit) {
        if (isReady(context)) return
        val out = dir(context).apply { mkdirs() }
        // A part-written donor is worse than none: it passes isReady() on the
        // files that landed and fails at render time on the ones that did not.
        for (f in out.listFiles().orEmpty()) f.delete()

        val conn = (URL(URL_).openConnection() as HttpURLConnection).apply {
            connectTimeout = 30_000
            readTimeout = 60_000
            instanceFollowRedirects = true
        }
        try {
            if (conn.responseCode !in 200..299) {
                throw Converter.Failure("download failed: HTTP ${conn.responseCode}")
            }
            val total = conn.contentLengthLong
            var seen = 0L
            val counting = object : java.io.FilterInputStream(conn.inputStream) {
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
                    if (e.isDirectory || base !in KEEP) {
                        zin.closeEntry()
                        continue
                    }
                    File(out, base).outputStream().use { zin.copyTo(it, 1 shl 20) }
                    zin.closeEntry()
                    kept++
                    Log.i(TAG, "extracted $base")
                }
            }
            if (kept != KEEP.size) {
                for (f in out.listFiles().orEmpty()) f.delete()
                throw Converter.Failure("archive had $kept of ${KEEP.size} expected files")
            }
        } finally {
            conn.disconnect()
        }
    }
}

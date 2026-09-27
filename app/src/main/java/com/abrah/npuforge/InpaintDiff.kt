package com.abrah.npuforge

import android.content.Context
import android.net.Uri
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import java.io.File
import java.security.MessageDigest

/**
 * The SD1.5 inpainting difference: `sd-v1-5-inpainting − v1-5-pruned-emaonly`
 * for every UNet tensor, float16. Added to a plain 4-channel checkpoint by
 * `tplconv --inpaint-diff`, it yields a 9-channel inpainting UNet
 * (docs/SD15-INPAINT.md). Downloaded once, on first use, or imported from a
 * file the user downloaded some other way.
 *
 * ⚠ 1.7 GB: the download resumes a partial file and alternates between
 * the chosen source and the other one ([HfDownload]). Either way the file is
 * used only after its size and SHA-256 match; the hash is checked once and
 * recorded in a stamp beside the file.
 */
object InpaintDiff {
    const val NAME = "sd15inp_diff_f16.safetensors"
    private const val PATH = "AbrahamPJ/npuforge-sd15-inpaint-diff/resolve/main/$NAME"
    const val SIZE = 1_719_165_856L
    private const val SHA256 = "9f08e2684fc2c6261e705e28ff437fd0fae569b1dcf62336afef5994beb6d487"

    private fun dir(context: Context) = File(context.filesDir, "inpaint_diff")
    fun file(context: Context) = File(dir(context), NAME)
    private fun stamp(context: Context) = File(dir(context), "$NAME.sha256")

    fun isReady(context: Context): Boolean {
        val f = file(context)
        return f.isFile && f.length() == SIZE && stamp(context).let { it.isFile && it.readText().trim() == SHA256 }
    }

    /** Downloads (or resumes) and verifies the diff. [onProgress] gets a short status. */
    suspend fun ensure(context: Context, onProgress: (String) -> Unit): File {
        if (isReady(context)) return file(context)
        val d = dir(context).apply { mkdirs() }
        val target = file(context)
        val tmp = File(d, "$NAME.part")
        if (target.isFile && target.length() == SIZE) {
            // Present but unverified: hash it rather than download it again.
            target.renameTo(tmp)
        }
        requireSpace(d, tmp.length())
        if (tmp.length() < SIZE) {
            var lastPct = -1
            try {
                HfDownload.fetch(context, PATH, tmp, SIZE, onProgress = { done, total ->
                    val pct = (done * 100 / total).toInt()
                    if (pct != lastPct) {
                        lastPct = pct
                        onProgress("$pct% of ${SIZE / 1_000_000} MB")
                    }
                }, onRetry = { onProgress("retrying ($it)") })
            } catch (e: java.io.IOException) {
                throw Converter.Failure(
                    "Inpainting download failed: ${e.message}. Progress is kept; retry, or download " +
                        "$NAME yourself and choose it with \"Use downloaded file\".",
                )
            }
        }
        onProgress("verifying")
        verifyAndStore(context, tmp)
        return target
    }

    /**
     * Copies a user-supplied copy of the diff (from a browser, a download
     * manager or another mirror) into place, hashing while it copies.
     */
    suspend fun import(context: Context, uri: Uri, onProgress: (String) -> Unit): File {
        val d = dir(context).apply { mkdirs() }
        val tmp = File(d, "$NAME.import")
        tmp.delete()
        requireSpace(d, 0)
        val md = MessageDigest.getInstance("SHA-256")
        val input = context.contentResolver.openInputStream(uri)
            ?: throw Converter.Failure("could not open the selected inpainting difference")
        input.use {
            tmp.outputStream().use { out ->
                val buffer = ByteArray(1 shl 20)
                var done = 0L
                var lastPct = -1
                while (true) {
                    currentCoroutineContext().ensureActive()
                    val n = input.read(buffer)
                    if (n < 0) break
                    if (done + n > SIZE) {
                        tmp.delete()
                        throw Converter.Failure("The selected file is not $NAME: it is larger than ${SIZE} bytes.")
                    }
                    out.write(buffer, 0, n)
                    md.update(buffer, 0, n)
                    done += n
                    val pct = (done * 100 / SIZE).toInt()
                    if (pct != lastPct) {
                        lastPct = pct
                        onProgress("copying $pct%")
                    }
                }
            }
        }
        store(context, tmp, md.digest())
        return file(context)
    }

    private fun requireSpace(d: File, have: Long) {
        if (d.usableSpace + have < SIZE + 256L * 1024 * 1024) {
            throw Converter.Failure(
                "Inpainting needs ${SIZE / 1_000_000} MB free for the inpainting difference; " +
                    "${d.usableSpace / 1_000_000} MB is available.",
            )
        }
    }

    private suspend fun verifyAndStore(context: Context, tmp: File) {
        val md = MessageDigest.getInstance("SHA-256")
        tmp.inputStream().use { input ->
            val buffer = ByteArray(1 shl 20)
            while (true) {
                currentCoroutineContext().ensureActive()
                val n = input.read(buffer)
                if (n < 0) break
                md.update(buffer, 0, n)
            }
        }
        store(context, tmp, md.digest())
    }

    private fun store(context: Context, tmp: File, digest: ByteArray) {
        val hex = digest.joinToString("") { "%02x".format(it) }
        val length = tmp.length()
        if (length != SIZE || hex != SHA256) {
            tmp.delete()
            throw Converter.Failure(
                "The inpainting difference failed verification ($length bytes, SHA-256 ${hex.take(12)}…); " +
                    "it was removed, please retry.",
            )
        }
        if (!tmp.renameTo(file(context))) throw Converter.Failure("could not store the inpainting difference")
        stamp(context).writeText(SHA256)
    }
}

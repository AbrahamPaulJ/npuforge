package com.abrah.npuforge

import android.content.Context
import android.net.Uri
import android.provider.DocumentsContract
import android.util.Log
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.nio.file.Files
import java.nio.file.StandardCopyOption.ATOMIC_MOVE
import java.nio.file.StandardCopyOption.REPLACE_EXISTING
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream
import java.util.zip.ZipOutputStream
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive

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
    suspend fun ensure(
        context: Context,
        model: CheckpointInfo.Model,
        archive: Uri?,
        onProgress: (Long, Long) -> Unit,
    ) {
        val operation = currentCoroutineContext()
        val out = File(context.filesDir, model.donorDirectory)
        val staging = Files.createTempDirectory(context.filesDir.toPath(), "components-").toFile()

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
                    operation.ensureActive()
                    val n = super.read(b, off, len)
                    if (n > 0) {
                        val previous = seen / (16 * 1024 * 1024)
                        seen += n
                        if (seen / (16 * 1024 * 1024) != previous) onProgress(seen, total)
                    }
                    return n
                }
            }
            val kept = mutableSetOf<String>()
            ZipInputStream(counting.buffered(1 shl 20)).use { zin ->
                while (true) {
                    val e = zin.nextEntry ?: break
                    val base = e.name.substringAfterLast('/')
                    if (e.isDirectory || base !in model.components) {
                        zin.closeEntry()
                        continue
                    }
                    File(staging, base).outputStream().use { zin.copyTo(it, 1 shl 20) }
                    zin.closeEntry()
                    kept.add(base)
                    Log.i(TAG, "extracted $base")
                }
            }
            if (kept != model.components || model.components.any { File(staging, it).length() == 0L }) {
                throw Converter.Failure("archive had ${kept.size} of ${model.components.size} expected nonempty files")
            }
            operation.ensureActive()
            out.mkdirs()
            for (name in model.components) {
                Files.move(File(staging, name).toPath(), File(out, name).toPath(), ATOMIC_MOVE, REPLACE_EXISTING)
            }
            onProgress(seen, total)
        } finally {
            conn?.disconnect()
            if (!staging.deleteRecursively()) Log.w(TAG, "Could not remove component import staging")
        }
    }

    suspend fun backup(
        context: Context,
        model: CheckpointInfo.Model,
        archive: Uri,
        onProgress: (Long, Long) -> Unit,
    ) {
        val operation = currentCoroutineContext()
        val directory = File(context.filesDir, model.donorDirectory)
        val total = model.components.sumOf { File(directory, it).length() }
        var copied = 0L
        try {
            context.contentResolver.openOutputStream(archive, "wt")!!.use { output ->
                ZipOutputStream(output.buffered(1 shl 20)).use { zip ->
                    // These model files are already compact; avoid expensive recompression.
                    zip.setLevel(0)
                    val buffer = ByteArray(1 shl 20)
                    for (name in model.components) {
                        operation.ensureActive()
                        zip.putNextEntry(ZipEntry(name))
                        File(directory, name).inputStream().use { input ->
                            while (true) {
                                operation.ensureActive()
                                val count = input.read(buffer)
                                if (count < 0) break
                                zip.write(buffer, 0, count)
                                val previous = copied / (16 * 1024 * 1024)
                                copied += count
                                if (copied / (16 * 1024 * 1024) != previous) onProgress(copied, total)
                            }
                        }
                        zip.closeEntry()
                    }
                }
            }
        } catch (e: Exception) {
            try {
                if (!DocumentsContract.deleteDocument(context.contentResolver, archive)) {
                    Log.w(TAG, "Could not remove incomplete component backup")
                }
            } catch (cleanup: Exception) {
                Log.w(TAG, "Could not remove incomplete component backup", cleanup)
            }
            throw e
        }
        onProgress(copied, total)
    }
}

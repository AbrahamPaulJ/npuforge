package com.abrah.npuforge

import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import java.io.File
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.delay
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.runInterruptible
import kotlinx.coroutines.withContext
import java.util.zip.Deflater
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import kotlin.time.Duration.Companion.seconds

/**
 * The conversion pipeline: a `.safetensors` in, a model directory out.
 *
 * QNN graphs are populated by [stageWeights], then compiled by [stageCompile].
 * Both model families write checkpoint-owned MNN text encoders via [stageClip].
 *
 * ⚠ These native tools are EXECUTABLES shipped as `lib*.so` in `jniLibs`. Android
 * refuses to execute anything in the writable app data directory, and
 * `nativeLibraryDir` is the one place it allows -- which is why they are named
 * that way and why `useLegacyPackaging = true` is not optional (build.gradle.kts).
 *
 * ⚠ The generator needs its QNN runtime unpacked to a readable directory and
 * pointed at by LD_LIBRARY_PATH / ADSP_LIBRARY_PATH, and the `config_file_path`
 * inside the backend-extension JSON must be an ABSOLUTE on-device path. A
 * relative one is resolved against the process CWD and fails confusingly.
 */
object Converter {

    private const val TAG = "Converter"

    class Failure(message: String, val log: String = "") : Exception(message)

    /**
     * Where converted models land: **`Download/npuforge/<name>.zip`**.
     *
     * Not the app's own external files directory. That is private to this app,
     * so a model written there is invisible to every file manager and to the
     * generator that is supposed to load it -- the conversion succeeded and the
     * result was unreachable. Downloads is written through MediaStore, which
     * needs no storage permission on API 29+ and leaves the files where a
     * person can actually find them.
     */
    const val OUTPUT_SUBDIR = "npuforge"

    /** Copies checkpoint or LoRA data into this conversion's work directory. */
    suspend fun importFile(context: Context, uri: Uri, dst: File, onBytes: (Long) -> Unit): File {
        context.contentResolver.openInputStream(uri)!!.use { input ->
            dst.outputStream().use { out ->
                val buf = ByteArray(1 shl 20)
                var total = 0L
                while (true) {
                    currentCoroutineContext().ensureActive()
                    val n = input.read(buf)
                    if (n <= 0) break
                    out.write(buf, 0, n)
                    total += n
                    onBytes(total)
                }
            }
        }
        return dst
    }

    /**
     * Runs a tool, reporting progress as it goes.
     *
     * Streamed rather than read at the end: the compile is ~95 s during which
     * the UI would otherwise say nothing at all, and the generator prints a
     * percentage we can surface. Only the last 200 lines are kept -- the
     * generator draws progress bars and the full log is megabytes of them.
     */
    private suspend fun run(
        cmd: List<String>,
        env: Map<String, String>,
        cwd: File,
        report: ConversionReport,
        onLine: (String) -> Unit,
    ): String = coroutineScope {
        val pb = ProcessBuilder(cmd).directory(cwd).redirectErrorStream(true)
        pb.environment().putAll(env)
        report.record("command=${cmd.joinToString(" ")}; cwd=$cwd; environment=$env")
        val p = pb.start()
        // Android UNIXProcess exposes its PID in toString(), not Java 9 Process.pid().
        val pid = Regex("pid=(\\d+)").find(p.toString())?.groupValues?.get(1)?.toLongOrNull()
        report.record("Started pid=$pid")
        val monitor = launch(Dispatchers.IO) {
            while (true) {
                report.snapshot(pid)
                delay(5.seconds)
            }
        }
        val tail = ArrayDeque<String>()
        try {
            val output = launch(Dispatchers.IO) {
                p.inputStream.bufferedReader().useLines { lines ->
                    lines.forEach { line ->
                        if (tail.size >= 200) tail.removeFirst()
                        tail.addLast(line)
                        report.record(line)
                        Log.i(TAG, line)
                        onLine(line)
                    }
                }
            }
            val rc = runInterruptible(Dispatchers.IO) { p.waitFor() }
            output.join()
            monitor.cancelAndJoin()
            report.snapshot()
            val signal = when (rc) {
                134 -> "SIGABRT (abort)"
                135 -> "SIGBUS (bus error)"
                137 -> "SIGKILL (killed; not proof of OOM)"
                139 -> "SIGSEGV (segmentation fault)"
                else -> ""
            }
            val outcome = "pid=$pid exit=$rc $signal"
            report.record(outcome)
            onLine(outcome)
            if (rc != 0 && pid != null) report.exitDetails(pid.toInt())
            val log = tail.joinToString(System.lineSeparator())
            Log.i(TAG, "rc=$rc for ${cmd.first().substringAfterLast('/')}")
            if (rc != 0) throw Failure("${cmd.first().substringAfterLast('/')} failed (rc $rc) $signal", log)
            log
        } finally {
            // A cancelled service must release the compiler and its native memory.
            p.destroyForcibly()
            withContext(NonCancellable + Dispatchers.IO) {
                p.waitFor()
                monitor.cancelAndJoin()
            }
        }
    }


    /**
     * Copies the hexagon skels where the DSP can read them.
     *
     * Every arch is copied rather than just this chip's: which skel FastRPC
     * wants is decided by the DEVICE, not by the graph being compiled, and
     * detecting that here would be a second thing to get wrong. ~145 MB of app
     * storage, written once.
     *
     * These are public libraries copied from the APK, never checkpoints or user
     * data. FastRPC runs outside the app UID and requires read/traverse access;
     * owner-only permissions break DSP loading (see docs/ANDROID.md, section 3).
     */
    @SuppressLint("SetWorldReadable")
    internal fun stageDspLibs(context: Context, from: File): File {
        val into = File(context.filesDir, "qnnlibs").apply { mkdirs() }
        val wanted = from.listFiles().orEmpty().filter {
            it.name.startsWith("libQnnHtpV") && !it.name.endsWith("Stub.so")
        }
        if (wanted.isEmpty()) throw Failure("no hexagon libraries in nativeLibraryDir")
        for (f in wanted) {
            val out = File(into, f.name)
            if (!out.exists() || out.length() != f.length()) f.copyTo(out, overwrite = true)
            // ⚠⚠ World-readable, every time. File.copyTo creates 0600 and the DSP
            // runs outside this UID, so a 0600 skel gives "Failed to load skel,
            // error: 4000" -- byte-identical to a working one and sitting in the
            // right directory. An `adb cp` preserved 0755 and worked, which is
            // what made the two cases look the same when they were not.
            out.setReadable(true, false)
        }
        into.setReadable(true, false)
        into.setExecutable(true, false)
        return into
    }

    /** Stage 1: checkpoint -> weight pack. */
    suspend fun stageWeights(
        context: Context,
        ckpt: File,
        work: File,
        model: CheckpointInfo.Model,
        report: ConversionReport,
        loras: List<Pair<File, Float>> = emptyList(),
        templateDirectory: String = model.templateDirectory,
        onLine: (String) -> Unit = {},
    ): File {
        val tpl = File(work, "template").apply { mkdirs() }
        for (name in context.assets.list(templateDirectory).orEmpty()) {
            context.assets.open("$templateDirectory/$name").use { input ->
                File(tpl, name).outputStream().use { input.copyTo(it) }
            }
        }
        val exe = File(context.applicationInfo.nativeLibraryDir, "libtplconv.so")
        val pack = File(work, "out.pack")
        // Merged into the weights before quantizing, not applied at runtime:
        // UPDATEABLE_STATIC costs 2.8x inference on this hardware. The result is
        // an ordinary model. Several adapters stack in one pass.
        val loraArgs = loras.flatMap { (f, s) -> listOf("--lora", "${f.absolutePath}:$s") }
        run(
            listOf(
                exe.absolutePath,
                File(tpl, "recipe.bin").absolutePath,
                File(tpl, "tpl_trim.pack").absolutePath,
                ckpt.absolutePath,
                pack.absolutePath,
            ) + loraArgs,
            emptyMap(), work, report, onLine,
        )
        if (!pack.isFile || pack.length() == 0L) throw Failure("tplconv produced no pack")
        return pack
    }

    /** Stage 2: weight pack -> context binary. */
    suspend fun stageCompile(
        context: Context,
        pack: File,
        work: File,
        model: CheckpointInfo.Model,
        report: ConversionReport,
        component: String = "unet",
        onLine: (String) -> Unit = {},
    ): File {
        // ⚠⚠ THE DSP AND THE CPU NEED THE LIBRARIES IN DIFFERENT PLACES.
        //
        // Measured the hard way, one location at a time:
        //   /data/app/.../lib/arm64  exec YES, dlopen YES, DSP **NO**
        //   filesDir                 exec no,  dlopen yes, DSP **YES**
        //   cacheDir                 DSP no
        //   /sdcard/Android/data     noexec, and DSP no (FUSE)
        //
        // So: the executables run from nativeLibraryDir (the only exec-able
        // place), the CPU-side libraries load from there too, and the hexagon
        // skels are COPIED into filesDir because that is the only directory the
        // DSP will read. Getting this wrong gives "Failed to load skel,
        // error: 4000" then "Device Creation failure", which reads like a
        // broken device rather than a path problem.
        val libs = File(context.applicationInfo.nativeLibraryDir)
        val dspLibs = stageDspLibs(context, libs)
        val tpl = File(work, "template")
        val exe = File(libs, "libqnncontextgen.so")
        val outDir = File(work, "out").apply { mkdirs() }

        // ⚠ absolute on-device path, see the class comment
        val cfg = File(tpl, "htp_config.json")
        report.record("QNN graph configuration: ${cfg.readText()}")
        val backend = File(work, "htp_backend.json")
        backend.writeText(
            """{"backend_extensions":{"shared_library_path":""" +
                """"${File(libs, "libQnnHtpNetRunExtensions.so").absolutePath}",""" +
                """"config_file_path":"${cfg.absolutePath}"}}"""
        )

        run(
            listOf(
                exe.absolutePath,
                "--model", File(tpl, "libqnn_model.so").absolutePath,
                "--backend", File(libs, "libQnnHtp.so").absolutePath,
                "--output_dir", outDir.absolutePath,
                "--binary_file", component,
                "--config_file", backend.absolutePath,
                "--log_level", "info",
            ),
            buildMap {
                if (model == CheckpointInfo.Model.SDXL) {
                    put("LD_PRELOAD", File(libs, "libcompiler_heap.so").absolutePath)
                    put("QNN_COMPILER_HEAP_DIR", work.absolutePath)
                }
                // ⚠⚠ /vendor/lib64 is NOT optional. libQnnHtpV<arch>Stub.so links
                // against the vendor FastRPC client libcdsprpc.so, which is not
                // in an app's default linker namespace -- dlopen fails, the
                // transport never comes up, and QNN reports the useless
                // "Device Creation failure". As shell it resolves, which is why
                // every /data/local/tmp run worked. The project's own
                // run_base.sh has had these paths all along.
                put("LD_LIBRARY_PATH", "${libs.absolutePath}:/system/lib64:/vendor/lib64:/vendor/lib64/egl")
                // ⚠ The app's own directory is deliberately NOT here. The DSP
                // reads this path from its own side and cannot open app-private
                // storage, so naming it produced "Device Creation failure" --
                // the /data/local/tmp runs worked only because that directory is
                // world-readable. The shipping inference server sets this
                // variable not at all; the vendor paths are what the DSP needs.
                // The DSP reads this path itself, so it names filesDir -- NOT
                // nativeLibraryDir, which it cannot open.
                put("ADSP_LIBRARY_PATH",
                    "${dspLibs.absolutePath};/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp")
                put("QNN_TPL_PACK", pack.absolutePath)
            },
            work, report, onLine,
        )
        val output = File(outDir, "$component.bin")
        if (!output.isFile || output.length() == 0L) throw Failure("the generator produced no $component.bin")
        return output
    }

    /** Reconstructs the model family's text encoder(s) with row-sized weight buffers. */
    suspend fun stageClip(        context: Context,
        ckpt: File,
        work: File,
        output: File,
        model: CheckpointInfo.Model,
        report: ConversionReport,
        onLine: (String) -> Unit,
    ) {
        val assets = File(work, "clip").apply { mkdirs() }
        output.mkdirs()
        context.assets.open("${model.componentDirectory}/clip_recipe.bin").use { input ->
            File(assets, "clip_recipe.bin").outputStream().use { input.copyTo(it) }
        }
        run(
            listOf(
                File(context.applicationInfo.nativeLibraryDir, "libcomponentconv.so").absolutePath,
                assets.absolutePath, ckpt.absolutePath, output.absolutePath,
            ),
            emptyMap(), work, report, onLine,
        )
        context.assets.open("${model.componentDirectory}/tokenizer.json").use { input ->
            File(output, "tokenizer.json").outputStream().use { input.copyTo(it) }
        }
        assets.deleteRecursively()
    }

    /** Packages the selected component directory without substituting weights. */
    suspend fun assemble(
        context: Context,
        unet: File,
        name: String,
        model: CheckpointInfo.Model,
        components: File,
        onFile: (String) -> Unit,
    ): String {
        // One zip, not seven loose files: it is what a generator's import
        // expects, and seven 150-900 MB files strewn through Downloads is
        // nobody's idea of a result. Entry names are flat basenames because
        // that is what importers key on.
        val files = listOf(unet to "unet.bin") +
            model.components.map { File(components, it) to it }
        for ((file, entryName) in files) {
            if (!file.isFile || file.length() == 0L) {
                throw Failure("missing converted component: $entryName")
            }
        }

        val values = ContentValues().apply {
            put(MediaStore.Downloads.DISPLAY_NAME, "$name.zip")
            put(MediaStore.Downloads.MIME_TYPE, "application/zip")
            put(MediaStore.Downloads.RELATIVE_PATH, "${Environment.DIRECTORY_DOWNLOADS}/$OUTPUT_SUBDIR")
            put(MediaStore.Downloads.IS_PENDING, 1)
        }
        // MediaStore assigns a distinct name on collisions, preserving existing exports.
        val uri = context.contentResolver
            .insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
            ?: throw Failure("could not create $name.zip in Downloads")

        return try {
            val displayName = context.contentResolver.query(
                uri, arrayOf(MediaStore.Downloads.DISPLAY_NAME), null, null, null,
            )!!.use { cursor ->
                cursor.moveToFirst()
                cursor.getString(0)
            }
            context.contentResolver.openOutputStream(uri)?.use { raw ->
                ZipOutputStream(raw.buffered(1 shl 20)).use { zip ->
                    // ⚠ No compression on purpose. These are quantized weights and
                    // context binaries -- deflate saves almost nothing and would
                    // add a minute of CPU to a 1.27 GB archive on a phone.
                    zip.setLevel(Deflater.NO_COMPRESSION)
                    if (model == CheckpointInfo.Model.SDXL) {
                        for ((entryName, text) in mapOf("SDXL" to "", "qnn_context.txt" to "231_masked_v1")) {
                            zip.putNextEntry(ZipEntry(entryName))
                            zip.write(text.toByteArray(Charsets.UTF_8))
                            zip.closeEntry()
                        }
                    }
                    for ((src, entryName) in files) {
                        onFile(entryName)
                        zip.putNextEntry(ZipEntry(entryName))
                        src.inputStream().use { input ->
                            val buffer = ByteArray(1 shl 20)
                            while (true) {
                                currentCoroutineContext().ensureActive()
                                val count = input.read(buffer)
                                if (count < 0) break
                                zip.write(buffer, 0, count)
                            }
                        }
                        zip.closeEntry()
                    }
                }
            } ?: throw Failure("could not open $name.zip for writing")
            currentCoroutineContext().ensureActive()
            context.contentResolver.update(
                uri, ContentValues().apply { put(MediaStore.Downloads.IS_PENDING, 0) }, null, null,
            )
            "Download/$OUTPUT_SUBDIR/$displayName"
        } catch (e: Exception) {
            context.contentResolver.delete(uri, null, null)
            throw e
        }
    }
}

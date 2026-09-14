package com.abrah.npuforge

import android.content.Context
import android.net.Uri
import android.util.Log
import java.io.File

/**
 * The conversion pipeline: a `.safetensors` in, a model directory out.
 *
 * Two stages, both native and both measured on an SM8750 (HTP v79):
 *
 *   1. [stageWeights]  tplconv: checkpoint -> an 863 MB TPLPACK1 weight pack   ~24 s
 *   2. [stageCompile]  the QNN context-binary generator: pack -> unet.bin      ~93 s
 *
 * ⚠ Both binaries are EXECUTABLES shipped as `lib*.so` in `jniLibs`. Android
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

    /** Everything the device needs that is not the user's checkpoint. */
    private val TEMPLATE_ASSETS = "template"

    data class Progress(val stage: String, val detail: String = "", val fraction: Float = -1f)

    class Failure(message: String, val log: String = "") : Exception(message)

    /** Where converted models land: `<external files>/models/<name>/`. */
    fun modelsRoot(context: Context): File = File(context.getExternalFilesDir(null), "models")

    private fun nativeExe(context: Context, name: String): File {
        val f = File(context.applicationInfo.nativeLibraryDir, name)
        if (!f.exists()) throw Failure("$name is missing from nativeLibraryDir")
        // Packaged read-only; exec permission comes from the directory, not us.
        return f
    }

    /**
     * Unpacks an asset directory into app storage, skipping files already there
     * at the right size.
     *
     * Size, not a hash: these are our own assets and cannot change without the
     * APK changing, so the cheap check is the correct one. A truncated unpack
     * from a killed process is what the size catches.
     */
    private fun unpackAssets(context: Context, dir: String, into: File): File {
        into.mkdirs()
        val names = context.assets.list(dir).orEmpty()
        if (names.isEmpty()) throw Failure("no assets under '$dir'")
        for (name in names) {
            val out = File(into, name)
            val want = context.assets.open("$dir/$name").use { it.available().toLong() }
            if (out.exists() && out.length() == want) continue
            context.assets.open("$dir/$name").use { input ->
                out.outputStream().use { input.copyTo(it) }
            }
        }
        return into
    }

    /**
     * Copies the user's checkpoint out of SAF into a real path.
     *
     * ⚠ Unavoidable: tplconv mmaps the file by path, and a `content://` URI has
     * none. It costs ~2 GB of storage for the duration of the conversion.
     */
    fun importCheckpoint(context: Context, uri: Uri, onBytes: (Long) -> Unit): File {
        val dst = File(context.cacheDir, "ckpt.safetensors")
        context.contentResolver.openInputStream(uri)
            ?: throw Failure("cannot open the selected file")
        context.contentResolver.openInputStream(uri)!!.use { input ->
            dst.outputStream().use { out ->
                val buf = ByteArray(1 shl 20)
                var total = 0L
                while (true) {
                    val n = input.read(buf)
                    if (n <= 0) break
                    out.write(buf, 0, n)
                    total += n
                    onBytes(total)
                }
            }
        }
        if (dst.length() < 1_000_000_000L) {
            throw Failure("that file is ${dst.length() / 1_000_000} MB; an SD1.5 checkpoint is ~2 GB")
        }
        return dst
    }

    private fun run(cmd: List<String>, env: Map<String, String>, cwd: File): String {
        val pb = ProcessBuilder(cmd).directory(cwd).redirectErrorStream(true)
        pb.environment().putAll(env)
        val p = pb.start()
        val log = p.inputStream.bufferedReader().readText()
        val rc = p.waitFor()
        Log.i(TAG, "rc=$rc for ${cmd.first().substringAfterLast('/')}")
        if (rc != 0) throw Failure("${cmd.first().substringAfterLast('/')} failed (rc $rc)", log)
        return log
    }


    /**
     * Copies the hexagon skels where the DSP can read them.
     *
     * Every arch is copied rather than just this chip's: which skel FastRPC
     * wants is decided by the DEVICE, not by the graph being compiled, and
     * detecting that here would be a second thing to get wrong. ~145 MB of app
     * storage, written once.
     */
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
    fun stageWeights(context: Context, ckpt: File, work: File): File {
        val tpl = unpackAssets(context, TEMPLATE_ASSETS, File(work, "template"))
        val exe = nativeExe(context, "libtplconv.so")
        val pack = File(work, "out.pack")
        run(
            listOf(
                exe.absolutePath,
                File(tpl, "recipe.bin").absolutePath,
                File(tpl, "tpl_trim.pack").absolutePath,
                ckpt.absolutePath,
                pack.absolutePath,
            ),
            emptyMap(), work,
        )
        if (!pack.isFile || pack.length() == 0L) throw Failure("tplconv produced no pack")
        return pack
    }

    /** Stage 2: weight pack -> context binary. */
    fun stageCompile(context: Context, pack: File, work: File): File {
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
        val exe = nativeExe(context, "libqnncontextgen.so")
        val outDir = File(work, "out").apply { mkdirs() }

        // ⚠ absolute on-device path, see the class comment
        val cfg = File(work, "htp_config.json")
        cfg.writeText(
            """{"graphs":[{"vtcm_mb":8,"graph_names":["model"],"O":3.0}],""" +
                """"devices":[{"dsp_arch":"v73","soc_model":43,""" +
                """"cores":[{"perf_profile":"burst","rpc_control_latency":100}]}]}"""
        )
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
                "--binary_file", "unet",
                "--config_file", backend.absolutePath,
            ),
            mapOf(
                // ⚠⚠ /vendor/lib64 is NOT optional. libQnnHtpV<arch>Stub.so links
                // against the vendor FastRPC client libcdsprpc.so, which is not
                // in an app's default linker namespace -- dlopen fails, the
                // transport never comes up, and QNN reports the useless
                // "Device Creation failure". As shell it resolves, which is why
                // every /data/local/tmp run worked. The project's own
                // run_base.sh has had these paths all along.
                "LD_LIBRARY_PATH" to "${libs.absolutePath}:/system/lib64:/vendor/lib64:/vendor/lib64/egl",
                // ⚠ The app's own directory is deliberately NOT here. The DSP
                // reads this path from its own side and cannot open app-private
                // storage, so naming it produced "Device Creation failure" --
                // the /data/local/tmp runs worked only because that directory is
                // world-readable. The shipping inference server sets this
                // variable not at all; the vendor paths are what the DSP needs.
                // The DSP reads this path itself, so it names filesDir -- NOT
                // nativeLibraryDir, which it cannot open.
                "ADSP_LIBRARY_PATH" to
                    "${dspLibs.absolutePath};/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp",
                "QNN_TPL_PACK" to pack.absolutePath,
            ),
            work,
        )
        val unet = File(outDir, "unet.bin")
        if (!unet.isFile || unet.length() == 0L) throw Failure("the generator produced no unet.bin")
        return unet
    }

    /**
     * Assembles the finished model directory.
     *
     * ⚠ CLIP, the VAE and the tokenizer come from the TEMPLATE, not from the
     * user's checkpoint: the recipe covers the UNet only. The UNet carries the
     * style, so this works, but a converted model is not a complete port of the
     * checkpoint and the UI should not claim otherwise.
     */
    fun assemble(context: Context, unet: File, name: String): File {
        if (!Donor.isReady(context)) {
            // Emitting unet.bin alone produces a directory that looks like a
            // model and cannot load. Fail here instead.
            throw Failure("the shared text encoder and VAE are missing")
        }
        val dir = File(modelsRoot(context), name).apply { mkdirs() }
        unet.copyTo(File(dir, "unet.bin"), overwrite = true)
        for (f in Donor.dir(context).listFiles().orEmpty()) {
            f.copyTo(File(dir, f.name), overwrite = true)
        }
        return dir
    }
}

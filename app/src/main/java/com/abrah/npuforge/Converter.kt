package com.abrah.npuforge

import android.annotation.SuppressLint
import android.app.ActivityManager
import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.os.Build
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
    /** In `template_swap/` and in every SD1.5 Swap export ([CheckpointInfo.Model.SD15_SWAP]). */
    const val SWAP_MARKER = "lora_targets.json"
    /** Swap v2: the IP-Adapter layers, in the UNet's `ipk_i` / `ipv_i` input order. */
    const val IP_MARKER = "ip_targets.json"
    /**
     * Swap v3: in `template_swap/`, the features its feature-gated `libqnn_model.so` can leave out
     * (`{"features":["lora","cn","ip","inp"]}`); in an export, the ones its UNet actually has. A
     * template without it (Swap v2) builds every feature and ignores `QNN_TPL_DROP`.
     */
    const val SWAP_FEATURES_MARKER = "swap_features.json"
    /** LocalDream/Nightmare marker: launch the backend with `--use_v_pred`. */
    const val V_PRED_MARKER = "V_PRED"

    internal fun predictionMarkers(type: CheckpointInfo.PredictionType): Map<String, String> =
        if (type == CheckpointInfo.PredictionType.V_PREDICTION) mapOf(V_PRED_MARKER to "")
        else emptyMap()
    /**
     * Every Swap feature name, in `QNN_TPL_DROP` spelling: LoRA, ControlNet, IP-Adapter, inpaint, and
     * SDXL Swap v2's conv LoRA, FreeU, PAG and attention coupling (regional prompts).
     * ⚠ A template feature missing from this list is never dropped: it would stay in every graph.
     */
    val SWAP_FEATURES = listOf("lora", "cn", "ip", "inp", "conv", "freeu", "pag", "couple")

    /**
     * Features a template has but this build does not OFFER, so every compile leaves them out. SDXL
     * Swap v2's conv LoRA, FreeU, PAG and coupling: no app feeds them yet, a kept feature costs render
     * time even unused, and coupling with the other features does not fit one HTP process on an
     * S25 (4.02 GB; docs/SDXL-SWAP-TEMPLATE.md §7c).
     */
    // ⚠ SDXL Swap's Inpaint stays unoffered (the user's call, 2026-10-06): it renders blotchy,
    // photographic fills on Illustrious, unresolved -- the quantized context reproduces fp32 on
    // held-out rows, so the add-difference itself is the suspect (docs/SDXL-SWAP-TEMPLATE.md §7a).
    // ControlNet and IP-Adapter are offered again from 1.0.12 as a PREVIEW (the user, 2026-10-07):
    // the test conversion that Nightmare's SDXL ControlNet / IP support is built against.
    // SD1.5 Swap keeps every feature (walked on the phone in v3).
    private val UNOFFERED = mapOf(
        CheckpointInfo.Model.SDXL_SWAP to setOf("inp", "conv", "freeu", "pag", "couple"),
    )

    /** The SDXL Swap template's name and prompt length, from its `swap_features.json` (`sdxl_swap_v2`, 462). */
    fun sdxlSwapTemplate(context: Context): Pair<String, Int> = runCatching {
        val text = context.assets.open("${CheckpointInfo.Model.SDXL_SWAP.templateDirectory}/$SWAP_FEATURES_MARKER")
            .use { it.readBytes().toString(Charsets.UTF_8) }
        val name = Regex("\"template\"\\s*:\\s*\"(\\w+)\"").find(text)?.groupValues?.get(1) ?: "sdxl_swap_v1"
        val tokens = Regex("\"text_tokens\"\\s*:\\s*(\\d+)").find(text)?.groupValues?.get(1)?.toInt() ?: 231
        name to tokens
    }.getOrDefault("sdxl_swap_v1" to 231)

    /** The features a Swap template's gated lib can leave out; empty for one that cannot (Swap v2). */
    fun swapTemplateFeatures(context: Context, model: CheckpointInfo.Model): List<String> = runCatching {
        val text = context.assets.open("${model.templateDirectory}/$SWAP_FEATURES_MARKER")
            .use { it.readBytes().toString(Charsets.UTF_8) }
        val listed = Regex("\"(\\w+)\"").findAll(text.substringAfter("[")).map { it.groupValues[1] }.toSet()
        SWAP_FEATURES.filter { it in listed }
    }.getOrDefault(emptyList())

    /** The features the user may keep: the template's, minus any this build does not offer. */
    fun swapFeaturesSupported(
        context: Context,
        model: CheckpointInfo.Model = CheckpointInfo.Model.SD15_SWAP,
    ): List<String> = swapTemplateFeatures(context, model) - UNOFFERED[model].orEmpty()

    fun swapFeaturesJson(kept: Collection<String>, template: String = "swap_v3"): String =
        "{\"template\":\"$template\",\"features\":[" +
            SWAP_FEATURES.filter { it in kept }.joinToString(",") { "\"$it\"" } + "]}"

    /**
     * ⭐⭐ What a kept feature needs to be fed, per family -- the template's contract
     * (docs/SD15-LORA-CN-TEMPLATE.md, docs/SDXL-SWAP-TEMPLATE.md §1, §7c), so an importer
     * sizes its inputs from the export instead of from a guess. Scalars only (see [manifestJson]).
     */
    private fun featureDetail(model: CheckpointInfo.Model, feature: String): String? {
        val xl = model == CheckpointInfo.Model.SDXL_SWAP
        return when (feature) {
            "lora" -> "{\"targets\":\"$SWAP_MARKER\",\"rank\":64}"
            "cn" -> if (xl) "{\"residuals\":10,\"hint\":1024}" else "{\"residuals\":13,\"hint\":512}"
            "ip" -> "{\"targets\":\"$IP_MARKER\",\"layers\":${if (xl) 70 else 16}}"
            "inp" -> "{\"mask\":${if (xl) 128 else 64}}"
            "conv" -> "{\"targets\":\"conv_targets.json\",\"rank\":32}"
            else -> null
        }
    }

    /**
     * ⭐⭐⭐ `swap_features.json` -- what this conversion IS, in EVERY export since 1.0.12 (the name
     * is historical: it began as the Swap v3 feature list). Importers decide what to offer from
     * THIS, never from the folder or zip name, which the user may change.
     *
     * `schema` 2 adds, beside v1's `template` and `features`: `producer`, `family` (sd15 | sdxl),
     * `kind` (plain | inpaint | swap), `prediction` (eps | v), `text_tokens`, `size` (the square
     * render), `soc` (the chip the context was compiled on -- on-phone compiles target the device's
     * own arch) and `detail` (per kept feature, [featureDetail]).
     *
     * ⚠⚠ `features` is the LAST key and nothing after it may quote a feature name: Nightmare's
     * backend (patch 020) looks for `"inp"` ANYWHERE after `"features"`. `detail` therefore comes
     * BEFORE it. A schema-1 reader (Nightmare up to 1.6.099) still finds `template` and `features`.
     */
    fun manifestJson(
        model: CheckpointInfo.Model,
        kept: Collection<String>,
        template: String,
        textTokens: Int,
        prediction: CheckpointInfo.PredictionType,
        producer: String,
        soc: String,
    ): String {
        val family = when (model) {
            CheckpointInfo.Model.SDXL, CheckpointInfo.Model.SDXL_SWAP -> "sdxl"
            else -> "sd15"
        }
        val kind = when (model) {
            CheckpointInfo.Model.SD15_SWAP, CheckpointInfo.Model.SDXL_SWAP -> "swap"
            CheckpointInfo.Model.SD15_INPAINT -> "inpaint"
            else -> "plain"
        }
        val features = SWAP_FEATURES.filter { it in kept }
        val detail = features.mapNotNull { f -> featureDetail(model, f)?.let { "\"$f\":$it" } }
        fun q(s: String) = "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"") + "\""
        return "{\"schema\":2,\"producer\":${q(producer)},\"template\":${q(template)}," +
            "\"family\":\"$family\",\"kind\":\"$kind\"," +
            "\"prediction\":\"${if (prediction == CheckpointInfo.PredictionType.V_PREDICTION) "v" else "eps"}\"," +
            "\"text_tokens\":$textTokens,\"size\":${if (family == "sdxl") 1024 else 512},\"soc\":${q(soc)}," +
            "\"detail\":{${detail.joinToString(",")}}," +
            "\"features\":[" + features.joinToString(",") { "\"$it\"" } + "]}"
    }

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
        extraArgs: List<String> = emptyList(),
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
            ) + loraArgs + extraArgs,
            emptyMap(), work, report, onLine,
        )
        if (!pack.isFile || pack.length() == 0L) throw Failure("tplconv produced no pack")
        return pack
    }

    /**
     * Phones below this much RAM compile the SD1.5 UNet with the storage-backed
     * allocator that SDXL always uses.
     *
     * ⚠ An 8 GB SM8450 (7.5 GB reported) had the whole app killed at the UNet
     * compile's ~4.3 GB anonymous peak with 0.6 GB left; a run that happened to
     * survive the same peak had 0.66 GB left (September 2026 reports). On the
     * S25 Ultra the allocator took the compiler's anonymous peak from 2.89 GB to
     * 0.03 GB, cost 70 s -> 157 s and ~4.6 GB of temporary storage, and the
     * context binary matched a normal compile except for the 2 bytes that differ
     * between any two normal compiles. The VAE compiles are small and stay on
     * the normal allocator.
     */
    private const val LOW_RAM_BYTES = 10L * 1024 * 1024 * 1024

    fun isLowRam(context: Context): Boolean {
        val memory = ActivityManager.MemoryInfo()
        context.getSystemService(ActivityManager::class.java).getMemoryInfo(memory)
        return memory.totalMem < LOW_RAM_BYTES
    }

    /**
     * Which per-chip `htp_config_<tier>.json` a template may carry.
     *
     * ⚠ On the phone the context is built for the DEVICE's own arch: contexts
     * compiled on an SM8750 report dspArch 79 / socModel 69 whether the config
     * said v73 or v69 (September 2026). The config's graph options (VTCM, O,
     * finalize settings) still apply. What 8 Gen 1 needed was the V69 HTP
     * libraries; its `htp_config_8gen1.json` (v69, soc 36) is the tested
     * configuration and matches the device anyway.
     */
    fun socTier(): String = when (Build.SOC_MODEL) {
        "SM8750" -> "8elite"
        "SM8850" -> "8gen5"
        "SM8450", "SM8475" -> "8gen1"
        else -> Build.SOC_MODEL.lowercase()
    }

    /** Stage 2: weight pack -> context binary. */
    suspend fun stageCompile(
        context: Context,
        pack: File,
        work: File,
        model: CheckpointInfo.Model,
        report: ConversionReport,
        component: String = "unet",
        tplDrop: String = "",
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
        val socConfig = File(tpl, "htp_config_${socTier()}.json")
        val cfg = if (socConfig.isFile) socConfig else File(tpl, "htp_config.json")
        report.record("QNN graph configuration (${cfg.name}): ${cfg.readText()}")
        val backend = File(work, "htp_backend.json")
        backend.writeText(
            """{"backend_extensions":{"shared_library_path":""" +
                """"${File(libs, "libQnnHtpNetRunExtensions.so").absolutePath}",""" +
                """"config_file_path":"${cfg.absolutePath}"}}"""
        )

        try {
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
                    if (model.isSdxl || (component == "unet" && isLowRam(context))) {
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
                    // Swap v3: the features left out of this graph. The feature-gated lib skips
                    // their ops at compose time, so they cost nothing per render.
                    if (tplDrop.isNotEmpty()) put("QNN_TPL_DROP", tplDrop)
                },
                work, report, onLine,
            )
        } catch (e: Failure) {
            // ⚠ SM8735 (8s Gen 4) is v73 without fp16 execution, and the SD1.5 VAE graphs
            // compute in fp16: the device rejects their float32 -> float16 conversion
            // node while composing the graph (September 2026 reports).
            if ("FLOAT_16" in e.log && "VALIDATION_ERROR" in e.log) {
                throw Failure(
                    "This phone's NPU (${Build.SOC_MODEL}) cannot run fp16 operations, which the " +
                        "SD1.5 $component graph needs. SD1.5 conversion is not supported on this chip yet.",
                    e.log,
                )
            }
            throw e
        }
        val output = File(outDir, "$component.bin")
        if (!output.isFile || output.length() == 0L) throw Failure("the generator produced no $component.bin")
        return output
    }

    /** Reconstructs the model family's text encoder(s) with row-sized weight buffers. */
    suspend fun stageClip(
        context: Context,
        ckpt: File,
        work: File,
        output: File,
        model: CheckpointInfo.Model,
        report: ConversionReport,
        clipDirectory: String = model.componentDirectory,
        onLine: (String) -> Unit,
    ) {
        val assets = File(work, "clip").apply { mkdirs() }
        output.mkdirs()
        report.record("CLIP recipe: $clipDirectory/clip_recipe.bin")
        context.assets.open("$clipDirectory/clip_recipe.bin").use { input ->
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
        swapFeatures: Collection<String>? = null,
        predictionType: CheckpointInfo.PredictionType = CheckpointInfo.PredictionType.EPSILON,
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
                    // Family markers read by the importing app. `INPAINT` is not an
                    // upstream Local Dream name: it tells Nightmare Mobile to launch
                    // the 9-channel inpaint pipeline for this SD1.5 folder.
                    // `lora_targets.json` is both the Swap marker and the data a
                    // LoRA packer needs: the order of the UNet's 160 LoRA inputs.
                    val familyMarkers = when (model) {
                        CheckpointInfo.Model.SDXL -> mapOf("SDXL" to "", "qnn_context.txt" to "231_masked_v1")
                        // SDXL Swap v2 reads a 462-token prompt (6 chunks) whatever is kept, so every
                        // export carries `qnn_context.txt` = `462_masked_v1`, the Swap marker
                        // (`lora_targets.json`, which also identifies a Swap folder) and the features
                        // marker naming `sdxl_swap_v2`: only an importer with SDXL Swap v2 support
                        // renders it. IP / conv target lists come with their features.
                        CheckpointInfo.Model.SDXL_SWAP -> {
                            val (tplName, tokens) = sdxlSwapTemplate(context)
                            val kept = swapFeatures.orEmpty()
                            mapOf("SDXL" to "", "qnn_context.txt" to "${tokens}_masked_v1") +
                                (listOf(SWAP_MARKER) + (if ("ip" in kept) listOf(IP_MARKER) else emptyList()) +
                                    (if ("conv" in kept) listOf("conv_targets.json") else emptyList()))
                                    .associateWith { m ->
                                        context.assets.open("${model.templateDirectory}/$m")
                                            .use { it.readBytes().toString(Charsets.UTF_8) }
                                    } + mapOf(SWAP_FEATURES_MARKER to swapFeaturesJson(kept, tplName))
                        }
                        CheckpointInfo.Model.SD15_INPAINT -> mapOf("INPAINT" to "")
                        // `ip_targets.json` (Swap v2): the template takes IP-Adapter K/V
                        // inputs; Nightmare offers a reference picture only when it is here.
                        // Swap v3: `swap_features.json` names what this UNet kept, and a
                        // dropped IP-Adapter drops its marker too. `lora_targets.json` stays
                        // either way -- it is what identifies a Swap model.
                        CheckpointInfo.Model.SD15_SWAP -> (listOf(SWAP_MARKER) +
                            (if (swapFeatures == null || "ip" in swapFeatures) listOf(IP_MARKER) else emptyList())
                            ).mapNotNull { m ->
                            runCatching {
                                context.assets.open("${model.templateDirectory}/$m")
                                    .use { it.readBytes().toString(Charsets.UTF_8) }
                            }.getOrNull()?.let { m to it }
                        }.toMap().also { require(SWAP_MARKER in it) { "template_swap has no $SWAP_MARKER" } } +
                            (swapFeatures?.let { mapOf(SWAP_FEATURES_MARKER to swapFeaturesJson(it)) } ?: emptyMap())
                        CheckpointInfo.Model.SD15 -> emptyMap()
                    }
                    // ⭐ The manifest for EVERY export ([manifestJson]); a Swap export's v1 list is
                    // replaced by it. ⚠ An SD1.5 Swap export from a template that cannot drop features
                    // (v2, `swapFeatures` null) keeps writing none, as before: it would have to guess.
                    val manifest = if (model == CheckpointInfo.Model.SD15_SWAP && swapFeatures == null) {
                        emptyMap()
                    } else {
                        val swapTemplate = when (model) {
                            CheckpointInfo.Model.SDXL_SWAP -> sdxlSwapTemplate(context)
                            CheckpointInfo.Model.SD15_SWAP -> "swap_v3" to 77
                            CheckpointInfo.Model.SDXL -> model.templateDirectory to 231
                            else -> model.templateDirectory to 77
                        }
                        val producer = "npuforge " + runCatching {
                            context.packageManager.getPackageInfo(context.packageName, 0).versionName
                        }.getOrNull().orEmpty()
                        mapOf(
                            SWAP_FEATURES_MARKER to manifestJson(
                                model, swapFeatures.orEmpty(), swapTemplate.first, swapTemplate.second,
                                predictionType, producer, Build.SOC_MODEL,
                            ),
                        )
                    }
                    val markers = familyMarkers + manifest + predictionMarkers(predictionType)
                    for ((entryName, text) in markers) {
                        zip.putNextEntry(ZipEntry(entryName))
                        zip.write(text.toByteArray(Charsets.UTF_8))
                        zip.closeEntry()
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
                        // The zip now holds it; dropping the source keeps peak storage
                        // near one copy of the model instead of two. A failure after
                        // this point loses the conversion, which the work-directory
                        // cleanup would have discarded anyway.
                        src.delete()
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

package com.abrah.npuforge

import android.content.Context
import android.net.Uri
import org.json.JSONObject
import java.io.DataInputStream
import java.io.File
import java.io.InputStream

/**
 * What is actually inside a `.safetensors`, read before converting anything.
 *
 * A safetensors file begins with a little-endian u64 header length followed by a
 * JSON object naming every tensor, its dtype and shape. That header is a few
 * hundred KB at the front of the file, so this costs a short read over SAF — no
 * 2 GB copy, no native call, no waiting.
 *
 * Two jobs:
 *
 *  1. **Say what will and will not carry over.** A single-file SD1.5 checkpoint
 *     holds three models: the UNet (`model.diffusion_model.*`), the VAE
 *     (`first_stage_model.*`) and the text encoder (`cond_stage_model.*`).
 *     Both families convert all three components; SDXL has two text encoders.
 *  2. **Refuse what cannot work, early.** Each recipe names its exact tensors. If
 *     any is absent the conversion fails partway through; checking the header
 *     turns that into an instant, specific answer — and catches SD2 and
 *     diffusers-layout files, which otherwise look plausible right up until they
 *     do not.
 */
object CheckpointInfo {

    enum class Model(
        val templateDirectory: String,
        val componentDirectory: String,
        val donorDirectory: String,
        val components: Set<String>,
    ) {
        SD15("template", "components_sd15", "donor", setOf(
            "clip_v2.mnn", "pos_emb.bin", "token_emb.bin", "tokenizer.json",
            "vae_encoder.bin", "vae_decoder.bin",
        )),
        SDXL("template_sdxl", "components_sdxl", "donor_sdxl", setOf(
            "clip.mnn", "clip_2.mnn", "clip_2.mnn.weight", "tokenizer.json",
            "pos_emb.bin", "token_emb.bin", "pos_emb_2.bin", "token_emb_2.bin",
            "vae_encoder.bin", "vae_decoder.bin",
        )),
    }

    private const val UNET = "model.diffusion_model."
    private const val VAE = "first_stage_model."
    private const val CLIP = "cond_stage_model."
    private const val SDXL_CLIP = "conditioner.embedders."
    private const val EMA = "model_ema."

    data class Part(val present: Boolean, val tensors: Int, val bytes: Long)

    data class Report(
        val unet: Part,
        val vae: Part,
        val clip: Part,
        val ema: Part,
        val totalTensors: Int,
        val dtypes: Map<String, Int>,
        /** Empty when every tensor the recipe needs is present. */
        val missing: List<String>,
        val architecture: String,
        val model: Model,
        val fatal: String? = null,
    ) {
        val convertible: Boolean get() = fatal == null && missing.isEmpty()
    }

    /** Reads the header only: an 8-byte length then that many bytes of JSON. */
    private fun readHeader(input: InputStream): JSONObject {
        val d = DataInputStream(input.buffered(1 shl 16))
        val raw = ByteArray(8).also { d.readFully(it) }
        var len = 0L
        for (i in 7 downTo 0) len = (len shl 8) or (raw[i].toLong() and 0xff)
        if (len <= 0 || len > 100L * 1024 * 1024) {
            throw Converter.Failure("this does not look like a .safetensors file")
        }
        val json = ByteArray(len.toInt()).also { d.readFully(it) }
        return JSONObject(String(json, Charsets.UTF_8))
    }

    fun inspect(context: Context, uri: Uri): Report {
        val header = context.contentResolver.openInputStream(uri)?.use { readHeader(it) }
            ?: throw Converter.Failure("cannot open the selected file")
        return inspectHeader(context, header)
    }

    /** Rechecks the imported file, including conversions started outside the UI. */
    fun validate(context: Context, file: File, expectedModel: Model): Report {
        val report = file.inputStream().use { inspectHeader(context, readHeader(it)) }
        if (report.model != expectedModel) {
            throw Converter.Failure("The checkpoint is ${report.model.name}, not ${expectedModel.name}.")
        }
        report.fatal?.let { throw Converter.Failure(it) }
        if (report.missing.isNotEmpty()) {
            throw Converter.Failure(
                "Required checkpoint tensors are missing: ${report.missing.take(3).joinToString()}",
            )
        }
        return report
    }

    private fun inspectHeader(context: Context, header: JSONObject): Report {
        var unetN = 0; var unetB = 0L
        var vaeN = 0; var vaeB = 0L
        var clipN = 0; var clipB = 0L
        var emaN = 0; var emaB = 0L
        var sdxlClip = false
        val dtypes = mutableMapOf<String, Int>()
        val names = HashSet<String>(header.length() * 2)

        for (key in header.keys()) {
            if (key == "__metadata__") continue
            val o = header.optJSONObject(key) ?: continue
            names.add(key)
            val offs = o.optJSONArray("data_offsets")
            val size = if (offs != null && offs.length() == 2) {
                offs.getLong(1) - offs.getLong(0)
            } else 0L
            val dt = o.optString("dtype", "?")
            dtypes[dt] = (dtypes[dt] ?: 0) + 1
            when {
                key.startsWith(EMA) -> { emaN++; emaB += size }
                key.startsWith(UNET) -> { unetN++; unetB += size }
                key.startsWith(VAE) -> { vaeN++; vaeB += size }
                key.startsWith(CLIP) -> { clipN++; clipB += size }
                key.startsWith(SDXL_CLIP) -> { sdxlClip = true; clipN++; clipB += size }
            }
        }

        val model = if (sdxlClip || "${UNET}label_emb.0.0.weight" in names) Model.SDXL else Model.SD15
        val unetRequired = context.assets.open("${model.templateDirectory}/sources.txt").use {
            it.bufferedReader().readLines().filter(String::isNotBlank)
        }
        val required = unetRequired.toMutableSet()
        var invalid: String? = null
        for (name in unetRequired) {
            val tensor = header.optJSONObject(name) ?: continue
            if (tensor.optString("dtype") !in setOf("F16", "F32", "BF16") && invalid == null) {
                invalid = "$name has unsupported dtype ${tensor.optString("dtype")}; expected F16, F32 or BF16."
            }
        }
        for (manifest in listOf(
            "${model.componentDirectory}/clip_requirements.json",
            "${model.componentDirectory}/vae_encoder/requirements.json",
            "${model.componentDirectory}/vae_decoder/requirements.json",
        )) {
            val specification = try {
                context.assets.open(manifest).use {
                    JSONObject(it.bufferedReader().readText()).getJSONArray("tensors")
                }
            } catch (e: Exception) {
                throw Converter.Failure(
                    "This app build is missing valid ${model.name} conversion assets ($manifest). " +
                        "Rebuild with the complete component assets. ${e.message.orEmpty()}",
                )
            }
            if (specification.length() == 0) {
                throw Converter.Failure("This app build has an empty ${model.name} component manifest: $manifest")
            }
            for (index in 0 until specification.length()) {
                val requirement = specification.getJSONObject(index)
                val name = requirement.getString("name")
                required.add(name)
                val tensor = header.optJSONObject(name) ?: continue
                val shape = requirement.getJSONArray("shape")
                val actual = tensor.optJSONArray("shape")
                val matches = actual != null && actual.length() == shape.length() &&
                    (0 until shape.length()).all { actual.optLong(it, -1) == shape.getLong(it) }
                val dtypes = requirement.getJSONArray("dtypes")
                val dtype = tensor.optString("dtype")
                if (!matches && invalid == null) {
                    invalid = "$name has shape $actual; expected $shape."
                } else if ((0 until dtypes.length()).none { dtypes.getString(it) == dtype } && invalid == null) {
                    invalid = "$name has unsupported dtype $dtype; expected $dtypes."
                }
            }
        }
        val missing = required.filterNot { it in names }

        // Report the specific architecture problem before missing tensor names.
        val fatal = when {
            unetN == 0 && names.any { it.startsWith("down_blocks.") || it.startsWith("mid_block.") } ->
                "This is a diffusers-layout folder file, not a single-file checkpoint."
            unetRequired.none { it in names } && unetN > 0 ->
                "This UNet does not match the SD1.5 or SDXL template."
            unetN == 0 ->
                "No UNet found (no model.diffusion_model.* tensors)."
            else -> invalid
        }

        val arch = when {
            model == Model.SDXL -> "SDXL"
            fatal != null -> "unrecognised"
            missing.isEmpty() -> "SD 1.5"
            else -> "SD 1.5 variant"
        }

        return Report(
            unet = Part(unetN > 0, unetN, unetB),
            vae = Part(vaeN > 0, vaeN, vaeB),
            clip = Part(clipN > 0, clipN, clipB),
            ema = Part(emaN > 0, emaN, emaB),
            totalTensors = names.size,
            dtypes = dtypes,
            missing = missing,
            architecture = arch,
            model = model,
            fatal = fatal,
        )
    }
}

package com.abrah.npuforge.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

/** Conversion scope and the measurements available for each model family. */
@Composable
fun InfoScreen() {
    Column(
        Modifier.padding(16.dp).verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        Section(
            "What it does",
            "Converts SD1.5 and SDXL checkpoints into Qualcomm NPU models on this phone.",
            "It ships a template that already carries the expensive " +
                "part -- the calibrated activation ranges -- so the phone only has to " +
                "re-quantize the weights and compile.",
        )

        Section(
            "Scope",
            "• Single-file .safetensors checkpoints",
            "• SD1.5: 512 × 512, Snapdragon 8 Gen 2 target (v73)",
            "• SDXL: 1024 × 1024, Snapdragon 8 Gen 3 target (v75)",
            "• QAIRT 2.50; SDXL UNet uses INT8 weights and 16-bit activations",
            "• Shared VAE encoder and decoder are included for image-to-image and text-to-image",
        )

        Section(
            "What is converted, and what is borrowed",
            "Conversion replaces the UNet from your checkpoint. SD1.5 downloads the " +
                "shared DreamShaper text encoder and VAE once.",
            "For SDXL, the shared components download once from Mr-J-369's " +
                "SDXL-OnDevice-Conversion repository (about 1 GB). The app saves its " +
                "MNN text encoders and QNN VAE encoder/decoder for reuse.",
            "The current SDXL components use madebyollin/sdxl-vae-fp16-fix, QAIRT 2.50 " +
                "and the 1024 × 1024 v75 VAE. Prompt interpretation and colour follow " +
                "these shared components.",
        )

        Warning(
            "Checkpoint compatibility",
            "CyberRealistic, DreamShaper 8 and AbsoluteReality have working conversions. " +
                "MistoonAnime produced noise in the documented test despite converting successfully.",
            "The template reuses calibrated activation ranges. Their suitability varies " +
                "by checkpoint; the MistoonAnime result does not establish that all anime models fail.",
        )

        Section(
            "LoRA",
            "Adapters are merged into the weights during conversion, so they cost " +
                "nothing at generation time. Several stack in one pass.",
            "Supported: standard kohya LoRAs (lora_down / lora_up / alpha).",
            "Not supported: diffusers/PEFT format, LoCon, LyCORIS, LoHa, DoRA, IA3. " +
                "These fail with a clear message rather than producing a bad model.",
            "Only attention modules have been tested; convolution adapters should work " +
                "but are unverified.",
            "The text-encoder half of a LoRA is dropped, so style adapters work better " +
                "than trigger-word ones.",
            "Strength is baked in, not a slider you can move later: two strengths of one " +
                "adapter are two separate models.",
        )

        Section(
            "What it needs from your phone",
            "• arm64, Android 13 or newer",
            "• Earlier SD1.5 conversion measured about 4.8 GB of RAM at peak",
            "• SD1.5 needs about 4 GB of working storage",
            "• SD1.5 output is roughly 1.3 GB; the tested SDXL ZIP was about 3.5 GB",
            "• Tested SDXL O=3 conversion completed in 437 seconds",
            "• SDXL uses temporary file-backed compiler memory and needs additional free storage",
            "• Peak SDXL phone RAM has not been measured",
        )

        Section(
            "How to tell what went wrong",
            "Saturated, blotchy noise can indicate that the checkpoint does not fit " +
                "the template's activation ranges, as in the MistoonAnime test.",
            "Model loading depends on the target chip and the QNN runtime in your generator.",
            "Good image, but the prompt is interpreted oddly: that is the borrowed text " +
                "encoder.",
            "Colours slightly off, detail soft: that is the borrowed VAE.",
        )

        Section(
            "Tested on",
            "One device: Samsung Galaxy S25 Ultra (SM8750, Hexagon v79). SD1.5 and " +
                "SDXL conversion and generation both worked on this phone.",
            "The SDXL O=3 output generated in Aura in 15 seconds at 1024 × 1024, " +
                "8 steps, CFG 1, LCM/Karras. This is one run, not a cross-device benchmark.",
        )
    }
}

@Composable
private fun Section(title: String, vararg body: String) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text(title, style = MaterialTheme.typography.titleSmall)
        body.forEach { Text(it, style = MaterialTheme.typography.bodySmall) }
    }
}

@Composable
private fun Warning(title: String, vararg body: String) {
    Card(Modifier.fillMaxWidth()) {
        Column(
            Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            Text(
                title,
                style = MaterialTheme.typography.titleSmall,
                color = MaterialTheme.colorScheme.error,
            )
            body.forEach { Text(it, style = MaterialTheme.typography.bodySmall) }
        }
    }
}

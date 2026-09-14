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

/**
 * What this app does, what it does not, and where it is known to fail.
 *
 * Written plainly and kept honest on purpose. Every limitation here was
 * measured rather than assumed, and several of them look like bugs when you
 * meet them unprepared -- a model that will not load on a newer phone, or an
 * anime checkpoint that converts "successfully" into noise.
 */
@Composable
fun InfoScreen() {
    Column(
        Modifier.padding(16.dp).verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        Section(
            "What it does",
            "Converts a Stable Diffusion 1.5 checkpoint into a Qualcomm NPU model on " +
                "this phone, in about two minutes. On a PC the same job takes hours: " +
                "roughly 50 minutes of calibration and a 2 h 20 m quantize.",
            "It avoids that by shipping a template that already carries the expensive " +
                "part -- the calibrated activation ranges -- so the phone only has to " +
                "re-quantize the weights and compile.",
        )

        Section(
            "Scope",
            "• Realistic SD 1.5 checkpoints, in single-file .safetensors form",
            "• 512 x 512, text-to-image",
            "• The UNet only (see below)",
            "• Output is a QAIRT 2.49 context binary built for the 8 Gen 2 tier",
        )

        Warning(
            "⚠ The model may not load on every phone",
            "The output is built with QAIRT 2.49, which stamps an fp16 requirement into " +
                "the context binary. Some chips reject that and refuse to load the " +
                "model -- including some NEWER than 8 Gen 2. It is a property of the " +
                "toolchain, not of your phone, and it cannot be configured away.",
            "Building under QAIRT 2.28 instead removes the stamp, at no measured cost " +
                "to quality. That is the known fix and it is not implemented here yet.",
            "Separately, the graph targets the 8 Gen 2 tier (v73, 8 MB VTCM), so " +
                "Snapdragon 888 (v68) and 8 Gen 1 (v69) cannot load it either.",
            "So \"8 Gen 2 or newer\" is necessary but NOT sufficient. If a converted " +
                "model fails to load, this is the first thing to suspect.",
        )

        Section(
            "What is converted, and what is borrowed",
            "Conversion replaces the UNet -- which is what carries style, subject and " +
                "anatomy. The text encoder and VAE come from the built-in template, not " +
                "from your checkpoint.",
            "Measured: across checkpoints these two barely differ (0.13-0.39% median), " +
                "and pairing a good UNet with a deliberately mismatched text encoder and " +
                "VAE produced visually identical output. So this is a fidelity limit, " +
                "not a correctness problem.",
            "What it does cost: prompt interpretation follows the template. A checkpoint " +
                "that relies on a heavily trained text encoder, or on clip skip 2, will " +
                "not behave exactly as it does elsewhere.",
        )

        Warning(
            "⚠ Anime checkpoints do not work yet",
            "They convert without error and render as saturated noise. The cause is " +
                "measured: the template's activation ranges come from a photoreal " +
                "checkpoint, and an anime model's weights diverge far past them -- up to " +
                "49x on individual tensors, against 1.02-1.12x for photoreal ones.",
            "This is not fixable by retrying. It needs a template built from anime " +
                "calibration data, which does not exist yet.",
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
            "• arm64, Android 12 or newer",
            "• About 4.8 GB of free RAM at peak -- 8 GB devices are unproven",
            "• About 4 GB of free storage while it runs",
            "• Each finished model is roughly 1.3 GB",
        )

        Section(
            "How to tell what went wrong",
            "Saturated, blotchy noise means the checkpoint sits outside the template's " +
                "range -- most likely an anime or heavily merged model.",
            "A model that will not load at all points at the fp16 stamp or the chip tier.",
            "Good image, but the prompt is interpreted oddly: that is the borrowed text " +
                "encoder.",
            "Colours slightly off, detail soft: that is the borrowed VAE.",
        )

        Section(
            "Tested on",
            "One device: Samsung Galaxy S25 Ultra (SM8750, Hexagon v79). Everything " +
                "above was measured there. Behaviour on other chips is projection.",
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

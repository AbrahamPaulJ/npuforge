package com.abrah.npuforge

import android.app.Activity
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlin.math.roundToInt
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme { Surface(Modifier.fillMaxSize()) { ConvertScreen() } }
        }
    }
}

@Composable
private fun ConvertScreen() {
    val context = LocalContext.current
    val state by ConvertService.state.collectAsState()
    var picked by remember { mutableStateOf<Uri?>(null) }
    var pickedName by remember { mutableStateOf("") }
    var modelName by remember { mutableStateOf("") }
    var report by remember { mutableStateOf<CheckpointInfo.Report?>(null) }
    val loras = remember { mutableStateListOf<Triple<Uri, String, Float>>() }  // uri, name, strength
    var nameEdited by remember { mutableStateOf(false) }

    /**
     * Suggests `<checkpoint>+<lora>@<strength>`.
     *
     * The LoRA belongs in the name because it is baked in: two strengths of the
     * same adapter are two different models, and "dreamshaper" twice in
     * Downloads tells you nothing about which is which. Stops suggesting the
     * moment the user types their own.
     */
    fun suggestedName(): String {
        fun clean(v: String) = v.substringBeforeLast(".").take(28)
            .replace(Regex("[^A-Za-z0-9_.-]"), "_").trim('_')
        val base = clean(pickedName)
        val adapters = loras.joinToString("") { (_, n, st) -> "+" + clean(n) + "@" + fmt(st) }
        return (base + adapters).take(60)
    }
    /**
     * Why the name is validated at all: it becomes a FILENAME in Downloads via
     * MediaStore. An invalid one fails at the very end, after ~2 minutes of
     * conversion, which is the worst possible moment to find out.
     */
    val nameError: Int? = when {
        modelName.isBlank() -> R.string.name_blank
        modelName.length > 80 -> R.string.name_long
        modelName.any { it in ("/:*?<>|" + Char(34) + Char(92)) } -> R.string.name_chars
        modelName.startsWith(".") -> R.string.name_dot
        modelName != modelName.trim() -> R.string.name_space
        else -> null
    }

    if (!nameEdited && picked != null) modelName = suggestedName()
    var inspectError by remember { mutableStateOf<String?>(null) }

    // Reading the safetensors HEADER is a short read, not a 2 GB copy -- so the
    // user learns what is in the file, and whether it can convert at all,
    // immediately rather than two minutes in.
    LaunchedEffect(picked) {
        report = null
        inspectError = null
        val uri = picked ?: return@LaunchedEffect
        try {
            report = withContext(Dispatchers.IO) { CheckpointInfo.inspect(context, uri) }
        } catch (e: Exception) {
            inspectError = e.message ?: e.javaClass.simpleName
        }
    }

    val picker = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null) {
            picked = uri
            pickedName = displayName(context as Activity, uri)
            nameEdited = false     // a new file gets a new suggestion
            loras.clear()          // and its own LoRA list
        }
    }

    val loraPicker = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null) loras.add(Triple(uri, displayName(context as Activity, uri), 0.8f))
    }

    Column(
        Modifier.padding(16.dp).verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text(stringResource(R.string.app_name), style = MaterialTheme.typography.headlineSmall)
        Text(stringResource(R.string.blurb), style = MaterialTheme.typography.bodyMedium)

        when (val s = state) {
            is ConvertService.State.Running -> {
                Card(Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text(
                            if (s.steps > 0) "${s.step}/${s.steps}  ${s.stage}" else s.stage,
                            style = MaterialTheme.typography.titleMedium,
                        )
                        // A real bar when the tool reports a percentage, an
                        // indeterminate one otherwise -- a fake bar that creeps
                        // is worse than an honest spinner.
                        val pct = s.detail.removeSuffix("%").toFloatOrNull()
                            ?.takeIf { s.detail.endsWith("%") }
                        if (pct != null) {
                            LinearProgressIndicator(
                                progress = { (pct / 100f).coerceIn(0f, 1f) },
                                modifier = Modifier.fillMaxWidth(),
                            )
                        } else {
                            LinearProgressIndicator(Modifier.fillMaxWidth())
                        }
                        if (s.detail.isNotBlank()) {
                            Text(s.detail, style = MaterialTheme.typography.bodyMedium)
                        }
                        // One Text per line: no embedded newlines to escape,
                        // and each line elides independently.
                        s.log.takeLast(4).forEach { line ->
                            Text(
                                line,
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                maxLines = 1, overflow = TextOverflow.Ellipsis,
                            )
                        }
                        Text(stringResource(R.string.keep_open), style = MaterialTheme.typography.bodySmall)
                    }
                }
            }

            is ConvertService.State.Done -> {
                Card(Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text(stringResource(R.string.done_title, s.seconds),
                            style = MaterialTheme.typography.titleMedium)
                        Text(stringResource(R.string.done_where, s.dir),
                            style = MaterialTheme.typography.bodySmall,
                            maxLines = 3, overflow = TextOverflow.Ellipsis)
                        Text(stringResource(R.string.done_note), style = MaterialTheme.typography.bodySmall)
                        TextButton(onClick = { ConvertService.reset() }) {
                            Text(stringResource(R.string.again))
                        }
                    }
                }
            }

            is ConvertService.State.Failed -> {
                Card(Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text(stringResource(R.string.failed_title),
                            style = MaterialTheme.typography.titleMedium,
                            color = MaterialTheme.colorScheme.error)
                        Text(s.message, style = MaterialTheme.typography.bodyMedium)
                        if (s.log.isNotBlank()) {
                            Text(s.log.takeLast(600), style = MaterialTheme.typography.bodySmall,
                                maxLines = 12, overflow = TextOverflow.Ellipsis)
                        }
                        TextButton(onClick = { ConvertService.reset() }) {
                            Text(stringResource(R.string.again))
                        }
                    }
                }
            }

            ConvertService.State.Idle -> {
                Button(onClick = { picker.launch(arrayOf("*/*")) }, modifier = Modifier.fillMaxWidth()) {
                    Text(stringResource(R.string.pick))
                }
                if (picked != null) {
                    Text(pickedName, style = MaterialTheme.typography.bodySmall)
                    inspectError?.let {
                        Text(it, style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.error)
                    }
                    report?.let { CheckpointCard(it) }
                    OutlinedTextField(
                        value = modelName,
                        onValueChange = { modelName = it; nameEdited = true },
                        label = { Text(stringResource(R.string.model_name)) },
                        singleLine = true,
                        isError = nameError != null,
                        supportingText = nameError?.let { { Text(stringResource(it)) } },
                        modifier = Modifier.fillMaxWidth(),
                    )
                    LoraList(loras) { loraPicker.launch(arrayOf("*/*")) }
                    Button(
                        onClick = {
                            ConvertService.start(
                                context, picked!!, modelName,
                                loras.map { it.first to it.third },
                            )
                        },
                        // Never offer to convert a file already known not to fit:
                        // the failure would arrive after a 2 GB copy.
                        enabled = nameError == null && report?.convertible == true,
                        modifier = Modifier.fillMaxWidth(),
                    ) { Text(stringResource(R.string.convert)) }
                }
                Text(stringResource(R.string.caveats), style = MaterialTheme.typography.bodySmall)
            }
        }
    }
}


/** What the picked checkpoint contains, and what conversion will keep. */
@Composable
private fun CheckpointCard(r: CheckpointInfo.Report) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(
                stringResource(R.string.inside_title, r.architecture, r.totalTensors),
                style = MaterialTheme.typography.titleSmall,
            )
            PartRow(stringResource(R.string.part_unet), r.unet, kept = true)
            PartRow(stringResource(R.string.part_vae), r.vae, kept = false)
            PartRow(stringResource(R.string.part_clip), r.clip, kept = false)
            if (r.ema.present) {
                Text(stringResource(R.string.part_ema, r.ema.tensors),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            if (r.dtypes.isNotEmpty()) {
                Text(r.dtypes.entries.joinToString(", ") { "${it.value} x ${it.key}" },
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            r.fatal?.let {
                Text(it, style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.error)
            }
            if (r.fatal == null && r.missing.isNotEmpty()) {
                Text(stringResource(R.string.missing_title, r.missing.size),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.error)
                Text(r.missing.take(3).joinToString(", "),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            if (r.convertible) {
                Text(stringResource(R.string.will_swap),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
    }
}

@Composable
private fun PartRow(label: String, p: CheckpointInfo.Part, kept: Boolean) {
    val mark = when {
        !p.present -> "—"
        kept -> "✓ converted"
        else -> "✗ not converted"
    }
    val size = if (p.bytes > 0) " · ${p.bytes / 1_000_000} MB" else ""
    Text("$label: $mark$size", style = MaterialTheme.typography.bodyMedium)
}

/**
 * Optional LoRAs, merged into the weights at conversion time.
 *
 * Several stack in one pass -- the merge is additive -- so this is a list, not
 * a single slot. Strength is baked into the model, which is why it is typed
 * here rather than offered as a slider at generation time: changing it means
 * converting again.
 */
@Composable
private fun LoraList(
    loras: androidx.compose.runtime.snapshots.SnapshotStateList<Triple<Uri, String, Float>>,
    onAdd: () -> Unit,
) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        loras.forEachIndexed { i, (uri, name, strength) ->
            Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        name, style = MaterialTheme.typography.bodyMedium,
                        maxLines = 1, overflow = TextOverflow.Ellipsis,
                        modifier = Modifier.weight(1f),
                    )
                    Text(fmt(strength), style = MaterialTheme.typography.bodyMedium)
                    TextButton(onClick = { loras.removeAt(i) }) {
                        Text(stringResource(R.string.remove))
                    }
                }
                // -1..2 in 0.1 steps. Negative is deliberate: detail-tweaker
                // style adapters are used inverted, and a slider that cannot go
                // below zero would quietly forbid that. Above 2 a merge tends to
                // leave the template's activation ranges and produce noise, so
                // the bound is a guard rail, not decoration.
                Slider(
                    value = strength,
                    onValueChange = { loras[i] = Triple(uri, name, (it * 10).roundToInt() / 10f) },
                    valueRange = -1f..2f,
                    steps = 29,
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        }
        OutlinedButton(onClick = onAdd, modifier = Modifier.fillMaxWidth()) {
            Text(stringResource(R.string.add_lora))
        }
        if (loras.isNotEmpty()) {
            Text(stringResource(R.string.lora_note),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

/** One decimal, so a name reads "@0.8" and not "@0.800000011920929". */
private fun fmt(v: Float): String = ((v * 10).roundToInt() / 10f).toString()

private fun displayName(activity: Activity, uri: Uri): String {
    activity.contentResolver.query(uri, null, null, null, null)?.use { c ->
        val i = c.getColumnIndex(OpenableColumns.DISPLAY_NAME)
        if (i >= 0 && c.moveToFirst()) return c.getString(i)
    }
    return uri.lastPathSegment ?: "checkpoint"
}

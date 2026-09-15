package com.abrah.npuforge

import android.Manifest
import android.app.ActivityManager
import android.app.NotificationManager
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.os.SystemClock
import android.os.PowerManager
import android.provider.OpenableColumns
import android.provider.Settings
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Tab
import androidx.compose.material3.PrimaryTabRow
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.runtime.saveable.listSaver
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.saveable.rememberSaveableStateHolder
import androidx.compose.runtime.snapshots.SnapshotStateList
import androidx.compose.runtime.toMutableStateList
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.res.pluralStringResource
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.core.net.toUri
import com.abrah.npuforge.ui.InfoScreen
import com.abrah.npuforge.ui.ComponentsScreen
import java.io.File
import kotlinx.coroutines.launch
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.CancellationException
import kotlin.math.roundToInt
import kotlin.time.Duration.Companion.milliseconds
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // A killed process cannot run service cleanup; clear its retained Android service record on reopen.
        if (ConvertService.state.value == ConvertService.State.Idle) {
            stopService(Intent(this, ConvertService::class.java))
            getSystemService(NotificationManager::class.java).cancel(ConvertService.NOTE_ID)
        }
        enableEdgeToEdge()
        setContent {
            val conversion by ConvertService.state.collectAsState()
            DisposableEffect(conversion is ConvertService.State.Running) {
                if (conversion is ConvertService.State.Running) {
                    window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                }
                onDispose { window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON) }
            }
            val colors = if (isSystemInDarkTheme()) dynamicDarkColorScheme(this)
                else dynamicLightColorScheme(this)
            MaterialTheme(colorScheme = colors) {
                Surface(Modifier.fillMaxSize()) { AppScreen() }
            }
        }
    }
}

/**
 * Two tabs: the job, and the truth about the job.
 *
 * The Info tab is not an "about" page. This converter has real limits that look
 * like defects when you meet them blind -- a model that will not load on a
 * phone newer than the one it targets, an anime checkpoint that converts
 * cleanly into noise -- and the only place a user can read about them is in the
 * app.
 */
@Composable
private fun AppScreen() {
    var tab by rememberSaveable { mutableIntStateOf(0) }
    val savedTabs = rememberSaveableStateHolder()
    val focus = LocalFocusManager.current
    BackHandler(enabled = tab != 0) { tab = 0 }
    Column(Modifier.fillMaxSize().safeDrawingPadding()) {
        PrimaryTabRow(selectedTabIndex = tab) {
            Tab(selected = tab == 0, onClick = { focus.clearFocus(); tab = 0 },
                text = { Text(stringResource(R.string.tab_convert)) })
            Tab(selected = tab == 1, onClick = { focus.clearFocus(); tab = 1 },
                text = { Text(stringResource(R.string.tab_info)) })
            Tab(selected = tab == 2, onClick = { focus.clearFocus(); tab = 2 },
                text = { Text(stringResource(R.string.tab_components)) })
        }
        savedTabs.SaveableStateProvider(tab) {
            when (tab) {
                0 -> ConvertScreen()
                1 -> InfoScreen()
                else -> ComponentsScreen { tab = 0 }
            }
        }
    }
}

@Composable
private fun ConvertScreen() {
    val context = LocalContext.current
    val state by ConvertService.state.collectAsState()
    val exportScope = rememberCoroutineScope()
    var exportSource by rememberSaveable { mutableStateOf("") }
    var exportError by remember { mutableStateOf<String?>(null) }
    val latestReport = remember(state) {
        File(context.filesDir, "conversion-reports").listFiles()?.maxByOrNull { it.lastModified() }
    }
    val exportReport = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("text/plain")) { uri ->
        if (uri != null) exportScope.launch {
            try {
                withContext(Dispatchers.IO) {
                    File(exportSource).inputStream().use { input ->
                        context.contentResolver.openOutputStream(uri)!!.use { output -> input.copyTo(output) }
                    }
                }
                exportError = null
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                exportError = e.message ?: e.javaClass.simpleName
            }
        }
    }
    val activityManager = remember(context) { context.getSystemService(ActivityManager::class.java) }
    var memory by remember(activityManager) {
        mutableStateOf(ActivityManager.MemoryInfo().also(activityManager::getMemoryInfo))
    }
    LaunchedEffect(activityManager) {
        while (true) {
            delay(1000.milliseconds)
            memory = ActivityManager.MemoryInfo().also(activityManager::getMemoryInfo)
        }
    }
    val power = remember(context) { context.getSystemService(PowerManager::class.java) }
    var unrestricted by remember { mutableStateOf(power.isIgnoringBatteryOptimizations(context.packageName)) }
    val batteryAccess = rememberLauncherForActivityResult(ActivityResultContracts.StartActivityForResult()) {
        unrestricted = power.isIgnoringBatteryOptimizations(context.packageName)
    }
    var picked by rememberSaveable { mutableStateOf<Uri?>(null) }
    var pickedName by rememberSaveable { mutableStateOf("") }
    var editedName by rememberSaveable { mutableStateOf<String?>(null) }
    var report by remember { mutableStateOf<CheckpointInfo.Report?>(null) }
    val loras = rememberSaveable(
        saver = listSaver<SnapshotStateList<Triple<Uri, String, Float>>, Any>(
            save = { items -> items.flatMap { listOf(it.first.toString(), it.second, it.third) } },
            restore = { values -> values.chunked(3).map {
                Triple((it[0] as String).toUri(), it[1] as String, it[2] as Float)
            }.toMutableStateList() },
        ),
    ) { mutableStateListOf<Triple<Uri, String, Float>>() }
    // Derive the suggestion without writing state during composition.
    val names = (listOf(pickedName) + loras.map { it.second }).map {
        it.substringBeforeLast(".").take(28).replace(Regex("[^A-Za-z0-9_.-]"), "_").trim('_')
    }
    val modelName = editedName ?: (names.first() + loras.mapIndexed { i, lora ->
        "+${names[i + 1]}@${lora.third}"
    }.joinToString("")).take(60)
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
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            inspectError = e.message ?: e.javaClass.simpleName
        }
    }

    val picker = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null) {
            context.contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
            picked = uri
            pickedName = displayName(context, uri)
            editedName = null      // a new file gets a new suggestion
            loras.clear()          // and its own LoRA list
        }
    }

    val loraPicker = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null) {
            context.contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
            loras.add(Triple(uri, displayName(context, uri), 0.8f))
        }
    }
    val notifications = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) {}

    Column(
        Modifier.padding(16.dp).verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text(stringResource(R.string.app_name), style = MaterialTheme.typography.headlineSmall)
        Text(stringResource(R.string.blurb), style = MaterialTheme.typography.bodyMedium)
        Card(Modifier.fillMaxWidth()) {
            val used = memory.totalMem - memory.availMem
            val fraction = used.toFloat() / memory.totalMem
            Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Text(
                    stringResource(R.string.phone_ram_used, (fraction * 100).roundToInt()),
                    style = MaterialTheme.typography.titleSmall,
                )
                LinearProgressIndicator(progress = { fraction }, modifier = Modifier.fillMaxWidth())
                Text(
                    stringResource(
                        R.string.phone_ram_details,
                        used / 1073741824.0, memory.totalMem / 1073741824.0, memory.availMem / 1073741824.0,
                    ),
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }

        if (latestReport != null) {
            OutlinedButton(onClick = {
                exportSource = latestReport.path
                exportReport.launch(latestReport.name)
            }) { Text(stringResource(R.string.export_conversion_report)) }
        }
        exportError?.let { Text(it, color = MaterialTheme.colorScheme.error) }

        when (val s = state) {
            is ConvertService.State.Running -> {
                var elapsed by remember(s.startedAt) {
                    mutableLongStateOf((SystemClock.elapsedRealtime() - s.startedAt) / 1000)
                }
                LaunchedEffect(s.startedAt) {
                    while (true) {
                        delay(1000.milliseconds)
                        elapsed = (SystemClock.elapsedRealtime() - s.startedAt) / 1000
                    }
                }
                Card(Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text(
                            if (s.steps > 0) "${s.step}/${s.steps}  ${s.stage}" else s.stage,
                            style = MaterialTheme.typography.titleMedium,
                        )
                        Text(stringResource(R.string.elapsed, elapsed / 60, elapsed % 60), style = MaterialTheme.typography.bodySmall)
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
                        Text(stringResource(R.string.keep_open), style = MaterialTheme.typography.bodySmall)
                    }
                }
            }

            is ConvertService.State.Done -> {
                Card(Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text(stringResource(R.string.done_title, s.seconds / 60, s.seconds % 60),
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
                        TextButton(onClick = { ConvertService.reset() }) {
                            Text(stringResource(R.string.again))
                        }
                    }
                }
            }

            ConvertService.State.Idle, is ConvertService.State.ComponentsDone -> {
                if (!unrestricted) {
                    OutlinedButton(onClick = {
                        batteryAccess.launch(Intent(
                            Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS,
                            "package:${context.packageName}".toUri(),
                        ))
                    }, modifier = Modifier.fillMaxWidth()) {
                        Text(stringResource(R.string.unrestricted_background))
                    }
                }
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
                        onValueChange = { editedName = it },
                        label = { Text(stringResource(R.string.model_name)) },
                        singleLine = true,
                        isError = nameError != null,
                        supportingText = nameError?.let { { Text(stringResource(it)) } },
                        modifier = Modifier.fillMaxWidth(),
                    )
                    LoraList(loras) { loraPicker.launch(arrayOf("*/*")) }
                    Button(
                        onClick = {
                            val model = report!!.model
                            notifications.launch(Manifest.permission.POST_NOTIFICATIONS)
                            ConvertService.start(
                                context, picked!!, modelName, loras.map { it.first to it.third },
                                model = model,
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
        val log = when (val s = state) {
            is ConvertService.State.Running -> s.log.joinToString("\n")
            is ConvertService.State.Done -> s.log
            is ConvertService.State.Failed -> s.log
            ConvertService.State.Idle, is ConvertService.State.ComponentsDone -> null
        }
        if (log != null) {
            val logScroll = rememberScrollState()
            var followLog by remember { mutableStateOf(true) }
            LaunchedEffect(logScroll.maxValue, followLog) {
                if (followLog) logScroll.scrollTo(logScroll.maxValue)
            }
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(stringResource(R.string.conversion_log), style = MaterialTheme.typography.titleMedium)
                    TextButton(onClick = { followLog = !followLog }) {
                        Text(stringResource(if (followLog) R.string.pause_scroll else R.string.follow_log))
                    }
                    SelectionContainer {
                        Text(
                            log.ifEmpty { stringResource(R.string.waiting_output) },
                            modifier = Modifier.fillMaxWidth().height(240.dp).verticalScroll(logScroll),
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }
                }
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
                pluralStringResource(R.plurals.inside_title, r.totalTensors, r.architecture, r.totalTensors),
                style = MaterialTheme.typography.titleSmall,
            )
            PartRow(stringResource(R.string.part_unet), r.unet, kept = true)
            PartRow(stringResource(R.string.part_vae), r.vae, kept = false)
            PartRow(stringResource(R.string.part_clip), r.clip, kept = false)
            if (r.model == CheckpointInfo.Model.SDXL) {
                Text(stringResource(R.string.sdxl_components), style = MaterialTheme.typography.bodySmall)
            }
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
                Text(pluralStringResource(R.plurals.missing_title, r.missing.size, r.missing.size),
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
    loras: SnapshotStateList<Triple<Uri, String, Float>>,
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
                    Text(strength.toString(), style = MaterialTheme.typography.bodyMedium)
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

private fun displayName(context: Context, uri: Uri): String {
    context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
        val i = c.getColumnIndex(OpenableColumns.DISPLAY_NAME)
        if (i >= 0 && c.moveToFirst()) return c.getString(i)
    }
    return uri.lastPathSegment ?: "checkpoint"
}

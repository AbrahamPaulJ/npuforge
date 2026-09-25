package com.abrah.npuforge

import android.Manifest
import android.annotation.SuppressLint
import android.app.ActivityManager
import android.app.LocaleManager
import android.app.NotificationManager
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.LocaleList
import android.os.PowerManager
import android.os.SystemClock
import android.provider.OpenableColumns
import android.provider.Settings
import android.view.WindowManager
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CenterAlignedTopAppBar
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.PrimaryTabRow
import androidx.compose.material3.Slider
import androidx.compose.material3.SuggestionChip
import androidx.compose.material3.SuggestionChipDefaults
import androidx.compose.material3.Surface
import androidx.compose.material3.Tab
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.listSaver
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.saveable.rememberSaveableStateHolder
import androidx.compose.runtime.setValue
import androidx.compose.runtime.snapshots.SnapshotStateList
import androidx.compose.runtime.toMutableStateList
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.res.pluralStringResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.core.net.toUri
import com.abrah.npuforge.ui.InfoScreen
import com.abrah.npuforge.ui.theme.NpuForgeTheme
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.zip.Deflater
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream
import java.util.zip.ZipOutputStream
import kotlin.math.roundToInt
import kotlin.time.Duration.Companion.milliseconds

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
            NpuForgeTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background,
                ) {
                    AppScreen()
                }
            }
        }
    }
}

/**
 * Three tabs: the job, the truth about the job, and the shared backups.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun AppScreen() {
    var tab by rememberSaveable { mutableIntStateOf(0) }
    val savedTabs = rememberSaveableStateHolder()
    val focus = LocalFocusManager.current

    BackHandler(enabled = tab != 0) { tab = 0 }
    Column(Modifier.fillMaxSize().safeDrawingPadding()) {
        CenterAlignedTopAppBar(
            title = {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        stringResource(R.string.app_name),
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.ExtraBold,
                    )
                    Spacer(Modifier.width(8.dp))
                    SuggestionChip(
                        onClick = {},
                        label = { Text("NPU: ${Build.SOC_MODEL}", style = MaterialTheme.typography.labelSmall, fontWeight = FontWeight.Bold) },
                        colors = SuggestionChipDefaults.suggestionChipColors(
                            containerColor = MaterialTheme.colorScheme.primaryContainer,
                            labelColor = MaterialTheme.colorScheme.onPrimaryContainer,
                        ),
                        border = null,
                    )
                }
            },
            colors = TopAppBarDefaults.topAppBarColors(
                containerColor = MaterialTheme.colorScheme.background,
            ),
        )

        PrimaryTabRow(
            selectedTabIndex = tab,
            containerColor = MaterialTheme.colorScheme.background,
        ) {
            Tab(
                selected = tab == 0,
                onClick = { focus.clearFocus(); tab = 0 },
                text = { Text(stringResource(R.string.tab_convert), fontWeight = if (tab == 0) FontWeight.Bold else FontWeight.Normal) },
            )
            Tab(
                selected = tab == 1,
                onClick = { focus.clearFocus(); tab = 1 },
                text = { Text(stringResource(R.string.tab_info), fontWeight = if (tab == 1) FontWeight.Bold else FontWeight.Normal) },
            )
            Tab(
                selected = tab == 2,
                onClick = { focus.clearFocus(); tab = 2 },
                text = { Text(stringResource(R.string.tab_utility), fontWeight = if (tab == 2) FontWeight.Bold else FontWeight.Normal) },
            )
        }

        savedTabs.SaveableStateProvider(tab) {
            when (tab) {
                0 -> ConvertScreen()
                1 -> InfoScreen()
                else -> UtilityScreen()
            }
        }
    }
}

private fun setAppLocale(context: Context, languageTag: String?) {
    val localeManager = context.getSystemService(LocaleManager::class.java)
    if (languageTag == null) {
        localeManager.applicationLocales = LocaleList.getEmptyLocaleList()
    } else {
        localeManager.applicationLocales = LocaleList.forLanguageTags(languageTag)
    }
}

@SuppressLint("BatteryLife")
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
    ) { mutableStateListOf() }

    val names = (listOf(pickedName) + loras.map { it.second }).map {
        it.substringBeforeLast(".").take(28).replace(Regex("[^A-Za-z0-9_.-]"), "_").trim('_')
    }
    val modelName = editedName ?: (names.first() + loras.mapIndexed { i, lora ->
        "+${names[i + 1]}@${lora.third}"
    }.joinToString("")).take(60)

    val nameError: Int? = when {
        modelName.isBlank() -> R.string.name_blank
        modelName.length > 80 -> R.string.name_long
        modelName.any { it in ("/:*?<>|" + Char(34) + Char(92)) } -> R.string.name_chars
        modelName.startsWith(".") -> R.string.name_dot
        modelName != modelName.trim() -> R.string.name_space
        else -> null
    }

    var inspectError by remember { mutableStateOf<String?>(null) }
    val isVaeMissing = remember {
        val vaeDir = File(context.filesDir, "vae_sdxl")
        !File(vaeDir, "vae_decoder.bin").isFile || !File(vaeDir, "vae_encoder.bin").isFile
    }
    var showVaeDownloadDialog by rememberSaveable { mutableStateOf(isVaeMissing) }
    var pendingConversionModel by remember { mutableStateOf<CheckpointInfo.Model?>(null) }

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
            editedName = null
            loras.clear()
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
        Modifier
            .padding(horizontal = 16.dp, vertical = 12.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text(
            stringResource(R.string.blurb),
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        // RAM status card
        val used = memory.totalMem - memory.availMem
        val fraction = (used.toFloat() / memory.totalMem).coerceIn(0f, 1f)
        val ramStatusText = when {
            fraction < 0.70f -> stringResource(R.string.ram_status_optimal)
            fraction < 0.85f -> stringResource(R.string.ram_status_moderate)
            else -> stringResource(R.string.ram_status_high)
        }
        val ramChipColor = when {
            fraction < 0.70f -> MaterialTheme.colorScheme.tertiary
            fraction < 0.85f -> MaterialTheme.colorScheme.primary
            else -> MaterialTheme.colorScheme.error
        }

        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.45f),
            ),
            border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(alpha = 0.2f)),
        ) {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                FlowRow(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalArrangement = Arrangement.spacedBy(4.dp),
                ) {
                    Text(
                        stringResource(R.string.phone_ram_used, (fraction * 100).roundToInt()),
                        modifier = Modifier.align(Alignment.CenterVertically),
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.SemiBold,
                    )
                    SuggestionChip(
                        onClick = {},
                        label = { Text(ramStatusText, style = MaterialTheme.typography.labelSmall) },
                        colors = SuggestionChipDefaults.suggestionChipColors(
                            containerColor = ramChipColor.copy(alpha = 0.15f),
                            labelColor = ramChipColor,
                        ),
                        border = null,
                    )
                }
                LinearProgressIndicator(
                    progress = { fraction },
                    modifier = Modifier.fillMaxWidth().height(6.dp),
                )
                Text(
                    stringResource(
                        R.string.phone_ram_details,
                        used / 1073741824.0, memory.totalMem / 1073741824.0, memory.availMem / 1073741824.0,
                    ),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }

        if (latestReport != null) {
            OutlinedButton(
                onClick = {
                    exportSource = latestReport.path
                    exportReport.launch(latestReport.name)
                },
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(12.dp),
            ) {
                Text(stringResource(R.string.export_conversion_report))
            }
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
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(16.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.2f),
                    ),
                    border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(alpha = 0.35f)),
                ) {
                    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                        // Give both labels the card width; long stages must not
                        // squeeze the status into a column one character wide.
                        Text(
                            if (s.steps > 0) "${s.step}/${s.steps} · ${s.stage}" else s.stage,
                            modifier = Modifier.fillMaxWidth(),
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.primary,
                        )
                        Surface(
                            shape = RoundedCornerShape(8.dp),
                            color = MaterialTheme.colorScheme.primary.copy(alpha = 0.15f),
                            contentColor = MaterialTheme.colorScheme.primary,
                        ) {
                            Text(
                                stringResource(R.string.active_process),
                                modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
                                style = MaterialTheme.typography.labelSmall,
                            )
                        }
                        Text(
                            stringResource(R.string.elapsed, elapsed / 60, elapsed % 60),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                        val pct = s.detail.removeSuffix("%").toFloatOrNull()
                            ?.takeIf { s.detail.endsWith("%") }
                        if (pct != null) {
                            LinearProgressIndicator(
                                progress = { (pct / 100f).coerceIn(0f, 1f) },
                                modifier = Modifier.fillMaxWidth().height(8.dp),
                            )
                        } else {
                            LinearProgressIndicator(
                                modifier = Modifier.fillMaxWidth().height(8.dp),
                            )
                        }
                        if (s.detail.isNotBlank()) {
                            Text(s.detail, style = MaterialTheme.typography.bodyMedium)
                        }
                        Text(
                            stringResource(R.string.keep_open),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
            }

            is ConvertService.State.Done -> {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(16.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.tertiary.copy(alpha = 0.12f),
                    ),
                    border = BorderStroke(1.dp, MaterialTheme.colorScheme.tertiary.copy(alpha = 0.4f)),
                ) {
                    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                        Text(
                            stringResource(R.string.done_title, s.seconds / 60, s.seconds % 60),
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.tertiary,
                        )
                        Text(
                            stringResource(R.string.done_where, s.dir),
                            style = MaterialTheme.typography.bodySmall,
                            maxLines = 3,
                            overflow = TextOverflow.Ellipsis,
                        )
                        FilledTonalButton(
                            onClick = { ConvertService.reset() },
                            shape = RoundedCornerShape(12.dp),
                        ) {
                            Text(stringResource(R.string.again))
                        }
                    }
                }
            }

            is ConvertService.State.Failed -> {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(16.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.errorContainer.copy(alpha = 0.25f),
                    ),
                    border = BorderStroke(1.dp, MaterialTheme.colorScheme.error.copy(alpha = 0.4f)),
                ) {
                    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                        Text(
                            stringResource(R.string.failed_title),
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.error,
                        )
                        Text(s.message, style = MaterialTheme.typography.bodyMedium)
                        FilledTonalButton(
                            onClick = { ConvertService.reset() },
                            shape = RoundedCornerShape(12.dp),
                        ) {
                            Text(stringResource(R.string.again))
                        }
                    }
                }
            }

            ConvertService.State.Idle -> {
                if (!unrestricted) {
                    OutlinedButton(
                        onClick = {
                            batteryAccess.launch(Intent(
                                Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS,
                                "package:${context.packageName}".toUri(),
                            ))
                        },
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(12.dp),
                    ) {
                        Text(stringResource(R.string.unrestricted_background))
                    }
                }

                Button(
                    onClick = { picker.launch(arrayOf("*/*")) },
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(14.dp),
                ) {
                    Text(
                        stringResource(R.string.pick),
                        style = MaterialTheme.typography.titleMedium,
                        modifier = Modifier.padding(vertical = 4.dp),
                    )
                }

                if (picked != null) {
                    Text(
                        pickedName,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.primary,
                    )
                    inspectError?.let {
                        Text(
                            it,
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.error,
                        )
                    }
                    report?.let { CheckpointCard(it) }

                    OutlinedTextField(
                        value = modelName,
                        onValueChange = { editedName = it },
                        label = { Text(stringResource(R.string.model_name)) },
                        singleLine = true,
                        shape = RoundedCornerShape(12.dp),
                        isError = nameError != null,
                        supportingText = nameError?.let { { Text(stringResource(it)) } },
                        modifier = Modifier.fillMaxWidth(),
                    )

                    LoraList(loras) { loraPicker.launch(arrayOf("*/*")) }

                    val startConversion = {
                        val model = report!!.model
                        notifications.launch(Manifest.permission.POST_NOTIFICATIONS)
                        ConvertService.start(
                            context, picked!!, modelName, loras.map { it.first to it.third },
                            model = model,
                        )
                    }

                    Button(
                        onClick = {
                            val vaeDir = File(context.filesDir, "vae_sdxl")
                            val missing = !File(vaeDir, "vae_decoder.bin").isFile || !File(vaeDir, "vae_encoder.bin").isFile
                            if (report?.model == CheckpointInfo.Model.SDXL && missing) {
                                pendingConversionModel = CheckpointInfo.Model.SDXL
                                showVaeDownloadDialog = true
                            } else {
                                startConversion()
                            }
                        },
                        enabled = nameError == null && report?.convertible == true,
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(14.dp),
                    ) {
                        Text(
                            stringResource(R.string.convert),
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.padding(vertical = 4.dp),
                        )
                    }
                }

                if (showVaeDownloadDialog) {
                    AlertDialog(
                        onDismissRequest = {
                            showVaeDownloadDialog = false
                            pendingConversionModel = null
                        },
                        title = { Text(stringResource(R.string.vae_download_dialog_title)) },
                        text = { Text(stringResource(R.string.vae_download_dialog_message)) },
                        confirmButton = {
                            TextButton(onClick = {
                                showVaeDownloadDialog = false
                                notifications.launch(Manifest.permission.POST_NOTIFICATIONS)
                                val pending = pendingConversionModel
                                pendingConversionModel = null
                                if (pending != null) {
                                    val model = report!!.model
                                    ConvertService.start(
                                        context, picked!!, modelName, loras.map { it.first to it.third },
                                        model = model,
                                    )
                                } else {
                                    ConvertService.downloadVae(context)
                                }
                            }) {
                                Text(stringResource(R.string.vae_download_dialog_confirm))
                            }
                        },
                        dismissButton = {
                            TextButton(onClick = {
                                showVaeDownloadDialog = false
                                pendingConversionModel = null
                            }) {
                                Text(stringResource(R.string.vae_download_dialog_cancel))
                            }
                        },
                    )
                }
            }
        }

        val log = when (val s = state) {
            is ConvertService.State.Running -> s.log.joinToString("\n")
            is ConvertService.State.Done -> s.log
            is ConvertService.State.Failed -> s.log
            ConvertService.State.Idle -> null
        }
        if (log != null) {
            val logScroll = rememberScrollState()
            var followLog by remember { mutableStateOf(true) }
            LaunchedEffect(logScroll.maxValue, followLog) {
                if (followLog) logScroll.scrollTo(logScroll.maxValue)
            }
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(
                    containerColor = Color(0xFF0F172A),
                ),
                border = BorderStroke(1.dp, Color(0xFF334155)),
            ) {
                Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(
                        stringResource(R.string.conversion_log),
                        modifier = Modifier.fillMaxWidth(),
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.SemiBold,
                        color = Color(0xFFE2E8F0),
                    )
                    FlowRow(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp, Alignment.End),
                        verticalArrangement = Arrangement.spacedBy(4.dp),
                    ) {
                        TextButton(onClick = {
                            val clipboard = context.getSystemService(ClipboardManager::class.java)
                            clipboard?.setPrimaryClip(ClipData.newPlainText("npuforge log", log))
                            Toast.makeText(context, R.string.log_copied, Toast.LENGTH_SHORT).show()
                        }) {
                            Text(
                                stringResource(R.string.copy_log),
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.primary,
                            )
                        }
                        TextButton(onClick = { followLog = !followLog }) {
                            Text(
                                stringResource(if (followLog) R.string.pause_scroll else R.string.follow_log),
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.primary,
                            )
                        }
                    }
                    SelectionContainer {
                        Text(
                            log.ifEmpty { stringResource(R.string.waiting_output) },
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(220.dp)
                                .background(Color(0xFF020617), RoundedCornerShape(8.dp))
                                .padding(10.dp)
                                .verticalScroll(logScroll),
                            style = MaterialTheme.typography.bodySmall.copy(
                                fontFamily = FontFamily.Monospace,
                                color = Color(0xFF94A3B8),
                            ),
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
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
        ),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(alpha = 0.25f)),
    ) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            FlowRow(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                Text(
                    pluralStringResource(R.plurals.inside_title, r.totalTensors, r.architecture, r.totalTensors),
                    modifier = Modifier.align(Alignment.CenterVertically),
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.SemiBold,
                )
                SuggestionChip(
                    onClick = {},
                    label = { Text(r.architecture, style = MaterialTheme.typography.labelSmall) },
                    colors = SuggestionChipDefaults.suggestionChipColors(
                        containerColor = MaterialTheme.colorScheme.primary.copy(alpha = 0.15f),
                        labelColor = MaterialTheme.colorScheme.primary,
                    ),
                    border = null,
                )
            }

            PartRow(stringResource(R.string.part_unet), r.unet)
            PartRow(stringResource(R.string.part_vae), r.vae)
            PartRow(stringResource(R.string.part_clip), r.clip)
            if (r.ema.present) {
                Text(
                    stringResource(R.string.part_ema, r.ema.tensors),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            if (r.dtypes.isNotEmpty()) {
                Text(
                    r.dtypes.entries.joinToString(", ") { "${it.value} x ${it.key}" },
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            r.fatal?.let {
                Text(
                    it,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.error,
                )
            }
            if (r.fatal == null && r.missing.isNotEmpty()) {
                Text(
                    pluralStringResource(R.plurals.missing_title, r.missing.size, r.missing.size),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.error,
                )
                Text(
                    r.missing.take(3).joinToString(", "),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

@Composable
private fun PartRow(label: String, p: CheckpointInfo.Part) {
    val size = if (p.bytes > 0) " · ${p.bytes / 1_000_000} MB" else ""

    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            text = "$label$size",
            style = MaterialTheme.typography.bodyMedium,
            modifier = Modifier.weight(1f),
        )
        if (!p.present) {
            Text(
                text = stringResource(R.string.part_status_absent),
                style = MaterialTheme.typography.bodySmall,
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

/**
 * Optional LoRAs, merged into the weights at conversion time.
 */
@Composable
private fun LoraList(
    loras: SnapshotStateList<Triple<Uri, String, Float>>,
    onAdd: () -> Unit,
) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        if (loras.isNotEmpty()) {
            Text(
                stringResource(R.string.lora_adapters),
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.SemiBold,
            )
        }
        loras.forEachIndexed { i, (uri, name, strength) ->
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.35f),
                ),
            ) {
                Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                        Text(
                            name,
                            style = MaterialTheme.typography.bodyMedium,
                            fontWeight = FontWeight.Medium,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                            modifier = Modifier.weight(1f),
                        )
                        SuggestionChip(
                            onClick = {},
                            label = { Text(stringResource(R.string.lora_weight, strength), style = MaterialTheme.typography.labelSmall) },
                            colors = SuggestionChipDefaults.suggestionChipColors(
                                containerColor = MaterialTheme.colorScheme.secondaryContainer,
                                labelColor = MaterialTheme.colorScheme.onSecondaryContainer,
                            ),
                            border = null,
                        )
                        TextButton(onClick = { loras.removeAt(i) }) {
                            Text(stringResource(R.string.remove), color = MaterialTheme.colorScheme.error)
                        }
                    }
                    Slider(
                        value = strength,
                        onValueChange = { loras[i] = Triple(uri, name, (it * 10).roundToInt() / 10f) },
                        valueRange = -1f..2f,
                        steps = 29,
                        modifier = Modifier.fillMaxWidth(),
                    )
                }
            }
        }
        OutlinedButton(
            onClick = onAdd,
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(12.dp),
        ) {
            Text(stringResource(R.string.add_lora))
        }
        if (loras.isNotEmpty()) {
            Text(
                stringResource(R.string.lora_note),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
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

@Composable
fun UtilityScreen() {
    val context = LocalContext.current
    var showLanguageMenu by remember { mutableStateOf(false) }
    val vaeNotFoundStr = stringResource(R.string.vae_not_found)
    val vaeExportedStr = stringResource(R.string.vae_exported)
    val importFailedStr = stringResource(R.string.import_failed)
    val vaeImportedStr = stringResource(R.string.vae_imported)
    val cacheClearedStr = stringResource(R.string.cache_cleared)

    val exportLauncher = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("application/zip")) { uri ->
        if (uri != null) {
            val vaeDir = File(context.filesDir, "vae_sdxl")
            val encoder = File(vaeDir, "vae_encoder.bin")
            val decoder = File(vaeDir, "vae_decoder.bin")
            if (!encoder.exists() || !decoder.exists()) {
                Toast.makeText(context, vaeNotFoundStr, Toast.LENGTH_SHORT).show()
                return@rememberLauncherForActivityResult
            }
            try {
                context.contentResolver.openOutputStream(uri)?.use { os ->
                    ZipOutputStream(os.buffered()).use { zip ->
                        zip.setLevel(Deflater.NO_COMPRESSION)
                        zip.putNextEntry(ZipEntry("vae_encoder.bin"))
                        encoder.inputStream().use { it.copyTo(zip) }
                        zip.closeEntry()
                        
                        zip.putNextEntry(ZipEntry("vae_decoder.bin"))
                        decoder.inputStream().use { it.copyTo(zip) }
                        zip.closeEntry()
                    }
                }
                Toast.makeText(context, vaeExportedStr, Toast.LENGTH_SHORT).show()
            } catch (_: Exception) {
                Toast.makeText(context, importFailedStr, Toast.LENGTH_SHORT).show()
            }
        }
    }

    val importLauncher = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            try {
                val vaeDir = File(context.filesDir, "vae_sdxl").apply { mkdirs() }
                context.contentResolver.openInputStream(uri)?.use { input ->
                    ZipInputStream(input.buffered()).use { zip ->
                        var entry = zip.nextEntry
                        while (entry != null) {
                            if (entry.name == "vae_encoder.bin" || entry.name == "vae_decoder.bin") {
                                val out = File(vaeDir, entry.name)
                                out.outputStream().use { zip.copyTo(it) }
                            }
                            entry = zip.nextEntry
                        }
                    }
                }
                Toast.makeText(context, vaeImportedStr, Toast.LENGTH_SHORT).show()
            } catch (_: Exception) {
                Toast.makeText(context, importFailedStr, Toast.LENGTH_SHORT).show()
            }
        }
    }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(32.dp),
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(
                stringResource(R.string.language),
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.SemiBold,
            )
            Text(
                "Change the application language",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            
            Box {
                OutlinedButton(
                    onClick = { showLanguageMenu = true },
                    modifier = Modifier.padding(top = 8.dp)
                ) {
                    Text(stringResource(R.string.language))
                }
                DropdownMenu(
                    expanded = showLanguageMenu,
                    onDismissRequest = { showLanguageMenu = false },
                ) {
                    DropdownMenuItem(
                        text = { Text(stringResource(R.string.language_system)) },
                        onClick = {
                            showLanguageMenu = false
                            setAppLocale(context, null)
                        },
                    )
                    DropdownMenuItem(
                        text = { Text(stringResource(R.string.language_en)) },
                        onClick = {
                            showLanguageMenu = false
                            setAppLocale(context, "en")
                        },
                    )
                    DropdownMenuItem(
                        text = { Text(stringResource(R.string.language_ru)) },
                        onClick = {
                            showLanguageMenu = false
                            setAppLocale(context, "ru")
                        },
                    )
                    DropdownMenuItem(
                        text = { Text(stringResource(R.string.language_zh)) },
                        onClick = {
                            showLanguageMenu = false
                            setAppLocale(context, "zh-CN")
                        },
                    )
                }
            }
        }

        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(
                stringResource(R.string.utility_storage_title),
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.SemiBold,
            )
            Text(
                stringResource(R.string.utility_storage_desc),
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            
            OutlinedButton(
                onClick = {
                    context.cacheDir.deleteRecursively()
                    context.codeCacheDir.deleteRecursively()
                    File(context.noBackupFilesDir, "conversion-work").deleteRecursively()
                    Toast.makeText(context, cacheClearedStr, Toast.LENGTH_SHORT).show()
                },
                modifier = Modifier.padding(top = 8.dp)
            ) {
                Text(stringResource(R.string.clear_cache))
            }
        }

        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(
                stringResource(R.string.utility_vae_title),
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.SemiBold,
            )
            Text(
                stringResource(R.string.utility_vae_desc),
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
                modifier = Modifier.padding(top = 8.dp)
            ) {
                FilledTonalButton(
                    onClick = {
                        exportLauncher.launch("vae_sdxl.zip")
                    }
                ) {
                    Text(stringResource(R.string.export_vae))
                }
                FilledTonalButton(
                    onClick = {
                        importLauncher.launch(arrayOf("application/zip", "application/x-zip-compressed"))
                    }
                ) {
                    Text(stringResource(R.string.import_vae))
                }
            }
        }
    }
}

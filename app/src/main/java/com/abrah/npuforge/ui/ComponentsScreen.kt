package com.abrah.npuforge.ui

import android.Manifest
import android.content.Intent
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Card
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.abrah.npuforge.CheckpointInfo
import com.abrah.npuforge.ConvertService
import com.abrah.npuforge.Donor
import com.abrah.npuforge.R

/** Owns component-family selection, SAF pickers and backup/restore progress. */
@Composable
fun ComponentsScreen(onBack: () -> Unit) {
    BackHandler(onBack = onBack)
    val context = LocalContext.current
    val state by ConvertService.state.collectAsState()
    val running = state is ConvertService.State.Running
    var selected by rememberSaveable { mutableStateOf(CheckpointInfo.Model.SDXL.name) }
    val notifications = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) {}
    val backup = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("application/zip")) { uri ->
        if (uri != null) {
            context.contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_WRITE_URI_PERMISSION)
            notifications.launch(Manifest.permission.POST_NOTIFICATIONS)
            context.startForegroundService(Intent(context, ConvertService::class.java)
                .setAction(ConvertService.ACTION_BACKUP)
                .putExtra(ConvertService.EXTRA_URI, uri)
                .putExtra(ConvertService.EXTRA_NAME, selected)
                .putExtra(ConvertService.EXTRA_MODEL, selected))
        }
    }
    val restore = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            context.contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
            notifications.launch(Manifest.permission.POST_NOTIFICATIONS)
            context.startForegroundService(Intent(context, ConvertService::class.java)
                .setAction(ConvertService.ACTION_RESTORE)
                .putExtra(ConvertService.EXTRA_URI, uri)
                .putExtra(ConvertService.EXTRA_NAME, selected)
                .putExtra(ConvertService.EXTRA_MODEL, selected))
        }
    }

    Column(Modifier.padding(16.dp).verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(16.dp)) {
        Text(stringResource(R.string.components_title), style = MaterialTheme.typography.headlineSmall)
        Text(stringResource(R.string.components_description), style = MaterialTheme.typography.bodyMedium)
        for (model in CheckpointInfo.Model.entries) {
            val ready = remember(state, model) { Donor.isReady(context, model) }
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(if (model == CheckpointInfo.Model.SD15) "SD1.5" else "SDXL",
                        style = MaterialTheme.typography.titleMedium)
                    Text(stringResource(if (ready) R.string.components_ready else R.string.components_missing),
                        style = MaterialTheme.typography.bodyMedium)
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        OutlinedButton(enabled = ready && !running, onClick = {
                            selected = model.name
                            backup.launch("npuforge-${model.name.lowercase()}-components.zip")
                        }) { Text(stringResource(R.string.components_backup)) }
                        OutlinedButton(enabled = !running, onClick = {
                            selected = model.name
                            restore.launch(arrayOf("application/zip", "application/octet-stream", "application/x-zip-compressed"))
                        }) { Text(stringResource(R.string.components_restore)) }
                    }
                }
            }
        }
        when (val current = state) {
            is ConvertService.State.Running -> {
                Text(current.stage, style = MaterialTheme.typography.titleMedium)
                LinearProgressIndicator(Modifier.fillMaxWidth())
                Text(current.detail, style = MaterialTheme.typography.bodyMedium)
                Text(stringResource(R.string.keep_open), style = MaterialTheme.typography.bodySmall)
            }
            is ConvertService.State.ComponentsDone -> Text(current.message)
            is ConvertService.State.Failed -> Text(current.message, color = MaterialTheme.colorScheme.error)
            else -> Unit
        }
    }
}

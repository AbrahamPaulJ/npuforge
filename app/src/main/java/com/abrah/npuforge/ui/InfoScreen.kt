package com.abrah.npuforge.ui

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.abrah.npuforge.R

@Composable
fun InfoScreen() {
    val uriHandler = LocalUriHandler.current
    Column(
        Modifier
            .fillMaxWidth()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(24.dp),
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(
                stringResource(R.string.info_use_title),
                modifier = Modifier.semantics { heading() },
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.SemiBold,
            )
            Text(stringResource(R.string.info_use_body), style = MaterialTheme.typography.bodyMedium)
        }

        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = MaterialTheme.shapes.large,
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceVariant,
                contentColor = MaterialTheme.colorScheme.onSurfaceVariant,
            ),
            border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(alpha = 0.4f)),
        ) {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text(
                    stringResource(R.string.info_black_title),
                    modifier = Modifier.semantics { heading() },
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.onSurface,
                )
                Text(stringResource(R.string.info_black_body), style = MaterialTheme.typography.bodyMedium)
                Text(stringResource(R.string.info_black_step1), style = MaterialTheme.typography.bodyMedium)
                Text(stringResource(R.string.info_black_step2), style = MaterialTheme.typography.bodyMedium)
                Text(stringResource(R.string.info_black_step3), style = MaterialTheme.typography.bodyMedium)
                FilledTonalButton(
                    onClick = {
                        uriHandler.openUri("https://huggingface.co/Mr-J-369/Fancy-AI/resolve/main/VAE.zip?download=true")
                    },
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text(stringResource(R.string.info_black_download), textAlign = TextAlign.Center)
                }
            }
        }

        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(
                stringResource(R.string.info_formats_title),
                modifier = Modifier.semantics { heading() },
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
            )
            Text(stringResource(R.string.info_formats_body), style = MaterialTheme.typography.bodyMedium)
        }

        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(
                stringResource(R.string.info_help_title),
                modifier = Modifier.semantics { heading() },
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
            )
            Text(stringResource(R.string.info_help_body), style = MaterialTheme.typography.bodyMedium)
        }
    }
}

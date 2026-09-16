package com.abrah.npuforge.ui

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.SuggestionChip
import androidx.compose.material3.SuggestionChipDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.abrah.npuforge.R

/** Conversion scope and the measurements available for each model family. */
@Composable
fun InfoScreen() {
    Column(
        Modifier
            .padding(horizontal = 16.dp, vertical = 12.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        InfoSectionCard(
            title = stringResource(R.string.info_what_title),
            badge = stringResource(R.string.chip_qualcomm_npu),
            paragraphs = listOf(
                stringResource(R.string.info_what_p1),
                stringResource(R.string.info_what_p2),
            ),
        )

        InfoSectionCard(
            title = stringResource(R.string.info_scope_title),
            paragraphs = listOf(
                stringResource(R.string.info_scope_b1),
                stringResource(R.string.info_scope_b2),
                stringResource(R.string.info_scope_b3),
                stringResource(R.string.info_scope_b4),
                stringResource(R.string.info_scope_b5),
            ),
        )

        InfoSectionCard(
            title = stringResource(R.string.info_converted_title),
            paragraphs = listOf(
                stringResource(R.string.info_converted_p1),
                stringResource(R.string.info_converted_p2),
                stringResource(R.string.info_converted_p3),
            ),
        )

        WarningCard(
            title = stringResource(R.string.info_compatibility_title),
            paragraphs = listOf(
                stringResource(R.string.info_compatibility_p1),
                stringResource(R.string.info_compatibility_p2),
            ),
        )

        InfoSectionCard(
            title = stringResource(R.string.info_lora_title),
            paragraphs = listOf(
                stringResource(R.string.info_lora_p1),
                stringResource(R.string.info_lora_p2),
                stringResource(R.string.info_lora_p3),
                stringResource(R.string.info_lora_p4),
                stringResource(R.string.info_lora_p5),
                stringResource(R.string.info_lora_p6),
            ),
        )

        InfoSectionCard(
            title = stringResource(R.string.info_requirements_title),
            paragraphs = listOf(
                stringResource(R.string.info_requirements_b1),
                stringResource(R.string.info_requirements_b2),
                stringResource(R.string.info_requirements_b3),
                stringResource(R.string.info_requirements_b4),
                stringResource(R.string.info_requirements_b5),
                stringResource(R.string.info_requirements_b6),
                stringResource(R.string.info_requirements_b7),
            ),
        )

        InfoSectionCard(
            title = stringResource(R.string.info_troubleshooting_title),
            paragraphs = listOf(
                stringResource(R.string.info_troubleshooting_p1),
                stringResource(R.string.info_troubleshooting_p2),
                stringResource(R.string.info_troubleshooting_p3),
                stringResource(R.string.info_troubleshooting_p4),
            ),
        )

        InfoSectionCard(
            title = stringResource(R.string.info_tested_title),
            paragraphs = listOf(
                stringResource(R.string.info_tested_p1),
                stringResource(R.string.info_tested_p2),
            ),
        )
    }
}

@Composable
private fun InfoSectionCard(
    title: String,
    badge: String? = null,
    paragraphs: List<String>,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
        ),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(alpha = 0.2f)),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    text = title,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.primary,
                )
                if (badge != null) {
                    SuggestionChip(
                        onClick = {},
                        label = { Text(badge, style = MaterialTheme.typography.labelSmall) },
                        colors = SuggestionChipDefaults.suggestionChipColors(
                            containerColor = MaterialTheme.colorScheme.primaryContainer,
                            labelColor = MaterialTheme.colorScheme.onPrimaryContainer,
                        ),
                        border = null,
                    )
                }
            }
            paragraphs.forEach { text ->
                Text(
                    text = text,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurface,
                )
            }
        }
    }
}

@Composable
private fun WarningCard(title: String, paragraphs: List<String>) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.errorContainer.copy(alpha = 0.25f),
        ),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.error.copy(alpha = 0.4f)),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(
                text = title,
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.error,
            )
            paragraphs.forEach { text ->
                Text(
                    text = text,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurface,
                )
            }
        }
    }
}

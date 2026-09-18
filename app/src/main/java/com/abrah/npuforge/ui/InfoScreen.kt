package com.abrah.npuforge.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.LinkAnnotation
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextLinkStyles
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.unit.dp
import com.abrah.npuforge.R

@Composable
fun InfoScreen() {
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
            val fullText = stringResource(R.string.info_use_body)
            val linkStyle = TextLinkStyles(
                style = SpanStyle(
                    color = MaterialTheme.colorScheme.primary,
                    textDecoration = TextDecoration.Underline,
                    fontWeight = FontWeight.SemiBold,
                )
            )
            val fancyAi = "Fancy-Ai"
            val nightmare = "Nightmare Mobile"
            val fancyIdx = fullText.indexOf(fancyAi)
            val nightmareIdx = fullText.indexOf(nightmare)

            val annotated = buildAnnotatedString {
                if (fancyIdx != -1 && nightmareIdx != -1 && fancyIdx < nightmareIdx) {
                    append(fullText.substring(0, fancyIdx))
                    addLink(
                        LinkAnnotation.Url("https://github.com/Mr-J-369/Fancy-Ai", linkStyle),
                        length,
                        length + fancyAi.length,
                    )
                    append(fancyAi)
                    append(fullText.substring(fancyIdx + fancyAi.length, nightmareIdx))
                    addLink(
                        LinkAnnotation.Url("https://github.com/AbrahamPaulJ/nightmare-mobile", linkStyle),
                        length,
                        length + nightmare.length,
                    )
                    append(nightmare)
                    append(fullText.substring(nightmareIdx + nightmare.length))
                } else {
                    append(fullText)
                }
            }
            Text(annotated, style = MaterialTheme.typography.bodyMedium)
        }

        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text(
                stringResource(R.string.info_formats_title),
                modifier = Modifier.semantics { heading() },
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
            )
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                InfoSpecItem(stringResource(R.string.info_spec_sd15))
                InfoSpecItem(stringResource(R.string.info_spec_sdxl))
                InfoSpecItem(stringResource(R.string.info_spec_weights))
                InfoSpecItem(stringResource(R.string.info_spec_loras))
            }
        }

        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(
                stringResource(R.string.info_help_title),
                modifier = Modifier.semantics { heading() },
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
            )
            Text(stringResource(R.string.info_help_body), style = MaterialTheme.typography.bodyMedium)
            val repoUrl = "https://github.com/AbrahamPaulJ/npuforge"
            val repoLink = buildAnnotatedString {
                addLink(
                    LinkAnnotation.Url(
                        repoUrl,
                        TextLinkStyles(
                            style = SpanStyle(
                                color = MaterialTheme.colorScheme.primary,
                                textDecoration = TextDecoration.Underline,
                                fontWeight = FontWeight.SemiBold,
                            )
                        )
                    ),
                    0,
                    repoUrl.length,
                )
                append(repoUrl)
            }
            Text(repoLink, style = MaterialTheme.typography.bodyMedium)
        }
    }
}

@Composable
private fun InfoSpecItem(text: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Text(
            text = "—",
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.primary,
        )
        Text(
            text = text,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurface,
            modifier = Modifier.weight(1f),
        )
    }
}

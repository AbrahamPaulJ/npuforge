package com.abrah.npuforge.ui.theme

import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext

val NeuralDarkBackground = Color(0xFF0B0F19)
val NeuralDarkSurface = Color(0xFF131B2E)
val NeuralDarkSurfaceVariant = Color(0xFF1E293B)
val NeuralDarkPrimary = Color(0xFF38BDF8)
val NeuralDarkOnPrimary = Color(0xFF082F49)
val NeuralDarkPrimaryContainer = Color(0xFF0C4A6E)
val NeuralDarkOnPrimaryContainer = Color(0xFFBAE6FD)
val NeuralDarkSecondary = Color(0xFFA78BFA)
val NeuralDarkOnSecondary = Color(0xFF2E1065)
val NeuralDarkSecondaryContainer = Color(0xFF4C1D95)
val NeuralDarkOnSecondaryContainer = Color(0xFFDDD6FE)
val NeuralDarkTertiary = Color(0xFF34D399)
val NeuralDarkOnTertiary = Color(0xFF064E3B)
val NeuralDarkError = Color(0xFFF87171)
val NeuralDarkOnError = Color(0xFF450A0A)
val NeuralDarkOnSurface = Color(0xFFF1F5F9)
val NeuralDarkOnSurfaceVariant = Color(0xFF94A3B8)
val NeuralDarkOutline = Color(0xFF334155)

val NeuralLightBackground = Color(0xFFF8FAFC)
val NeuralLightSurface = Color(0xFFFFFFFF)
val NeuralLightSurfaceVariant = Color(0xFFF1F5F9)
val NeuralLightPrimary = Color(0xFF0284C7)
val NeuralLightOnPrimary = Color(0xFFFFFFFF)
val NeuralLightPrimaryContainer = Color(0xFFE0F2FE)
val NeuralLightOnPrimaryContainer = Color(0xFF0369A1)
val NeuralLightSecondary = Color(0xFF7C3AED)
val NeuralLightOnSecondary = Color(0xFFFFFFFF)
val NeuralLightSecondaryContainer = Color(0xFFEDE9FE)
val NeuralLightOnSecondaryContainer = Color(0xFF6D28D9)
val NeuralLightTertiary = Color(0xFF059669)
val NeuralLightOnTertiary = Color(0xFFFFFFFF)
val NeuralLightError = Color(0xFFDC2626)
val NeuralLightOnError = Color(0xFFFFFFFF)
val NeuralLightOnSurface = Color(0xFF0F172A)
val NeuralLightOnSurfaceVariant = Color(0xFF475569)
val NeuralLightOutline = Color(0xFFCBD5E1)

private val DarkColorScheme = darkColorScheme(
    primary = NeuralDarkPrimary,
    onPrimary = NeuralDarkOnPrimary,
    primaryContainer = NeuralDarkPrimaryContainer,
    onPrimaryContainer = NeuralDarkOnPrimaryContainer,
    secondary = NeuralDarkSecondary,
    onSecondary = NeuralDarkOnSecondary,
    secondaryContainer = NeuralDarkSecondaryContainer,
    onSecondaryContainer = NeuralDarkOnSecondaryContainer,
    tertiary = NeuralDarkTertiary,
    onTertiary = NeuralDarkOnTertiary,
    background = NeuralDarkBackground,
    surface = NeuralDarkSurface,
    surfaceVariant = NeuralDarkSurfaceVariant,
    onSurface = NeuralDarkOnSurface,
    onSurfaceVariant = NeuralDarkOnSurfaceVariant,
    outline = NeuralDarkOutline,
    error = NeuralDarkError,
    onError = NeuralDarkOnError,
)

private val LightColorScheme = lightColorScheme(
    primary = NeuralLightPrimary,
    onPrimary = NeuralLightOnPrimary,
    primaryContainer = NeuralLightPrimaryContainer,
    onPrimaryContainer = NeuralLightOnPrimaryContainer,
    secondary = NeuralLightSecondary,
    onSecondary = NeuralLightOnSecondary,
    secondaryContainer = NeuralLightSecondaryContainer,
    onSecondaryContainer = NeuralLightOnSecondaryContainer,
    tertiary = NeuralLightTertiary,
    onTertiary = NeuralLightOnTertiary,
    background = NeuralLightBackground,
    surface = NeuralLightSurface,
    surfaceVariant = NeuralLightSurfaceVariant,
    onSurface = NeuralLightOnSurface,
    onSurfaceVariant = NeuralLightOnSurfaceVariant,
    outline = NeuralLightOutline,
    error = NeuralLightError,
    onError = NeuralLightOnError,
)

@Composable
fun NpuForgeTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    dynamicColor: Boolean = true,
    content: @Composable () -> Unit,
) {
    val context = LocalContext.current
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        }
        darkTheme -> DarkColorScheme
        else -> LightColorScheme
    }

    MaterialTheme(
        colorScheme = colorScheme,
        content = content,
    )
}

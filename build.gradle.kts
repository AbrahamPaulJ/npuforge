// Pinned versions: keep them explicit so a clone builds without resolving to
// whatever is newest today.
plugins {
    id("com.android.application") version "9.4.1" apply false
    id("org.jetbrains.kotlin.plugin.compose") version "2.4.20" apply false
}

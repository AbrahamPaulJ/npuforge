plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "com.abrah.npuforge"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.abrah.npuforge"
        // 31: the QNN runtime and the exec-from-nativeLibraryDir trick both
        // match the generator ecosystem's floor, and there is no point supporting a device
        // that cannot run the models this produces.
        minSdk = 31
        targetSdk = 35
        versionCode = 2
        versionName = "0.2"
        ndk { abiFilters += "arm64-v8a" }
    }

    buildTypes {
        debug {
            // Unminified: this is a tool for one person so far, and a readable
            // stack trace is worth more than the megabytes. Revisit if it ships.
            isMinifyEnabled = false
        }
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    buildFeatures { compose = true }

    packaging {
        jniLibs {
            // ⚠ The whole design depends on this. Android blocks executing a
            // file from the writable app data dir; nativeLibraryDir is the one
            // allowed location, and useLegacyPackaging=true is what makes the
            // entries real files there instead of staying inside the APK.
            // tplconv and the QNN context-binary generator are both EXECUTABLES
            // shipped under lib*.so names for exactly this reason.
            useLegacyPackaging = true
            keepDebugSymbols += "**/libtplconv.so"
            keepDebugSymbols += "**/libqnncontextgen.so"
            // ⚠⚠ And every QNN library. AGP strips native libs by default, which
            // silently ALTERS them: the packaged libQnnHtpV79Skel.so came out
            // with a different md5 from the SDK file, and the DSP then refused
            // it -- "Failed to load skel, error: 4000" / "Device Creation
            // failure". The file SIZES were identical, which is why listing them
            // looked fine. Only md5 caught it.
            keepDebugSymbols += "**/libQnn*.so"
            keepDebugSymbols += "**/libstable_diffusion_core.so"
            keepDebugSymbols += "**/libqnnnetrun.so"
        }
        resources { excludes += "/META-INF/{AL2.0,LGPL2.1}" }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.4")
    implementation("androidx.activity:activity-compose:1.9.1")
    implementation(platform("androidx.compose:compose-bom:2024.09.02"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("androidx.documentfile:documentfile:1.0.1")
}

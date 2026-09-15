import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "com.abrah.npuforge"
    compileSdk = 37
    ndkVersion = "29.0.14206865"

    defaultConfig {
        applicationId = "com.abrah.npuforge"
        // Android 13 is the app's minimum for its Snapdragon 8 Gen 2+ audience.
        minSdk = 33
        targetSdk = 37
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

// Owns compilation of the executable that Android runs from nativeLibraryDir.
abstract class CompileTplconv @javax.inject.Inject constructor(
    private val process: ExecOperations,
) : DefaultTask() {
    @get:InputFile abstract val source: RegularFileProperty
    @get:InputFile abstract val compiler: RegularFileProperty
    @get:InputFile abstract val ndkRevision: RegularFileProperty
    @get:OutputDirectory abstract val outputDirectory: DirectoryProperty

    @TaskAction
    fun compile() {
        val output = outputDirectory.get().file("arm64-v8a/libtplconv.so").asFile
        output.parentFile.mkdirs()
        process.exec {
            commandLine(
                compiler.get().asFile,
                "-O2", "-std=c++17", "-ffp-contract=off", "-static-libstdc++",
                "-Wl,-z,max-page-size=16384", "-Wl,-z,common-page-size=16384",
                source.get().asFile, "-o", output,
            )
        }
    }
}

val converterNdk = androidComponents.sdkComponents.ndkDirectory
val compileTplconv = tasks.register<CompileTplconv>("compileTplconv") {
    source.set(rootProject.layout.projectDirectory.file("native/tplconv.cpp"))
    ndkRevision.set(converterNdk.map { it.file("source.properties") })
    compiler.set(converterNdk.map {
        it.file("toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android31-clang++")
    })
    outputDirectory.set(layout.buildDirectory.dir("generated/tplconv/jniLibs"))
}

// Owns the allocator DSO preloaded only by the SDXL compiler subprocess.
abstract class CompileCompilerHeap @javax.inject.Inject constructor(
    private val process: ExecOperations,
) : DefaultTask() {
    @get:InputFile abstract val source: RegularFileProperty
    @get:InputFile abstract val symbols: RegularFileProperty
    @get:InputFile abstract val compiler: RegularFileProperty
    @get:InputFile abstract val ndkRevision: RegularFileProperty
    @get:OutputDirectory abstract val outputDirectory: DirectoryProperty

    @TaskAction
    fun compile() {
        val output = outputDirectory.get().file("arm64-v8a/libcompiler_heap.so").asFile
        output.parentFile.mkdirs()
        process.exec {
            commandLine(
                compiler.get().asFile,
                "-O2", "-std=c11", "-fPIC", "-shared", "-fno-builtin", "-Wall", "-Wextra",
                "-Wl,-z,max-page-size=16384", "-Wl,-z,common-page-size=16384",
                "-Wl,-z,defs", "-Wl,--version-script=${symbols.get().asFile}",
                source.get().asFile, "-ldl", "-o", output,
            )
        }
    }
}

val compileCompilerHeap = tasks.register<CompileCompilerHeap>("compileCompilerHeap") {
    source.set(rootProject.layout.projectDirectory.file("native/compiler_heap.c"))
    symbols.set(rootProject.layout.projectDirectory.file("native/compiler_heap.map"))
    ndkRevision.set(converterNdk.map { it.file("source.properties") })
    compiler.set(converterNdk.map {
        it.file("toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android33-clang")
    })
    outputDirectory.set(layout.buildDirectory.dir("generated/compilerHeap/jniLibs"))
}
androidComponents.onVariants { variant ->
    variant.sources.jniLibs?.addGeneratedSourceDirectory(compileTplconv, CompileTplconv::outputDirectory)
    variant.sources.jniLibs?.addGeneratedSourceDirectory(compileCompilerHeap, CompileCompilerHeap::outputDirectory)
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.material3)
}

# Build npuforge

The source checkout can run the host tests without an Android SDK, Qualcomm
SDK, checkpoint, or generated model bundle. A **working conversion APK** also
needs the external runtime and generated assets listed below. Gradle does not
download or generate those assets automatically; assembling an APK alone does
not establish that it contains everything needed for conversion.

## Host requirements

The Android native build tasks currently target **Linux x86_64**.

| Dependency | Version used by the project |
| --- | --- |
| Gradle | 9.7.1, supplied through `./gradlew` |
| Android Gradle Plugin | 9.4.0 |
| Gradle daemon | JetBrains JDK 21; application bytecode targets Java 17 |
| Kotlin Compose plugin | 2.4.20 |
| Android SDK | Platform 37, Build Tools 37.0.0 |
| Android NDK | 30.0.16248370 |
| Qualcomm AI Runtime | QAIRT 2.50.0.260828 |

Install Android command-line tools and put `sdkmanager` on `PATH`, then install
the SDK packages:

```sh
export ANDROID_HOME="$HOME/Android/Sdk"
sdkmanager --sdk_root="$ANDROID_HOME" \
  'platform-tools' 'platforms;android-37' 'build-tools;37.0.0' \
  'ndk;30.0.16248370'
sdkmanager --sdk_root="$ANDROID_HOME" --licenses
```

Set `JAVA_HOME` to a JDK 21 installation. The tracked Gradle daemon criteria
select JetBrains JDK 21 and can provision it when necessary. The first build
needs network access for Gradle, Android dependencies and toolchain downloads.

## External runtime and model assets

Read [NOTICE](../NOTICE) before supplying or redistributing these files. They
have separate provenance and terms from the repository's original source.
Keep generated model files matched to their recipes and SDK version.

### Qualcomm runtime

Place the QAIRT Android ARM64 files in
`app/src/main/jniLibs/arm64-v8a/`:

- `qnn-context-binary-generator`, renamed to `libqnncontextgen.so`;
- `libQnnHtp.so`, `libQnnHtpPrepare.so`, `libQnnSystem.so` and
  `libQnnHtpNetRunExtensions.so`;
- the matching `libQnnHtpV<arch>.so`, `libQnnHtpV<arch>Stub.so` and
  `libQnnHtpV<arch>Skel.so` files for the device architectures being supported.

The tested runtime bundle contains v68, v69, v73, v75, v79 and v81 files. These
are not a guarantee that every corresponding device supports the current
templates. See [device/runtime packaging](ANDROID.md) and [limits](LIMITS.md).

### UNet templates

| Asset directory | Files to supply |
| --- | --- |
| `app/src/main/assets/template/` | SD1.5 `libqnn_model.so`; copy `recipe.bin` and `tpl_trim.pack` from the repository's `template/` directory |
| `app/src/main/assets/template_sdxl/` | Matching SDXL `libqnn_model.so`, `recipe.bin`, `tpl_trim.pack` |
| `app/src/main/assets/template_inpaint/` | SD1.5 9-channel inpaint `libqnn_model.so`, `recipe.bin`, `tpl_trim.pack` ([SD15-INPAINT.md](SD15-INPAINT.md)) |
| `app/src/main/assets/template_swap/` | SD1.5 Swap v2 `libqnn_model.so`, `recipe.bin`, `tpl_trim.pack` ([SD15-LORA-CN-TEMPLATE.md](SD15-LORA-CN-TEMPLATE.md) §8); tracks `lora_targets.json` and `ip_targets.json` too |

The asset directories track `sources.txt` and `htp_config.json`. Without
`template_inpaint/`, a 9-channel checkpoint is reported as unsupported by this
build; 4-channel conversion is unaffected.
The SD1.5 recipe and trimmed pack have separate provenance restrictions
documented in [template/README.md](../template/README.md).

### Checkpoint component templates

Supply this tree under **each** of `app/src/main/assets/components_sd15/` and
`app/src/main/assets/components_sdxl/`:

```text
clip_recipe.bin
clip_requirements.json
clip_skip1/                 (components_sd15 only: the clip-skip-1 recipe)
  clip_recipe.bin
  clip_requirements.json
tokenizer.json
vae_encoder/
  libqnn_model.so
  recipe.bin
  tpl_trim.pack
  requirements.json
  sources.txt
  htp_config.json
vae_decoder/
  libqnn_model.so
  recipe.bin
  tpl_trim.pack
  requirements.json
  sources.txt
  htp_config.json
```

The families use different graphs and resolutions. Authoring instructions are
in [CLIP components](CLIP-COMPONENTS.md), [VAE templates](VAE-TEMPLATES.md),
[SD1.5 components](SD15-COMPONENTS.md) and [SDXL components](SDXL-COMPONENTS.md).
Authoring requires additional host tools and source model assets; it is a
separate workflow from compiling the Android app.

## Build and inspect

From the repository root:

```sh
./gradlew :app:assembleDebug :app:lintDebug --console=plain
```

- Debug APK: `app/build/outputs/apk/debug/app-debug.apk`
- Lint report: `app/build/reports/lint-results-debug.html`
- Release build: `./gradlew :app:assembleRelease`
- Shareable test build: `./gradlew :app:assemblePreview` →
  `app/build/outputs/apk/preview/app-preview.apk`. It is the release variant
  signed with the local debug key, installed as `com.abrah.npuforge.preview`
  ("npuforge preview") beside the signed app, with its own data. ⚠ Share this,
  not the debug APK: the debug build carries the debug-only DSP probe, its canary
  contexts and `app/src/debug/jniLibs` (Local Dream's CC BY-NC backend).

The Gradle tasks that compile `tplconv`, `componentconv` and `libcompiler_heap`
use the NDK's clang wrappers for the host OS, so the build also runs on Windows.

Release signing reads `Keys/signing.properties` in the project root, or the file
named by `NPUFORGE_SIGNING_PROPERTIES`. The `Keys/` directory is excluded from
Git. Store the local keystore there as well. The properties are:

```properties
storeFile=/absolute/path/to/npuforge/Keys/npuforge-release.p12
storePassword=YOUR_KEYSTORE_PASSWORD
keyAlias=npuforge-release
keyPassword=YOUR_KEYSTORE_PASSWORD
```

Use the same password for the PKCS12 keystore and key. Without a default signing
file or an explicit path, the build produces an unsigned APK. An explicitly
configured path must exist. With signing configured, the output is
`app/build/outputs/apk/release/app-release.apk`.

Back up the keystore and credentials securely; future updates must use the same
signing key. Signing credentials are never included in source control.

The debug source set exports diagnostic
services and is intended for development. Its optional DSP probe additionally
needs the external probe backend and canary assets described in [app/README.md](../app/README.md).
They are not used by ordinary conversion.

Gradle builds `libtplconv.so`, `libcomponentconv.so` and
`libcompiler_heap.so` from the native source and registers them as generated
JNI inputs. Do not place prebuilt copies in `src/main/jniLibs`.

Preserve the existing native build and packaging settings:

- `-ffp-contract=off` keeps LoRA arithmetic consistent with the reference.
- The native outputs use 16 KB ELF alignment.
- `useLegacyPackaging = true` extracts executables to `nativeLibraryDir`.
- Qualcomm libraries must retain their original bytes; stripping DSP libraries
  can prevent device creation. Compare packaged files with the SDK originals.

The compiler allocator is preloaded into SDXL compiler subprocesses. It is
not an application-wide allocator replacement.

## Verification and authoring

Run the [host tests](TESTING.md) before changing the converter. Full integration
also needs a packaged runtime, generated templates and a supported phone.

To author a pack-loading model library, patch the QAIRT-generated C++ using
`tools/tpl_patch.py`, then run the SDK model library generator:

```sh
export ANDROID_NDK_ROOT="$ANDROID_HOME/ndk/30.0.16248370"
python "$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-model-lib-generator" \
  -c model_tpl.cpp -t aarch64-android -o lib_arm
```

Omit `-b`: the generated `libqnn_model.so` reads the external weight pack through
`QNN_TPL_PACK` instead of embedding checkpoint weights. See
[template authoring](TEMPLATE-AUTHORING.md) for the complete workflow.

Once a complete APK is installed and the checkpoint/adapters are local,
SD1.5 text-to-image conversion needs no network. SDXL downloads its VAE
contexts once and plain-SD1.5 → inpainting downloads the inpainting difference
once; both have an offline import ([LIMITS.md](LIMITS.md) §Downloads).

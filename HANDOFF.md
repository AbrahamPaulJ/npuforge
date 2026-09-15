# Current state — 2026-09-15

SDXL INT8 conversion and generation work end to end on the tested Galaxy S25
Ultra. The current SDXL configuration is O=3 with both source-destructive
reuse options disabled and the storage-backed compiler allocator enabled.

- O=1: compiler exited 0 in about 254 seconds; Aura generated in 20 seconds.
- O=3: user reported 437 seconds total conversion; Aura generated in 15 seconds.
- Both images: 1024 × 1024, 8 steps, CFG 1, LCM/Karras. Seeds differed.
- User observed 41% RAM during O=3 conversion; this is not a peak measurement.
- SD1.5, LoRA merging and its separate template remain present.

The evidence, exact settings, failed approaches, allocation mechanism and
remaining limitations are recorded in [docs/SDXL.md](docs/SDXL.md).

## Implementation

- QAIRT 2.50.0.260828; SD1.5 v73, SDXL v75/soc57. Targets are fixed.
- `native/compiler_heap.c` owns SDXL compiler allocation backing, with indexed
  ownership lookup, a 64 KiB cutoff and less than 2 MiB of idle reuse storage.
- `Converter.kt` sets the allocator environment only for SDXL, selects the
  model-specific template, and exports the Aura model markers.
- `ConvertService.kt` owns conversion, foreground lifetime, cancellation and
  child cleanup. Main UI includes live RAM, elapsed time and selectable logs.
- Shared SDXL CLIP/VAE components now download from Mr-J-369/SDXL-OnDevice-Conversion
  once, reusing the existing cache. Explicit local service imports remain supported.
  SDK/model assets remain outside source control. Download flow needs a phone test.
- Android uses AGP 9.4.0, Gradle 9.7.1, Kotlin Compose plugin 2.4.20, SDK 37,
  minSdk 33, NDK 29.0.14206865. Gradle builds the converter and allocator on Linux.

## Working constraints

The user builds and installs through Android Studio. Do not install, deploy or
launch APKs on their phone. Do not start PC model conversion, extra validation
jobs, benchmarks or tests. Preserve the successful configuration during further
work. A source push does not publish model assets, SDK binaries or an APK release.

Historical build/lint checks preceded the final configuration change. The O=3
APK was built and tested by the user. Documentation/UI result text and the automatic shared-component download were
updated for the source push without another build or device run.

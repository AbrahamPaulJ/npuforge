# The app

`com.abrah.npuforge` — a standalone Android app that runs the whole converter on
the phone. Pick a `.safetensors`, optionally stack LoRAs, get
`Download/npuforge/<name>.zip`. Existing exports are preserved: MediaStore assigns
numbered filenames for duplicate names, and the completion screen shows the
actual filename.

The UI follows the system theme and applies safe drawing insets, including the
keyboard. Checkpoint selection, model name, LoRAs and strengths survive tab
changes and activity recreation. Back from Info returns to Convert.

Each conversion keeps its checkpoint, adapters, pack and compiled binary in one
work directory, cleaned on success, failure or coroutine cancellation. Native
tool processes are terminated when their owning conversion is cancelled.

⛔ It is **not** a feature of any one generator and must not become one. A
generator renders; this writes models that any generator can import.

## The pipeline, as the app runs it

| stage | file | note |
|---|---|---|
| inspect | `CheckpointInfo.kt` | safetensors **header only** — a short read, not a 2 GB copy. Selects SD1.5 or SDXL; checks the selected recipe sources; SD2 and diffusers layouts remain unsupported |
| donor | `Donor.kt` | downloads SD1.5 components once; SDXL downloads the shared component ZIP once, reusing the existing cache |
| weights | `Converter.kt` → `libtplconv.so` | applies the selected recipe, merging LoRAs before quantization |
| compile | `Converter.kt` → `libqnncontextgen.so` | SD1.5 or SDXL config; SDXL preloads the storage-backed allocator. Foreground service, screen on |
| assemble | `Converter.kt` | one **uncompressed** zip via MediaStore — these are quantized weights, deflate would cost a minute of CPU to save nothing |

`ConvertService.kt` owns the foreground service and the progress parsing; its
LoRA extras are plain `"uri|strength"` strings so the whole flow can be driven
from `adb shell am` for testing.

## Two things that will bite

1. **Packaging.** Getting QNN to run inside an app took four separate fixes,
   each surfacing as the identical "Device Creation failure". Read
   `../docs/ANDROID.md` before touching `jniLibs`, `packaging {}`, or the
   library paths. In short: `/vendor/lib64` must be on `LD_LIBRARY_PATH`, skels
   must live in `filesDir` and be world-readable, executables must ship as
   `lib*.so` in `nativeLibraryDir`, and AGP stripping silently corrupts the
   skels (same size, different md5).
2. **The Info tab is documentation.** `ui/InfoScreen.kt` is the user-facing copy
   of `../docs/LIMITS.md`. When a limit changes, change both in the same edit —
   a stale limits screen is worse than none.

## Debug source set

`src/debug/` exports `ConvertService` and a `ProbeService` that load-tests a
54 KB canary context binary to separate "cannot reach the DSP" from "cannot
prepare a graph". ⚠ It keys off a positive signal (`"ok":true`) because the
first version passed without ever creating a device. It never ships in release.

## SDXL phone result

O=3 conversion completed in 437 seconds; the exported model generated in Aura
at 1024 × 1024, 8 steps, CFG 1, LCM/Karras in 15 seconds. These are individual
Galaxy S25 Ultra results. See [SDXL findings](../docs/SDXL.md) for configuration,
measurement limits and the separate compiler-memory investigation.

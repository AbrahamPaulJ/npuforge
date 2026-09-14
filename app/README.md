# The app

`com.abrah.npuforge` — a standalone Android app that runs the whole converter on
the phone. Pick a `.safetensors`, optionally stack LoRAs, get
`Download/npuforge/<name>.zip`.

⛔ It is **not** a feature of any one generator and must not become one. A
generator renders; this writes models that any generator can import.

## The pipeline, as the app runs it

| stage | file | note |
|---|---|---|
| inspect | `CheckpointInfo.kt` | safetensors **header only** — a short read, not a 2 GB copy. Refuses SDXL / SD2 / diffusers layouts and confirms all 686 recipe sources are present |
| donor | `Donor.kt` | streams the shared CLIP/VAE once (1.03 GB fetched, 395 MB kept) |
| weights | `Converter.kt` → `libtplconv.so` | ~24 s, merges any LoRAs on the way through |
| compile | `Converter.kt` → `libqnncontextgen.so` | ~93 s, ~4.8 GB peak. Foreground service, screen on |
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

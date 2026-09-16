# Android app

`com.abrah.npuforge` converts a local SD1.5 or SDXL `.safetensors` checkpoint
into an archive for [Local Dream](https://github.com/xororz/local-dream) or
[Fancy-Ai](https://github.com/Mr-J-369/Fancy-Ai), the only supported output
consumers. It supports optional UNet
LoRA merging and writes `Download/npuforge/<name>.zip` through MediaStore.
Existing exports are preserved; the completion screen shows the actual filename.

See [Build](../docs/BUILD.md) for prerequisites and external runtime/assets.
The app requires Android 13 or later and targets ARM64 Snapdragon devices with
compatible Qualcomm HTP support. Hardware compatibility and checkpoint limits
are documented in [Limits](../docs/LIMITS.md).

## Conversion pipeline

| Stage | Implementation | Output |
| --- | --- | --- |
| Import and inspect | `CheckpointInfo.kt`, `ConvertService.kt` | Local checkpoint and adapter copies |
| Text encoders | `Converter.kt` → `libcomponentconv.so` | Checkpoint-owned MNN encoder(s), token and position embeddings |
| VAE encoder | `Converter.kt` → `libtplconv.so` → `libqnncontextgen.so` | Compiled QNN encoder context |
| VAE decoder | Same tools, separate compiler process | Compiled QNN decoder context |
| UNet | `libtplconv.so` → `libqnncontextgen.so` | LoRA-merged weight pack, then QNN context |
| Export | `Converter.kt` | Uncompressed ZIP with model components and tokenizer |

Both families use the selected checkpoint's CLIP and VAE weights. SDXL has two
text encoders; SD1.5 has one. The tokenizer and graph templates are packaged
assets. Conversion does not download shared model weights. `Donor.kt` remains
only for legacy component archive backup/restore.

LoRAs apply to supported UNet modules before quantization. Unmatched adapter
tensors produce warnings while matched layers continue to merge; an adapter
with no effective layers cannot be applied. Text-encoder LoRA merging is not
implemented. See [LoRA details](../docs/LIMITS.md#lora-behavior).

## Process and storage ownership

`ConvertService.kt` owns conversion in a foreground service. Active checkpoints,
adapters, compiler backing files and outputs live in
`noBackupFilesDir/conversion-work`, outside Android's cache directory. The service
explicitly cleans this workspace on success, failure or cancellation. Cancellation
also terminates the native subprocess before cleanup.

VAE encoder, VAE decoder and UNet compile sequentially. SDXL compilation
preloads the storage-backed allocator to reduce pressure on the compiler's
anonymous heap. This does not establish a universal RAM or storage minimum.

The UI retains checkpoint selection, name, LoRAs and strengths across tab
changes and activity recreation. `ui/InfoScreen.kt` displays the supported
formats and limits; keep it consistent with [Limits](../docs/LIMITS.md).

## Native packaging

Android launches the converter executables from `nativeLibraryDir`, packaged
under `lib*.so` names. Qualcomm DSP libraries are staged separately where the
DSP can read them. Library search paths, extraction and byte-preserving
packaging are required parts of this integration. Read
[Android runtime integration](../docs/ANDROID.md) before changing them.

## Debug diagnostics

The debug manifest exports `ConvertService` and `ProbeService` for development.
The optional probe requires a separately supplied `libstable_diffusion_core.so`
and `probe/canary_249.bin` / `probe/canary_228.bin` assets. These external artifacts
are excluded from source control and have separate terms; see [NOTICE](../NOTICE).
The probe is absent from release builds and is not a conversion dependency.

The in-app troubleshooting log records the conversion stages, compiler output,
device information and memory/storage observations. See
[Testing](../docs/TESTING.md) for reproducible checks and reporting guidance.

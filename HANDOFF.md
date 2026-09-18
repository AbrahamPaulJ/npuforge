# Developer overview

npuforge converts SD1.5 and SDXL checkpoints into importable Qualcomm NPU model
bundles on Android for [Fancy-Ai](https://github.com/Mr-J-369/Fancy-Ai) and
[Nightmare Mobile](https://github.com/AbrahamPaulJ/nightmare-mobile). Output compatibility is limited
to those two apps. Start with [README.md](README.md) for the project overview
and [docs/BUILD.md](docs/BUILD.md) for dependencies and build instructions.

## Current implementation

- Both families use the selected checkpoint's UNet, text encoder weights and
  embeddings, VAE encoder and VAE decoder. SDXL has two text encoders. New
  conversions do not download or consume shared component weights.
- The app writes MNN text encoders and compiles QNN UNet/VAE contexts. Graph
  templates and tokenizer assets are prepared separately and bundled with the
  app; conversion does not recalibrate the UNet for each checkpoint.
- QAIRT 2.50.0.260828 targets are fixed: SD1.5 v73 and SDXL v75/soc57,
  both with 8 MB VTCM. SDXL retains O=3 with source-destructive reuse disabled.
- The SDXL compiler subprocess uses storage-backed allocation, including
  compact small-object slabs. Active conversion files live under
  `noBackupFilesDir/conversion-work` and are explicitly cleaned up.
- UNet LoRA merging supports standard kohya attention, ResNet, convolution,
  sampling and embedding-layer mappings. Unmatched tensors produce a warning;
  matched layers are merged. Retained merged weights are capped at 128 MiB.
  Text-encoder LoRA and BF16 input remain unsupported.

## Evidence and remaining scope

Phone results include SD1.5/SDXL generation on a Galaxy S25 Ultra, a successful
Pony CLIP-only diagnostic and full-component Illustrious output. The latest
test APK also received a successful field report after allocator, workspace
and LoRA compatibility fixes. No post-fix device-specific logs accompany that
confirmation; it does not establish universal Vivo/Nubia compatibility.

[docs/LIMITS.md](docs/LIMITS.md) defines current support and measurement scope.
[docs/SDXL-INVESTIGATION.md](docs/SDXL-INVESTIGATION.md) separates historical
failure evidence, source changes and reported results. Broader checkpoint
coverage, the new SD1.5 component path and image-to-image need further phone
results. BF16 work is deferred.

## Source navigation

| Area                                    | Entry point                                                       |
|-----------------------------------------|-------------------------------------------------------------------|
| Android conversion lifecycle            | `app/src/main/java/com/abrah/npuforge/ConvertService.kt`          |
| Native process staging and export       | `app/src/main/java/com/abrah/npuforge/Converter.kt`               |
| Model family and component requirements | `app/src/main/java/com/abrah/npuforge/CheckpointInfo.kt`          |
| Weight conversion and UNet LoRA         | `native/tplconv.cpp`, `tools/tpl_apply.py`, `tools/lora_merge.py` |
| CLIP component writer                   | `native/componentconv.cpp`, `tools/clip_recipe.py`                |
| VAE template authoring                  | `tools/vae_template.py`                                           |
| Compiler allocation backing             | `native/compiler_heap.c`, `native/compiler_heap.map`              |
| Regression coverage                     | `tests/`                                                          |

## Development conventions

- Preserve native/Python weight-pack parity and `-ffp-contract=off`.
  QNN context binaries are not byte-reproducible; a different checksum alone
  does not establish a numerical regression.
- Keep compatibility decisions tied to evidence. Do not turn unmatched LoRA
  tensors or an uncalibrated quality heuristic into a blanket rejection.
- Use focused host tests and build/lint checks for changed paths. Device
  deployment and expensive full-model experiments are separate activities.
- Keep SDK binaries, generated models, checkpoints, local reports and private
  device identifiers outside source control. [NOTICE](NOTICE) records the
  third-party provenance and distribution restrictions.

See [ROADMAP.md](ROADMAP.md) for proposed work and [CLAUDE.md](CLAUDE.md) for
the documentation index.

# Developer overview

npuforge converts SD1.5 and SDXL checkpoints into importable Qualcomm NPU model
bundles on Android for [Fancy-Ai](https://github.com/Mr-J-369/Fancy-Ai) and
[Nightmare Mobile](https://github.com/AbrahamPaulJ/nightmare-mobile). Output compatibility is limited
to those two apps. Start with [README.md](README.md) for the project overview
and [docs/BUILD.md](docs/BUILD.md) for dependencies and build instructions.

## Current implementation

- Both families use the selected checkpoint's UNet, text encoder weights and
  embeddings, VAE encoder and VAE decoder. SDXL has two text encoders.
  Exception: SDXL downloads Mr.J's precompiled VAE contexts
  (`Mr-J-369/Fancy-AI`) instead of compiling the checkpoint's VAE.
- Downloads (SDXL VAE, inpaint difference) go through `HfDownload.kt`:
  resumable, starting from the Utility tab's *Download source* (huggingface.co
  or `hf-mirror.com`; a China timezone defaults to the mirror) and falling back
  to the other. Each has an offline import ([docs/LIMITS.md](docs/LIMITS.md)
  §Downloads). The SDXL VAE is asked for only when an SDXL conversion starts.
- The app writes MNN text encoders and compiles QNN UNet/VAE contexts. Graph
  templates and tokenizer assets are prepared separately and bundled with the
  app; conversion does not recalibrate the UNet for each checkpoint.
- QAIRT 2.50.0.260828 targets are fixed: SD1.5 v73 and SDXL v75/soc57,
  both with 8 MB VTCM. SDXL retains O=3 with source-destructive reuse disabled.
- The SDXL compiler subprocess uses storage-backed allocation, including
  compact small-object slabs; so does the SD1.5 UNet compile on phones under
  10 GiB (`Converter.isLowRam`). Active conversion files live under
  `noBackupFilesDir/conversion-work` and are explicitly cleaned up.
- UNet LoRA merging supports standard kohya attention, ResNet, convolution,
  sampling and embedding-layer mappings. Unmatched tensors produce a warning;
  matched layers are merged. Retained merged weights are capped at 128 MiB.
  Text-encoder LoRA remains unsupported; BF16 checkpoints are read as FP32.
- SD1.5 clip skip 1 or 2 (default 2) selects `components_sd15/` or
  `components_sd15/clip_skip1/` for the text-encoder recipe.
- SD1.5 inpainting: a 9-channel `conv_in` selects `CheckpointInfo.Model.SD15_INPAINT`
  and the `template_inpaint/` UNet template (QAIRT 2.50, DreamShaper 8
  inpainting calibration); exports carry an `INPAINT` marker. A plain SD1.5
  checkpoint can instead be converted **as** an inpainting model: the app
  downloads the official SD1.5 inpainting difference once
  (`AbrahamPJ/npuforge-sd15-inpaint-diff`, 1.72 GB, SHA-256 checked) or
  imports a user-supplied copy, and runs `tplconv --inpaint-diff`.
  [docs/SD15-INPAINT.md](docs/SD15-INPAINT.md).

## State and next steps — 27 September 2026

**Last release: v1.0.4** (27 September 2026, tag at `61f9927`, asset
`npuforge-1.0.4.apk`, signed). Nightmare Mobile 1.6.033 is out, so SD1.5 exports
work there. The signed release is installed on the test phone; the old debug
build (same application id, debug key) had to be uninstalled first.

| | |
|---|---|
| Signed release | `NPUFORGE_SIGNING_PROPERTIES=<.secrets>/npuforge-keystore/signing.properties ./gradlew :app:assembleRelease` (key: CLAUDE.md). Fingerprint matches v1.0.3 |
| Shareable test build | `./gradlew :app:assemblePreview`: debug-key signed, `com.abrah.npuforge.preview`, installs beside the release. ⚠ Never share the debug APK ([docs/BUILD.md](docs/BUILD.md)) |
| Toolchain | NDK **30.0.16248370** (Mr.J's bump). Phone `tplconv` from the NDK 30 APK gave the same DreamShaper 8 pack md5 as the host (`042b99cf…`). 56/56 host tests pass |

What 1.0.4 adds over v1.0.3, with its evidence:

| Change | Evidence |
|---|---|
| SD1.5 inpainting: 9-channel checkpoints and *Convert as → Inpainting* | Phone renders and field reports, [docs/SD15-INPAINT.md](docs/SD15-INPAINT.md); user re-tested 27 September |
| Clip skip 1/2 | Encoder parity on host; one phone conversion and a "decent" image (user) |
| Low-RAM compile (<10 GiB) | Allocator measured on the S25 via shell; **not yet run in the app on an 8 GB phone** |
| Downloads: resume, mirror, source setting, file import | Ranged requests checked from outside China; **unconfirmed in China** |
| File pickers list other file managers | User picked the checkpoint through a file manager |
| Checkpoint deleted before compile; outputs deleted as zipped | Code change; user conversions completed |

**Field reports behind this (27 September):** a user in China could not finish
an in-app download (probably the SDXL VAE, whose old downloader had no timeout,
retry or resume); an 8 GB SM8450 lost inpaint conversions to Android's memory
killer at the same UNet-compile peak a plain conversion survived. That phone is
8 Gen 1, which stays unsupported (decision, 27 September): the V68/V69
libraries Mr.J removed in v1.0.3 stay out.

**Next:** (1) send 1.0.4 to the China and
8 GB reporters; low-RAM mode needs an 8 GB phone on 8 Gen 2+; (2) Mr.J: SDXL
inpaint template, [docs/SDXL-INPAINT-TEMPLATE.md](docs/SDXL-INPAINT-TEMPLATE.md);
other SD1.5 resolutions remain his too (below).

**Other resolutions** are new templates, not app changes: each (architecture,
resolution) needs its own ONNX export, calibration at that resolution, quantize,
recipe and bundle ([docs/TEMPLATE-AUTHORING.md](docs/TEMPLATE-AUTHORING.md); the
inpaint runbook is [docs/SD15-INPAINT.md](docs/SD15-INPAINT.md) §Reproduction).
`tplconv`, the inpaint difference and the app path are resolution-agnostic.
Every resolution needs its own activation-range measurement, and above 512×768
the host quantize hits a memory cliff (LocalDream's `docs/CONVERSION.md` §2).
Phone compile memory at larger graphs is unmeasured.

**Do not redo:** the DreamShaper-inpaint calibration rows in LocalDream cover
only timesteps 77–913 (recalibrate); the QAIRT SDK downloads with plain `curl`;
`tools/mk_inpaint_diff.py` rebuilds the difference; the CLIP authoring env is
WSL `~/npuconvert/.venv` plus pip `MNN==3.6.1` (`~/clipskip/`), which reproduces
the shipped skip-2 recipe except MNN's random model UUID.

Left behind: WSL `inp9/` (~25 GB inpaint authoring workspace), `~/clipskip/`
(CLIP authoring), `~/ramtest/`; on the test phone `/data/local/tmp/ramtest`
(~2.6 GB, safe to delete) and `DreamShaper_8_pruned.safetensors` in Download.

## Evidence and remaining scope

Phone results include SD1.5/SDXL generation on a Galaxy S25 Ultra, a successful
Pony CLIP-only diagnostic and full-component Illustrious output. The latest
test APK also received a successful field report after allocator, workspace
and LoRA compatibility fixes. No post-fix device-specific logs accompany that
confirmation; it does not establish universal Vivo/Nubia compatibility.

[docs/LIMITS.md](docs/LIMITS.md) defines current support and measurement scope.
[docs/SDXL-INVESTIGATION.md](docs/SDXL-INVESTIGATION.md) separates historical
failure evidence, source changes and reported results. Broader checkpoint
coverage and plain SD1.5 text-to-image/image-to-image in the consuming apps
need further phone results; SD1.5 components have run on the phone only through
inpaint models so far. BF16 work is deferred.

## Source navigation

| Area                                    | Entry point                                                       |
|-----------------------------------------|-------------------------------------------------------------------|
| Android conversion lifecycle            | `app/src/main/java/com/abrah/npuforge/ConvertService.kt`          |
| Native process staging and export       | `app/src/main/java/com/abrah/npuforge/Converter.kt`               |
| Model family and component requirements | `app/src/main/java/com/abrah/npuforge/CheckpointInfo.kt`          |
| Weight conversion and UNet LoRA         | `native/tplconv.cpp`, `tools/tpl_apply.py`, `tools/lora_merge.py` |
| Inpaint difference                      | `InpaintDiff.kt` (download), `tools/mk_inpaint_diff.py` (build)   |
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

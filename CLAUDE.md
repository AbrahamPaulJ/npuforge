# npuforge development index

Read [HANDOFF.md](HANDOFF.md) for the current implementation and source map.
Read [README.md](README.md) for the public overview. This file is a navigation
aid; measured results and limitations belong in the topic documents below.

## Documentation

| Document | Contents |
|---|---|
| [docs/BUILD.md](docs/BUILD.md) | Toolchain, required external artifacts, native and Android builds |
| [docs/LIMITS.md](docs/LIMITS.md) | Supported inputs, LoRA behavior, device and quality limits |
| [docs/SDXL.md](docs/SDXL.md) | SDXL runtime contract, compiler configuration and phone measurements |
| [docs/SDXL-COMPONENTS.md](docs/SDXL-COMPONENTS.md) | Checkpoint-owned SDXL CLIP/VAE conversion and precision limits |
| [docs/SD15-COMPONENTS.md](docs/SD15-COMPONENTS.md) | SD1.5 CLIP, clip-skip 2 and 512px VAE components |
| [docs/SDXL-INVESTIGATION.md](docs/SDXL-INVESTIGATION.md) | Compiler allocation, workspace, LoRA and conditioning findings |
| [docs/PIPELINE.md](docs/PIPELINE.md) | Template derivation, tensor mapping and quantization measurements |
| [docs/ANDROID.md](docs/ANDROID.md) | Android packaging, native library and DSP loading requirements |
| [docs/CHECKPOINT-FAMILIES.md](docs/CHECKPOINT-FAMILIES.md) | Historical SD1.5 checkpoint measurements and calibration questions |
| [docs/TEMPLATE-AUTHORING.md](docs/TEMPLATE-AUTHORING.md) | External template authoring dependencies and reproduction steps |
| [ROADMAP.md](ROADMAP.md) | Priorities, deferred experiments and evidence needed |
| [NOTICE](NOTICE) | Third-party artifact provenance and distribution restrictions |

## Engineering constraints

- Native weight conversion must match the Python reference byte-for-byte.
  Preserve `-ffp-contract=off`; FMA contraction changes LoRA rounding.
- Compare a recipe-derived pack against the corresponding reference pack.
  A context-binary checksum or a successful compile is not an image-quality
  measurement.
- The app uses checkpoint-owned CLIP and both VAE components for both model
  families. Shared-component archives are legacy backup/restore inputs only.
- Extra unmatched LoRA tensors warn and leave the recognized UNet layers
  usable. Text-encoder LoRA and BF16 remain unsupported; BF16 is deferred.
- SDK libraries, generated templates/models and checkpoints stay excluded
  from source control. A fresh clone needs externally supplied artifacts for
  a working APK; see [docs/BUILD.md](docs/BUILD.md) and [NOTICE](NOTICE).
- Published documentation should distinguish host checks, historical phone
  measurements and reported field results. Avoid broad compatibility claims
  from a single checkpoint or device.
- Keep device serials, network endpoints, private paths, credentials and local
  diagnostic reports out of published changes. Review the actual diff before
  committing; historical repository rewriting requires separate coordination.

Use the repository's current Gradle and NDK configuration. Build commands and
test dependencies are maintained in [docs/BUILD.md](docs/BUILD.md).

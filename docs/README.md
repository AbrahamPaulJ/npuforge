# Documentation

## Start here

| Document | Purpose |
|---|---|
| [Technical overview](TECHNICAL-OVERVIEW.md) | Architecture and topics for Qualcomm technical review |
| [Results](RESULTS.md) | Measured results, tester reports and evaluation procedure |
| [Related work](RELATED-WORK.md) | Scope and evidence for the on-device SDXL conversion claim |
| [Build](BUILD.md) | Developer prerequisites and external assets |
| [Testing](TESTING.md) | Reproducible host tests and device verification |
| [Limitations](LIMITS.md) | Supported inputs and known compatibility limits |
| [Roadmap](../ROADMAP.md) | Current priorities and deferred work |
| [License and provenance](../NOTICE) | Source and artifact ownership boundaries |

## Implementation guides

- [Android packaging](ANDROID.md): executing the compiler and loading QNN/DSP libraries.
- [SDXL](SDXL.md): graph contract, compiler configuration and memory behavior.
- [SDXL components](SDXL-COMPONENTS.md) and [SD1.5 components](SD15-COMPONENTS.md):
  checkpoint-owned text encoders and VAEs.
- [CLIP recipes](CLIP-COMPONENTS.md) and [VAE templates](VAE-TEMPLATES.md):
  component authoring and reference comparisons.
- [Template authoring](TEMPLATE-AUTHORING.md): preparing new graph assets.

## Research records

These retain experiment-specific measurements and historical configurations.
Use the current overview, results and limitations for present capabilities.

- [Original SD1.5 pipeline](PIPELINE.md).
- [SDXL component and device investigation](SDXL-INVESTIGATION.md).
- [Checkpoint family analysis](CHECKPOINT-FAMILIES.md).
- [Dated working notes](../notes/).

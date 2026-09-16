# Roadmap

The current pipeline converts checkpoint-owned UNet, CLIP and both VAE
components for SD1.5 and SDXL. The latest test APK received a successful field
report after LoRA compatibility and compiler workspace fixes. Priorities below
extend that evidence while preserving the working path.

## Near-term priorities

### 1. Establish a reproducible device and checkpoint matrix

Record the app version, device/SoC, checkpoint and adapter identifiers,
conversion report, generation settings and output for each test. Cover:

- The latest build on devices with earlier Vivo/Nubia failures. Current field
  confirmation has no accompanying post-fix device-specific logs.
- SD1.5 with checkpoint-owned components, additional SDXL derivatives and
  image-to-image using the converted VAE encoder.
- DMD2 F16/F32 and representative style adapters, including a controlled
  same-seed comparison with and without the adapter.
- Continuous peak memory and temporary-storage use, including lower-memory
  devices. Historical SD1.5 figures are not SDXL requirements.

Completion means published, scoped results rather than a blanket compatibility
claim. Existing evidence is in [docs/LIMITS.md](docs/LIMITS.md) and
[docs/SDXL-INVESTIGATION.md](docs/SDXL-INVESTIGATION.md).

### 2. Clarify SDK and template distribution

Resolve the QAIRT runtime redistribution terms and template provenance before
distributing a production bundle or proposing commercial use. The source
repository's MIT licence does not relicense SDK binaries, generated artifacts
or checkpoints. [NOTICE](NOTICE) records the current restrictions, including
the non-commercial template derivation chain.

### 3. Evaluate device-specific compiler targets

Current targets are fixed: SD1.5 v73, SDXL v75/soc57, both with 8 MB VTCM.
Investigate supported device/SDK combinations before selecting architecture,
SoC and VTCM configuration automatically. A higher Snapdragon model number or
more RAM alone does not prove context compatibility. Each proposed target
needs compilation, loading, numerical and memory measurements.

### 4. Extend component and adapter fidelity

Text-encoder LoRA merging remains unsupported even though CLIP base weights now
come from the checkpoint. Add it only with encoder-output comparisons and
regressions for existing UNet-only adapters. Additional adapter formats require
their actual merge arithmetic; accepting an unfamiliar tensor name does not
implement that format.

BF16 support is deferred. F16/F32 remain the supported input dtypes.

## Research proposals

### Checkpoint calibration coverage

The historical SD1.5 MistoonAnime failure points to a UNet activation-range
mismatch, while the Pony SDXL case was resolved by replacing shared CLIPs.
These are different failure mechanisms. Before adding a separate template:

1. Compare checkpoint-owned components and fixed-tensor UNet outputs.
2. Measure where the quantized graph first diverges from its floating reference.
3. Survey a broader checkpoint population before choosing calibration data.
4. Verify any calibration change against the known working checkpoints.

`tools/span_probe.py` supplies a low-bandwidth weight-span diagnostic. The
historical one-dimensional SD1.5 probe reads 0.89 MB of tensor data, but weight
spans alone do not establish activation clipping or a safe rejection threshold.
Keep this an investigative measurement, not an automatic conversion gate.

A proposed pack-driven activation-range experiment would move 3,623 activation
encodings out of the generated SD1.5 library, adding approximately 29 KB of
pack data. It has not been validated. Semantic ranges, such as Sigmoid outputs,
and lower-precision attention tensors require separate treatment. Details and
the historical measurements are in
[docs/CHECKPOINT-FAMILIES.md](docs/CHECKPOINT-FAMILIES.md).

### Older QAIRT compatibility variant

Historical QAIRT 2.49 SD1.5 experiments found a context FP16 requirement that
was absent in a 2.28 build. The five-prompt comparison recorded mean
`extreme_frac` 0.0197 for 2.28 versus 0.0193 for 2.49, with approximately 12%
more latency. These measurements do not establish the behavior of the current
2.50 runtime. An older-SDK variant is deferred until a current compatibility
failure identifies a need; obtaining the SDK and testing the affected hardware
are prerequisites.

### Workflow improvements

- Document the existing Local Dream/Fancy-Ai import contract before extending
  output compatibility to other generation apps.
- Investigate resuming completed stages after interruption.
- Measure compilation memory and quality before adding resolutions. Current
  graphs are fixed at 512 × 512 for SD1.5 and 1024 × 1024 for SDXL; a new
  resolution requires its own authored graph and evidence.

## Approaches not selected

| Approach | Existing evidence |
|---|---|
| Runtime LoRA through `UPDATEABLE_STATIC` | Historical inference measurement increased from 83 to 281 ms/pass; 24 updated tensors incurred the same cost as 768. Merging before conversion avoids that runtime path. |
| Full workstation quantization on the phone | The original SD1.5 authoring run required over 11 GB RAM plus swap and approximately 2 h 20 m for 400 calibration passes. The reusable template/recipe split avoids repeating it per checkpoint. |
| QNN context checksum as a correctness test | Recompiling the same pack can produce different context bytes with identical renders. Compare deterministic weight packs and numerical outputs instead. |
| Shared CLIP/VAE weights as a universal substitute | The successful Pony CLIP-only replacement demonstrated a real conditioning mismatch. Both families now use checkpoint-owned weights. |
| Rejecting an entire adapter for extra unmatched tensors | This rejected previously usable DMD2 adapters. The converter now warns, merges recognized layers and reports unsupported content. |

These decisions can be revisited when new measurements identify a specific
benefit and preserve the existing working cases.

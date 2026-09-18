# Scope and limitations

This page distinguishes implemented behavior from measured device results.
The current implementation includes the compiler, workspace and LoRA fixes
reported working in test APK 3. That latest confirmation did not include
post-fix device-specific logs, so it does not establish support for every
Vivo/Nubia device or checkpoint.

## Output application compatibility

Output models work only with [Fancy-Ai](https://github.com/Mr-J-369/Fancy-Ai)
and [Nightmare Mobile](https://github.com/AbrahamPaulJ/nightmare-mobile). Their custom-model import
path implements the exported component filenames, conditioning interfaces and
QNN context contract. Device/runtime requirements apply in both apps.

## Current conversion contract

|                               | SD1.5                                                  | SDXL                                                        |
|-------------------------------|--------------------------------------------------------|-------------------------------------------------------------|
| Input                         | Single-file `.safetensors`, supported LDM layout       | Single-file `.safetensors`, supported SDXL-base layout      |
| Input weight dtypes           | F16/F32/BF16                                           | F16/F32/BF16                                                |
| Image size                    | 512 × 512                                              | 1024 × 1024                                                 |
| Checkpoint-owned components   | UNet, CLIP and embeddings, VAE encoder and decoder     | UNet, CLIP-L/CLIP-G and embeddings, VAE encoder and decoder |
| Fixed compiler target         | v73, 8 MB VTCM                                         | v75 / soc57, 8 MB VTCM                                      |
| Runtime                       | QAIRT 2.50.0.260828                                    | QAIRT 2.50.0.260828                                         |
| Full-component phone evidence | New component path still needs a specific phone result | Reported successful Illustrious conversion and generation   |

The graph templates and tokenizer are shared; component weights come from the
selected checkpoint. New conversions do not download donor CLIP/VAE weights.
Both VAE graphs are included, even for a text-to-image workload that uses only
the decoder. Legacy component backup/restore is separate from conversion.

SD2, diffusers-layout checkpoints and arbitrary architectures are
unsupported. Required component names, shapes and dtypes must match the bundled
graphs. Standard architecture compatibility does not guarantee image quality:
the UNet retains the template's calibration rather than being recalibrated for
each checkpoint. VAE internal arithmetic is FP16 despite float32 external I/O;
a checkpoint requiring float32 VAE arithmetic may not work faithfully.
BF16 checkpoint and adapter inputs are expanded into FP32 before the existing
weight conversion; this does not enable BF16 graph execution.

See [SD1.5 components](SD15-COMPONENTS.md),
[SDXL components](SDXL-COMPONENTS.md) and [SDXL.md](SDXL.md) for contracts.

## LoRA behavior

Adapters are merged into UNet weights before quantization. Their strength is
baked into the exported model, with no separate adapter computation at
inference. Multiple adapters can be stacked; each strength combination produces
a separate exported model. The app's strength slider spans −1.0 to 2.0.

| Implemented                                                                                          | Outside current support                                                       |
|------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| Standard kohya `lora_down`, `lora_up`, optional `alpha`                                              | General PEFT/diffusers adapter file layouts                                   |
| Attention, ResNet, input/output convolution, down/up sampling and time/additional embedding mappings | Arbitrary adapter naming schemes                                              |
| Linear and convolution down weights with a 1×1 up kernel                                             | Spatial up kernels and format-specific LoCon/LyCORIS/LoHa/DoRA/IA3 arithmetic |
| F16/F32/BF16 adapters                                                                                | Other weight dtypes                                                           |
| UNet merging                                                                                         | Text-encoder LoRA, including checkpoint-owned CLIPs                           |

Kohya adapters may use module names derived from diffusers; that naming support
is distinct from accepting a PEFT/diffusers adapter file format. DMD2-related
ResNet and sampling mappings are implemented. Native and Python tests cover
those mappings with both F16 and F32 weights.

**Extra unmatched or unsupported tensors warn; they do not reject the whole
adapter.** Recognized UNet layers continue to merge. Text-encoder modules are
reported and skipped. An adapter with no matched UNet pairs still fails, as do
invalid shapes that cannot be multiplied. A mixed-format file may therefore
produce a partial merge; warning-only behavior is not full support for its
unsupported features.

The merged-weight cache retains at most 128 MiB of payloads. Temporary merge
buffers and source mappings are additional memory; this is not a process-wide
RAM cap. Evicted tensors are recomputed with the same arithmetic.

Historical nubia NX789J reports include successful DMD2 F16/F32 conversion and
generation before the component update. The latest test build also received a
successful report after mapping and compatibility fixes. Controlled comparisons
of adapter effect and broader adapter coverage remain outstanding.

## Device and resource limits

- Android 13+ (`minSdk 33`) and ARM64 are required.
- Compiler targets are fixed. A phone's brand, advertised RAM or newer chip
  number does not establish DSP/context compatibility.
- SDXL compilation uses storage-backed allocations, including small-object
  pooling. Available storage and I/O performance matter alongside physical RAM.
  Backing allocation totals are not resident-memory measurements.
- Active conversion inputs, backing files and outputs live under the app's
  `noBackupFilesDir/conversion-work`, outside Android's reclaimable cache.
  The service removes them explicitly after conversion.
- A foreground service, renewable wake lock and visible-screen keep-awake
  support long conversions. They do not prevent every allocation failure or
  process termination.

### Recorded measurements

| Measurement                                                           | Scope                                                                                           |
|-----------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| 117 seconds total conversion                                          | Original SD1.5 pipeline on Samsung SM-S938B; predates checkpoint-owned component conversion     |
| Approximately 4.8 GB peak compile RAM, 0.87 GB minimum `MemAvailable` | Historical SD1.5 run on the same 12 GB-class phone with normal apps open                        |
| 1.98 GB native weight-stage peak vs 5.07 GB Python                    | Historical SD1.5 comparison                                                                     |
| Approximately 4 GB free working storage; 1.3 GB finished model        | Historical SD1.5 pipeline; not current full-pipeline or SDXL requirements                       |
| 437 seconds total conversion, 15 seconds generation                   | Reported SDXL O=3 MOP run at 1024 × 1024, 8 steps, CFG 1, LCM/Karras; shared-component pipeline |
| 45.8 seconds generation                                               | Full-component `waiIllustriousSDXL_v170`, 1024 × 1024, 30 steps, CFG 7                          |

The reported 41% phone RAM during the SDXL O=3 run was a snapshot, not a peak.
Neither a universal RAM bound nor reliable operation on 8 GB devices has been
established. Earlier Vivo/Nubia failures and the latest fixes are detailed in
[the investigation](SDXL-INVESTIGATION.md).

## Historical checkpoint-quality findings

### MistoonAnime: SD1.5 UNet mismatch

The historical MistoonAnime conversion exited successfully and loaded, but
rendered saturated noise. Checkpoint weight spans relative to the DreamShaper 8
template differed substantially:

| Checkpoint                                     | Maximum weight-span ratio | Historical output |
|------------------------------------------------|--------------------------:|-------------------|
| DreamShaper 8                                  |                     1.000 | Recognizable      |
| AbsoluteReality                                |                     1.020 | Recognizable      |
| CyberRealistic                                 |                     1.117 | Recognizable      |
| DreamShaper + rank-128 watercolour LoRA at 0.8 |                     1.061 | Recognizable      |
| MistoonAnime                                   |                      49.4 | Noise             |

A component-swap experiment kept the faulty UNet noisy with its own CLIP/VAE
(saturated-pixel fractions 0.358 vs 0.331), while a working UNet stayed clean
with the substituted components (0.049 vs 0.029). This localized that failure
to the UNet and supports an activation-range mismatch. Weight-span ratios alone
are not a calibrated acceptance threshold or proof that all anime models fail.

The source VAE separately contained 516 non-finite values in
`decoder.up.3.block.0.conv1.weight`, with norm 2,858,648 versus 78.4 in ft-mse.
Checkpoint-owned conversion cannot repair those source weights. Historical
CLIP comparisons of only CyberRealistic and MistoonAnime showed median relative
differences of 0.204% and 0.363% from stock SD1.5; those small differences did
not justify substituting CLIPs for every family.

See [CHECKPOINT-FAMILIES.md](CHECKPOINT-FAMILIES.md) for the full measurements.

### Pony and Illustrious: SDXL components

A Pony diagnostic replaced only its seven CLIP files with checkpoint-owned
weights. Hash checks preserved the baseline UNet, both VAEs, tokenizer and model
markers. The reported successful output supports CLIP substitution as the
cause of that failure. The later full-component Illustrious result establishes
another working checkpoint; it does not isolate CLIP versus VAE effects or
establish compatibility with every derivative.

## Historical QAIRT 2.49 compatibility findings

Earlier SD1.5 experiments found an FP16 requirement in QAIRT 2.49 contexts that
some chips rejected. An 11-variant configuration sweep retained it; a 2.28
rebuild removed it. A five-prompt comparison recorded mean `extreme_frac`
0.0197 for 2.28 versus 0.0193 for 2.49, with approximately 12% greater latency.

These findings concern those SDK builds. They do not establish the same
behavior for current QAIRT 2.50. A load failure requires the actual device,
context and runtime diagnostics; “8 Gen 2 or newer” is not a support guarantee.

## Interpreting results

A successful weight stage, compilation and model load are separate from correct
image generation. Compare images or numerical outputs with a known reference
using the same checkpoint, conditioning, sampler, prediction type, seed and
steps. Correct latency alone is not evidence of correct model computation.

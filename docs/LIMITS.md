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
selected checkpoint, with one exception: SDXL conversions download Mr.J's
precompiled SDXL VAE contexts (`Mr-J-369/Fancy-AI`, ~192 MB, once) instead of
compiling the checkpoint's VAE; the Utility tab exports and imports them.
Both VAE graphs are included, even for a text-to-image workload that uses only
the decoder.

### Downloads

| File | When | Source |
|---|---|---|
| SDXL VAE contexts, ~192 MB | First SDXL conversion (the app also offers it at launch) | `huggingface.co/Mr-J-369/Fancy-AI`, or a VAE zip imported in the Utility tab |
| SD1.5 inpainting difference, 1.72 GB | First plain-SD1.5 → inpainting conversion | `huggingface.co/AbrahamPJ/npuforge-sd15-inpaint-diff`, or a copy chosen with *Use downloaded file* |

Both downloads resume a partial file and alternate between huggingface.co and
`hf-mirror.com`, starting from the source chosen in the Utility tab (*Download
source*); until one is chosen, a mainland-China timezone starts with the mirror. The mirror was added for a field report from China
(27 September 2026: "downloads a little, then stops, dozens of times"); the
previous VAE download had no timeout, retry or resume. The mirror path is
checked from outside China only (it redirects to huggingface.co there); an
in-China result is still needed.

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

### SD1.5 inpainting checkpoints

A checkpoint whose `conv_in` takes 9 channels selects the separate inpaint
template (`template_inpaint/`) and exports an `INPAINT` marker file. Host and
phone measurements are in [SD15-INPAINT.md](SD15-INPAINT.md): weight packs
match byte-for-byte, and phone renders of two inpaint checkpoints score
29.9–32.3 dB inside the mask against their own-calibrated builds. A test APK
converted DreamShaper 8 inpainting in-app and the export inpainted correctly in
Nightmare Mobile 1.6.033, the first version that reads the `INPAINT` marker and
runs npuforge's float32 SD1.5 VAE (earlier versions fail every npuforge SD1.5
export at `vae_encode`). Fancy-Ai inpaint support is unknown.

Converting an ordinary 4-channel checkpoint into an inpainting model
(add-difference) was measured but is not implemented. LoRA modules on
`conv_in` do not fit the 9-channel weight and stop the conversion.

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
- **Low-RAM mode:** phones reporting under 10 GiB compile the SD1.5 UNet with the
  same allocator. Two reports from one 8 GB SM8450 (7.5 GB reported) showed the
  plain and inpaint UNet compiles with the same anonymous-memory profile, both
  peaking near 4.3 GB; the inpaint run was killed at 0.60 GB available, the plain
  run survived the same peak at 0.66 GB. On the S25 Ultra (shell, same app
  binaries, DreamShaper 8 pack) the allocator took the compiler's anonymous peak
  from 2.89 GB to 0.03 GB and lowest `MemAvailable` from 1.28 to 5.81 GB, cost
  70 s → 157 s and ~4.6 GB of temporary storage, and produced a context binary
  identical to a normal compile apart from the 2 bytes that differ between any
  two normal compiles. **Not yet run inside the app on an 8 GB phone.**
- The checkpoint and LoRA copies are deleted after the weight stage, before the
  compile, and each output is deleted once it is in the export ZIP. Measured
  peak-storage figures above predate both changes.
- **Snapdragon 8 Gen 1 and older are unsupported**, for converting and running:
  graphs target v73/v75, and v1.0.3 removed the V68/V69 HTP libraries.
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

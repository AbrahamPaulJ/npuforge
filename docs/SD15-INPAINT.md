# SD1.5 9-channel inpaint template

A second SD1.5 UNet template for inpainting checkpoints, whose `conv_in` takes
9 channels: `[noisy latent 4 | mask 1 | masked-image latent 4]`. Everything
after the template — recipe, `tplconv`, phone compile, export — is the same
machinery as the text-to-image template ([PIPELINE.md](PIPELINE.md)).

Built and measured on 25 September 2026. Host: WSL, QAIRT **2.50.0.260828**.
Device: Galaxy S25 Ultra (SM8750, HTP v79), running the v73 / 8 MB VTCM
context through the 2.50 runtime.

## Status

| Stage | Result |
|---|---|
| Template authoring (host) | Done: encodings gate, Phase 2 identity, Phase 1 recipe |
| Native `tplconv` parity | Byte-identical to Python on host and on the phone |
| Phone conversion (installed 1.0.0 binaries) | 19 s weights + 62 s compile; renders identical to the PC compile |
| Render quality vs a known-good inpaint build | 31.3–32.3 dB inside the mask (5 prompts) |
| Second checkpoint through the template | AbsoluteReality inpainting: 29.9 dB vs its own-calibrated build |
| Add-difference (4-channel checkpoint → inpaint) | In the app (`inpaint-test2`): the phone downloaded and verified the hosted difference, converted DreamShaper 8 base in ~2.5 min, and its UNet renders 5/5 byte-identical to the host build |
| App detection and export marker | Test APK converted DreamShaper 8 inpainting in-app; export carries `INPAINT` |
| Consuming app | Nightmare Mobile 1.6.033: marker read, float32 SD1.5 VAE supported, inpaint confirmed by the user; Fancy-Ai unknown |

## The template

Calibration checkpoint: `DreamShaper_8_INPAINTING.inpainting.safetensors`
(2,132,703,862 B, F16), the inpainting sibling of the text-to-image template's
checkpoint. The diffusers folder used for export and calibration was built
from that same single file, so graph weights and checkpoint weights agree
exactly (spot-checked `conv_in` and `conv_out`: `torch.equal`).

**Calibration: 400 rows from 480 captured UNet calls**, 12 base prompts with a
full-canvas mask plus 12 brush-mask edits, CFG 7, 20 steps. ~53 min on CPU.

⚠ **Reusing LocalDream's existing DreamShaper-inpaint rows would have clipped
the schedule.** They carried only **12 distinct timesteps, 77–913**. Neither
the first denoising step (t≈999) nor the last (t<77) was represented. The rows
used here come from DPM-Solver with `linspace` spacing: **20 distinct
timesteps, 50–999**. Check the timestep coverage of any borrowed calibration
set before quantizing.

Quantize: w8a16, per-channel, 32-bit bias, `--preserve_io layout`, 400 rows at
~21 s/row → **2 h 30 m**.

| Gate | Result |
|---|---|
| `_time_proj_Cast_output_0` / `Mul` | [0, 999] / [0, 999] |
| `_time_proj_Sin` / `Cos` | [−1, 1] / [−1, 1] |
| Quantized (u16) tensors with `scale == 0` | **0** |
| `sample` input | [1, 9, 64, 64], range [−12.79, 8.57] |
| Phase 2: pack-loading context vs stock context | **0 differing bytes** |
| Phase 1 discovery | matched **1358**, unmatched **1025**, ambiguous **0**; 686 sources each used once |
| Rule counts | template 257, u8_asym 197, i32_scalar 197, i8_axis 866, i32_axis_bias 98, i32_axis_zero 768 |
| Same-checkpoint round trip | **0 scale mismatches**, 226 byte mismatches, all int32 biases |

The mapping and rule counts are identical to the text-to-image template; only
`conv_in`'s channel count differs. The round-trip residue is the known float32
bias-rounding class (the text-to-image control had 234, bias-only).

Bundle, from the 2.50 `qnn-model-lib-generator` (`aarch64-android`, NDK r27c):

| File | Size |
|---|---|
| `libqnn_model.so` | 9,721,104 B (tagged `qaisw-v2.50.0.260828221209`) |
| `recipe.bin` | 397,199 B |
| `tpl_trim.pack` | 43,328 B (257 constants, 832 B payload) |
| `sources.txt` | 686 source tensors |

## Weight parity

| Built by | Checkpoint | md5 |
|---|---|---|
| Python `tpl_apply.py` (host) | DreamShaper 8 inpainting | `be956a84c136e6f2a87d33359daf3345` |
| C++ `tplconv` (host, trimmed bundle) | same | `be956a84…` — byte-identical |
| Installed npuforge 1.0.0 `libtplconv.so` (phone) | same | `be956a84…` — byte-identical |

Host `tplconv`: 42 s, 1.97 GB peak RSS. Phone: 19 s.

## Phone renders

Fixture: LocalDream's `inpaint_base.png` + `inpaint_mask_centre.png`, 512²,
20 steps, DPM, CFG 7, seed 12345, denoise 1.0, five prompts (sunflowers,
ramen, car, sailor, frog). Backend: Nightmare Mobile's
`libstable_diffusion_core.so --type sd15npu_inpaint` with the QAIRT 2.50
runtime. Only `unet.bin` varies between DreamShaper arms; the AbsoluteReality
arms use that checkpoint's own CLIP/VAE.

Generation is bit-exact at a fixed seed (a repeated arm scores 99 dB, i.e.
identical), so one render per arm is a valid comparison.

| Arm | What it is |
|---|---|
| `ref` | LocalDream's shipped DreamShaper inpaint UNet (QAIRT 2.49, own calibration) — known good |
| `stock` | This template's own weights, compiled with weights baked in |
| `recipe` | DreamShaper inpaint weights through the recipe pack (PC compile) |
| `phone` | The same, converted and compiled **on the phone** by the installed app binaries |
| `arref` | LocalDream's AbsoluteReality inpaint UNet (2.49, own calibration) |
| `ar` | AbsoluteReality 1.6525 INPAINTING through this template |
| `addiff` | AbsoluteReality 1.6525 base + (DreamShaper 8 inpainting − DreamShaper 8 base) through this template |

Inside-mask PSNR, dB:

| Pair | sunflowers | ramen | car | sailor | frog | mean |
|---|---|---|---|---|---|---|
| ref ~ stock | 22.75 | 25.83 | 39.48 | 36.02 | 32.48 | **31.31** |
| ref ~ recipe | 22.84 | 27.45 | 41.06 | 36.27 | 33.85 | **32.29** |
| stock ~ recipe | 31.34 | 33.85 | 41.03 | 43.06 | 36.33 | **37.12** |
| recipe ~ phone | identical | identical | identical | identical | identical | **5/5 byte-identical PNGs** |
| arref ~ ar | 27.50 | 20.69 | 36.43 | 37.92 | 26.69 | **29.85** |
| arref ~ addiff | 24.42 | 24.65 | 32.82 | 35.87 | 24.26 | **28.40** |
| ar ~ addiff | 28.16 | 20.95 | 32.05 | 37.84 | 28.39 | **29.48** |

Every render: `extreme_frac` 0.0002–0.0006 (healthy is 0.00–0.05); mean
|diff| outside the mask vs the base image 0.82–1.44 / 255, i.e. VAE
round-trip noise. For scale, LocalDream's own DreamShaper-inpaint builds of
the *same checkpoint* scored 29.16 dB (own calibration vs reference) and
30.08 dB (borrowed ranges) on this fixture ([PIPELINE.md](PIPELINE.md) test 1).
The 20.7 dB AbsoluteReality ramen pair was inspected by eye: same composition,
differing detail, which is diffusion amplification rather than a defect.

Steady latency: 3.4–3.6 s per 512² / 20-step inpaint for every arm, matching
the reference. ⚠ Some runs on 25 September took 5–11 s (first `stock` pass,
`recipe`, `ar`), landing on different prompts each time; a repeat of the same
`stock` binary ran at 3.44 s. That points to device state rather than the
model, but it is not a controlled latency measurement.

## Add-difference: what the measurement shows

`addiff` gives the phone path a way to make an inpainting model from an
ordinary 4-channel checkpoint: `inpaint + (custom − base)` in float32 per
UNet tensor; for `conv_in` the difference goes to channels 0–3 and channels
4–8 stay the inpaint model's. CLIP and VAE are the custom checkpoint's.

Built from a DreamShaper difference, the AbsoluteReality result sits as close
to Lykon's real AbsoluteReality inpaint release (28.4 dB) as that release does
to itself through the template (29.9 dB). ⚠ One fixture, one centre mask, one
checkpoint pair, and possibly a favourable pair: both are Lykon photoreal
models, and the release may itself have been made by merging. Anime and other
distant fine-tunes are untested.

### The official difference (25 September 2026)

`sd-v1-5-inpainting.ckpt − v1-5-pruned-emaonly.safetensors` (the RunwayML pair,
both fp32; the `.ckpt` read with `torch.load(weights_only=True)`): 686 UNet
tensors, 859,535,364 values, max |d| 1.427. Stored as fp16 it is
**1,719,165,856 B**; the largest fp16 rounding error of any value is 0.000394.
`conv_in` is stored 9-wide: channels 0–3 `inpaint − base`, channels 4–8 the
inpaint model's own, so applying is always `zero_pad(custom) + diff`.

| Pair | sunfl | ramen | car | sailor | frog | mean |
|---|---|---|---|---|---|---|
| DS8 + official diff: fp32 ~ fp16 storage | 32.65 | 24.29 | 40.74 | 40.56 | 29.71 | **33.59** |
| Lykon DS8 inpainting ~ DS8 + official diff (fp16) | 15.94 | 15.89 | 23.86 | 21.67 | 11.86 | 17.85 |
| Lykon AR inpainting ~ AR + official diff (fp16) | 14.84 | 13.92 | 19.85 | 21.59 | 14.16 | 16.87 |
| Lykon AR inpainting ~ AR + DS8 diff | 24.42 | 24.65 | 32.82 | 35.87 | 24.26 | 28.40 |

- **fp16 storage is sufficient**: fp32 and fp16 differ by less than two builds
  of one model do (37.1 dB), and a contact sheet shows the same pictures.
- **The official difference makes a different, not worse, inpaint model.** The
  17–18 dB against Lykon's releases is two different fine-tunes, not an error:
  inspected by eye, every prompt paints the right subject and blends as cleanly
  as the releases. All renders healthy (`extreme_frac` ≤ 0.0039).
- A DreamShaper-derived difference reproduces Lykon's AbsoluteReality release
  closely (28.4 dB) while the official one does not, which suggests Lykon's
  inpaint releases share a difference of their own. Inference, not measured.

### In the app

Hosted at `huggingface.co/AbrahamPJ/npuforge-sd15-inpaint-diff`
(`sd15inp_diff_f16.safetensors`; the server's ETag equals the SHA-256 above).
`tplconv --inpaint-diff <file>` applies it after any LoRA, in float32; the Python
reference is `tpl_apply.py apply … --inpaint-diff <file>` and
`tests/test_inpaint_diff.py` checks both against the definition and each other.

| Check | Result |
|---|---|
| Native vs Python vs merge-to-file-then-convert, DS8 base + diff | all three `c87f0e0a3224787fb4aa1c051046343b` |
| Native without the flag | DS8-inpainting pack unchanged (`be956a84…`) |
| Host `tplconv` with the diff | 49 s, 3.66 GB peak RSS (1.97 GB without): both files are mmapped and counted resident |
| Phone, `1.0.1-inpaint-test2` | diff downloaded (~2 min on Wi-Fi) and SHA-256 verified once; conversion ~2.5 min; UNet renders 5/5 byte-identical to the host `ds8off16` arm |
| Field result (user, 25 September 2026) | The exported `ds8base_addiff_inpaint.zip` imported into Nightmare Mobile 1.6.033 and inpainted at denoise 1.0 with a "decent" result |
| Field result (user, 25 September 2026) | CuteYukiMix (anime SD1.5 merge) converted as Inpainting in ~3.5 min; inpainted clothing on an anime image well in Nightmare Mobile 1.6.033 |

For a plain SD1.5 checkpoint the UI offers **Text-to-image | Inpainting**; the
first inpainting conversion asks before the 1.7 GB download, which resumes if
interrupted, alternates between huggingface.co and `hf-mirror.com`, and is used
only after its size and hash match. The same dialog's *Use downloaded file*
imports a copy fetched any other way (browser, download manager), hashing it
while copying.

⚠ **The 8 GB report (27 September 2026) was not the 9-channel graph.** A user
with an 8 GB SM8450 had the inpainting conversion killed on five of five
attempts while a plain SD1.5 conversion completed, and read it as the 9-channel
graph needing more compile RAM. Their two conversion reports (a native
inpainting checkpoint, not add-difference) show the plain and inpaint UNet
compiles with the same memory profile to within ~0.1 GB, both peaking near
4.3 GB of anonymous memory; the inpaint run was killed with 0.60 GB available,
the plain one survived the same peak with 0.66 GB. Android killed the whole
app, not only the compiler. Phones under 10 GiB now compile the SD1.5 UNet with
the storage-backed allocator ([LIMITS.md](LIMITS.md) §Device and resource
limits). That user's SM8450 is itself unsupported.

⚠ The first on-phone attempt was killed by Android during the compile with
about 1.4 GB of RAM free (another NPU process and a video app were running).
The retry with those closed completed. Low free memory, not the difference,
is the constraint; the compile itself is unchanged.

## The SD1.5 VAE contract in Nightmare Mobile

npuforge's SD1.5 VAE contexts use FLOAT_32 I/O (fp16 inside); xororz's use
UFIXED_16. Nightmare's SD1.5 backend handled only the latter until 1.6.033
(backend patch 012), so **every npuforge SD1.5 export failed there at
`vae_encode`/`vae_decode`**, not only inpaint models. Measured after the fix on
SM8750: a quantized model renders 5/5 byte-identical before and after; the
float32 VAE costs encoder 124 ms vs 83 ms and decoder 329 ms vs 175 ms, about
+0.24 s per 512² inpaint (~7%). The UNet is unaffected.

## Known limits

- **LoRA touching `conv_in`.** `tplconv` rejects a LoRA delta whose shape does
  not fit the weight, and an inpaint `conv_in` is 9-wide where adapters train a
  4-wide one. Adapters without a `conv_in` module are unaffected. Untested on
  an inpaint checkpoint in either case.
- **Removal/erase fixtures untested.** Only the centre fill mask was scored.
- **Calibration is DreamShaper-derived.** The text-to-image template's anime
  limitation ([LIMITS.md](LIMITS.md)) should be assumed to carry over.
- **One device.** SM8750 only; the v73 / 8 MB VTCM / fp16 requirements are the
  same as the text-to-image template's.

## Reproduction outline

Authoring uses the same non-MIT export path as the text-to-image template
([TEMPLATE-AUTHORING.md](TEMPLATE-AUTHORING.md) §1): LocalDream's
`export_onnx_inpaint_unet.py`, `prepare_data_inpaint.py` and
`gen_quant_data_inpaint.py` with the redefined diffusers modules. Differences
from the text-to-image runbook:

1. Build a diffusers folder from the single-file inpaint checkpoint with
   `StableDiffusionInpaintPipeline.from_single_file`, so export and recipe
   discovery read identical weights.
2. Replace the calibration scheduler with
   `DPMSolverMultistepScheduler.from_config(..., timestep_spacing="linspace")`
   and assert the rows' timestep range before quantizing.
3. Tighten the encoding gate's `Cast` upper bound to ≥ 900 so a clipped
   schedule fails it.
4. The Phase 2 / Phase 1 / bundle steps are unchanged; `ndk-build` must be on
   `PATH` for the `aarch64-android` library.

Artifacts carry the same provenance restriction as the text-to-image template
([template/README.md](../template/README.md), [NOTICE](../NOTICE)).

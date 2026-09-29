# SDXL 9-channel inpaint template: runbook

For whoever builds the SDXL inpainting template (written 27 September 2026 as a
handover to Mr.J). It assumes a **native Linux** host; the SD1.5 inpaint
template was built in WSL, and §7 lists what changes.

Read first, in this order:

1. [SD15-INPAINT.md](SD15-INPAINT.md) — the same job done for SD1.5, with every
   gate and number. This runbook is that one, adjusted for SDXL.
2. [TEMPLATE-AUTHORING.md](TEMPLATE-AUTHORING.md) — Phase 0–3 and the gates.
3. [SDXL.md](SDXL.md) §Model contract and §Template preparation — the SDXL
   contract the new template must keep.

## 1. What is being built, and what is not

An SDXL UNet template whose `conv_in` takes **9 channels**:
`[noisy latent 4 | mask 1 | masked-image latent 4]`, at 1024² (latents
`[1, 9, 128, 128]`). Everything else in the SDXL contract stays as it is:

| | Keep exactly |
|---|---|
| Weights / activations | W8A16, per-channel, 32-bit bias, no INT4 overrides |
| Conditioning | masked 231-token `231_masked_v1` interface, text_embeds / time_ids as today |
| QNN target | QAIRT **2.50.0.260828**, 8 MB VTCM, O=3, source-destructive reuse disabled. The config names v75 / soc_model 57; host compiles honour it, on-phone compiles build for the phone's own arch (LIMITS.md) |
| Per-SoC configs | `htp_config_8elite.json` / `htp_config_8gen5.json` beside `htp_config.json`, as in `template_sdxl/` |
| Phone compile | storage-backed allocator (`libcompiler_heap.so`), as every SDXL compile |

Nothing downstream of the template changes: `tplconv` and the recipe tools are
resolution- and architecture-agnostic, and the app treats a template as data.
The only SDXL-specific tool flag is `tpl_recipe.py discover --head-dim 64`.

⚠ **Nothing can run the result yet.** Neither consuming app has a 9-channel
SDXL pipeline. Nightmare's SDXL "inpaint" path runs the normal 4-channel UNet
with mask blending; the 9-channel SD1.5 path is `PipelineSd15Inpaint`
(Nightmare backend patch 006). An SDXL equivalent needs:

- mask downsampled to one `[1,1,128,128]` plane,
- masked image `image * (mask < 0.5)` through the VAE encoder, scaled by
  **0.13025** (SDXL), not SD1.5's 0.18215,
- concatenation into the 9-channel `sample` in the order above, every step,
  and the unchanged text/time conditioning.

**Agree the export marker first** (proposal: the existing `SDXL` and
`qnn_context.txt` files plus the SD1.5 `INPAINT` marker file), so the Fancy-Ai
and Nightmare work can start in parallel with the long quantize.

## 2. Choose the calibration checkpoint

The template's activation ranges come from this checkpoint; every later
conversion reuses them.

| Option | For | Against |
|---|---|---|
| `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` | the one general SDXL inpaint model; also the natural add-difference source (§6) | diffusers folder only; convert the UNet to single-file (`model.diffusion_model.*`) keys before export so export and recipe discovery read identical weights |
| An anime/Illustrious or Pony inpaint release | most of npuforge's SDXL users convert anime checkpoints | fewer releases; less general |

SD1.5 lesson: the template was calibrated on one family (DreamShaper) and
generalised to a second photoreal checkpoint at 29.9 dB, but the text-to-image
template's anime limitation is still assumed to carry over. **Decide with the
user base in mind**, and put a checkpoint from the other family through the
template in Phase 3 either way.

⚠ Assert the checkpoint's byte size and SHA-256 before starting; a partial
download fails hours later.

## 3. Calibration rows

Use the pipeline that produced the current SDXL template (the `231_masked_v1`
export and row generator), with these changes:

1. **Timesteps 0–999.** The borrowed SD1.5 inpaint rows covered only 77–913 and
   would have clipped the first and last steps. Use a DPM-Solver-style scheduler
   with `timestep_spacing="linspace"` and assert the rows' min/max timestep before
   quantizing.
2. **Masks in the rows.** SD1.5 used 400 rows from 480 captured calls: 12 prompts
   with a full-canvas mask plus 12 brush-mask edits, CFG 7, 20 steps. Keep both
   kinds; the full-canvas case is what text-to-image through an inpaint model
   looks like.
3. **Token lengths.** The SDXL template's rows cover 77 / 154 / 231 active
   tokens; keep that coverage.
4. **Latent range.** The SDXL template's calibration clipped latents to
   [−7.2, 7.2]. Record the 9-channel `sample` range per channel group; the
   masked-image latent is VAE-encoder output and should be checked, not assumed.
5. **Row count.** The SDXL template used 30 rows (three trajectories). More rows
   cost host time linearly (§5). Take at least as many as the text-to-image
   template, and gate on coverage rather than count.

⚠ Check dtypes when writing the raw rows: a timestep written as int32 and read as
float32 once produced a model that passed four checks and computed nothing
(TEMPLATE-AUTHORING.md §2 step 3).

## 4. Export, quantize, gates

Export the UNet the same way as the SDXL text-to-image template, with `conv_in`
widened to 9 input channels by the checkpoint itself. Then:

```sh
qnn-onnx-converter -n --input_network ./unet/model.onnx \
    --preserve_io layout \
    --input_list ./input_list_unet.txt \
    --use_per_channel_quantization \
    --bias_bitwidth 32 \
    --act_bitwidth 16
```

⚠ `--preserve_io layout` is mandatory for conv graphs. **Keep `model.cpp` and
`model.bin`**; Phases 1 and 2 derive from them.

Gates before any compile (TEMPLATE-AUTHORING.md §2):

- 0 quantized (u16, dtype 1046) tensors with `scale == 0`.
- time-projection `Cast`/`Mul` reach [0, 999]; `Sin`/`Cos` [−1, 1]. Tighten the
  `Cast` upper bound to ≥ 900 so a clipped schedule fails.
- `sample` input is `[1, 9, 128, 128]`.

Run an **8-row probe build first**: it validates the whole pipeline in minutes
and reproduces the ranges closely enough to gate on.

Phases 1–2 (`tools/tpl_patch.py`, `tpl_pack.py`, `tpl_recipe.py discover
--head-dim 64`, `tpl_recipe_bin.py`, `tpl_pack_trim.py`), then
`qnn-model-lib-generator -t aarch64-android` without `-b`. Gates:

| Gate | Expected |
|---|---|
| Identity pack context vs stock context | 0 differing bytes |
| Negative control (XOR 4 KB of the pack) | context changes |
| Discovery | ambiguous **0**; every source tensor used once |
| Mapping vs `template_sdxl/` | same counts except `conv_in` (SD1.5 inpaint vs text-to-image matched exactly) |
| Same-checkpoint round trip | all scales exact; residue in int32 biases only |

## 5. Host resources

Unmeasured for SDXL inpaint; plan from these:

| Step | SD1.5 inpaint (measured) | SDXL inpaint (projection) |
|---|---|---|
| Calibration capture | 53 min CPU, 400 rows | far more per row: 1024² and ~3× the UNet |
| Quantize | ~21 s/row, 2 h 30 m for 400 rows | many times that per row; start from the SDXL template's own timing |
| Host RAM | >11 GB plus swap | at least what the SDXL template build needed; SD1.5 hit a memory cliff above 512×768 |
| Disk | ~25 GB workspace | fp32 ONNX alone is ~10 GB (2.6 B parameters); budget 150 GB+ |

Record wall time and peak RSS for each step. They are the numbers the next
template needs.

## 6. Add-difference (optional, after the template works)

`tplconv --inpaint-diff` is generic: it applies `zero_pad(custom) + diff` per
recipe source key and widens only a conv weight's input channels. What is
SD1.5-specific:

- `tools/mk_inpaint_diff.py` loads a `.ckpt` with `torch.load`; the SDXL inpaint
  release is diffusers-format, so convert it to single-file keys first. The
  `conv_in` 9-vs-4 logic carries over.
- `InpaintDiff.kt` hard-codes the SD1.5 file, size and SHA-256. SDXL needs its
  own entry and file; fp16 is about **5.1 GB**, three times SD1.5's 1.72 GB.
  Hosting, download time and phone storage all need a decision. The new
  `HfDownload` (resume, mirror) and *Use downloaded file* import apply as-is.

Measure the difference's quality the SD1.5 way: the same fixture rendered from
a real inpaint release and from base + difference, PSNR inside the mask, then
inspect the images by eye.

## 7. Linux host notes

The SD1.5 inpaint template was built in WSL from a Windows checkout. On native
Linux:

- **Paths.** The existing notes' `/mnt/c/...` paths do not apply. The rule that
  matters is unchanged: every list QAIRT reads (`input_list_unet.txt`) must hold
  **absolute paths, LF line endings, no BOM**. Relative or Windows paths make the
  converter report every row missing.
- **QAIRT 2.50.** The SDK downloads with plain `curl` from Qualcomm's URL
  without a login (a browser/PowerShell request got 403). Use the
  `x86_64-linux-clang` binaries directly; source `bin/envsetup.sh`.
- **`qnn-model-lib-generator -t aarch64-android`** needs `ndk-build` on `PATH`,
  not only `ANDROID_NDK_ROOT`. The app now pins **NDK 30.0.16248370**; build
  the model library with that NDK too.
- **Python.** The npuforge tools and the CLIP authoring path were validated
  with PyTorch 2.5.1, transformers 4.46.1, diffusers 0.31.0; tests run with
  `python -m unittest discover -s tests -p 'test_*.py'` **inside an activated
  venv** (one test shells out to a bare `python3`).
- **Gradle.** `app/build.gradle.kts` resolves NDK clang wrappers per host OS, so
  the same checkout builds on Linux and Windows. Signing reads
  `NPUFORGE_SIGNING_PROPERTIES` (absolute path) or `Keys/signing.properties`.
- **Line endings.** Commit with LF; this checkout is also edited on Windows
  with `core.autocrlf=true`.
- **Nothing generated is committed**: `libqnn_model.so`, recipes, packs,
  checkpoints and rows stay out of Git ([BUILD.md](BUILD.md), [NOTICE](../NOTICE)).
  The export path imports Local Dream's CC BY-NC modules, so the template
  carries the same non-commercial restriction as the others.

## 8. Putting it in the app

Assets: `app/src/main/assets/template_sdxl_inpaint/` with `libqnn_model.so`,
`recipe.bin`, `tpl_trim.pack`, `sources.txt`, `htp_config.json` and the per-SoC
configs.

Code, following the SD1.5 inpaint change:

| File | Change |
|---|---|
| `CheckpointInfo.kt` | `SDXL_INPAINT` model: SDXL detection **and** `conv_in` input channels == 9 (SD1.5 reads `model.diffusion_model.input_blocks.0.0.weight`; check the SDXL key is the same) |
| `Converter.kt` | markers for `SDXL_INPAINT`: `SDXL`, `qnn_context.txt`, `INPAINT` (per §1's agreement); the storage-backed allocator condition must include it |
| `ConvertService.kt` | treat `SDXL_INPAINT` like `SDXL` for CLIP/VAE and step counts |
| `MainActivity.kt` / strings | detection note; later, the add-difference choice |

Check every `== CheckpointInfo.Model.SDXL` comparison. Each is a place where
`SDXL_INPAINT` must be included or deliberately excluded.

## 9. Proving it (Phase 3)

There is **no known-good SDXL NPU inpaint build** to compare against, unlike
SD1.5 (Local Dream's). Use a host reference instead:

1. Render the fixture (512² base image and centre mask upscaled to 1024², plus
   one object mask) with diffusers `StableDiffusionXLInpaintPipeline`, CPU or
   GPU, fixed seed, 20 steps, CFG 7, five prompts.
2. Render the same with the phone-converted template in the consuming app.
3. Score PSNR inside the mask; check `extreme_frac` ≤ 0.05; inspect by eye. NPU
   vs float reference differs more than NPU vs NPU. The SD1.5 NPU builds scored
   29–32 dB against each other, so set the bar from a same-checkpoint NPU pair
   first.
4. Put a second checkpoint from the other family (§2) through the template.
5. Record phone compile time and peak memory from the conversion report. The
   9-channel graph should cost the same as the text-to-image one (SD1.5 showed
   identical profiles). Verify that for SDXL too.

Write the results into a `docs/SDXL-INPAINT.md` modelled on SD15-INPAINT.md,
keeping the gate table, the measurements, and what went wrong.

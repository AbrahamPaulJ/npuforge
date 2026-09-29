# SD1.5 template with swappable LoRA and ControlNet

An SD1.5 UNet template whose LoRA and ControlNet are **graph inputs** rather than
merged weights: one converted model takes any rank ≤ 64 LoRA at any strength, and
the residuals of any SD1.5 ControlNet, per render, with no recompile. Built and
measured 2026-09-29 (S25 Ultra, SM8750, QAIRT 2.50). The measurement log and
the scripts are in `notes/2026-09-29-input-lora-cn-probe.md`.

## Contents

1. Graph contract
2. Authoring
3. Gates
4. Accuracy and speed
5. Traps
6. Runtime (Nightmare Mobile), 6b. In the app: "SD1.5 Swap"
7. Open work

## 1. Graph contract

| input | shape | encoding (u16) | meaning |
|---|---|---|---|
| `sample`, `timestamp`, `text_embedding` | as the stock template | calibrated | unchanged |
| `lora_S` | [160] | [0, 0.25] | per-target scale = `(alpha/r) * max|A| * max|B|` × strength |
| `la_i` | [din, 64] | ±1 | LoRA A for target i, normalised to max |.| = 1 |
| `lb_i` | [64, dout] | ±1 | LoRA B for target i, normalised |
| `res_0..12` | NHWC, [1,64,64,320] … [1,8,8,1280] | ± 2 × the largest of canny/depth/openpose | ControlNet down-block and mid-block residuals |

337 inputs. Each target (160 = 16 transformer blocks × attn1 q/k/v/out, ff
proj/out, attn2 q/k/v/out) adds `S * ((x @ A) @ B)` to its Linear. q/k/v deltas are
computed once per projection and sliced per head. **Zero `lora_S` and zero residuals
are exactly the base model.**

⚠ The target ORDER is the export's hook order. It ships as `targets.json` beside
the template and must travel with every export (as `lora_targets.json`): a packer
with the wrong order puts every LoRA on the wrong layer and still renders.

## 2. Authoring

The stock pipeline ([PIPELINE.md](PIPELINE.md)) unchanged, on a different ONNX:

1. ONNX export with the branches and residual adds (`export_a1.py combo 64`).
2. Legacy `qnn-onnx-converter` (`--preserve_io layout`, per-channel, act 16 / bias
   32) on **4 calibration rows** spanning t = 999 / 4 / 846 / 724 (the encoding gate
   needs t ≈ 999), each with a rank-64 LoRA and one ControlNet type's residuals
   (none / canny / depth / openpose).
3. `--quantization_overrides`: the input windows above, plus every branch
   intermediate (x·A, ·S, ·B) at **4× its calibrated range** (§5).
4. `tpl_patch` → identity pack → pack-loading lib → `tpl_recipe discover` →
   `finalize` → bundle, as for the stock template.

Bundle: `recipe.bin` 404 KB, `tpl_trim.pack` 44 KB, aarch64 `libqnn_model.so`
14 MB, plus `targets.json`.

## 3. Gates

| gate | result |
|---|---|
| encodings | time_proj reaches 999, Sin/Cos ±1, 0 dead tensors; 798 / 814 overrides landed (16 k-branch ·B tensors renamed by the converter keep calibrated ranges) |
| identity pack vs stock build | 2,383 sites, byte-identical contexts |
| discover (DreamShaper 8) | matched 1,358, unmatched 1,025, ambiguous 0 — the stock template's numbers: the new inputs add no weight sites |
| round trip (DreamShaper 8) | 0 scale mismatches, 243 bias-only byte mismatches (the stock control) |
| second checkpoint (SD1.5 base) | 1,358 byte mismatches: weight 914, bias 295, MatMul 64, permute 24, other 61 — the stock Phase 3 breakdown |
| native `tplconv` from the trimmed bundle (host) | byte-identical to Python |
| `tplconv` on the phone | same md5, 36 s |
| context compile on the phone | 191 s (stock template: 93 s), 885,977,216 B |
| phone-built vs PC-built context | max abs output difference 0.0 on all four cases |

## 4. Accuracy and speed

Held-out row (t = 495), held-out LoRA (camera style, rank ≤ 64), canny residuals;
NPU vs ORT fp32 of the same checkpoint. Gain and cosine compare the NPU's *delta*
(case − base) with fp32's.

| model | base vs fp32 | LoRA gain / cos / δSNR | canny gain / cos / δSNR | both gain / cos / δSNR |
|---|---|---|---|---|
| DreamShaper 8 (the template's own) | 35.3 dB | 0.987 / 0.989 / 16.6 dB | 1.013 / 0.997 / 22.1 dB | 1.009 / 0.997 / 21.9 dB |
| SD1.5 base, converted on the phone | 38.1 dB | 1.004 / 0.992 / 18.0 dB | 1.006 / 0.998 / 24.7 dB | 1.006 / 0.998 / 24.9 dB |

Renders through Nightmare Mobile's backend (20 steps, 512², DreamShaper template):

| | 20-step sample | UNet call |
|---|---|---|
| stock DreamShaper (3 inputs) | 3.2 s | 70 ms |
| template, no LoRA | 3.7 s | 82 ms |
| + LoRA | 3.8–4.0 s | 86–96 ms |
| + canny ControlNet (AI Hub context) | 6.1 s | — |
| + both | 6.4–6.7 s | — |

Template-no-LoRA vs stock: same composition, 24.1 dB (the 4-row calibration).
LoRA strength scales visibly; canny follows its hint and stacks with a LoRA.

## 5. Traps

- **Branch intermediates must be widened.** Calibrated as-is, LoRA gain was 0.51
  (A1 probe) and 0.76 here: CLIP's BOS-token outliers set ranges the LoRA delta
  then clips against. 4× on x·A, ·S and ·B fixes it (gain 0.99–1.00).
- **The legacy converter re-quantizes `la_i` / `lb_i` to 8-bit** before their
  MatMul (`…_converted_UFIXED_POINT_8`). Measured harmless (LoRA δSNR 16.6 vs
  16.8 dB for a 16-bit build), and probably why this build is faster than the DLC
  route (min 303–343 ms vs 397 ms in `qnn-net-run`).
- **Do not borrow the stock template's encodings by tensor name.** The ONNX op
  counters shift once branches are inserted: of this graph's outputs, 773 match
  the stock `model_net.json` by name and shape, 2,299 by name with a different
  shape, 8,262 not at all.
- **Rank > 64** needs an SVD truncation at pack time (a CivitAI LoRA at rank 128
  is common). The product is unique, the factors are not: compare packers on
  `A·B·S`, not bytes.
- **Strength headroom.** Measured `lora_S` max at strength 1: 0.072 (a rank-128
  style LoRA), 0.001 (a small one). The [0, 0.25] window holds strength 2.0 for both.
- **Residual windows** are 2× the largest of three types because residual
  magnitudes differ up to 7.7× between canny, depth and openpose.

## 6. Runtime (Nightmare Mobile)

Nightmare's backend patch 015 binds every UNet input past the standard three by
name, zero-filled unless set, and re-quantizes an input only when it changes.
Per request: `lora_dir` (a directory of `<input>.raw` float32 files), `lora_strength`
(scales `lora_S`), `controlnet` (a ControlNet context path) and `control_image`.
One launch swapped LoRA, strength and ControlNet across six requests with pixel
hashes identical to per-launch runs. Its app-side packer (`TemplateLora.kt`) is
byte-identical to `pack_lora.py` at rank ≤ 64 and within 1e-3 on the product at
rank 128.

## 6b. In the app: "SD1.5 Swap"

The user's names (2026-09-29): **SD1.5 Swap** for this template, "v1" for the
merged-weight exports when a distinction is needed. It is the third *Convert
as* chip beside Text-to-image (still the default) and Inpainting; the baked-LoRA
list is v1 only and hides for Swap (the service refuses baked LoRAs with Swap).
The checkpoint validates as SD15; everything after is `SD15_SWAP`
(`template_swap/`). The export's name ends `_npuforge_swap` (a bare `_v2` would
collide with checkpoint names), and it carries `lora_targets.json`
(`Converter.SWAP_MARKER`), which is both the marker Nightmare keys on and the
packer's target order. `sources.txt` and both `htp_config*.json` are the stock
template's: the same 686 source tensors, and the graph is still named `model`.

✅ Proven in the app (2026-09-29, S25 Ultra, preview build of branch `sd15-swap`):
cuteyukimix converted as Swap in about 8 minutes; the export's `unet.bin` is
885,977,216 B (the same size as the hand-built Swap context) and its
`lora_targets.json` is the bundled one. Rendered through Nightmare's patch-015
backend, one launch, 20 steps at 512²: base 3.8 s, + Colorwater 1.0 4.0 s,
+ canny 6.6 s, + both 5.8 s; five distinct pixel hashes; the checkpoint's own
style survives, the LoRA and canny apply and stack.

## 7. Open work

- ControlNet contexts per type (only Qualcomm AI Hub's canny is on hand) and a
  ControlNet template, so a CivitAI ControlNet converts the same way.
- Production calibration (more rows); the authoring scripts into `tools/`.

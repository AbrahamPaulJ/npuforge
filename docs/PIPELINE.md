# How the template and recipe were built — the measured record

This is the derivation of `template/recipe.bin` and `template/tpl_trim.pack`, and
the evidence that each step is correct. It is kept because re-deriving any of it
costs hours of compute, and because several of the dead ends here look like
reasonable ideas until you have paid for them.

Everything was measured on 2026-09-13 / 2026-09-14. Device: Samsung SM-S938B
(SM8750, HTP **v79**). Host: WSL Ubuntu, QAIRT **2.49**, Linux NDK r27c.

⚠ **The authoring scripts are not in this repo**, and cannot be: they import a
modified `diffusers` UNet whose licence is incompatible with this one
(`NOTICE` §2). What this repo carries is the *product* — `tools/tpl_*.py`,
`native/tplconv.cpp`, and the template bundle. To rebuild the authoring side from
scratch, see `docs/TEMPLATE-AUTHORING.md`.

## The idea, in one paragraph

A full PC conversion is ~50 min of calibration, ~20 min of ONNX export and a
**2 h 18 m** quantize. But almost all of that expense is in the **activation**
ranges, and activation ranges *transfer between checkpoints*. So: quantize one
SD1.5 graph once on a PC and keep its activation encodings fixed (the
**template**); separately record a numeric map from checkpoint tensors to graph
tensors plus the exact quantization rule for each (the **recipe**); then a phone
only has to re-quantize weights and compile.

## Test 1 — do borrowed activation ranges work at all?

DreamShaper-8-inpainting weights, QAIRT 2.49, 8gen2 tier, three arms identical
except for where the activation ranges came from. Rendered on device, 512²,
20 steps, seed 12345, scored inside the mask against the shipped build
(`ref249`). `own~ref` is the noise floor; `borrow` is judged against that, never
against zero.

| prompt | own~ref PSNR | **borrow~ref PSNR** | xf ref / own / borrow |
|---|---|---|---|
| sunflowers | 18.62 | **20.88** | 0.0133 / 0.0152 / 0.0098 |
| ramen | 25.52 | **27.62** | 0.0019 / 0.0013 / 0.0015 |
| car | 39.92 | **39.99** | 0 / 0 / 0 |
| sailor | **38.58** | 34.88 | 0.0007 / 0.0007 / 0.0007 |
| frog | 23.16 | **27.03** | 0.0164 / 0.0566 / 0.0291 |
| **mean** | 29.16 (SSIM 0.970) | **30.08 (SSIM 0.983)** | 0.0065 / 0.0147 / 0.0082 |

All three rows are visually indistinguishable, including `ref249`'s quirk of
painting a dog's face for "a small red sports car" — which both other arms
reproduce.

⭐ **Why it works:** over 4,638 tensors the borrowed/own span ratio is within
**×1.033 at p50**, ×1.139 at p90, ×1.385 at p99. Two photoreal SD1.5 finetunes
genuinely agree.

⚠ **And why it does not generalise**: a checkpoint that moved further from the
base does not. See `LIMITS.md` § anime.

⚠ The borrow arm was not 100% borrowed — activations the override did not reach
were calibrated from 50 DreamShaper rows (719 overrides landed, 48 correctly
overruled on Sigmoid outputs). A phone has no rows at all. A zero-row build would
settle it and has not been run.

## Test 2 — can the phone compile a context binary?

Yes. `libmodel.so` cross-built for `aarch64-android`, generator run on device.

| | |
|---|---|
| time | **70 s**, exit 0 |
| peak RSS | **~4.8 GB** (phone `MemAvailable` bottomed at 0.87 GB) |
| size | 882,686,608 B vs the PC's 882,174,632 B for the same graph |
| renders | **byte-identical** to the PC build |
| inference | unchanged, ~4.8–5.2 s per 512²/20-step inpaint |

Latency is safe by construction: it is a property of the compiled graph, and a
phone build has the same topology and w8a16 format.

⚠ `config_file_path` inside the backend-extension JSON must be **absolute** on
device — a relative path is read against the process CWD and fails confusingly.

## Phase 0 — the template

`DreamShaper_8_pruned.safetensors` (2,132,625,894 B), 20 prompts → 942 valid →
**400 calibration rows** (52 min CPU), stock QAIRT 2.49 w8a16.

The first build took **2 h 39 m** and was **dead** — see below. The rebuild
(STEP 3→7 only; `data.pkl` had always been correct, so the 52-minute calibration
and the ONNX export were reused) took **2 h 18 m at 20.7 s/row** and produced
`unet.bin` **881,826,392 B**, md5 `94032107756d2b182ce6ec131baf9155`.

The rebuild gates on the encodings **before** spending the compile:

| tensor | dead build | 8-row probe | rebuilt (400 rows) |
|---|---|---|---|
| `_time_proj_Cast_output_0` | [0, 0] | [0, 929] | **[0, 999]** |
| `_time_proj_Mul_output_0` | [0, 0] | [0, 929] | **[0, 999]** |
| `_time_proj_Sin_output_0` | [0, 0] | [−1, 1] | **[−1, 1]** |
| `_time_proj_Cos_output_0` | [0, 1] const | [−1, 1] | **[−1, 1]** |

plus **0 quantized (u16) tensors with `scale == 0`**, against 3 in the dead
build. The range reaches 999 rather than the probe's 929 because 400 rows cover
the whole schedule.

⚠ **Gate only on quantized tensors.** The first version of this gate failed a
*healthy* build by asserting on non-quantized ones (dtype 50 int32, and a 562
float alias). Check dtype 1046 and nothing else — a check that reports the
opposite of the truth is worse than no check.

## Phase 2 — the pack-loading template: byte-identical

`tools/tpl_patch.py` rewrites every STATIC tensor's data and quantizeParams into
`tpl::data/len/axis/scalar(<binvar>)` calls (`tools/tpl_runtime.hpp`, inlined):
**2,383 sites** — 1,732 per-axis (int8 conv weights, int32 biases) and 651 scalar
(u8 linear weights, norm constants, int32 biases). Count-asserted, 0 raw
references left. `tools/tpl_pack.py identity` then repacks the template's own
`model.bin` into an 863,626,496 B `TPLPACK1` file, 2,383 entries, 0 unused.

Gate: stock context, stock context regenerated (a determinism control) and
template+pack context all produced **the same md5**. Negative control: XOR 4 KB
inside the pack → 6,922 bytes of the output differ, proving the pack actually
drives the build.

The library built **without** `model.bin` is 10,107,296 B instead of ~873 MB.

## Phase 1 — the weight recipe

**Mapping: all 686 checkpoint UNet tensors ↔ 1,358 graph entries, 0 ambiguous,
0 unused.** Found **numerically** — sorted-value fingerprint plus permutation
search — never by name, because the exporter emits norm gamma/beta as anonymous
`onnx__Mul_*` / `onnx__Add_*`. Conv weights permute `[2,3,1,0]`; q/k/v split 8
ways by rows. The 1,025 entries with no checkpoint source are 768 zero biases the
converter adds to the per-head q/k/v convs and 257 u16 graph constants.

⚠⚠ **Arithmetic order is the specification**, and the obvious alternative is
wrong in every case:

| rule | exact | the trap |
|---|---|---|
| i8 conv weights: `s=f32(max|w_c|/127)`, `q=rha(w*127/max)` in f64 | **866/866** | `w/(max/127)` fails 632 tensors — a weight at exactly `max/2` must land on 63.5 |
| u8 linear/norm: `q=floor(f32(w*255/(hi−lo) − o) + 0.5)` | **197/197** | the `+0.5` happens in **float32** |
| i32 conv biases: `s_c=f32(f32(in_act_scale)*f32(s_w_c))` with the NEW weight scale, `q=rha(b/s)` in f64 | scales exact | 39 tensors differ by 1–2 bytes |
| i32 per-tensor biases: `s=f32(max|b|/2^31)`, `q=rha(b/s)` in f64 | scales exact | bytes within **±126 units** on all 197 |

`rha` is round-half-away-from-zero. **All 2,383 scales reproduce exactly.**

The ±126-unit bias residue was chased and is **not** a rounding rule: 15 variants
were tried. The converter's own float bias differs from the checkpoint's by
exactly **one float32 ulp**; 126 units ≈ 8e−9 absolute, below float32 resolution
of the bias. Stop looking for a better rule.

**Render gate (device, txt2img 512²/20 steps, seed 12345, 5 prompts, only
`unet.bin` varying):**

| | PSNR | SSIM | note |
|---|---|---|---|
| recipe ~ template | **27.91** | **0.967** | the recipe's own quantization |
| xororz ~ template | 20.47 | 0.898 | control: an independent build of the same checkpoint |

Latency unchanged (template 3.9–4.6 s, recipe 3.9–4.2 s, control 3.7–4.0 s), and
`extreme_frac` 0.000–0.050 against the dead build's 0.317.

⭐ **The 20-step gap is amplification, not a wrong rule — proven by a step
sweep** (same prompt, same seed):

| steps | 1 | 2 | 4 | 8 | 20 |
|---|---|---|---|---|---|
| PSNR | **46.05** | 39.29 | 39.40 | 31.58 | 31.99 |
| max channel diff | **16** | 99 | 105 | 166 | 207 |

After **one** UNet call the two agree to 46 dB with a max difference of 16/255.
A wrong quantization rule would disagree at step 1. Diffusion is chaotic, so
2,163 bytes of int32-bias float noise in 881,826,392 is enough to pick a
different fine-detail path by step 20 while staying the same image.

⚠ **An earlier run of this gate scored 51.10 dB / SSIM 1.0000 — on the DEAD
template.** Two binaries computing the same *nothing* agree perfectly. Recorded
so the number is never mistaken for a regression when a real build scores lower.

⭐ **Re-deriving Phases 1 and 2 against the rebuilt template took 9 minutes and
reproduced everything exactly** — the patched `model.cpp`, the identity pack, the
pack-loading library and the recipe's `finalize`. Nothing structural moved:
discover again matched 1358 / unmatched 1025 / ambiguous 0, and the recipe pack's
context binary again differed from stock by **exactly 2,163 bytes**. The mapping
is stable across a full re-quantize; the bias noise is systematic, not
incidental.

## Phase 3 — the recipe generalises to a second checkpoint

AbsoluteReality's weights through the **DreamShaper** template, which is exactly
what a phone conversion does. Apply took 43 s / 5.1 GB RSS, 2,383 entries.

⭐ **The proof is in the pack, not the pixels:**

| | scale mismatches | byte mismatches | kinds |
|---|---|---|---|
| AbsoluteReality vs the template's own pack | **2,111** / 2,383 | **1,358** | weight 914, bias 295, MatMul 64, permute 24, other 61 |
| DreamShaper via recipe vs the same (control) | **0** | 234 | **bias only** |

**1,358 is exactly the mapping's `matched` count** — every entry with a
checkpoint source got new weights *and* recomputed per-channel scales, and all
1,025 template constants were left untouched. The same-checkpoint control
reproduces all 2,383 scales exactly. Those two rows together are what make this a
test rather than a demo.

⚠ **Pixel metrics do not discriminate between genuinely different models here,
and the 1-step trick does not rescue them.** At 20 steps every pair lands at
13–20 dB; at 1 step everything lands at 16–19 dB, because a 1-step image is
mostly noise and PSNR then measures noise patterns:

| | 20 steps | 1 step | 4 steps |
|---|---|---|---|
| ar_recipe ~ ar_xororz | 16.37 | 19.41 | 20.06 |
| ar_recipe ~ ds_template | 14.38 | 16.41 | 19.52 |
| ar_xororz ~ ds_template | 14.63 | 18.57 | 19.99 |

The 1-step test was decisive in Phase 1 only because those two models were
*numerically almost identical* (46 dB). Do not re-run it expecting separation.

⚠ Nor is the third-party build the right bar: it is a **2.28** graph with
**AbsoluteReality's own** ranges; ours is **2.49** with **DreamShaper's borrowed**
ranges. Legitimately different models.

## Phase 4 — the phone converts a checkpoint by itself

`native/tplconv.cpp` (a C++ port of `tools/tpl_apply.py`), `tools/tpl_recipe_bin.py`
(recipe → the binary the C++ reads) and `tools/tpl_pack_trim.py`.

**The gate needed no device and no render** — the PC's reference packs already
existed, so correctness is decided by `cmp`:

| built by | checkpoint | md5 | verdict |
|---|---|---|---|
| Python (PC) | DreamShaper 8 | `042b99cf38b37cecc15483f37118bb46` | reference |
| C++ (PC) | DreamShaper 8 | `042b99cf…` | **byte-identical** |
| Python (PC) | AbsoluteReality | `52ca4f492e9f43634d08698ea9a5360b` | reference |
| C++ (PC) | AbsoluteReality | `52ca4f49…` | **byte-identical** |
| C++ (**phone**) | DreamShaper 8 | `042b99cf…` | **byte-identical** |

First run, no iteration, no tolerance.

⭐ **`tplconv` peaks at 1.98 GB against Python's 5.07 GB**, which is what makes it
viable on a phone: two passes — compute every scale first (a few MB, since
payload lengths follow from `dims`), then stream each entry's bytes. The phone
ran it in **24 s** against the PC's 43 s.

### ⭐⭐ The template bundle is 43 KB, not 863 MB

The full pack is ~863 MB because it carries the template checkpoint's own
quantized weights — and the recipe **overwrites every one of them**. The only
payloads that survive are the 257 `template`-rule entries: u16 graph constants
with no checkpoint source. Measured: **832 bytes of payload, a 43,328-byte
pack**, and `tplconv` built from it produces a **byte-identical** output pack.

| | |
|---|---|
| trimmed template pack | 43 KB |
| `recipe.bin` | 397 KB |
| `libqnn_model.so` (aarch64, pack-loading) | 9.7 MB |

Against 863 MB for the naive bundle: the difference between shippable and not.

### End to end on the phone

| step | time |
|---|---|
| `tplconv_arm`: checkpoint → pack (863,626,496 B) | **24 s** |
| `qnn-context-binary-generator`: pack → `unet.bin` (882,780,736 B) | **93 s** |
| **total** | **117 s** |

✅ The phone-built model renders **5/5 PNGs byte-identical** to the PC's compile
of the same pack.

⚠ **Compare against the right arm.** The phone builds from the *recipe* pack, so
`recipe` is its twin. Comparing it against the stock `template` arm reproduces
the recipe~template numbers exactly (34.27 / 22.33 / 30.50 / 31.99 / 20.44 dB)
and reads as a failure when nothing is wrong. This cost a false "0/5" once.

⚠ **The context binary is not byte-reproducible and must not be a gate.** Two
runs on the same phone from the same pack produced 882,780,736 B both times with
**different md5s**, and rendered byte-identical PNGs. A PC compile differs again
in size (881,826,392 B). The weight pack is the thing with an exact hash, which
is why that is what `tplconv` is verified against.

## 🐞 The dead first template — what a broken template looks like

The first Phase 0 template rendered **pure noise** on all five prompts while a
third-party build of the same checkpoint rendered perfectly through the identical
rig. The tell was statistical, not visual: mean 116.8 / std 105.9,
`extreme_frac` **0.317**, and *near-identical across different prompts* — the
signature of an output that does not depend on the conditioning at all. A healthy
render scores 0.00–0.05.

**Cause: the calibration timestep was written as int32 and read as float32.** The
whole time-embedding path quantized to `[0, 0]` and the UNet denoised blind.
Cost: **2 h 39 m** of quantize. The checkpoint, `data.pkl` and the ONNX export
were all reusable, and the mapping and rules are independent of calibration.

⚠ **The diagnosis is worth more than the fix.** Four checks passed on a model
that was completely dead, and they are the obvious ones to reach for next time:

1. the converter exited 0;
2. the IO contract — names, dims, dtypes — matched a known-good binary exactly;
3. the IO quantization ranges were sane and close (`sample` ±7.2 vs ±6.4,
   `text_embedding` equal to 4 significant figures);
4. the model loaded and ran on the NPU **at the correct speed** (4.0 s vs 3.9 s).

**Latency proves a graph compiled, never that it computes anything.** The one
check that catches it — scanning the encodings for a zero scale — takes seconds
and now runs inside the build.

This is the second time a dtype crossing a tool boundary read as "bad model"
rather than "one wrong integer". Check dtypes wherever a number crosses a tool
boundary.

## Template families

A template is one fixed graph: **one per (architecture, resolution)**, possibly
per style. The machinery — patcher, pack, recipe tool, phone compile — is
generic.

- **SD1.5 inpaint** — cheap; test 1 already proved borrowing on this graph.
- **Anime SD1.5** — the known gap. `LIMITS.md`.
- **SDXL** — gated on phone RAM: its UNet is ~3× SD1.5's and SD1.5's compile
  already peaked at ~4.8 GB. Measure an SDXL compile on the phone before planning
  anything else.

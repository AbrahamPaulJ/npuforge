# npuforge

**Convert SD1.5 and SDXL checkpoints into Qualcomm NPU models on the phone.**

SDXL INT8 conversion and generation now work on the tested Galaxy S25 Ultra:
**437 seconds total conversion at O=3**, then **15 seconds** for a 1024 × 1024
image in Aura (8 steps, CFG 1, LCM/Karras). These are individual phone results,
not a cross-device benchmark. [SDXL findings and configuration](docs/SDXL.md)
record the memory fix, O=1 comparison and required local artifacts.

The measurements below describe the original **SD1.5** pipeline.

Today, putting a new SD1.5 checkpoint on an NPU means a developer running a
multi-hour conversion on a workstation: ~50 min of calibration, ~20 min of ONNX
export and a **2 h 18 m** quantize. npuforge replaces all of that, at conversion
time, with a template plus a weight recipe — so the phone only has to re-quantize
the weights and compile.

Measured on a Samsung SM-S938B (SM8750, HTP v79), 2026-09-14:

| step | time |
|---|---|
| `tplconv`: `.safetensors` → weight pack (863,626,496 B) | **24 s** |
| `qnn-context-binary-generator`: pack → `unet.bin` (882,780,736 B) | **93 s** |
| **total, on the phone** | **117 s** |

The result renders **byte-identical** to the same pack compiled on a PC.

## Why it is small enough to ship

A naive "ship the template" approach means a 863 MB pack. But the recipe
**overwrites every weight in the template** — the only payloads that survive are
257 graph constants with no checkpoint source, totalling **832 bytes**.

| what the app ships | size |
|---|---|
| `template/tpl_trim.pack` | 43 KB |
| `template/recipe.bin` | 397 KB |
| `libqnn_model.so` (aarch64, pack-loading) | 9.7 MB |

Verified: a pack built from the 43 KB trimmed template is byte-identical to one
built from the full 863 MB pack.

## How it works

1. **Template (PC, once per architecture).** One fixed SD1.5 graph, quantized with
   one checkpoint's *activation* ranges. Activation ranges are what needs
   calibration, and they transfer between checkpoints — two photoreal SD1.5
   finetunes agree to ×1.033 at p50 over 4,638 tensors.
2. **Recipe (PC, once per template).** A numeric map from checkpoint tensors to
   graph tensors — 686 ↔ 1,358 entries, found by value fingerprint and permutation
   search, never by name, because the exporter names norms anonymously. Plus the
   exact quantization rule for each.
3. **Convert (phone).** Read the user's `.safetensors`, apply the recipe's
   transforms and per-channel quantization, write a weight pack.
4. **Compile (phone).** `qnn-context-binary-generator` links the pack into a
   context binary for *this* chip.

## Correctness is decided by `cmp`, not by looking at pictures

The PC pipeline produces reference packs, so the on-device converter has an exact
target. `tplconv` reproduces them byte-for-byte:

| built by | checkpoint | md5 |
|---|---|---|
| Python (PC) | DreamShaper 8 | `042b99cf38b37cecc15483f37118bb46` |
| C++ (PC) | DreamShaper 8 | `042b99cf…` ✅ |
| C++ (**phone**) | DreamShaper 8 | `042b99cf…` ✅ |
| Python (PC) | AbsoluteReality | `52ca4f492e9f43634d08698ea9a5360b` |
| C++ (PC) | AbsoluteReality | `52ca4f49…` ✅ |

Two different checkpoints, first run, no tolerance.

⚠ **Arithmetic order is the specification.** Each rule was measured against the
reference converter's own output, and the obvious-looking alternative is wrong:

- `i8`: `q = rha(w * 127 / max)` — *not* `w / (max/127)`. A weight at exactly
  `max/2` must land on 63.5.
- `u8`: `x = f32(w*255/(hi-lo) - o)` then `floor(f64(x + f32(0.5)))` — the `+0.5`
  happens in **float32**.
- `i32`: divide in **float64** by the **float32** scale. Doing it in float32 moves
  results in 128-unit steps.

`rha` is round-half-away-from-zero. The header of `native/tplconv.cpp` repeats
this, because getting one wrong is silent — the model still builds and still runs.

## Memory

`tplconv` peaks at **1.98 GB** where the Python reference peaks at 5.07 GB. It
makes two passes: compute every scale first (a few MB — payload lengths follow
from the dims), then stream each entry's bytes. That is what makes it run on a
phone at all.

⚠ The **compile** step is the heavy one: ~4.8 GB peak. 8 GB phones are unproven.

## LoRA: merge, then convert

Runtime LoRA needs `UPDATEABLE_STATIC` tensors, and marking even **one** costs
2.8x inference on this hardware -- a cliff, not a slope (83 ms/pass -> 281, and
24 vs 768 tensors cost the same). Merging instead produces an ordinary model at
full speed. That route was previously rejected only because re-converting cost
1-1.5 h on a PC; on-device conversion makes it ~2 min.

`tools/lora_merge.py` does `W' = W + strength * (alpha/rank) * (up @ down)` in
checkpoint space, which also sidesteps the per-head attention split -- merged
`to_q`/`to_k`/`to_v` are split by the recipe like any base weight.

✅ **Measured 2026-09-14**: DreamShaper + a rank-128 watercolour LoRA at 0.8,
192/192 attention modules merged, converted and rendered on device. The style
transfer is unmistakable and there are no artifacts.

✅ **Wired into the app**: add one or more LoRAs, each with its own strength.
They stack in a single pass -- the merge is additive. `tplconv --lora
<file>[:<strength>]` reproduces `lora_merge.py` + convert **byte for byte**,
on x86 and on the phone, and the app's output renders byte-identical PNGs to
the PC build.

⚠ The merge is cached per source tensor: the recipe reads an attention weight
once per head, so a q/k/v tensor is fetched 8 times and recomputing the rank-R
product each time cost more than the rest of the conversion (126 s -> 55 s).

| | |
|---|---|
| inference cost | **none** -- it is an ordinary model |
| conversion cost | seconds on top of the usual ~117 s |
| **storage** | **~1.3 GB per (checkpoint x LoRA x strength)** |

Storage is the real constraint: strength is baked in, not a slider.

⚠ kohya names SD1.5 LoRAs with **diffusers** block names while the checkpoint
uses **LDM** ones, so the block prefix needs translating. Build the kohya name
forwards from each checkpoint key -- reversing it is ambiguous, because
`to_out_0`, `ff_net_0_proj` and `transformer_blocks_0` all collide under
`"." -> "_"`.
⚠ The text-encoder half (`lora_te_*`) is dropped, since CLIP comes from the
template. For the style LoRA tested, the UNet half still carried the effect.
⚠ Standard kohya LoRA only; LoCon/LoHa/DoRA need their own merge formulas.

## Is borrowing the template's CLIP and VAE a problem? Measured: no

Conversion covers the UNet, so a converted model keeps the **template's** text
encoder and VAE. The obvious worry is that this mismatch degrades or breaks
output. It does not, and the test was run on the worst mismatch available -- an
anime checkpoint whose VAE differs from the template's by up to **8x**:

| UNet | CLIP + VAE | saturated px | result |
|---|---|---|---|
| ours (converted) | template | 0.331 | noise |
| ours (converted) | **the checkpoint's own** | 0.358 | **still noise** |
| official build | the checkpoint's own | 0.029 | clean |
| official build | **template (mismatched)** | 0.049 | **still clean** |

Giving a broken UNet its matched CLIP/VAE does not rescue it; giving a good UNet
a mismatched one does not break it. The two clean rows are visually
indistinguishable.

Weight-level agreement says the same thing more cheaply -- across checkpoints the
text encoder and VAE barely move (median 0.13-0.39% relative difference), because
SD1.5 finetunes train the UNet. **Converting them is not worth building.**

⚠ What this does NOT cover: a converted model inherits the template's *prompt
interpretation*, so a checkpoint relying on a heavily-trained text encoder --
or on `clip_skip 2` -- will not behave exactly like it does elsewhere. That is a
fidelity limit, not a failure.

## Status

✅ **Working end to end as an Android app** (2026-09-14, one device). Pick a
`.safetensors`, get a loadable model directory. The app reproduces the
adb-driven pipeline: its model renders **byte-identical PNGs** to both the
adb-built and the PC-built versions. Output is a single
**`Download/npuforge/<name>.zip`** written via MediaStore — no storage
permission needed, somewhere a person can actually find it, and in the shape a
generator's custom-model import expects. Verified: 7 entries, uncompressed
(these are quantized weights; deflate would cost a minute of CPU to save
nothing).

Before picking a file it reads the safetensors **header** (a short read, not a
2 GB copy) and shows what the checkpoint contains — UNet, VAE, text encoder —
which of those conversion keeps, and whether all 686 tensors the recipe needs
are present. An SDXL, SD2 or diffusers-layout file is refused immediately
instead of failing two minutes in.

⚠ **Getting QNN to run inside an app took four separate fixes**, each producing
the same "Device Creation failure". If you touch the packaging or the library
paths, read `docs/ANDROID.md` first.

An **Info tab** in the app states the scope and every known limit in plain
language, so the person holding the phone reads the same caveats this repo does.

**What is not done — and it matters.** Two of the limits look like defects when
you meet them blind: the output carries QAIRT 2.49's **fp16 stamp** (some chips
*newer* than the target refuse to load it), the tier is hardcoded to `_8gen2`,
and **the tested MistoonAnime checkpoint rendered noise**, with evidence pointing
to a mismatch with the template's borrowed activation ranges. All three are measured, with causes and
dead hypotheses, in **`docs/LIMITS.md`**; what is planned about them is in
**`ROADMAP.md`**.

⛔ **Shipping is gated on the QAIRT redistribution question.** See `NOTICE`.

## Build and run

⚠ **A fresh clone does not build a working APK.** Three things are deliberately
absent because they are not ours to ship, and the build needs all three:

| You must supply | Where it goes | Why it is absent |
|---|---|---|
| QAIRT SDK device runtime (`libQnnHtp*.so`, `libQnnSystem.so`, …) and `qnn-context-binary-generator` | `app/src/main/jniLibs/arm64-v8a/` | Qualcomm's, redistribution restricted |
| `libqnn_model.so` (9.7 MB) | `app/src/main/assets/template/` | generated from QAIRT converter output |
| an SD1.5 `.safetensors` to convert | the phone | ~2 GB, and licences vary per checkpoint |

Full steps, including the `-ffp-contract=off` requirement that is **not
optional**, are in **`docs/BUILD.md`**. Android packaging has four separate traps
that all surface as the same error — **`docs/ANDROID.md`** before you touch
`jniLibs`.

## Where to start reading

| If you want to | Read |
|---|---|
| know what this supports and where it fails | `docs/LIMITS.md` |
| build it | `docs/BUILD.md`, then `docs/ANDROID.md` |
| understand how the template/recipe split was derived | `docs/PIPELINE.md` |
| build a **new** template (other resolution, SDK, or calibration) | `docs/TEMPLATE-AUTHORING.md` |
| know which checkpoints work, and why two templates may be the wrong plan | `docs/CHECKPOINT-FAMILIES.md` |
| pick something to work on | `ROADMAP.md` — it also lists what was **rejected on measurement** |
| see the live state and what not to redo | `HANDOFF.md` |

`CLAUDE.md` is the index that ties them together. Findings live in `docs/`, one
file per topic, each claim carrying its measurement; when a claim is refuted the
section stating it is rewritten rather than annotated.

## Licence

**MIT** (`LICENSE`) for this repository's own source: `native/`, `tools/`,
`app/`, and the documentation.

⛔ **`template/` is not MIT and is not commercial-use clean.** `recipe.bin` and
`tpl_trim.pack` inherit a **CC BY-NC 4.0** derivation chain — see
`template/README.md` and `NOTICE` §2. A clean-room regeneration removes this;
`docs/TEMPLATE-AUTHORING.md` says how.

⚠ **`NOTICE` is not optional reading.** The QAIRT runtime is
redistribution-restricted, the app fetches a CLIP/VAE donor at runtime from the
same non-commercial project, and checkpoints carry their own terms.

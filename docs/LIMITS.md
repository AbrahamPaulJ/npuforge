# Scope and limits — what npuforge converts, and where it fails

Every claim here is measured on one device, a Samsung SM-S938B (SM8750, Hexagon
v79), during 2026-09-13 and 2026-09-14. Anything not measured is labelled a
projection. The app's **Info** tab is a plain-language summary of this file; when
one changes, change the other in the same edit.

## In scope

| | |
|---|---|
| architecture | Stable Diffusion **1.5**, single-file `.safetensors` |
| task | txt2img, **512×512**, 4-channel |
| converted | the **UNet** only |
| adapters | standard kohya LoRA, merged at conversion time, stackable |
| output | a QAIRT **2.49** context binary for the **`_8gen2` tier** (v73, 8 MB VTCM) |
| time | ~117 s + a few seconds per LoRA |

A checkpoint is checked before conversion starts, from the safetensors **header**
(`CheckpointInfo.kt`): SDXL, SD2 and diffusers-layout files are refused outright,
and all 686 tensors the recipe needs are confirmed present. That check costs a
short read rather than a 2 GB copy, so a bad file fails in a second instead of
two minutes in.

## ⚠ The output may not load on a phone that is newer than the target

Two independent reasons, and neither is visible before the model is downloaded
and tried:

1. **The fp16 stamp.** QAIRT 2.49 writes an fp16 requirement into every context
   binary it produces. Some Qualcomm chips — including some *newer* than 8 Gen 2,
   e.g. SM8735 — reject it. It is stamped by the SDK, not demanded by the graph,
   and it **cannot be configured away**: an 11-variant sweep of the whole HTP
   backend-extension schema left the signature intact.
   **The fix is known and is a rebuild, not a redesign**: the same graph built
   under QAIRT **2.28** loses the stamp at **no measured quality cost** (mean
   `extreme_frac` 0.0197 vs 2.49's 0.0193 on a 5-prompt set), at about **12%**
   more latency. It needs the 2.28 SDK, which is behind a Qualcomm login and is
   not currently on this machine. See `ROADMAP.md`.
   `tplconv` itself is SDK-agnostic — the stamp comes from the template library
   and the generator — so this is a template rebuild plus a second bundled
   runtime.
2. **The tier is hardcoded.** The graph compiles for `_8gen2` (v73, 8 MB VTCM),
   so v68 (Snapdragon 888) and v69 (8 Gen 1) cannot load it at all. An on-device
   converter *should* compile for the chip it is running on — that was one of the
   original motivations and it is not implemented.

So "8 Gen 2 or newer" is necessary but **not sufficient**, and a load failure is
the first symptom to attribute here rather than to the conversion.

## ⚠ Anime checkpoints convert cleanly and render noise

MistoonAnime converted with exit 0, produced a loadable model, ran at the correct
speed, and rendered saturated noise on every prompt.

**Cause, measured:** the template's activation ranges come from a photoreal
checkpoint (DreamShaper 8), and an anime checkpoint's weights leave them. The
per-tensor ratio of checkpoint weight span to template weight span:

| checkpoint | max ratio | converts |
|---|---|---|
| DreamShaper 8 (the template's own) | 1.000 | ✅ |
| AbsoluteReality | 1.02 | ✅ |
| CyberRealistic | 1.117 | ✅ |
| DreamShaper + rank-128 watercolour LoRA @0.8 | 1.061 | ✅ |
| **MistoonAnime** | **49.4** | ❌ noise |

Three-for-three as a predictor, and the gap between 1.2 and 49 is **untested** —
so this is a usable smell test, not a calibrated threshold. A guard that refuses
to convert above some ratio is cheap to add and is on the roadmap; the honest
version reports the number rather than inventing a cutoff.

**Dead hypotheses, so they are not re-run:**

- *"The checkpoint is corrupt."* It is not. Its UNet weights are healthy; the
  NaNs it contains are in the **VAE**, and NaNs in an fp16 SD1.5 merge's VAE are
  common and authentic. This was over-called once and retracted after measuring.
- *"The borrowed CLIP/VAE break it."* Ruled out by a 2×2 (`README.md`): giving the
  broken UNet the checkpoint's **own** CLIP and VAE leaves it noise (0.358 vs
  0.331 saturated pixels), and giving a good UNet the **mismatched** template
  CLIP/VAE leaves it clean (0.049 vs 0.029). The failure is in the converted
  UNet.

What is left is range borrowing, and the fix is a second template calibrated on
anime data — a PC job of ~50 min calibration plus a ~2 h 20 m quantize.

## The text encoder and VAE are borrowed, on purpose

A converted model keeps the **template's** CLIP and VAE. This was measured before
it was accepted:

- The **text encoder** barely moves, across the whole population. Median relative
  difference against stock SD1.5's CLIP: **0.204%** (CyberRealistic) and
  **0.363%** (MistoonAnime) — anime included, because SD1.5 finetunes train the
  UNet.
- **Photoreal checkpoints' VAEs are the same file.** CyberRealistic's baked VAE
  *is* `vae-ft-mse-840000-ema-pruned`, to **0.021%**. Baking that file is what
  checkpoint authors do.
- The 2×2 above shows a deliberate mismatch is visually indistinguishable.

⚠ **The VAE half of that does not generalise, and an earlier version of this
section overstated it.** MistoonAnime's baked VAE is **207%** from base SD1.5's,
**213%** from ft-mse and **212%** from the anime VAE `kl-f8-anime2` — 0 of its 248
tensors are within 1% of ft-mse. It is not a style choice but fp16 overflow in a
merge: `decoder.up.3.block.0.conv1.weight` has norm 2,858,648 against ft-mse's
78.4, 516 non-finite values, and a span of 59,200 against fp16's 65,504 ceiling.
For such a checkpoint the borrowed VAE is an **improvement**, not a compromise —
the converted model gets a decoder the source file no longer has.
`docs/CHECKPOINT-FAMILIES.md` §5.

**Cost:** prompt interpretation follows the template. A checkpoint that leans on
a heavily-trained text encoder, or on `clip_skip 2`, will not behave exactly as
it does elsewhere. That is a fidelity limit, not a failure — and converting CLIP
and the VAE is therefore **not worth building** at present.

## LoRA

Merged into the weights before quantization, so there is **no inference cost** —
the result is an ordinary model. Runtime LoRA was measured and rejected:
`UPDATEABLE_STATIC` tensors cost **2.8×** inference (83 ms/pass → 281), and it is
a cliff, not a slope — 24 tensors cost the same as 768.

| supported | not supported |
|---|---|
| kohya `lora_down` / `lora_up` / `alpha` | diffusers / PEFT layout |
| several stacked in one pass | LoCon, LyCORIS, LoHa, DoRA, IA3 |
| linear and 1×1/k×k conv shapes | — |

Only **attention** modules have been exercised (192/192 on the tested adapter);
conv adapters are implemented and unverified. An unsupported file is reported,
not silently half-merged.

⚠ **The text-encoder half (`lora_te_*`) is dropped** — CLIP comes from the
template — so style adapters carry over better than trigger-word ones.

⚠ **Strength is baked in.** Each (checkpoint × LoRA set × strength) is its own
~1.3 GB model. Storage, not time, is the practical limit.

⚠ The slider is bounded **−1.0 … 2.0**. Negative is deliberate (detail-tweaker
adapters are used inverted); above 2 a merge tends to leave the template's
activation ranges and land in the noise regime above.

## Device requirements

| | |
|---|---|
| CPU | arm64 |
| Android | 12+ (`minSdk 31`) |
| RAM | peak **~4.8 GB** during the compile; phone `MemAvailable` bottomed at 0.87 GB with normal apps open on a 12 GB device. **8 GB phones are unproven.** |
| storage | ~4 GB free while running; ~1.3 GB per finished model |

The compile is the heavy step, not the weight stage: `tplconv` peaks at
**1.98 GB** (against the Python reference's 5.07 GB).

⚠ Leave the screen on. The compile is a foreground service, but Android will
still kill a multi-gigabyte job on a sleeping device.

## Reading a failure

| symptom | cause |
|---|---|
| model will not load in the generator at all | fp16 stamp, or the chip is below v73 |
| saturated, blotchy noise on every prompt, statistics near-identical **across different prompts** | the checkpoint left the template's activation ranges (anime / heavy merge) |
| clean image, prompt read oddly | the borrowed text encoder |
| clean image, colour slightly off or detail soft | the borrowed VAE |
| conversion refused before it starts | wrong architecture, or missing tensors — the reason is on the checkpoint card |

⚠ **Latency proves a graph compiled, never that it computes anything.** The dead
first template ran at the correct speed (4.0 s vs a known-good 3.9 s), exited 0,
matched a good binary's IO contract exactly, and produced pure noise. Never
accept "it ran" as evidence.

## Verified on exactly one device

Everything above is one phone. Behaviour on other chips — including whether the
fp16 stamp actually blocks them — is projection. That is the single largest gap
in this record, and no amount of local testing closes it.

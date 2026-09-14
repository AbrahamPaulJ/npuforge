# Roadmap

Ordered by value per hour, not by ambition. Each item says what it costs and
what evidence would close it. Things that were considered and **rejected on
measurement** are at the bottom — those are the expensive ones to re-propose.

Status as of 2026-09-14: the converter works end to end as an Android app on one
device, with LoRA. `HANDOFF.md` has the live state.

---

## ⏸ Parked by decision (2026-09-14) — not blocked, deliberately not now

These two are the largest remaining gaps and both are PC-side template work. The
current focus is the app plus LoRA on realistic checkpoints, so they wait.

### P1. A QAIRT 2.28 build variant

**Why it matters more than anything else here:** every model this app produces
carries 2.49's fp16 stamp, and some chips — including some newer than the target
— refuse to load a stamped binary. It is the difference between "works on my
phone" and "works on phones".

- **Known to work.** The same graph rebuilt under 2.28 loses the stamp at no
  measured quality cost (mean `extreme_frac` 0.0197 vs 2.49's 0.0193 over a
  5-prompt set) for about **12% more latency** — which is why a 2.28 build should
  ship *beside* a 2.49 one, not replace it.
- **Cost:** re-run Phase 0/1/2 under 2.28 (a ~2 h quantize plus ~9 min of
  re-derivation), then bundle a second device runtime and a second template.
  `tplconv` is SDK-agnostic and does not change.
- ⛔ **Blocked on an artefact, not on knowledge**: the 2.28 SDK is no longer on
  this machine and re-downloading it needs a Qualcomm login.
- **Evidence that closes it:** a 2.28-built model loading on a chip that rejects
  the 2.49 one. Nobody here owns such a chip, so the honest close is
  `contextBlobVersion 3.2.0` / spillFill 0 / buildId 2.28 plus a quality gate —
  the same bar the last 2.28 rebuild met.

### P2. An anime template

Anime checkpoints convert cleanly and render noise; the cause is measured
(`docs/LIMITS.md`). The fix is a second template calibrated on anime prompts.

- **Cost:** ~50 min calibration + ~2 h 20 m quantize on the PC, then the same
  9-minute Phase 1/2 re-derivation. No new machinery — the pipeline is generic
  over templates.
- **Then:** the app picks a template by measuring the checkpoint's weight-span
  ratio against each, which is exactly the number P3 already computes.
- **Evidence that closes it:** MistoonAnime rendering clean through the anime
  template, with the photoreal template as the negative control.

⚠ **Its premise is now in doubt — do not start this before item 1a below**
(`docs/CHECKPOINT-FAMILIES.md`, 2026-09-14):

- **"Anime" may not be the failing category.** ReV Animated, an anime-lineage
  2.5D model, profiles like a photoreal one (max span ratio **1.27** against base
  SD1.5, against CyberRealistic's 1.22). MistoonAnime's 48 may be one merge, not
  a family. A template calibrated for the wrong category costs 3 h and fixes one
  checkpoint.
- **There is no boundary to switch on.** Linear blends of the two measured
  checkpoints move the max span ratio smoothly from 1.22 to 20.58 with no gap —
  a 50/50 merge sits at 10.30 — so "photoreal or anime" is a cut through a
  continuum, and the real population (ChilloutMix, majicMIX, 2.5D) lives in it.
- **A cheaper lever may exist.** Activation encodings are literals compiled into
  `libqnn_model.so`, but `tpl_patch.py` already redirects 2,383 static sites to
  the pack; extending that to the 3,623 activation tensors costs ~29 KB of pack
  and would let `tplconv` *stretch* the ranges per checkpoint. Untested, and one
  afternoon to test.

---

## Next, in order

### 1. Warn about a checkpoint that will render noise — cheap

The predictor already exists and is 3-for-3: the max per-tensor ratio of
checkpoint weight span to template weight span was 1.00 / 1.02 / 1.117 / 1.061
for the four checkpoints that converted, and **49.4** for the one that failed.

⚠ **Not from the header** — spans need the weights. But they need almost none of
them: the 404 one-dimensional UNet tensors give the same separation as all 686
for **0.89 MB** (CyberRealistic 1.22, MistoonAnime 20.58), so the check is a
sub-megabyte read plus a **~1.6 KB** reference table added to the bundle
(`tools/span_probe.py`, `docs/CHECKPOINT-FAMILIES.md` §4).

⚠ **Gate on a count, not the maximum.** The max is a one-tensor statistic —
MistoonAnime's is driven by a single `skip_connection` weight. "Tensors above 2×"
separates far more robustly: **0** for CyberRealistic, **91 of 686** for
MistoonAnime.

⚠ **Report the number, do not invent a threshold.** The region between 1.2 and 49
is untested and is *populated* — a 50/50 merge lands at 10.3 — so a hard cutoff
would be a guess dressed as a gate.

### 1a. Survey the population before building a second anything — half a day

The template question is "how many sets of activation ranges", and two local
checkpoints cannot answer it. `tools/span_probe.py` profiles a checkpoint over
HTTP ranges **without downloading it** (~15 MB fetched, ~80 requests, ~3 min for
a 2 GB file; the 0.89 MB it actually needs is scattered). Fifty checkpoints
spanning both roots and the merged middle is an afternoon.

- **What it decides:** whether the span-ratio histogram is bimodal (two templates
  is a design) or continuous (two templates is a guess), and whether base SD1.5
  or a broad merge is the better template centroid.
- **It should run before P1's 2.28 rebuild**, because that rebuild is a free
  chance to change whose checkpoint the template is built from.
- ⚠ Civitai is HTTP 451 from here, so the candidate list has to come from
  Hugging Face mirrors and secondary sources.

### 1b. Build the template from base SD1.5, not DreamShaper — free, at the next rebuild

Measured (`docs/CHECKPOINT-FAMILIES.md` §5): every SD1.5 text encoder is within
**0.2–0.4%** of stock SD1.5's, anime included, and CyberRealistic's baked VAE
**is** `vae-ft-mse-840000-ema-pruned` to 0.021%. So stock CLIP + ft-mse VAE is
the neutral, licence-clean default, and it costs nothing to adopt at a rebuild
that is happening anyway.

⚠ One number points the other way and 1a should settle it: CyberRealistic's max
span ratio is 1.117 against DreamShaper and 1.22 against base, so a broad merge
may sit more centrally in range terms than the ancestor does.

### 1c. Make the activation ranges pack-driven — an afternoon, and it could retire P2

Measured (`docs/CHECKPOINT-FAMILIES.md` §6): activation encodings are
`scaleOffsetEncoding` literals compiled into `libqnn_model.so` — 3,623 NATIVE
tensors, of which 3,369 are `UFIXED_POINT_16`. `tpl_patch.py` already rewrites
2,383 **static** sites into `tpl::` calls that read the pack; doing the same for
the activation encodings adds roughly **29 KB** to a 43 KB pack and lets
`tplconv` scale each range by the span ratio it is already computing for item 1.

- **Why it might work:** for `y = Wx + b`, a 10× wider weight span is a
  first-order reason to expect a wider `y`. The anime failure is range clipping.
- **Why it is cheap:** activations are 16-bit, so headroom costs about one bit in
  sixteen per doubling, while a too-narrow range clips catastrophically.
- ⚠ **Where it could hurt:** the 709 `UFIXED_POINT_8` tensors — the graph
  converts some attention inputs down to 8 bits, and 4× there costs two bits of
  eight. And per `PIPELINE.md`, Sigmoid outputs have semantically fixed ranges
  that must not be widened at all.
- **Evidence that closes it:** MistoonAnime rendering clean through the existing
  photoreal template, with the photoreal checkpoints as the regression control.
  ⚠ Judge it on renders — latency proves nothing.

### 2. Compile for the chip that is running, not for `_8gen2`

The tier is hardcoded to v73 / 8 MB VTCM, which is the one thing an *on-device*
converter has no excuse for. The device's real `vtcmSize` and `arch` are readable
before compiling. Unlocks v69 and v75+ tiers and removes half of the "will it
load" question.

### 3. A NaN/Inf guard in `tplconv`

A checkpoint with NaNs in the UNet would quantize to garbage silently. Cheap to
add in the scale pass, which already touches every value. ⚠ Guard the **UNet**
only — NaNs in an fp16 SD1.5 merge's VAE are common and authentic, and treating
them as corruption over-called a checkpoint once.

### 4. Hand the model to a generator instead of leaving a zip in Downloads

Today the app writes `Download/npuforge/<name>.zip` and the user imports it by
hand. A share/open-with hand-off, or a documented import intent, would finish the
loop. Deliberately **not** a feature of any one generator — the converter is its
own app and must stay generator-agnostic.

### 5. 8 GB devices

The compile peaks at ~4.8 GB and has only ever run on a 12 GB phone. Either
measure one or say clearly that it is unproven. Currently it is unproven and the
app says so.

### 6. Resume after an interrupted compile

93 s of compile thrown away by a phone call is annoying but survivable; it
becomes serious if the tier work makes compiles longer.

### 7. A second resolution — measure one compile before planning it

The graph is 512² by construction (`model_tpl.cpp`: `dimensions_sample[] = {1, 4,
64, 64}`). Another resolution is a full Phase 0 at that size (~3 h PC) plus
**~10.1 MB** of bundle, and the output model stays ~1.3 GB because it is mostly
weights. Arithmetic grows 1.61× at 512×768 and 2.68× at 768²
(`docs/CHECKPOINT-FAMILIES.md` §7).

⚠ **The 4.8 GB compile peak is what decides this, and it is unmeasured above
512².** Run one 768² compile for its peak RSS before planning anything else —
the same discipline `PIPELINE.md` applies to SDXL.

⚠ And the honest alternative is to build none of them: hires-fix at generation
time buys the same large outputs with no second template and no higher peak. A
second *resolution* buys composition, not detail.

---

## ⛔ Shipping blocker, independent of all of the above

**The QAIRT redistribution question** (`NOTICE` §1). The app cannot work without
Qualcomm's device runtime, and whether that may be bundled is unresolved. It
decides whether this is distributable at all, not merely how it is licensed.
Second, the template's derivation chain passes through a CC BY-NC 4.0 project
(`NOTICE` §2) — treat the bundle as non-commercial until someone qualified says
otherwise.

Neither blocks development. Both block release.

---

## Considered and rejected — do not re-propose without new evidence

| idea | why not |
|---|---|
| **Runtime LoRA** via `UPDATEABLE_STATIC` tensors | **2.8× slower** inference (83 ms/pass → 281), and a cliff, not a slope — 24 tensors cost the same as 768. Merging before conversion is free at generation time |
| **Converting the text encoder and VAE** | Measured unnecessary: across checkpoints they differ by a median of 0.13–0.39%, and a deliberate CLIP/VAE mismatch is visually indistinguishable (the 2×2 in `README.md`). Real but small fidelity cost, large build cost |
| **Running the full QAIRT converter/quantizer on the phone** | The PC recipe needs >11 GB RAM plus swap, and 400 calibration passes take ~2 h 20 m. The template+recipe split exists precisely to avoid it |
| **Gating correctness on the context binary's md5** | The compile is not byte-reproducible — same phone, same pack, two different md5s, identical renders. Gate on the weight pack, which *is* exact |
| **Using 1-step renders to separate two models** | Only works when the models are already numerically near-identical (46 dB). Between genuinely different models everything lands at 16–19 dB because a 1-step image is mostly noise |
| **Widening the 2.28 quantization-override filter** | Only ~460 of ~4,600 overrides survive 2.28's fusion; the rest cannot land. Widening further is wasted work |

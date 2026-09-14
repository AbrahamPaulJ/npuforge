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

---

## Next, in order

### 1. Refuse (or warn about) a checkpoint that will render noise — cheap

The predictor already exists and is 3-for-3: the max per-tensor ratio of
checkpoint weight span to template weight span was 1.00 / 1.02 / 1.117 / 1.061
for the four checkpoints that converted, and **49.4** for the one that failed.

Compute it during the header inspection and show it. ⚠ **Report the number,
do not invent a threshold** — the region between 1.2 and 49 is untested, so a
hard cutoff would be a guess dressed as a gate. A warning above, say, 2 with the
measured table beside it is honest; a refusal is not, yet.

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
loop. Deliberately **not** a DreamUI feature — the converter is its own app and
must stay generator-agnostic.

### 5. 8 GB devices

The compile peaks at ~4.8 GB and has only ever run on a 12 GB phone. Either
measure one or say clearly that it is unproven. Currently it is unproven and the
app says so.

### 6. Resume after an interrupted compile

93 s of compile thrown away by a phone call is annoying but survivable; it
becomes serious if the tier work makes compiles longer.

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

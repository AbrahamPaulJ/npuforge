# Handoff — 2026-09-14

**Rewrite this file each session. Do not append to it.**

## Where things stand

The converter works end to end as a standalone Android app on one phone. You
pick an SD1.5 `.safetensors`, optionally stack LoRAs with a strength each, and
about two minutes later `Download/npuforge/<name>.zip` is a model a generator
can import. It has been verified the only way that means anything: the app's
weight pack is **byte-identical** to the PC reference, and its model renders
**byte-identical PNGs** to both the adb-driven and the PC-built pipelines. The
repo is `github.com/AbrahamPaulJ/npuforge` (git `main`, MIT — but read `NOTICE`
before redistributing anything built from it).
What is *not* done is everything about other people's phones — the output
carries QAIRT 2.49's fp16 stamp and targets a hardcoded `_8gen2` tier, so
"8 Gen 2 or newer" is necessary but not sufficient, and anime checkpoints still
convert cleanly into noise.

## The numbers

| | |
|---|---|
| `tplconv` weight stage | **24 s**, 1.98 GB peak RSS (Python reference: 43 s, 5.07 GB) |
| compile on device | **93 s**, ~4.8 GB peak |
| total | **117 s** |
| shipped template | 43 KB pack + 397 KB recipe + 9.7 MB `libqnn_model.so` |
| output | ~1.3 GB zip, 7 entries, stored uncompressed |
| reference pack md5s | DreamShaper 8 `042b99cf38b37cecc15483f37118bb46`, AbsoluteReality `52ca4f492e9f43634d08698ea9a5360b` |
| weight-span ratio vs template | 1.00 / 1.02 / 1.117 / 1.061 convert ✅ · **49.4 renders noise** ❌ |
| APK on device | `0.2 (2)`, `/sdcard/Download/npuforge-0.2.apk`, md5 `254d290afbaf043a99ff2643f7f394d4`, installed and verified |

## What changed this session

- LoRA support in the app: pick several adapters, each with a **bounded slider**
  (−1.0 … 2.0 in 0.1 steps), merged in one pass at conversion time. `tplconv
  --lora <file>[:<strength>]` reproduces `tools/lora_merge.py` + convert byte for
  byte, on x86 **and** on the phone.
- Model-name validation, and a suggested name built from the checkpoint plus its
  adapters (`base+lora@0.8`) — because strength is baked in and two strengths are
  two different models.
- Live progress with a real percentage where the tool reports one.
- **New: an Info tab** (`app/.../ui/InfoScreen.kt`) stating the scope and every
  known limit in plain language, including the 2.49 fp16 caveat. It is the
  user-facing copy of `docs/LIMITS.md` — **change both in the same edit**.
- Documentation: `CLAUDE.md`, this file, `ROADMAP.md`, `docs/LIMITS.md`,
  `docs/PIPELINE.md`. The converter's history has been moved out of
  the PC conversion tree it grew out of, into this repo.

## ⚠ Added 2026-09-14, later the same day: read `docs/CHECKPOINT-FAMILIES.md`

The plan below said "decide between P1 and P2". **P2's premise is now in doubt**
and the section after this one is superseded on that point:

- ReV Animated — anime-lineage, 2.5D — profiles like a *photoreal* checkpoint
  (max weight-span ratio **1.27** against base SD1.5, vs CyberRealistic's 1.22).
  MistoonAnime's 48 may be one merge, not a family.
- Linear blends of the two local checkpoints move the ratio **smoothly** from
  1.22 to 20.58 — a 50/50 merge lands at 10.30 — so there is no boundary for a
  two-template switch to sit on.
- A **0.89 MB** read decides a checkpoint's profile, and it works over HTTP
  ranges without downloading the file (`tools/span_probe.py`, new).
- Activation encodings are literals inside `libqnn_model.so`; making them
  pack-driven costs ~29 KB and could let one template stretch to fit —
  `ROADMAP.md` item 1c, an afternoon, and it could retire P2 entirely.

**So: run the survey (item 1a) before spending 3 h on an anime template**, and
if P1's 2.28 rebuild happens, build it from **base SD1.5 + stock CLIP + ft-mse
VAE** rather than DreamShaper (item 1b) — measured free, and it takes a specific
finetune out of the middle of the pipeline.

## The one thing to do next

**Decide between the two parked items and start it** (`ROADMAP.md` P1/P2). Both
are PC-side template work and both are ~2–3 h of quantize plus ~9 min of
re-derivation:

- **P1, a QAIRT 2.28 build** — removes the fp16 stamp, which is what currently
  stands between "works on my phone" and "works on phones". ⛔ Needs the 2.28
  SDK, which is no longer on this machine and needs a Qualcomm login to fetch.
  That download is the first step and it needs the user.
- **P2, an anime template** — closes the one checkpoint family that fails.
  Nothing blocks it: the pipeline is generic over templates.

Doing both under 2.28 in one sitting is the efficient shape — the anime
calibration and the 2.28 rebuild are the same job run twice with different
inputs, and building the anime template under 2.49 would only have to be redone.

If neither is started, the cheap win is **roadmap item 1**: compute the
weight-span ratio during header inspection and show it, so a checkpoint that
will render noise is flagged before two minutes are spent. The predictor is
3-for-3 and the code already walks every tensor.

## What NOT to redo

- **Do not re-derive the quantization rules.** All 2,383 scales reproduce
  exactly; the ±126-unit int32 bias residue is a float32 ulp in the converter's
  own bias, not a rounding rule — **15 variants were tried**. `docs/PIPELINE.md`.
- **Do not gate on the context binary's md5.** The compile is not
  byte-reproducible. Gate on the weight pack.
- **Do not compare a recipe-built model against the stock template arm.** It
  reads as 0/5 when nothing is wrong.
- **Do not re-run the 1-step render trick** to separate two different models. It
  only worked in Phase 1 because those models were numerically near-identical.
- **Do not investigate the anime checkpoint for corruption.** Its UNet is
  healthy; the NaNs are in the VAE and are authentic for an fp16 SD1.5 merge.
  Nor is the borrowed CLIP/VAE the cause — the 2×2 in `README.md` ruled it out.
- **Do not convert the text encoder or VAE.** Measured unnecessary (0.13–0.39%
  median difference, and a deliberate mismatch is visually indistinguishable).
- **Do not fold the converter into a generator's UI.** It is a separate app by
  decision: it writes models, any generator imports them.

## Rebuild commands

```sh
# host reference binary  (⚠ -ffp-contract=off is not optional)
c++ -O2 -std=c++17 -ffp-contract=off -o tplconv native/tplconv.cpp

# device binary
$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++ \
    -O2 -std=c++17 -ffp-contract=off -static-libstdc++ -o tplconv_arm native/tplconv.cpp

# the app
./gradlew.bat assembleDebug
```

Install and push (always pass `-s <serial>` — `adb devices` lists them; a phone
on both USB and TCP appears twice):

```sh
adb -s "$SERIAL" install -r app/build/outputs/apk/debug/app-debug.apk
adb -s "$SERIAL" push app/build/outputs/apk/debug/app-debug.apk /sdcard/Download/npuforge-<ver>.apk
adb -s "$SERIAL" shell md5sum /sdcard/Download/npuforge-<ver>.apk   # must match the local md5
```

Correctness gate, before trusting any change to `native/tplconv.cpp`:

```sh
./tplconv template/recipe.bin template/tpl_trim.pack DreamShaper_8_pruned.safetensors out.pack
md5sum out.pack     # 042b99cf38b37cecc15483f37118bb46
```

`libqnn_model.so`, the QAIRT runtime and the checkpoints are gitignored and must
be regenerated or supplied — `docs/BUILD.md`, `NOTICE`.

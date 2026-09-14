# npuforge — project index

Convert an SD1.5 `.safetensors` into a Qualcomm NPU model **on the phone**, in
~117 s, with no PC. A standalone converter: it writes models that any generator
can import, and is deliberately not a feature of any one generator.

📌 **START HERE: `HANDOFF.md`.** It is rewritten every session and says what
state things are in and what to do next. This file is only an index.

## Status

✅ **Working end to end as an Android app** (2026-09-14, one device): pick a
checkpoint, optionally stack LoRAs, get `Download/npuforge/<name>.zip` that a
generator can import. The app's output renders **byte-identical PNGs** to both
the adb-driven and the PC-built pipelines.

| | |
|---|---|
| weight stage (`tplconv`) | **24 s**, 1.98 GB peak RSS |
| compile (`qnn-context-binary-generator`) | **93 s**, ~4.8 GB peak |
| shipped template bundle | 43 KB pack + 397 KB recipe + 9.7 MB library |
| output model | ~1.3 GB zip |
| verified on | exactly **one** phone: SM-S938B / SM8750 / HTP v79 |

⚠ Known limits that look like bugs: the 2.49 **fp16 stamp** (some newer chips
refuse to load the output), the hardcoded **`_8gen2`** tier, and **anime
checkpoints converting cleanly into noise**. All three are measured and
explained in `docs/LIMITS.md`; the app's Info tab is the user-facing copy of it.

⛔ **Not shippable yet** — the QAIRT redistribution question in `NOTICE` decides
whether it can be distributed at all.

## Read on demand

| Doc | Read it when |
|---|---|
| `HANDOFF.md` | **always, first** — live state, next step, what not to redo |
| `docs/LIMITS.md` | **before saying what this supports**, or when a conversion produced something wrong. Scope, the fp16 stamp, the anime failure and its dead hypotheses, LoRA support matrix, how to read a failure |
| `ROADMAP.md` | planning. Also lists what is **parked by decision** (2.28 build, anime template) and what was **rejected on measurement** — check before proposing anything |
| `docs/CHECKPOINT-FAMILIES.md` | **before planning a second template, a second resolution, or changing whose CLIP/VAE ships**. Which SD1.5 checkpoints exist and where they came from, the 0.9 MB span probe, why "photoreal vs anime" may be the wrong split, and what another resolution costs |
| `docs/PIPELINE.md` | touching the template, the recipe, or the quantization rules. The full derivation with every measurement, plus the dead first template and the four checks that passed on it |
| `docs/BUILD.md` | building `tplconv`, the pack-loading library, or running the adb-driven pipeline. ⚠ `-ffp-contract=off` and the md5-gating trap live here |
| `docs/ANDROID.md` | **anything touching packaging, `jniLibs`, or library paths.** Four separate causes of one "Device Creation failure" |
| `README.md` | the public face: results, why the bundle is small, the CLIP/VAE 2×2 |
| `NOTICE` | before publishing anything, or adding a dependency |

## Key files

| Path | What |
|---|---|
| `native/tplconv.cpp` | the weight stage. safetensors reader, quantization rules, TPLPACK1 writer, LoRA merge. **Its header comment is the arithmetic spec** |
| `tools/tpl_apply.py` | the Python reference `tplconv` must match byte-for-byte |
| `tools/tpl_recipe.py`, `tpl_patch.py`, `tpl_pack.py` | template authoring (PC, once) |
| `tools/tpl_recipe_bin.py`, `tpl_pack_trim.py` | what turns authoring output into the 43 KB + 397 KB the app ships |
| `tools/lora_merge.py` | the PC reference for the LoRA merge |
| `tools/span_probe.py` | weight-span profile of a checkpoint, local **or over HTTP ranges** — 0.9 MB decides whether a checkpoint will survive the template's activation ranges |
| `template/` | `recipe.bin`, `tpl_trim.pack`, `sources.txt` (686 LDM keys) |
| `app/src/main/java/com/abrah/npuforge/Converter.kt` | the staged pipeline: weights → compile → DSP libs → zip via MediaStore |
| `…/ConvertService.kt` | foreground service, progress parsing, `--esa` LoRA extras so adb can drive it |
| `…/CheckpointInfo.kt` | header-only inspection; refuses SDXL/SD2/diffusers before a 2 GB copy |
| `…/MainActivity.kt`, `…/ui/InfoScreen.kt` | Compose UI; Info tab mirrors `docs/LIMITS.md` |
| `app/src/debug/…/ProbeService.kt` | DSP reachability probe, debug source set only |

## Rules that cost time to learn

- **Correctness is decided by `cmp` on the weight pack, never by looking at
  renders and never by the context binary's md5** — the compile is not
  byte-reproducible. Two reference checkpoints with known pack md5s are in
  `docs/BUILD.md`.
- **`-ffp-contract=off` is mandatory** on both host and device builds. Without
  it clang fuses FMA on aarch64, the LoRA merge diverges, and the phone produces
  a different pack from x86 (renders drifted 29 dB after 20 steps).
- **Compare against the right arm.** A model built from the *recipe* pack is
  compared against the PC's build of the *recipe* pack. Comparing against the
  stock template arm reads as a total failure when nothing is wrong.
- **Latency proves a graph compiled, never that it computes anything.** See the
  dead template in `docs/PIPELINE.md`.
- **When a check reports a failure, suspect the check first.** In this project an
  encoding gate failed a healthy build, a DSP probe passed without creating a
  device, and a render comparison scored 0/5 against the wrong baseline.
- ⛔ **Never commit** `libstable_diffusion_core.so`, `libQnn*.so`,
  `qnn-context-binary-generator`, `libqnn_model.so`, checkpoints or packs. The
  first is CC BY-NC 4.0; the rest are Qualcomm's or huge. `.gitignore` covers
  them — do not "fix" it.
- ⛔ **This repo is public. Sanitise before every push, not before the first
  one.** Anything written here for local convenience gets published the moment
  it is pushed, and a push cannot be taken back — a scrub after the fact still
  leaves the value in someone's clone. So check *before* `git push`:

  | never publish | write instead |
  |---|---|
  | device serials, `adb -s R5…` | `-s <serial>`, or `"$SERIAL"` |
  | Tailscale / LAN addresses, `:5555` endpoints | `<device-ip>` |
  | absolute paths under a home directory (`C:\Users\…`, `/home/…`, `/mnt/c/…`) | a repo-relative path, or say what the thing is and how to get it |
  | paths into private sibling checkouts | name the artefact, not its location |
  | personal emails, tokens, keystores, launcher scripts | nothing — `.gitignore` them |

  ```sh
  git grep -nEi "R5CY|100\.99\.|C:.Users|/home/[a-z]|:5555|@gmail" -- $(git ls-files)
  ```

  ⚠ **The working tree is not the deliverable — history is.** Scrubbing a file
  leaves the old blob reachable. If something identifying has already been
  committed, rewrite (`git filter-branch --tree-filter`), drop `refs/original`,
  expire the reflog and `gc --prune=now`, then verify by scanning **every blob**,
  not just `HEAD`. That was done once (2026-09-14) and the verification is the
  part that took the time.

  ⚠ Commits are authored as `AbrahamPaulJ@users.noreply.github.com`, and
  `Claude-Session:` trailers are **not** published — they link to private
  transcripts. Keep `Co-Authored-By`.

## What this repo does NOT contain

It is self-contained for **building and verifying the converter**. It is not
self-contained for **authoring a new template** or for **rendering a model to
look at it**. Those need things that are deliberately absent — some because they
are Qualcomm's, some because their licence is incompatible with this one
(`NOTICE`), some because they are gigabytes.

| What | Why absent | Needed for | How to get it |
|---|---|---|---|
| QAIRT SDK — `qnn-context-binary-generator`, `libQnnHtp*`, the device runtime | Qualcomm's, redistribution restricted (`NOTICE` §1) | the app's `jniLibs`, and every rebuild | Qualcomm's developer site; needs an account |
| `libqnn_model.so` (9.7 MB) | generated from QAIRT converter output (`NOTICE` §3) | the app's assets | regenerate — `docs/BUILD.md` |
| Template authoring scripts and the modified `diffusers` UNet they import | CC BY-NC 4.0 upstream (`NOTICE` §2) | building a **new** template | `docs/TEMPLATE-AUTHORING.md` names every step and the upstream |
| Render + score harnesses | not written for a public audience | judging a converted model on device | `docs/TEMPLATE-AUTHORING.md` §5 says what they must do |
| Checkpoints | ~2 GB each, mixed licences (`NOTICE` §4) | any conversion | the user supplies their own |

## Environment

- The app's `jniLibs` and `assets/template/libqnn_model.so` are **gitignored**, so
  a fresh clone does not build a working APK until they are put back. That is a
  licence boundary, not an oversight — `docs/BUILD.md` and `NOTICE`.
- Build: JDK 17 + Android SDK 35, `./gradlew.bat assembleDebug` (or `./gradlew`).
- adb: always pass `-s <serial>` — there is usually more than one transport when
  a phone is on both USB and TCP. Check `adb shell settings get global wifi_on`
  before a large push, and verify every push by md5: `adb push` reports bytes
  **sent**, not bytes **landed**.
- ⚠ **`\` collapses inside quoted heredocs in some shells.** Kotlin/C++/Gradle
  written that way comes out with literal newlines and invalid escapes. Write
  source files with an editor, or use `Char(92)` / `System.lineSeparator()`.

## Documentation convention

`HANDOFF.md` is rewritten each session, never appended. `CLAUDE.md` stays a thin
index (< ~350 lines). Findings live in `docs/`, one file per topic, each claim
carrying its measurement. **No self-contradiction** — when a claim is refuted,
rewrite the section that states it rather than appending a correction. Record
**what was wrong and why**: the dead hypotheses are the most valuable part of
this record.

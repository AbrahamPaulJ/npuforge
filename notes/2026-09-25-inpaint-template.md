# 2026-09-25 — SD1.5 9-channel inpaint template

Outcome: template authored on QAIRT 2.50, all host gates passed, phone
conversion with the installed 1.0.0 binaries verified, five arms rendered on
the S25 Ultra. Measurements live in `docs/SD15-INPAINT.md`; this note records
the process and what went wrong.

## Decisions (user, this session)

- Calibrate on DreamShaper 8 inpainting (rows existed; sibling of the
  text-to-image template's checkpoint).
- Support existing inpaint checkpoints first, add-difference second.
- Defer how the add-difference delta reaches the phone.
- Author on QAIRT 2.50, not the 2.49 on disk.

## What was wrong, and why

- **LocalDream's existing inpaint calibration rows covered 12 timesteps,
  77–913.** A template built on them would clip the first and last steps.
  Re-calibrated with DPM `linspace` (20 timesteps, 50–999). The old txt2img
  gate would have passed the clipped build: its `Cast` bound was `>= 100`.
- **The QAIRT 2.50 SDK looked login-gated.** PowerShell's
  `Invoke-WebRequest` got 403 from the Software Center URL for every version;
  plain `curl` downloads the same URL without a login. The qai-appbuilder
  "QAIRT_v2.50" release zip contains only Genie Windows libraries.
- **The local `jniLibs/` are QAIRT 2.49**, while the installed 1.0.0 release
  and `docs/BUILD.md` are 2.50 (verified from the installed app's libraries).
  The local SDXL template library is also absent. The release was evidently
  built from assets not in this working tree. The local copies were replaced
  from the installed release later the same day (below).
- `qnn-model-lib-generator -t aarch64-android` needs `ndk-build` on `PATH`,
  not only `ANDROID_NDK_ROOT`.
- Two self-inflicted shell failures: a PowerShell-quoted `tr -d "\r"` deleted
  every letter `r` from a script, and `pkill -f` over `adb shell` killed its
  own shell. Script files avoided both afterwards.

## Later the same day

- **In-app test (debug APK, adb-driven via `run-as`):** conversion succeeded,
  but Nightmare failed at `vae_encode`. Cause: npuforge's SD1.5 VAE is float32
  I/O and Nightmare's SD1.5 backend handled only xororz's uint16 — so **every**
  npuforge SD1.5 export had been failing there, unnoticed because no end-to-end
  SD1.5 render had been recorded. Fixed in Nightmare backend patch 012
  (1.6.033); quantized models verified byte-identical before/after.
- **Add-difference:** built the official `sd-v1-5-inpainting − v1-5` diff
  (fp16 1.72 GB; fp16 vs fp32 storage within build noise), hosted it on HF,
  added `tplconv --inpaint-diff` with Python parity and tests, and the app's
  Text-to-image | Inpainting choice. User confirmed DreamShaper 8 base and
  CuteYukiMix (anime) as inpaint models in Nightmare.
- **Mistakes that cost time:** a PowerShell `Get-Content`/`Set-Content` round
  trip mangled every `⚠` in `build.gradle.kts` into mojibake and added a BOM
  (restored from HEAD); the first on-phone add-difference conversion was killed
  by Android because my test server was still holding NPU memory; the test2 debug
  APK was nearly shared although it bundles Local Dream's CC BY-NC backend —
  hence the `preview` build type.
- Windows builds need the host-aware NDK path now in `app/build.gradle.kts`;
  the local `jniLibs/` were stale 2.49 copies and were replaced from the
  installed 1.0.0 release (2.50).

## Left behind

- Phone: `/data/local/tmp/inp9` (6.3 GB test rig, safe to delete); npuforge
  debug `1.0.1-inpaint-test2` and `…preview` both installed.
- Host: WSL `inp9/` authoring workspace (~25 GB, `scripts/README.txt`), QAIRT
  2.50 in WSL. The WSL disk has not been compacted since; C: had 17 GB free.

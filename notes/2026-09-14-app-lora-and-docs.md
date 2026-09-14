# 2026-09-14 — the app, LoRA, and what went wrong on the way

Session history. The findings themselves live in `docs/`; this is the order
things happened and the mistakes, which are the part that does not fit a topic
file.

## Sequence

1. Phase 1's render gate ran and failed 0/5 — **against the wrong arm**. The
   phone builds from the *recipe* pack, so `recipe` is its twin; compared
   correctly it was 5/5 byte-identical. The first real lesson of the day was
   about the check, not the converter.
2. Scope correction from the user: the converter is **a separate app, never a
   feature of a generator app**. Those changes were reverted and the work moved into
   `CC/npuforge` as its own MIT repo.
3. The Android app: checkpoint picker → header inspection → foreground-service
   conversion → a single zip in Downloads. Getting QNN to run inside an app cost
   four separate fixes (`docs/ANDROID.md`).
4. LoRA, first on the PC (`tools/lora_merge.py`), then in `tplconv`, then in the
   app with a bounded strength slider.
5. An anime checkpoint converted cleanly and rendered noise. Localised to the
   converted UNet by a 2×2 and explained by weight-span ratio (`docs/LIMITS.md`).
6. Documentation split: converter history moved out of the PC conversion tree into this
   repo; an **Info tab** added to the app so the limits reach the person holding
   the phone. APK `0.2 (2)` installed and pushed to `/sdcard/Download/`.

## Mistakes worth keeping

- **"Corrupt checkpoint" was over-called.** MistoonAnime's NaNs are in the VAE,
  not the UNet, and NaNs in an fp16 SD1.5 merge's VAE are ordinary. Retracted
  after measuring. The cause was range borrowing all along.
- **The LoRA merge matched 0 modules at first.** kohya names SD1.5 LoRAs with
  **diffusers** block names while the checkpoint uses **LDM** ones. Fixed by
  building the kohya name *forwards* from each checkpoint key — reversing it is
  ambiguous, since `to_out_0`, `ff_net_0_proj` and `transformer_blocks_0` all
  collide under `"." → "_"`.
- **The LoRA byte gate then failed on arithmetic order**: summing into a zero
  buffer and scaling once is not the same as scaling each term. Same class of
  bug as the quantization rules.
- **And failed again across architectures**: the phone's pack differed from
  x86's and the renders drifted 29 dB. Cause was clang fusing multiply-add into
  FMA on aarch64. `-ffp-contract=off` made them identical. Without that flag the
  byte gate had only ever validated the host build.
- **Caching mattered more than expected**: the recipe reads an attention weight
  once per head, so a q/k/v tensor is fetched 8 times; recomputing the rank-R
  product each time cost more than the rest of the conversion (126 s → 55 s).
- **The DSP probe passed without creating a device** — `qnn-net-run` with an
  empty input list aborted during argument parsing. It now keys off `"ok":true`.
- **The first encoding gate failed a healthy build** by asserting on
  non-quantized tensors. Check dtype 1046 only.
- I described a needed rebuild in a way that implied the expensive part was
  avoidable and then quoted 2 h 50 m in the same breath. Say the cost plainly
  the first time.
- A blind `screencap` captured the user's personal content. Don't take
  screenshots that were not asked for.

## Environment traps hit repeatedly

- `\` collapses inside quoted heredocs in this shell, so generated Kotlin, C++
  and Gradle came out with literal newlines and invalid escapes. Write source
  files with the Write tool, or use `Char(92)` / `System.lineSeparator()`.
- adb must run from PowerShell with an explicit `-s`. Verify every push by md5:
  `adb push` prints bytes sent, not bytes landed.

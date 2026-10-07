# NPUforge handoff

NPuforge converts SD1.5 and SDXL checkpoints into Qualcomm NPU packages on
Android. Start with [README.md](README.md); build instructions are in
[docs/BUILD.md](docs/BUILD.md), measured limits in [docs/LIMITS.md](docs/LIMITS.md),
and the documentation index in [CLAUDE.md](CLAUDE.md).

## Where things are - 7 October 2026

**1.0.12-preview installed (branch `sdxl-swap`, uncommitted):** SDXL Swap offers ControlNet and IP-Adapter
again (preview; `Converter.UNOFFERED` keeps inp/conv/freeu/pag/couple), and EVERY export now carries the
schema-2 `swap_features.json` (`Converter.manifestJson`: producer, family, kind, prediction, text_tokens, size,
soc, `detail` per feature, `features` LAST -- Nightmare's backend 020 substring-searches after it).
`ManifestTest` pins it. The user is converting Illustrious XL with LoRA + ControlNet + IP; Nightmare's
side is its backend 023 / 024. Nothing below changed.

## Before that - 4 October 2026

**1.0.11-preview installed (uncommitted, branch `sdxl-swap`): SDXL Swap is the v2.1 template** (no longer
labelled preview). `template_sdxl_swap/` = the v2.1 bundle from the private HF archive
`AbrahamPJ/sdxl-swap-vm-archive` (`v21/bundle/`; lib 302 MB, gitignored like before). Chips: LoRA,
ControlNet, IP-Adapter and now **Inpaint** (downloads the hosted SDXL difference, `SdxlInpaintDiff` in
`InpaintDiff.kt`, 5.1 GB, size + SHA-256 pinned; the trade-off — better editing, weaker text-to-image — is
explained under the chip, in all four locales). v2's conv / FreeU / PAG / coupling are always dropped
(`Converter.UNOFFERED`; coupling + the rest exceeds the S25's HTP process memory). Every export carries
`qnn_context.txt` = `462_masked_v1` + `lora_targets.json` + `swap_features.json` (`sdxl_swap_v2`).
⚠ The first v2 conversion on the phone (LoRA only, Juggernaut) started 4 Oct 21:31 and was still
compiling at 22:52 (storage-backed compiler heap peak 15.2 GB; it needs that much free disk). Nightmare
1.6.078 reads 6-chunk prompts and v2's LoRA targets; ControlNet / IP / inpaint inputs are not fed on SDXL yet. v1 SDXL Swap was never released: no
compatibility kept. Everything below is the 3 October state.

**Newest, unpushed: branch `sdxl-swap` (`6a61838`, `415cdaf`, `0ad819b`), 1.0.10-preview installed on
the user's phone.** *Convert as → SDXL Swap* (preview) for SDXL checkpoints, from `template_sdxl_swap/`
(the SDXL Swap template, [docs/SDXL-SWAP-TEMPLATE.md](docs/SDXL-SWAP-TEMPLATE.md)): LoRA / ControlNet /
IP-Adapter chips, none ticked by default (a plain SDXL export); inpaint is in the template but not offered
(its SDXL difference is not hosted). Its 190 MB lib (`-O0`) and `recipe.bin` / `tpl_trim.pack` are
gitignored; the bundle lives in the authoring PC's WSL (`~/sdxl_swap/out/bundle`). Proven: the phone's
`tplconv` rebuilt the Juggernaut XL Ragnarok pack byte for byte (103 s); the template's on-phone compile
took 52 min. Not yet run end to end in the app — the user converted Juggernaut with 1.0.9 (features off)
and is testing a LoRA conversion with Nightmare 1.6.076. The next template (v2) and its plan: the doc's §7.

**v1.0.8 is published as a GitHub pre-release.** `main` and the local
`sd15-swap-v3` branch point to commit `d4905de`; annotated tag `v1.0.8` points to
the same commit. The release is
[NPUforge 1.0.8 - Swap v3 and V-prediction](https://github.com/AbrahamPaulJ/npuforge/releases/tag/v1.0.8).
The stable `Latest` release remains v1.0.7.

Release APK facts:

| Item | Value |
|---|---|
| Asset | `npuforge-1.0.8.apk` |
| Size | 129,054,305 bytes |
| SHA-256 | `8a94649e1ef27a693ec9a23101b9bd0834b1774102bf4277831f0a24f41f16ae` |
| Package | `com.abrah.npuforge`, version code 9, version name 1.0.8 |
| Signature | release certificate SHA-256 `901e78f6b49a382b1e53eb15449e7fa3df9533896026460d30f0c71b346ac26b` |

The release build and lint passed, as did the complete Android unit suite. The
Swap-v3 host suite was previously recorded as 60/60. The user explicitly chose
to publish without a final manual phone pass. The release APK was not installed
or pushed to the phone because the expected USB serial (in the private notes) was absent;
only a changing LAN transport was visible. Do not claim v1.0.8 has been rendered
on-device as a release build.

### What v1.0.8 contains

- **SD1.5 Swap v3:** conversion-time chips keep or remove LoRA, ControlNet,
  IP-Adapter and inpaint inputs. Omitted features are absent from the compiled
  graph. The inpaint path uses `--input-channel-prefix` and template constants.
  The evidence and authoring contract are in
  [docs/SD15-LORA-CN-TEMPLATE.md](docs/SD15-LORA-CN-TEMPLATE.md) section 9.
- **Prediction type:** `modelspec.prediction_type` values `v` and `epsilon` are
  detected from safetensors metadata; the UI permits an explicit override.
  Missing or unknown metadata remains epsilon. V-prediction exports contain an
  empty root `V_PRED` entry; epsilon exports do not. See
  [docs/PREDICTION-TYPES.md](docs/PREDICTION-TYPES.md).
- **SDXL Swap authoring work:** the checked-in plan and tiny-model evidence live
  in [docs/SDXL-SWAP-TEMPLATE.md](docs/SDXL-SWAP-TEMPLATE.md). The live Vast/WSL
  build is owned by `../nightmare-mobile/notes/HANDOFF.md`; do not restart it
  from this document.

Nightmare Mobile v1.6.075 understands the same `V_PRED` marker and was released
as a pre-release alongside this build. Its importer preserves the zero-byte
entry and its common model-launch path supplies `--use_v_pred` for SD1.5, SDXL
and Swap/template packages.

## One next step

Wait for field or manual reports before promoting v1.0.8. The smallest useful
release-candidate check is one known v-prediction checkpoint:

1. confirm metadata selects V-prediction, then override both choices once;
2. inspect the resulting ZIPs (`V_PRED` present only in the V-prediction one);
3. import the V-prediction package into Nightmare Mobile v1.6.075;
4. confirm `BackendProcess` logs the marker and `--use_v_pred`, then render and
   compare against an epsilon launch.

Promotion, only after that evidence:

```powershell
gh release edit v1.0.8 --repo AbrahamPaulJ/npuforge --prerelease=false --latest
```

Do not rebuild or republish merely to change the GitHub release flag. The asset
digest above is already verified against GitHub's uploaded digest.

## Do not redo

- Prediction type changes sampler interpretation, not QNN graph conversion.
- `V_PRED` is the established LocalDream package marker; the spelling is not
  `V_PRE`.
- SD1.5 community v-prediction checkpoints often lack prediction metadata, so
  the manual override is required even though SDXL files commonly carry
  ModelSpec metadata.
- Swap v3 feature removal happens at template composition/compile time through
  `QNN_TPL_DROP`; it is not a runtime toggle.
- The released Swap-v3 template and generated QNN/QAIRT assets are intentionally
  staged build inputs and are excluded where licensing requires it. Do not add
  signing material or restricted SDK binaries to git.
- Do not touch the active WSL `~/sdxl_swap/`, Vast VM, or the phone's
  `/data/local/tmp/probe` from this handoff. Their current state is documented
  in Nightmare's handoff.

## Rebuild and verify

Release signing properties live outside the repositories under
`<.secrets>\npuforge-keystore\`.

```powershell
$env:NPUFORGE_SIGNING_PROPERTIES = '<.secrets>\npuforge-keystore\signing.properties'
.\gradlew.bat :app:testDebugUnitTest --no-daemon
.\gradlew.bat :app:assembleRelease :app:lintRelease --no-daemon
```

Then verify `app/build/outputs/apk/release/app-release.apk` with the newest
Android SDK `apksigner.bat`. Never print the signing properties or copy them
into this repository.

## Source navigation

| Area | Entry point |
|---|---|
| Checkpoint family and prediction metadata | `CheckpointInfo.kt` |
| Conversion lifecycle and override/default choice | `ConvertService.kt` |
| Package markers and ZIP assembly | `Converter.kt` |
| Prediction-type UI | `MainActivity.kt` |
| Swap-v3 native packing | `native/tplconv.cpp`, `tools/tpl_apply.py`, `tools/tpl_features.py` |
| Prediction regression tests | `app/src/test/java/com/abrah/npuforge/PredictionTypeTest.kt` |
| Swap-v3 native regression tests | `tests/test_input_channel_prefix.py` |

## Existing limits that remain

QAIRT is 2.50.0.260828. Phone compilation targets the phone's own HTP
architecture, so exports are chip-specific. SDXL still uses the precompiled VAE
contexts described in [docs/LIMITS.md](docs/LIMITS.md). Text-encoder LoRA remains
unsupported, BF16 checkpoint data is read as FP32, and device-specific coverage
is not universal. Keep new compatibility claims tied to measured phone results.

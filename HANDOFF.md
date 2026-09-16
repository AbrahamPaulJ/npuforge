# Current state — 2026-09-16

SDXL INT8 conversion and generation work on the tested Galaxy S25 Ultra.
The user confirms that Pony's CLIP-only test fixed its output and the new
full-component app pipeline produced excellent waiIllustriousSDXL_v170 images.
Cross-device compilation and broader LoRA validation remain separate work in
[docs/SDXL-INVESTIGATION.md](docs/SDXL-INVESTIGATION.md).

## Latest findings and source changes

- **Pony CLIP-only diagnostic succeeded on the user's phone.**
  `v6-Pony-CLIP-test.zip` replaces all seven CLIP component files using the
  checkpoint's own weights. Its UNet, VAE, tokenizer and model markers preserve
  the exact 23:27 phone export. Every ZIP payload passed CRC/SHA-256 readback.
  The user reports "It worked!" after testing the replacement. This establishes
  a working fix for this Pony model by replacing the shared CLIPs, with no UNet
  or VAE change. It does not validate every SDXL checkpoint. The distinct test
  model path avoids reusing in-memory conditioning.
- The two CLIPs were exported sequentially with the existing FP16-L / INT8-G
  formats. Strict checkpoint loads and original-encoder wrapper checks passed;
  emitted embedding files reproduce fixture inputs exactly. Host MNN hidden
  relative RMSE is 0.080–0.090% for L and 1.92–2.92% for G; G EOS pooling is
  0.91–1.29%. All tested outputs are finite. Local scripts, logs and manifests
  are in the dated Pony CLIP test artifact directory, outside the public repo.
- **Vivo remains unresolved.** The 11:23 UTC V2307A / SM8650 report already uses
  the 8 KiB slab allocator and C++ hooks, yet aborts at exactly 65,530 mappings:
  64,575 Scudo-secondary, only 503 compiler backing mappings. This strongly
  supports mapping exhaustion, not an explanation based on advertised RAM.
  Do not report the slab or C++ changes as a verified fix.
- `ConversionReport` now reads Android's native tombstone when available and
  exports its abort message, signal and up to 24 crashing-thread frames.
  `NativeTombstone` bounds input and skips unrelated protobuf fields. The
  missing historical Vivo stack has not been recovered by this source change.
- **LoRA memory defect fixed in source.** `LoraSet` now caps retained FP32 merged
  weights at 128 MiB. Previously a broad SDXL transformer adapter could retain
  8.14 GiB across the two weight passes. Evicted tensors are recomputed with
  unchanged arithmetic; conversion time may increase.
- Native and Python LoRA merging now reject unmatched/unsupported adapter
  tensors and invalid ranks/up kernels, and report dropped text-encoder modules.
  Diffusers-style ResNet/conv-name coverage and BF16 support remain incomplete.
- **A tester's SDXL LoRA conversion succeeded.** `Lora-f16-success.txt` records
  nubia NX789J / SM8750, Android 16, roughly 24 GB reported RAM; adapter label
  `dmd2Sdxl4stepLora.LhSd`, strength 0.8, 722 matched modules, both native exits 0
  and completion in 687.678 seconds. F16 is reported by the filename, not verified
  from tensor metadata. Base checkpoint identity and rendered LoRA effect remain
  unknown; this log does not verify the new bounded cache implementation.
- The same tester subsequently reported **DMD2 F32 success**, with recognizable
  renders labeled `+lora_f32`. The user says both F16 and F32 tests preceded the
  component-conversion update. Record these as reported conversion and generation
  successes for that setup; no F32 tensor file or controlled adapter comparison
  was supplied. They do not independently validate the component update.
- **Full-component Illustrious conversion succeeded.** The user reports excellent quality after full app conversion of `waiIllustriousSDXL_v170`. The supplied screenshot shows 1024 × 1024, 30 steps, CFG 7, seed 418928922 and 45.8 seconds on NPU. This validates conversion and text-to-image for this checkpoint; it does not isolate CLIP versus VAE effects or establish compatibility with every derivative.
  No UNet template change was needed; xxmix calibration remains three trajectories / 30 rows.
- The original Pony v6 screenshot showed banded noise at 30 steps / CFG 7;
  the user confirms DPM and V-prediction off. Its checkpoint header specifies
  epsilon. The 23:27 conversion report records no LoRAs and both stages exiting
  successfully; the fresh export's shared-component ZIP metadata matches the
  donor. No tensor trace for this exact render was retained. All 1,680 UNet
  source keys are covered by the recipe; saved tensor layouts match the runtime.
- Shared CLIP embedding replacement is now measured: sampled token differences
  are 28.87% / 33.18% relative RMSE versus Pony, and full position tables differ
  by 58.61% / 62.37%. These quantify weight differences; the subsequent successful
  CLIP-only phone test supplies the image-quality evidence. Reproducible checks
  and JSON results are saved with the local inspection artifacts.
- **Checkpoint-owned SDXL component conversion is implemented in the app.**
  `componentconv` writes both MNN CLIPs and their embeddings from sparse graph
  recipes; `tplconv` has FP16/FP32 rules for both VAE templates. The service
  validates required tensor names/shapes/dtypes, runs each component sequentially,
  and assembles the result without a donor download. SD1.5 now has the same
  component ownership with separate CLIP and 512px VAE assets. See [component conversion](docs/SDXL-COMPONENTS.md) for checks,
  authoring and precision limits. The SDXL flow now has the Illustrious success above;
  SD1.5 full-component phone validation remains pending.
- Component checks pass: all seven native CLIP outputs exactly match the working
  Pony diagnostic files; both native VAE packs match the converter's original
  weights; both VAE HTP contexts compile with the required planar float32 I/O.
  All 30 Python/native tests, debug APK assembly and full app lint pass. Generated
  assets and native converter bytes were checked inside the APK. The user installed
  and tested the app; no deployment or generation was performed by the agent.

- **Checkpoint-owned SD1.5 components are implemented too.** The existing
  clip-skip 2 plus final normalization contract is preserved; native CLIP output
  matches all three verified reference files exactly. Both VAE packs cover all
  248 parameters without ambiguous mappings, and both v73/soc43 contexts compile
  with verified 512px interfaces. The 28 existing regressions and two new VAE
  authoring cases pass, as do debug APK assembly and full lint. The service uses family-specific
  component assets with 10 conversion stages and no donor download for either
  family. Legacy archive backup/restore remains available. See
  [SD1.5 components](docs/SD15-COMPONENTS.md) for validation and quality limits.
  The known damaged MistoonAnime VAE is rejected; own weights do not fix a UNet
  activation-range mismatch or corrupt source tensors.

## Implementation

- Conversion status and log controls now have separate layout rows; other card
  headers use FlowRow so long labels do not squeeze badges into narrow columns.
- The app inspection report was reviewed: blocking donor IO is confined to IO,
  the wake lock uses a renewable 10-minute lease, and unused resources/style
  findings were cleaned up. Five narrowly documented lint exclusions cover
  required ARM64/DSP access and invariant min/sec abbreviations. The IDE's Play
  publication advisory remains; the manifest already declares specialUse.
- Follow-up Kotlin compilation and full-app Gradle lint passed with no lint
  issues. This does not constitute phone or visual validation.
- QAIRT 2.50.0.260828; SD1.5 v73, SDXL v75/soc57. Targets are fixed.
- `native/compiler_heap.c` owns SDXL compiler allocation backing, with indexed
  ownership lookup, an 8 KiB cutoff and shared 8 MiB slabs through 1 MiB.
- Successful configuration retained: O=3 with source-destructive reuse disabled.
  Reported total conversion: 437 seconds; Aura generation: 15 seconds at
  1024 × 1024, 8 steps, CFG 1, LCM/Karras on the tested MOP checkpoint.
  The user's 41% RAM snapshot is not a measured peak.
- `Converter.kt` sets the allocator environment only for SDXL, selects the
  model-specific template, and exports the Aura model markers.
- `ConvertService.kt` owns conversion, foreground lifetime, cancellation and
  child cleanup. Main UI includes live RAM, elapsed time and selectable logs.
- Legacy shared-component backups remain available for both families; new
  conversions do not consume those caches. Generated component recipes, template libraries and
  tokenizer assets remain outside source control.
- Android uses AGP 9.4.0, Gradle 9.7.1, Kotlin Compose plugin 2.4.20, SDK 37,
  minSdk 33, NDK 29.0.14206865. Gradle builds the converter and allocator on Linux.

## Working constraints

The user builds and installs through Android Studio. Do not install, deploy or
launch APKs on their phone, or start expensive model validation jobs as routine
follow-up. Preserve the successful compiler configuration. The user authorized committing
and pushing the source after the successful Illustrious test. Generated artifacts
and local reports remain excluded from publication.

The original diagnostic used existing evidence, synthetic host regressions and
a focused local export/check of Pony's two CLIPs. The subsequent requested app
implementation additionally authored VAE templates and native component writers;
its checks are recorded in docs/SDXL-COMPONENTS.md. No UNet template rebuild or
phone deployment was performed by the agent. The user successfully tested both
the Pony CLIP ZIP and the expanded app pipeline with Illustrious.
SDXL LoRA conversion now has a successful tester report on nubia; rendered adapter
effect, full adapter coverage and the separate Vivo compilation failure remain pending.

For the phone failure, obtain an exported report with the native tombstone before
changing allocator thresholds again. Next validate the SD1.5 component pipeline,
image-to-image and additional checkpoints. Text-encoder LoRA merging remains
unsupported. The Pony result
is user-reported visual success; the exact rerun parameters and image were not
independently captured. Reserve further UNet comparisons for failures that
remain after supplying the correct encoders.
Also record the independent padding difference: QNN currently EOS-pads CLIP-G,
while the standard tokenizer and MNN path zero-pad after EOS. Calibration mirrors
the QNN padding; it was unchanged in the successful CLIP-only test. Exact failing adapter files/symptoms are
not known from the user's broad reports.

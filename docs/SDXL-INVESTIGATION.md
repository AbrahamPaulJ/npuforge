# SDXL failures: evidence and next steps

Investigation: 2026-09-15–16. Existing source, saved reports and authoring artifacts
were inspected, followed by a focused local Pony CLIP export and a user-reported
successful phone test, then full-component app conversion and a successful
Illustrious phone render. Exact files and
symptoms from the broader user reports are unknown; the findings below must not
be applied to every phone or adapter.

## Summary

| Report | Evidence | Conclusion |
|---|---|---|
| More RAM, but conversion crashes | Vivo SM8650 reaches exactly 65,530 mappings; 64,575 belong to Scudo | Strong evidence of mapping exhaustion, independent of advertised RAM; originating allocation path remains unknown |
| SDXL LoRA fails | Merged FP32 tensors were cached without eviction across both weight passes | Confirmed memory defect; a broad SDXL transformer adapter can retain 8.14 GiB in this cache alone |
| Pony output is striped noise | User reports success after replacing only the seven CLIP files; baseline UNet/VAE/tokenizer/markers are byte-identical | Checkpoint-owned CLIPs fix this Pony case; the shared-encoder substitution was not faithful enough |
| Illustrious output was poor | User reports excellent waiIllustriousSDXL_v170 output after full checkpoint-owned component conversion | This checkpoint now works without changing the UNet template; the test does not isolate CLIP versus VAE contributions |

## 1. Phone compilation: a mapping limit is different from a RAM limit

The latest saved Vivo report starts at 11:23:46 UTC, identifies V2307A / SM8650,
and ends at 11:31:47 UTC. It already uses the 8 KiB cutoff, slabs and C++ hooks.
Immediately before abort it has **65,530 mappings**, including **64,575 Scudo
secondary mappings** and only **503 compiler backing mappings**. In its final
seconds mapping count rises sharply while virtual-memory size falls.

Linux limits the number of mapped memory regions per process separately from
their total size. Its documented default is 65,530. More physical RAM therefore
does not resolve this limit. The device's actual setting was unreadable, so the
matching count strongly supports this explanation without proving the failed
syscall. [Linux kernel documentation](https://www.kernel.org/doc/html/latest/admin-guide/sysctl/vm.html#max-map-count)

The signal is consistent with the process aborting itself. It does not identify
an OEM watchdog or prove an ordinary low-memory kill. The older Samsung Scudo
stack predates the successful configuration change and is not the missing Vivo
stack. See [the existing evidence](../notes/vivo-existing-evidence.md).

The slab and C++ interception changes have **not** closed this failure. Reducing
the cutoff again or disabling MTE without the native caller would be guessing.
Also keep chip compatibility separate: the SDXL target is fixed at v75 / soc57;
reported RAM and a phone's brand do not identify its DSP compatibility.

The report already records Android exit reason 5 (`REASON_CRASH_NATIVE`), but
the original diagnostic code only printed exit metadata. Android 12+ can expose
the native tombstone through `ApplicationExitInfo.traceInputStream`, including
the abort message and native backtrace. The report now decodes a bounded summary
of the abort message, signal and up to 24 crashing-thread frames into its existing
text export. Availability depends on Android retaining the trace; this
investigation has not recovered the missing Vivo stack.
[Android native crash diagnostics](https://developer.android.com/ndk/guides/debug)

## 2. LoRA: a separate weight-stage memory defect

`LoraSet::apply` in `native/tplconv.cpp` retained a full FP32 merged tensor for
every matched source key. The cache survived both conversion passes and had no
size limit. The compiler's storage-backed allocator cannot help: it is preloaded
only into the later QNN compiler process, not `tplconv`.

Counting source shapes in the shipped recipes, a broad adapter covering all
700 SDXL transformer weight tensors retains **8,735,948,800 bytes (8.14 GiB)**.
The corresponding SD1.5 transformer subset is about **924 MiB**. These are
calculated cache sizes, not measured phone peaks; checkpoint pages, adapter
pages and temporary merge buffers add to memory use. A sparse adapter costs less.

The fix bounds the merged-tensor cache to **128 MiB**, retaining recently used
tensors for repeated attention-head reads. Evicted tensors are recomputed when
needed; merge order and FP16 rounding stay the same. This may increase conversion
time. The bound applies to retained cache payloads, not the entire process.

Other limits remain important:

- Standard SDXL kohya attention naming follows the expected block mapping;
  there is no evidence that every SDXL LoRA is mapped incorrectly.
- Text-encoder adapters (`lora_te`, `lora_te1`, `lora_te2`) are not merged into
  the CLIPs, including checkpoint-owned encoders. A successful UNet merge cannot
  reproduce those changes.
- Native and Python merging now reject unmatched adapter tensors and invalid
  ranks/spatial up kernels, and report dropped encoder modules. Previously,
  one matched pair could conceal unmatched UNet modules in the same file.
  Standard diffusers-style ResNet/conv names remain incompletely mapped;
  supporting a convolution's shape does not mean every naming scheme is supported.
- The native safetensors reader supports F16/F32, not BF16 adapters.
- Correct merging does not establish that an adapter matches the checkpoint's
  training family, or that merged weights fit the template's activation ranges.

The tester report `Lora-f16-success.txt` records a successful SDXL conversion on
a nubia NX789J / SM8750 running Android 16, with 24,230,260,736 bytes of reported
RAM. The adapter label is `dmd2Sdxl4stepLora.LhSd`, strength 0.8: 722 modules
matched, both native stages exited 0, and conversion finished in 687.678 seconds.
The filename reports F16, but the log does not independently establish the
adapter dtype or the base checkpoint's identity. It establishes conversion
success, not a rendered comparison or proof of the new cache limit's behavior.

The same tester later reported DMD2 F32 success and supplied recognizable renders
labeled `+lora_f32`. The user explicitly places both F16 and F32 tests before the
component-conversion update. Together these are tester-reported conversion and
generation successes for that setup. Adapter dtypes were not independently read
from tensor files, and no controlled comparison isolates the adapter's effect.
These results do not independently validate the new component pipeline. The
subsequent Illustrious success is a separate test, recorded below.

The reports still do not identify an actual failing adapter, so the cache defect
is a concrete fix, not proof of the cause of every reported LoRA failure. The
successful tester run also rules out describing SDXL LoRA conversion as generally
unsupported. Rendered LoRA effect and full adapter coverage remain unverified.

## 3. Image quality: what the existing evidence rules out

The SDXL authoring `plan.json` identifies **xxmix9realisticsdxl_v10** as the
template checkpoint. Calibration covers three prompt trajectories / 30 rows,
77/154/231 tokens, with latents clipped to [-7.2, 7.2]. This is narrow coverage
for transferring activation ranges to distant finetunes. No internal-activation
failure has been measured here. Pony now works with the same converted UNet
after replacing its CLIPs; that case does not require a calibration change.

Additional saved artifacts from `pony-inspection` narrow the diagnosis:

| Artifact | Result | What it does not prove |
|---|---|---|
| `pony-pack-arithmetic.json` and audit script | 19,464 entries inspected, including 9,060 mapped entries; zero reported arithmetic issues | Recipe correctness against the original floating-point graph, or compiler correctness |
| `conditioning-ranges.json` | No context values outside the input range; one of 1,280 pooled values outside in one prompt | Internal activations fit their ranges |
| `vae-comparison.json` | All 248 Pony VAE tensors / 83,653,863 values match the shared FP16-fix VAE after FP16 rounding | Every other checkpoint has the same VAE |
| `dense-quantization-comparison.json` | For Pony's 15 selected high-error tensors, median relative RMSE is 6.96% for Pony versus 7.01% for xxmix | Full-model accuracy or harmlessness of quantization |
| `aura-tensor.log` | Pony ran at 20 steps, CFG 7, Euler ancestral | A paired quality comparison against the original model |
| `illustrious-negative-prompt.log` | Seven runs use 20 steps, CFG 5, Euler ancestral, with the negative branch executed | Correct text conditioning or internal UNet computation |

The CLIP export code already selects the penultimate hidden state (`[-2]`) for
both encoders; CLIP-G pooling uses the final layer and projection. The output
name `last_hidden_state` is not evidence that the wrong layer was exported.
Shared encoder **weights** differ from Pony's own encoders. The embedding
comparison quantifies that difference, and the successful CLIP-only phone test
below establishes a working fix for this checkpoint.

Sampling settings must follow the specific checkpoint. Illustrious's official
early-release card suggests Euler ancestral, 20–28 steps and CFG 5–7.5. The
successful MOP run's 8 steps / CFG 1 / LCM cannot establish a universal preset.
However, the saved standard-sampler runs mean changing presets alone is not a
complete explanation here. [Illustrious model card](https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0)

Some derivatives use a different prediction type. NPUForge currently exports
architecture/context markers, not that model-specific setting; Aura accepts
the setting from the generation request. Verify it for an identified failing
checkpoint rather than assuming it from the name “anime” or “SDXL”.

### Original Pony v6 failure

The user's 23:32 screenshot shows banded noise at 1024 × 1024, 30 steps,
CFG 7, seed 434446878 and 45.7 seconds. The user confirms DPM and V-prediction
off; the exact DPM variant and schedule were not supplied. The phone checkpoint
header matches the previously audited local header: title `v6-full-te`, SDXL
base architecture, 1024 resolution and `modelspec.prediction_type=epsilon`.
This header comparison is not a full-file checksum.

The corresponding saved conversion report identifies v6, no LoRAs, successful
weight and compiler exits, and completion at 23:27. It retains O=3 and both
source-destructive settings disabled. The fresh ZIP's shared-component sizes
and ten recorded CRCs match the donor archive. This checked ZIP directory
metadata, not every payload byte or the release app's installed copy.

An independent header/recipe comparison covers all 1,680 checkpoint UNet keys.
Saved graph descriptors use float32 CHW UNet and VAE inputs/outputs, matching
the runtime's pixel assembly. Neither missing whole source tensors nor a
CHW/HWC mismatch was demonstrated. There is no retained tensor trace for this
exact render; the older Pony and XXMix logs must remain separately attributed.

The successful conversion alone did not establish a working Pony model.
The confirmed settings do not support blaming this failure on a low-step LCM
preset or the wrong prediction type.

### Shared text encoders: measured weight differences

`compare_clip_embedding_samples.py` and `clip-embedding-comparison.json` in
the local inspection artifacts compare the shared embeddings with checkpoint
weights. Token samples cover eight complete rows: 0, 1, 100, 320, 1125, 2368,
49406 and 49407. Position comparisons cover all 77 rows. Shared position
weights are rounded to FP16 before comparison; metrics use float64.

| Shared embedding compared with checkpoint | Pony | XXMix |
|---|---:|---:|
| CLIP-L token sample | 28.87% | 6.74% |
| CLIP-L positions | 58.61% | 17.64% |
| CLIP-G token sample | 33.18% | 0.45% |
| CLIP-G positions | 62.37% | 1.10% |

Each percentage is relative RMSE: the L2 norm of the weight difference divided
by the checkpoint sample's L2 norm. These are **weight differences, not image
quality losses**. They establish a concrete conditioning-weight substitution
and cannot be explained solely by FP16 rounding. They do not measure complete
encoder outputs or establish the cause of the striped render by themselves.
The subsequent encoder export/check and successful phone comparison provide
the relevant output evidence.

### Converting checkpoint-owned components on the phone

Standard SDXL-base derivatives share CLIP-L and CLIP-G network architectures,
while finetuning can change their weights. Architecture compatibility permits
reusing graph definitions; it does not justify substituting a donor's weights.
The app now uses reusable MNN encoder graphs and verified tensor maps to write
checkpoint-owned embeddings, transformer weights, normalization parameters and
CLIP-G projection on the phone. Native conversion preserves FP16 CLIP-L and the
verified INT8 CLIP-G representation. Text-encoder LoRA merging remains unsupported.

The VAE has separate encoder/decoder recipes and QNN template libraries. All 248
source parameters are mapped. Floating graph arithmetic on HTP remains FP16;
float32 I/O does not mean float32 internal computation. Pony's source VAE already
matches the shared FP16-fix weights after rounding. See
[component conversion](SDXL-COMPONENTS.md) for the new implementation and checks.
The user reports excellent quality after full app conversion of `waiIllustriousSDXL_v170`. The supplied screenshot shows 1024 × 1024, 30 steps, CFG 7, seed 418928922 and 45.8 seconds on NPU. This validates conversion and text-to-image for this checkpoint; it does not isolate CLIP versus VAE effects or establish compatibility with every derivative.
Other derivatives and image-to-image still need testing. Component fidelity and
UNet activation calibration remain separate verification tasks.

### CLIP-only diagnostic: user-confirmed success on September 16

`v6-Pony-CLIP-test.zip` contains checkpoint-owned CLIP-L and CLIP-G in the same
formats as the donor: FP16 for L and INT8 with external weights for G. Later
inspection of the actual MNN 3.6.1 artifact found symmetric per-channel INT8
(`readType=0`), despite the original command specifying asymmetric quantization.
The source was the local `v6.safetensors` with the previously matched header.
Both encoders were loaded with strict parameter matching, exported sequentially
and checked against the original checkpoint encoders. No UNet or VAE was
reconverted. Six other entries preserve the exact 23:27 phone export: UNet,
both VAEs, tokenizer and the two model markers. All ZIP payloads passed closed-file
CRC and SHA-256 readback verification.

The emitted token/position files reproduce reference inputs exactly. Host MNN
outputs were compared on one short prompt and an empty prompt; G additionally
covered both EOS and zero padding:

| Output | Relative RMSE versus checkpoint float32 reference |
|---|---:|
| CLIP-L hidden states | 0.080–0.090% |
| CLIP-G hidden states | 1.92–2.92% |
| CLIP-G EOS pooled vector | 0.91–1.29% |

All outputs were finite and the original-encoder wrapper comparisons passed.
These numerical results are host conditioning checks. The user subsequently
reported that the diagnostic model worked on the phone: "It worked!"
Replacing the shared CLIPs fixed the reported Pony failure while preserving
the exact compiled UNet and VAE. This does not establish universal SDXL support
or identify which individual encoder/tensor caused the degradation.
The reproducible `pony_clip_test.py`, `package_clip_test.py`, per-encoder results
and `package-verification.json` remain in local artifacts outside the public repo.

The requested comparison used the distinct test model, baseline prompt, seed
434446878, 30 steps, CFG 7, the same DPM variant/schedule and V-prediction off.
The exact rerun settings and output image were not independently captured;
the success is user-reported. A distinct model path matters because same-path
model reuse can retain conditioning in RAM.

The integration review also identified a separate conditioning difference:
the QNN path EOS-pads CLIP-G, while the standard tokenizer and MNN path use zero
padding after EOS. Current QNN calibration mirrors its runtime padding. The
causal mask keeps post-EOS padding from changing the EOS pooled vector, but the
padded hidden rows remain visible to UNet attention within the active chunk.
The successful test left that behavior unchanged, so no padding change was
needed for the observed Pony fix. Its broader effect remains unmeasured.

### Next implementation and remaining quality work

Extend phone validation to SD1.5, image-to-image and additional checkpoints,
then support applicable text-encoder LoRAs. The successful Pony diagnostic used
desktop export; the subsequent Illustrious app conversion provides separate
evidence for the new SDXL stages. Neither result required a new UNet template.

For failures that remain with their own encoders, use fixed tensors and settings:

1. Compare its original floating-point UNet with the exported floating-point
   graph using identical conditioning. This checks the graph transformation.
2. Compare that exported graph with the current quantized template. Inspect the
   first divergent internal activations and their clipping rates.
3. Compare checkpoint-owned and shared CLIP conditioning through the same
   floating-point UNet. This isolates the encoder mismatch.
4. If calibration is responsible, recalibrate the identified checkpoint and
   verify both its quality and the known working model before selecting broader
   calibration data or additional templates.

A weight-span ratio is only a screening heuristic. The historical SD1.5
MistoonAnime result neither proves the SDXL cause nor establishes that one
additional “anime template” covers Pony and Illustrious.

## Verification of the source changes

- Ten focused Python/native tests passed, including SD1.5/SDXL attention pack
  byte parity with synthetic LoRAs, cache eviction/recomputation, stacked signed
  strengths and rejection of partial/DoRA/invalid-shape adapters. The original
  native code accepted a tiny DoRA fixture while ignoring its magnitude tensor;
  the updated code rejects it.
- Standalone Kotlin compilation of `NativeTombstone` and `ConversionReport`
  against Android 37 succeeded. Tombstone fixtures passed for field ordering,
  signed signals, unknown fields, bounds, malformed input and PID matching.
- Independent source review and `git diff --check` found no actionable issues.
- No APK build/deployment, full checkpoint conversion or UNet template rebuild
  was performed in the diagnostic. The focused local CLIP export and subsequent
  user-reported phone success are recorded above.
  These checks do not establish a full SDXL LoRA fix or Vivo conversion success
  on a device; they validate the bounded source changes described above.

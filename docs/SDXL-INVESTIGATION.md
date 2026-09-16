# SDXL reliability and component fidelity

This record separates saved failure evidence, implemented fixes and reported
phone outcomes. Investigations covered compiler allocation failures, a LoRA
compatibility regression and poor output from substituted text encoders.
They did not identify a single cause applicable to every phone or checkpoint.

The latest test APK received a successful report after small-object pooling,
workspace relocation and LoRA compatibility fixes. No post-fix device-specific
logs were supplied with that confirmation; broader device support remains to
be measured.

## Summary

| Finding | Evidence | Current implementation or result |
|---|---|---|
| Compiler mapping exhaustion | Vivo report reaches 65,530 mappings, 64,575 belonging to Scudo; a later Nubia trace identifies a 40-byte allocation | Small objects now share storage-backed slabs instead of remaining in Scudo |
| Backing file creation fails despite free storage | Test APK 2 reports `openat` ENOENT during VAE decoder compilation; deleted-directory host reproduction matches it | Active work moved from cache to `noBackupFilesDir/conversion-work` |
| LoRA merge retains too much memory | Calculated broad-SDXL merged-weight cache payload is 8.14 GiB without eviction | Retained payload capped at 128 MiB; evicted tensors are recomputed |
| Previously accepted DMD2 adapters rejected | 722 modules matched, but 198 extra tensors triggered a new hard rejection | ResNet/sampling mappings added; remaining unmatched tensors warn while matched layers merge |
| Pony output is striped noise | Successful reported test replacing only seven CLIP files, with the UNet/VAE/tokenizer/markers preserved | Checkpoint-owned CLIPs resolve this reported case |
| Illustrious component fidelity | Successful full-component `waiIllustriousSDXL_v170` conversion and recognizable output | App converts checkpoint-owned CLIPs and both VAE graphs without changing the UNet template |

## 1. Compiler allocation and workspace failures

### Mapping exhaustion in earlier Vivo builds

The saved Vivo V2307A / SM8650 report starting at 11:23:46 UTC already uses
the earlier 8 KiB cutoff, backing slabs and C++ hooks. Before abort it has
**65,530 mappings**, including **64,575 Scudo secondary mappings** and only
**503 compiler backing mappings**. In its final seconds mapping count rises
sharply while virtual-memory size falls.

Linux limits mapped regions separately from their total size; its documented
default limit is 65,530. More physical RAM does not remove this limit. The
actual device setting was unreadable, so the matching count strongly supports
mapping exhaustion without identifying the failed syscall in that report.
[Linux kernel documentation](https://www.kernel.org/doc/html/latest/admin-guide/sysctl/vm.html#max-map-count)

The native-abort signal does not establish an OEM watchdog or low-memory kill.
Earlier slabs reduced compiler backing mappings but left small allocations with
libc/Scudo. Subsequent Nubia native-crash evidence identified an ordinary
40-byte C++ allocation on that remaining path. Current pooling covers small
objects from a 16-byte size class, using compact metadata and shared 8 MiB slabs.
Blocks through 1 MiB also share slabs; larger allocations retain separate
backing. [SDXL.md](SDXL.md) describes the allocator contract.

A host test with 300,000 simultaneous 40-byte allocations used three backing
slabs, with total process mappings increasing from 48 to 54. Regression tests
also exercise reuse, alignment, C/C++ allocation APIs, cross-thread freeing,
foreign libc pointers and crash-time mapping diagnostics. These results check
the mechanism, not every Android allocator or QNN workload.

### Test APK 2: active directory removed during compilation

The later report identifies `0.2.1-small-pool-lora-test2`, a Nubia NX789J /
SM8750 running Android 16, and a Pony/DMD2 F32 conversion. CLIP conversion and
VAE encoder compilation completed. VAE decoder compilation failed **before
UNet weight preparation or LoRA merging**.

The allocator reported:

```text
operation=openat backing file
bytes=1572864 alignment=16 page_size=4096
errno=2 (No such file or directory)
```

At that point the report recorded about 1,136 process mappings, no Scudo
secondary mappings, approximately 3.74 GiB RSS and 15.95 GiB available RAM.
Available storage was approximately 145 GiB. The small-object pool had handled
about 15.5 million allocations. This failure did not reproduce the earlier
mapping-limit abort.

The backing directory was inside `cache/work/vae_decoder`. Removing a directory
while the allocator retains its file descriptor reproduces this exact `openat`
failure: existing mapped data remains readable and filesystem capacity remains
available, but creating a new file fails with ENOENT. The saved report does not
identify the process or policy that removed the directory.

Active checkpoint, adapter, component and compiler files now live under
`noBackupFilesDir/conversion-work`. The foreground service cleans that workspace
explicitly after conversion. A host reproduction confirmed that deleting cache
does not interrupt allocations in the separate no-backup directory. Android
classifies cache files as reclaimable; persistent app-specific files use a
different lifecycle. [Android storage documentation](https://developer.android.com/training/data-storage/app-specific)

### Diagnostic coverage and device scope

Reports now include signal-time mapping counts, memory/storage snapshots and
Android native tombstone summaries when available. Tombstone decoding extracts
the abort message, signal and up to 24 crashing-thread frames from
`ApplicationExitInfo.traceInputStream`. Retention depends on Android; adding the
reader cannot recover an already missing trace.
[Android native crash diagnostics](https://developer.android.com/ndk/guides/debug)

The latest test build was reported working after the workspace and LoRA fixes.
No accompanying post-fix Vivo/Nubia logs identify the successful device matrix.
Target compatibility also remains separate: SDXL uses fixed v75/soc57 settings.

## 2. LoRA memory use and compatibility

### Bounded merged-weight cache

`LoraSet::apply` previously retained a full FP32 merged tensor for every matched
source key across both weight passes. Counting shipped recipe shapes, an
adapter covering all 700 SDXL transformer weights could retain
**8,735,948,800 bytes (8.14 GiB)**. The corresponding SD1.5 subset is about
924 MiB. These are calculated payload sizes, not measured phone peaks.
Checkpoint mappings, adapter mappings and temporary buffers add to them.

The current cache retains at most **128 MiB** of merged payloads. Adjacent
attention-head reads reuse cached weights; evicted tensors are recomputed with
the same merge order and FP16 rounding. Recomputing can increase conversion
time. The compiler allocator is separate and cannot bound this earlier weight
stage's RAM usage.

### DMD2 mapping and warning-only unmatched tensors

The pre-fix failure reports matched 722 UNet modules but rejected 198 unmatched
adapter tensors. The first reported tensor belonged to
`lora_unet_down_blocks_0_downsamplers_0_conv`. The newer rejection exposed an
incomplete mapping that earlier conversions had silently skipped.

Native and Python mappings now include ResNet normalization/convolution/shortcut
layers, down/up samplers, input/output convolution and time/additional embedding
layers, alongside attention. On the inspected SDXL source layout, 66 additional
modules with down/up/alpha entries account for the 198 previously unmatched
tensors. Mapping aliases were compared across 391 SD1.5 and 1,050 SDXL source
weights, with no collisions in those sets.

The whole-adapter rejection for extra tensors has also been removed. Recognized
UNet pairs merge, while remaining unsupported or unmatched tensors produce a
warning. Missing pairs can be skipped when other recognized pairs exist. A file
with no matched UNet pair still fails; invalid multiplication shapes remain
errors. This preserves useful partial merging without claiming the unsupported
parts were applied.

Remaining limits:

- Text-encoder adapters are reported and skipped, including with checkpoint-owned
  CLIPs. A UNet-only merge cannot reproduce their encoder changes.
- Supported input dtypes are F16/F32. BF16 support is deferred.
- Standard kohya module naming is supported; arbitrary PEFT/diffusers file
  layouts and format-specific DoRA/LyCORIS arithmetic are not implemented.
- Correct merge arithmetic does not establish adapter/checkpoint training-family
  compatibility or that the result fits the template's activation ranges.

### Historical and latest phone reports

The historical `Lora-f16-success.txt` report records a Nubia NX789J / SM8750,
Android 16, adapter label `dmd2Sdxl4stepLora.LhSd`, strength 0.8, 722 matched
modules, successful native exits and completion in **687.678 seconds**. F16 is
reported by the filename rather than verified tensor metadata. The base
checkpoint is not established by that report.

The same tester supplied recognizable renders labeled `+lora_f32` and reported
DMD2 F32 success. Both tests preceded checkpoint-owned component conversion and
the strict unmatched-tensor rejection. They establish working historical setups,
not full adapter coverage or a controlled measurement of the adapter's effect.

The latest test APK received a separate successful report after the mapping,
warning behavior and workspace fixes. It does not provide new adapter tensor
metadata or a same-seed LoRA-effect comparison.

## 3. Image quality and checkpoint-owned conditioning

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

Some derivatives use a different prediction type. npuforge exports
architecture/context markers; the generation request supplies the prediction
setting in Local Dream or Fancy-Ai. Verify it for the specific checkpoint rather
than assuming it from the name “anime” or “SDXL”.

### Original Pony v6 failure

The supplied 23:32 screenshot shows banded noise at 1024 × 1024, 30 steps,
CFG 7, seed 434446878 and 45.7 seconds. The reported settings were DPM with V-prediction
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
Full app conversion of `waiIllustriousSDXL_v170` subsequently produced a
reported successful result. Its screenshot records 1024 × 1024, 30 steps,
CFG 7, seed 418928922 and 45.8 seconds on NPU. This supports conversion and
text-to-image for this checkpoint; it does not isolate CLIP versus VAE effects
or establish compatibility with every derivative.
Other derivatives and image-to-image still need testing. Component fidelity and
UNet activation calibration remain separate verification tasks.

### CLIP-only diagnostic: reported phone success on September 16

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
These numerical results are host conditioning checks. A subsequent phone test
reported successful generation with the diagnostic model. Replacing the shared CLIPs fixed the reported Pony failure while preserving
the exact compiled UNet and VAE. This does not establish universal SDXL support
or identify which individual encoder/tensor caused the degradation.
The reproducible `pony_clip_test.py`, `package_clip_test.py`, per-encoder results
and `package-verification.json` remain in local artifacts outside the public repo.

The comparison protocol specified a distinct test model with the baseline
prompt, seed 434446878, 30 steps, CFG 7, the same DPM variant/schedule and
V-prediction off. The exact rerun settings and output image were not independently
captured, so success remains a reported phone result. A distinct model path
avoids retaining conditioning through same-path model reuse.

The integration review also identified a separate conditioning difference:
the QNN path EOS-pads CLIP-G, while the standard tokenizer and MNN path use zero
padding after EOS. Current QNN calibration mirrors its runtime padding. The
causal mask keeps post-EOS padding from changing the EOS pooled vector, but the
padded hidden rows remain visible to UNet attention within the active chunk.
The successful test left that behavior unchanged, so no padding change was
needed for the observed Pony fix. Its broader effect remains unmeasured.

### Remaining quality work

Extend phone evidence to the checkpoint-owned SD1.5 path, image-to-image and
additional checkpoints. Text-encoder LoRA is a separate future capability. The successful Pony diagnostic used
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

## Verification scope

Recorded checks for the implemented changes include:

- Native/Python LoRA pack parity with F16/F32 adapters, ResNet/sampler mappings,
  stacked signed strengths and bounded-cache eviction/recomputation.
- Warning-only behavior for unmatched or mixed-format tensors, continued merging
  of recognized pairs, and failure for zero matches or unusable shapes.
- Host compiler-allocation stress tests, alignment/reallocation, cross-thread
  freeing and crash-signal/mapping-count behavior.
- The deleted-directory backing-file reproduction and the separate persistent
  workspace case.
- Native CLIP outputs matching the working Pony diagnostic; VAE recipe/pack
  comparisons and compiled planar float32 component interfaces.
- Android build and lint checks, plus tombstone fixtures covering malformed
  input, bounds, unknown fields, field ordering and PID matching.

Host checks establish behavior of the exercised paths. Phone measurements and
reported results are attributed separately above. Neither class of evidence
establishes universal device, adapter or checkpoint support.

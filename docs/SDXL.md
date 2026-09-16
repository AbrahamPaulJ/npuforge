# SDXL on-device INT8 conversion

SDXL UNet weight conversion, QNN compilation, ZIP import and image generation
completed on a Samsung Galaxy S25 Ultra (SM-S938B, SM8750, HTP v79) on
2026-09-15. The current configuration is O=3 with source-destructive memory
reuse disabled. SD1.5 retains its separate template and conversion path.

The current app converts checkpoint-owned CLIP-L, CLIP-G, VAE encoder and VAE
decoder as well as the UNet. The latest test APK received a successful field
report after allocator, workspace and LoRA compatibility fixes. No post-fix
per-device logs accompanied that confirmation.

[SDXL investigation](SDXL-INVESTIGATION.md) records the Pony CLIP-only and
full-component Illustrious results, historical Vivo/Nubia failures and the
changes addressing them. These results do not establish universal device or
checkpoint compatibility.

## Historical compiler-configuration results

These runs predate checkpoint-owned component conversion. Checkpoint: `mopMixtureOfPerverts_instaV1`. Shared VAE source:
`madebyollin/sdxl-vae-fp16-fix`, already converted with QAIRT 2.50.

| Run | Conversion measurement | Generation |
|---|---|---|
| O=1, reuse disabled | Compiler exited 0; approximately 254 seconds compiling only | 20 seconds |
| O=3, reuse disabled | User reported 437 seconds total conversion | 15 seconds |

Both screenshots show QNN, 1024 × 1024, 8 steps, CFG 1, LCM and Karras, with
recognizable generated images. Seeds differ: O=1 used 3445391806; O=3 used
2262025881. The observed generation time fell by 25%, but these are individual
runs, not a controlled repeated benchmark. Total conversion and compiler-only
times must not be compared as the same measurement.

During O=3 conversion the user observed 41% phone RAM usage. This is a snapshot,
not peak RAM. The O=1 allocator log peaked at 11,220 MiB of storage-backed
allocations; that is allocated file backing, not resident RAM. No continuous
peak-RAM measurement was captured for either successful run.

## Model contract and checkpoint components

- UNet uses W8A16: INT8 weights, 16-bit activations, no INT4 overrides.
- Batch one; four-channel 128 × 128 latents for 1024 × 1024 images.
- Masked 231-token conditioning, identified by `qnn_context.txt` containing
  `231_masked_v1`; output also carries the `SDXL` marker.
- QAIRT 2.50.0.260828, v75 / soc_model 57, 8 MB VTCM, burst profile.
  The target is fixed; it is not selected automatically from the phone's chip.
- New conversions write checkpoint-owned MNN CLIP-L/CLIP-G and embeddings,
  and compile checkpoint-owned QNN VAE encoder/decoder graphs. Only the standard
  tokenizer is shared. No SDXL donor download is used.
- CLIP-L stores FP16 weights; CLIP-G uses the verified MNN INT8 representation.
  VAE graph arithmetic on HTP is FP16, with float32 planar external tensors.
  This does not establish compatibility with VAEs requiring float32 arithmetic.

The historical successful phone exports used shared components and were about
3.5 GB. The expanded component pipeline now has a successful user-reported
Illustrious text-to-image result; image-to-image still needs validation. Legacy shared-component backups are
retained but are not inputs to new SDXL conversions. See
[component conversion and authoring](SDXL-COMPONENTS.md).

### Pony CLIP-only test — reported success, 16 September 2026

The diagnostic `v6-Pony-CLIP-test.zip` received a successful phone report. This diagnostic package
replaced seven CLIP files with Pony's checkpoint-owned weights: `clip.mnn`,
`clip_2.mnn`, `clip_2.mnn.weight`, both token-embedding files and both
position-embedding files. SHA-256 checks verified that its UNet, VAE encoder and
decoder, tokenizer and model markers match the exact baseline phone export.

This supports shared CLIP substitution as the cause of this Pony failure.
The local exporter passed strict checkpoint loading, reference-encoder
comparisons and host MNN checks using the emitted embedding files. It preserves
the existing FP16 CLIP-L and INT8 CLIP-G formats and runtime padding behavior.
Native on-phone component conversion is implemented separately from this
diagnostic and was subsequently tested with Illustrious, below. Device
compilation and adapter coverage remain separate evidence; see
[the investigation](SDXL-INVESTIGATION.md) for current status.

### Full-component Illustrious test — reported success, 16 September 2026

A successful full app conversion of `waiIllustriousSDXL_v170` was reported,
with a recognizable generated image. The supplied screenshot records
1024 × 1024, 30 steps, CFG 7, seed 418928922 and 45.8 seconds on NPU. This
supports conversion and text-to-image for this checkpoint; it does not isolate
CLIP versus VAE effects or establish compatibility with every derivative.
No conversion timing or peak-memory measurement was supplied. Other derivatives
and image-to-image remain to be validated.

## Why earlier compilation ran out of memory

Repeated native aborts occurred along this preparation path:

```text
GraphPrepare::sequencing_stage
  -> link_source_destructive_operands
  -> FancyAllocator::allow_tensor_overlap
  -> FancyAllocator::link_blocks
  -> hash-table rehash / vector growth
  -> operator new -> libc / Scudo
```

Disassembly matched the table layout and growth to the supplied SDK's
`HTP/core/minihash.h`: eight-byte block-ID entries, with old and new storage
coexisting during rehash. This identified memory-reuse bookkeeping at the
failing allocation, not a convolution activation buffer. The exact failed
request size was not captured, and this does not prove every allocation in
that stage is excessive or that the SDK has a general algorithmic defect.

Reducing O=3 to O=2 and O=1 alone did not resolve failures. Foreground service
priority did not prevent allocation failure either. A 64 KiB file-backed
allocator alone also failed in the same linking path.

The successful configuration in `app/src/main/assets/template_sdxl/htp_config.json`
keeps O=3 and sets:

```json
"finalize_config": {
  "source_destructive_ops": 0,
  "source_destructive_custom_ops": 0
}
```

The SDK documents the generic `finalize_config` key/value interface. These two
specific option names were found in the supplied binary, not documented as
public settings. Source-destructive operations permit input/output memory reuse.
The configuration was present on the successful O=1 phone run; with the same
settings retained, the user completed O=3 conversion and generation. This is
end-to-end evidence for this configuration, not a trace proving every internal
option branch or compatibility with another SDK revision. Disabling reuse may
increase inference memory; its isolated cost has not been measured.

## Storage-backed compiler allocations

`native/compiler_heap.c` is preloaded only into SDXL compiler subprocesses,
including their VAE and UNet compilation stages. It hooks C and C++ allocation
APIs and backs ordinary allocations with unlinked, preallocated files using
`MAP_SHARED`. Direct SDK mappings and GPU/NPU-owned buffers are outside this
interception path.

The kernel can reclaim and reload file-backed pages. This requires free storage
and incurs I/O; it does not create physical RAM or eliminate allocation failures.
The current allocator handles three size ranges:

- Requests below 8 KiB use 32 small-object size classes, starting at 16 bytes,
  in shared 8 MiB slabs. Compact slot metadata, recycled slots and an address
  index avoid per-object mappings. At most one empty slab per small size class
  is retained; additional empty slabs are released.
- Blocks from 8 KiB through 1 MiB share 8 MiB backing slabs and metadata.
  Empty slabs release their mappings and backing.
- Requests outside the supported pooling size/alignment classes use individual
  mappings.

Indexed ownership lookup uses 4,096 buckets. Process exit releases remaining
unlinked backing files. The build must preserve `-fno-builtin` and the `LIBC`
symbol versions in `compiler_heap.map`.

The earlier allocator left requests below 8 KiB with libc/Scudo. Vivo reports
reached exactly 65,530 mappings, mostly Scudo secondary allocations, despite
having ample advertised RAM. A later Nubia native trace identified a failing
40-byte C++ allocation. Small-object pooling addresses that remaining path.
In a host stress test, 300,000 live 40-byte objects used three backing slabs;
total process mappings rose from 48 to 54. Host behavior does not by itself
establish successful compilation on every phone.

A subsequent test APK 2 failure occurred during VAE decoder compilation, before
UNet LoRA merging. The compiler reported `openat` ENOENT for a backing file in
`cache/work/vae_decoder`, with only 1,136 mappings and ample available storage.
A host reproduction matched this failure after deleting an allocator's open
backing directory. Active files now live in
`noBackupFilesDir/conversion-work`, outside reclaimable cache, and are explicitly
cleaned by the service. The report cannot identify who removed the original
cache directory. [Android app-specific storage](https://developer.android.com/training/data-storage/app-specific)

The latest test build received a successful report after this workspace change
and the LoRA compatibility fix. Per-device post-fix logs and continuous memory
measurements remain outstanding. See
[the investigation](SDXL-INVESTIGATION.md) for the complete evidence sequence.

Historical profiling also found that a linear ownership list consumed 93.71%
of sampled CPU time in `free`; indexing reduced it to 1.07% in a later short
sample. These are profile samples, not end-to-end speed benchmarks.

## Template preparation and reproducibility

The template was authored once on a workstation; converting a user's checkpoint
uses the existing recipe and performs weight conversion and QNN compilation on
the phone. No workstation conversion is required per checkpoint.

The template has 19,464 pack entries: 9,060 mapped entries, 7,800 injected zero
biases and 2,604 constants. Recipe discovery matched all 1,680 source tensors.
The recorded phone weight pack was 2,583,972,032 bytes. Calibration used an FP32
export and 30 rows from three trajectories, covering 77/154/231 active tokens
and timesteps 0–999. Latents were clipped to [-7.2, 7.2] by that calibration
pipeline. Coverage across other checkpoint families is not established.

SDXL attention uses 64-channel heads (ten or twenty per tested layer).
Native/Python recipe application derives slice width from recipe dimensions
and source stride; SD1.5 retains its eight-head discovery default. LoRA merges
in checkpoint space before slicing. For SDXL recipe discovery, use
`tools/tpl_recipe.py discover` with `--head-dim 64`.

Generated `libqnn_model.so`, binary recipes/packs, checkpoints, donor ZIPs and
QAIRT binaries are not uploaded with these source changes. Supply the SDXL
artifacts described in [BUILD.md](BUILD.md) to produce a working APK. Local
registration code was built with `-O0 -g0`; those C++ flags are separate from
QNN's O=3 graph preparation setting.

## Verification scope

The MOP phone runs establish the original compiler configuration on the tested
Samsung device. The Pony diagnostic separately checked local CLIP export and
received a successful phone report. Illustrious then supplied a full-component
SDXL result. Historical Nubia tests established DMD2 F16/F32 conversion and
recognizable generation before the component update; the latest test build has
separate reported success after the regression fixes.

Native/Python tests cover LoRA mapping and merge parity, including ResNet and
sampling layers, mixed files with unmatched tensors, bounded-cache recomputation
and F16/F32 weights. Extra unsupported tensors now warn while recognized UNet
layers merge. Text-encoder LoRA and BF16 remain unsupported.

Crash diagnostics include signal-time mapping counts, compact process status,
available RAM/storage and native tombstone summaries when Android retains them.
These diagnostics help separate resource failures from checkpoint or adapter
problems. Repeated quality comparisons, image-to-image, inference peak memory
and a broader device matrix remain open work.

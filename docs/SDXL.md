# SDXL on-device INT8 conversion

SDXL UNet weight conversion, QNN compilation, ZIP import and image generation
completed on a Samsung Galaxy S25 Ultra (SM-S938B, SM8750, HTP v79) on
2026-09-15. The current configuration is O=3 with source-destructive memory
reuse disabled. SD1.5 retains its separate template and conversion path.

## Recorded phone results

Checkpoint: `mopMixtureOfPerverts_instaV1`. Shared VAE source:
`madebyollin/sdxl-vae-fp16-fix`, already converted with QAIRT 2.50.

| Run | Conversion measurement | Aura generation |
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

## Model contract and shared components

- UNet uses W8A16: INT8 weights, 16-bit activations, no INT4 overrides.
- Batch one; four-channel 128 × 128 latents for 1024 × 1024 images.
- Masked 231-token conditioning, identified by `qnn_context.txt` containing
  `231_masked_v1`; output also carries the `SDXL` marker.
- QAIRT 2.50.0.260828, v75 / soc_model 57, 8 MB VTCM, burst profile.
  The target is fixed; it is not selected automatically from the phone's chip.
- Shared components are MNN CLIP-L/CLIP-G, embeddings and tokenizer, plus QNN
  VAE encoder and decoder. Their weights are borrowed, not converted from the
  selected checkpoint. Image-to-image with this SDXL output is not yet recorded.

At first SDXL conversion, the app downloads the approximately 1 GB
[shared component ZIP](https://huggingface.co/Mr-J-369/SDXL-OnDevice-Conversion/resolve/main/sdxl-shared-mnn-clips-qnn250-vae1024-v75.zip).
It extracts the shared files into its existing SDXL cache. Subsequent conversions
reuse that cache, including components previously imported locally. The service
still accepts an explicit local component URI for debug-driven imports.
The tested exported ZIP was about 3.5 GB. The automatic download was wired after
the successful phone runs; its endpoint was checked, but the new download flow
has not yet been tested on the phone.

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

`native/compiler_heap.c` is preloaded only into the SDXL compiler subprocess.
It backs allocations of at least 64 KiB with unlinked, preallocated files using
`MAP_SHARED`. Small allocations continue through libc. Aligned requests also
use backing when their alignment reaches the cutoff. Direct SDK mappings and
GPU/NPU-owned buffers are not intercepted.

The kernel can reclaim and reload file-backed pages. This requires free storage
and incurs I/O; it does not create physical RAM or eliminate OOM. Ownership
lookup uses 4,096 hash buckets. One idle mapping per size class from 64 KiB
through 1 MiB is retained for reuse, totaling less than 2 MiB. Freed mappings
otherwise release their backing; process exit releases remaining unlinked files.

The initial linear ownership list consumed 93.71% of sampled CPU time in `free`.
The indexed version reduced `free` to 1.07% in a later short sample. These are
profile samples, not conversion-speed benchmarks. Keep `-fno-builtin` in the
allocator build and preserve the `LIBC` symbol versions in `compiler_heap.map`.

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

The phone run is the validation for this change. No additional host conversion,
benchmark, build or phone deployment was performed while recording these
findings. Other phones, CFG > 1, SDXL LoRA, repeated quality comparisons and
inference peak memory remain unverified by the recorded runs.

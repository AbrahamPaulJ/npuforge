# Testing

## Host regressions

Use Linux with Python 3.10–3.12, GCC/G++ and `/proc` available. The CI job uses
Ubuntu 24.04 and Python 3.12. Tests create synthetic tensors and temporary
files; no checkpoint, Qualcomm runtime, generated template, Android SDK or
device is needed.

On 2026-09-16, all **44 tests passed with no skips** in a source-only checkout
using Python 3.12.3 and the dependencies in `requirements-test.txt`. That run
contained no generated model assets or Qualcomm binaries.

From the repository root:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -p 'test_*.py' -v
```

Activate the environment before running tests: one test launches `python3` as
a subprocess. The suite compiles its native test executables automatically.
Keep the environment outside source control.

| Area | Coverage |
| --- | --- |
| Weight conversion | FP16/FP32 rounding, convolution layout, attention reshaping, native/reference pack comparisons |
| CLIP writer | Quantization, QKV slicing, projection transpose, sparse recipe boundaries |
| LoRA | F16/F32 mappings, convolution/ResNet/embedding layers, stacked adapters, cache eviction, warning-and-continue behavior for unmatched tensors |
| VAE recipe authoring | Source attribution and ambiguous mappings after half-precision rounding |
| Compiler allocator | Pooled small objects, alignment, calloc/realloc, C++ allocation, cross-thread use, foreign pointers and crash reporting |

The allocator tests use `LD_PRELOAD` in child processes. Their crash-report case
deliberately aborts a child and checks that the original signal is preserved.
These tests are skipped without Linux procfs or a C/C++ compiler; a passing run
with skips is not equivalent to the Linux CI run.

To run one area, for example:

```sh
python -m unittest discover -s tests -p 'test_lora*.py' -v
python -m unittest discover -s tests -p 'test_compiler_heap.py' -v
```

## Android build and lint

With the toolchain described in [Build](BUILD.md):

```sh
./gradlew :app:assembleDebug :app:lintDebug --console=plain
```

Build and lint establish compilation and static Android checks. They do not
run Qualcomm HTP compilation or verify that all external runtime/model assets
are present. Public CI intentionally runs only the host suite; it has no SDK
binaries, checkpoint downloads, signing keys or phone access.

## Conversion and rendering checks

For native weight changes, compare the native weight pack with the Python
reference for the same recipe, checkpoint and adapters. Weight-pack bytes
should agree exactly. Synthetic tests cover this without external models;
representative full-checkpoint comparisons need separately supplied artifacts.

QNN context binary hashes are not a correctness test: repeated compilations can
produce different bytes. Compare compatible graph interfaces and inference
results using the same generation settings instead.

A phone check should record:

1. App/build version, device/SoC, Android version and available RAM/storage.
2. Checkpoint identity and family; adapter identity, dtype and strength.
3. Completion of CLIP, **both** VAE components, UNet and ZIP export.
4. The consuming generator's sampler, prediction mode, seed, steps, CFG and size.
5. A generated result and the full conversion log, including any warnings.

Text-to-image exercises the decoder but does not establish image-to-image
encoder correctness. Check image-to-image separately when changing the VAE
encoder. One successful model or phone does not establish support for every
derivative or device. Current evidence is recorded in [SDXL](SDXL.md),
[SDXL components](SDXL-COMPONENTS.md) and [SD1.5 components](SD15-COMPONENTS.md).

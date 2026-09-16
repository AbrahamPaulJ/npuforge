# Building and running npuforge

The current Android project uses AGP 9.4.0, Gradle 9.7.1, Kotlin Compose plugin
2.4.20, Android SDK 37 and NDK 29.0.14206865. The Gradle daemon toolchain is
JetBrains JDK 21; application bytecode targets Java 17. Native Gradle tasks
currently use the Linux x86_64 NDK toolchain.

## What you must supply

A source checkout does not contain all runtime/model artifacts (see `NOTICE`):

- **QAIRT SDK 2.50.0.260828** for the current Android runtime and SDXL template.
  Supply its Android context generator as
  `app/src/main/jniLibs/arm64-v8a/libqnncontextgen.so`, plus the QNN HTP,
  Prepare, System, NetRunExtensions, target stub and DSP skel libraries.
  Follow [ANDROID.md](ANDROID.md) for runtime placement and packaging.
- In `app/src/main/assets/template/`, supply `libqnn_model.so` and copy the
  SD1.5 `template/recipe.bin` and `template/tpl_trim.pack` from the repository.
- In `app/src/main/assets/template_sdxl/`, supply the matching SDXL
  `libqnn_model.so`, `recipe.bin` and `tpl_trim.pack`. The source list and
  O=3 configuration are tracked. These generated SDXL artifacts are not
  distributed by this source push; a fresh clone needs them before conversion.
- In `app/src/main/assets/components_sdxl/`, supply the sparse CLIP recipe,
  tokenizer, component manifests and both VAE template bundles described in
  [SDXL-COMPONENTS.md](SDXL-COMPONENTS.md). New SDXL conversions use checkpoint
  weights and do not download shared CLIP/VAE weights.
- In `app/src/main/assets/components_sd15/`, supply the SD1.5 CLIP recipe and
  tokenizer, component manifests and both 512px VAE template bundles described
  in [SD15-COMPONENTS.md](SD15-COMPONENTS.md). SD1.5 also converts the checkpoint's
  own component weights; it no longer downloads the DreamShaper donor archive.

Gradle builds the first-party converter and compiler allocator. SDK and model
binaries remain gitignored. The older SD1.5 measurements below came from the
original QAIRT 2.49 template; they are not measurements of every current build.

## 1. The converter binary

Android builds compile `native/tplconv.cpp` through the `:app:compileTplconv`
task using NDK r29 (`29.0.14206865`) on Linux. The resulting executable is
registered through AGP's generated JNI sources API and packaged as
`lib/arm64-v8a/libtplconv.so` for both debug and release builds. No prebuilt
`libtplconv.so` belongs in `app/src/main/jniLibs`.

```sh
./gradlew :app:assembleDebug
```

The task statically links libc++, retains `-ffp-contract=off`, and sets both
`max-page-size` and `common-page-size` to 16384. This replaces the original
4 KB-aligned prebuilt converter. Keep native library extraction enabled: the
converter is an executable launched from `nativeLibraryDir`.

For host comparisons against the Python reference:

```sh
c++ -O2 -std=c++17 -ffp-contract=off -o tplconv native/tplconv.cpp
```

No dependencies beyond libc++ and POSIX `mmap` for `tplconv`.

`:app:compileComponentconv` builds `native/componentconv.cpp` with the same NDK
and linker settings, packaging it as `libcomponentconv.so`. It reconstructs the
selected family's MNN text encoder(s) directly from the checkpoint and sparse recipe; PyTorch,
ONNX and the MNN converter are authoring tools, not phone dependencies.

`:app:compileCompilerHeap` also builds `native/compiler_heap.c` into
`libcompiler_heap.so` for debug and release. Its symbol map, `-fno-builtin` and
16 KB linker alignment are part of the tested integration. It is preloaded
only into the SDXL QNN compiler. See [SDXL.md](SDXL.md) for its operation.


`-ffp-contract=off` is essential to the existing LoRA merge arithmetic. Without
it, fused multiply-add can round differently across host and device builds.
The source and weight recipe are unchanged by the alignment fix.

## 2. The pack-loading library (generated, gitignored)

From the template's patched `model_tpl.cpp` (produced by `tools/tpl_patch.py`
against the QAIRT converter's `model.cpp`):

```sh
export ANDROID_NDK_ROOT="$ANDROID_SDK_ROOT/ndk/29.0.14206865"
python $QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-model-lib-generator \
    -c model_tpl.cpp -t aarch64-android -o lib_arm
```

⚠ **No `-b`.** Omitting the weights is the whole point: the library is 9.7 MB
instead of ~873 MB and takes its weights from `QNN_TPL_PACK` at generate time.
⚠ With no `-b` the generator names the output `libqnn_model.so`, **not**
`libmodel.so`. Takes ~72 s.

## 3. Run it on the phone

```sh
adb push tplconv_arm template/recipe.bin template/tpl_trim.pack \
         lib_arm/aarch64-android/libqnn_model.so  /data/local/tmp/npuforge/
adb push <your-checkpoint>.safetensors /data/local/tmp/npuforge/ckpt.safetensors
# plus the QAIRT device runtime listed above, into the same directory

adb shell
cd /data/local/tmp/npuforge
./tplconv_arm recipe.bin tpl_trim.pack ckpt.safetensors out.pack     # ~24 s

export LD_LIBRARY_PATH=/data/local/tmp/npuforge
export ADSP_LIBRARY_PATH="/data/local/tmp/npuforge;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp"
export QNN_TPL_PACK=/data/local/tmp/npuforge/out.pack
./qnn-context-binary-generator --model ./libqnn_model.so --backend ./libQnnHtp.so \
    --output_dir ./out --binary_file unet --config_file ./htp_backend.json      # ~93 s
```

`out/unet.bin` is the model.

⚠ **`config_file_path` inside the backend-extension JSON must be ABSOLUTE on the
device.** A relative path is read against the process CWD and fails confusingly.

## 4. Verifying a change

Never judge the converter by looking at renders. Build the same checkpoint with
the Python reference (`tools/tpl_apply.py apply`) and `cmp` the packs — they must
be **byte-identical**. Two reference checkpoints are known-good:

| checkpoint | pack md5 |
|---|---|
| DreamShaper 8 (`Lykon/DreamShaper`, `DreamShaper_8_pruned.safetensors`) | `042b99cf38b37cecc15483f37118bb46` |
| AbsoluteReality (`Lykon/AbsoluteReality`, `AbsoluteRealityV1.6525_pruned.safetensors`) | `52ca4f492e9f43634d08698ea9a5360b` |

⚠⚠ **Do NOT gate on the context binary's md5.** The compile is not
byte-reproducible: two runs on the same phone, from the same pack, produced
882,780,736 bytes both times with **different md5s** (`2fbde3d5…` and
`0acd5323…`) — and rendered **byte-identical PNGs**. A PC compile differs again
(881,826,392 bytes). Neither size nor hash is a gate at this stage; the render
is. The one place hashes ARE exact is the weight pack, which is why that is what
`tplconv` is verified against.

⚠ And compare against the right arm. A model built from the *recipe* pack must be
compared against the PC's build of the *recipe* pack — not against the stock
template build. Comparing against the template reads as a failure when nothing is
wrong.

## Authoring a new template

Out of scope for this document; it needs the full PC conversion pipeline (ONNX
export, calibration, a 2 h 18 m quantize). The steps are `tools/tpl_patch.py` →
`tools/tpl_pack.py identity` → `tools/tpl_recipe.py discover` →
`tools/tpl_apply.py finalize` → `tools/tpl_recipe_bin.py` →
`tools/tpl_pack_trim.py`.

⚠ **Gate the encodings before spending the compile.** A template whose timestep
path quantized to `[0, 0]` renders pure noise, and the converter exits 0, the IO
contract matches a known-good binary, the IO ranges look sane, and it runs on the
NPU at the correct speed. Latency proves a graph compiled, never that it computes
anything.

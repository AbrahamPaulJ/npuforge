# Building and running npuforge

Everything here was run on WSL Ubuntu with QAIRT 2.49 and the **Linux** NDK r27c.

## What you must supply

Neither is in this repo (see `NOTICE`):

- **QAIRT SDK 2.49** — for `qnn-model-lib-generator` (host) and, on the device,
  `qnn-context-binary-generator` plus `libQnnHtp.so`, `libQnnHtpPrepare.so`,
  `libQnnHtpV<arch>.so`, `libQnnHtpV<arch>Skel.so`, `libQnnHtpV<arch>Stub.so`,
  `libQnnSystem.so`, `libQnnHtpNetRunExtensions.so`.
- **Android NDK r27c, Linux build.** ⚠ The Windows NDKs cannot serve this:
  `Android.mk` hardcodes `toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-objcopy`.

## 1. The converter binary

```sh
# host, for testing against the Python reference
c++ -O2 -std=c++17 -o tplconv native/tplconv.cpp

# device
$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++ \
    -O2 -std=c++17 -static-libstdc++ -o tplconv_arm native/tplconv.cpp
```

No dependencies beyond libc++ and POSIX `mmap`.

## 2. The pack-loading library (9.7 MB, regenerate — it is gitignored)

From the template's patched `model_tpl.cpp` (produced by `tools/tpl_patch.py`
against the QAIRT converter's `model.cpp`):

```sh
export ANDROID_NDK_ROOT=$HOME/android-ndk-r27c
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

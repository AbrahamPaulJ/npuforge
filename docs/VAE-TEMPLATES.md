# Checkpoint-owned QNN VAE templates

The phone replaces every VAE encoder/decoder parameter from
`first_stage_model.*`, then compiles QNN context binaries. This path preserves
the generator's existing interfaces: float32 NCHW image/latent tensors,
decoder `output`, and encoder `mean`/`std`. Latent scaling remains the
generator's responsibility.

| Family | Image shape | Latent shape | Generator scale | QNN target |
|---|---|---|---|---|
| SD1.5 | `[1,3,512,512]` | `[1,4,64,64]` | 0.18215 | v73 / soc43 |
| SDXL | `[1,3,1024,1024]` | `[1,4,128,128]` | 0.13025 | v75 / soc57 |

Generated asset directories are
`app/src/main/assets/components_<family>/vae_encoder` and `vae_decoder`, where
`<family>` is `sd15` or `sdxl`. Each contains
`libqnn_model.so`, `recipe.bin`, `tpl_trim.pack`, `sources.txt`,
`requirements.json` and `htp_config.json`. Generated libraries, recipes and
constants remain outside source control; Qualcomm's terms still apply.

## Reproduction

Use the family's standard AutoencoderKL config, the user's checkpoint, the
QAIRT 2.50 SDK and Android NDK. The exporter imports stock `diffusers`, PyTorch
and safetensors; it does not import the UNet's modified authoring modules.
Set `CHECKPOINT`, `VAE_CONFIG`, `VAE_WORK`, `QNN_SDK_ROOT`, `ANDROID_NDK_ROOT`
and `PYTHON` to those local inputs, and `FAMILY` to `sd15` or `sdxl`.
Run from the repository root:

```sh
REPO_ROOT="$PWD"
image_size=1024
latent_size=128
template=template_sdxl
if [ "$FAMILY" = sd15 ]; then
  image_size=512
  latent_size=64
  template=template
fi
export PATH="$ANDROID_NDK_ROOT:$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH"
export PYTHONPATH="$QNN_SDK_ROOT/lib/python"
"$PYTHON" tools/vae_template.py export --checkpoint "$CHECKPOINT" \
  --config "$VAE_CONFIG" --output "$VAE_WORK"

for component in vae_encoder vae_decoder; do
  work="$VAE_WORK/$component"
  assets="$REPO_ROOT/app/src/main/assets/components_$FAMILY/$component"
  shape="1,3,$image_size,$image_size"
  if [ "$component" = vae_decoder ]; then shape="1,4,$latent_size,$latent_size"; fi
  "$PYTHON" "$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-onnx-converter" \
    --input_network "$work/model.onnx" --output_path "$work/model.cpp" \
    --input_dim input "$shape" --float_bitwidth 16 --float_bias_bitwidth 16 \
    --preserve_io --no_simplification
  "$PYTHON" tools/vae_template.py recipe --checkpoint "$CHECKPOINT" \
    --model "$work" --component "$component" --output "$assets"
  "$PYTHON" tools/tpl_patch.py "$work/model.cpp" "$work/model_tpl.cpp"
  (cd "$work" && "$PYTHON" "$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-model-lib-generator" \
    -c model_tpl.cpp -t aarch64-android x86_64-linux-clang -l qnn_model -o lib)
  cp "$work/lib/aarch64-android/libqnn_model.so" "$assets/"
  cp "app/src/main/assets/$template/htp_config.json" "$assets/"
done
```

The recipe author rejects ambiguous source mappings, checks complete component
coverage and exact byte agreement
between checkpoint-reconstructed and converter-emitted packs. The encoder uses
108 checkpoint tensors plus three scalar constants; the decoder uses 140
tensors plus two constants. No donor parameter payload remains in the trimmed
packs. Conv weights use OIHW-to-HWIO permutation; attention matrices squeeze
trailing singleton dimensions. Floating recipe rules preserve IEEE half
rounding and reject non-finite values or half overflow.

## Verification and limits

### SD1.5

The reference is the archived original FP16 SD1.5 checkpoint at
[revision `9cfd069`](https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/tree/9cfd069101959ca3828bf9c04a4419870832b74f).
Only its CLIP and VAE tensor bytes were fetched using validated HTTP ranges;
the source header, pinned revision and component-checkpoint hash are retained
with the local authoring artifacts. The reference is not shipped in the app.

All 248 VAE parameters mapped uniquely. Native, Python and converter-emitted
packs match exactly; both v73/soc43 host HTP context compilations succeeded and
their metadata confirms the 512px interfaces above. Exported-graph checks
through host MNN against PyTorch had relative RMSE of 0.00284% for encoder
mean, 0.00199% for encoder std and 0.01294% for decoder output.

A saved image fixture resized to 512px produced finite encoder and decoder
outputs in CUDA float32 and float16. Float16 relative RMSE was 0.4815% for
encoder mean, 0.3982% for std and 0.04780% for decoder output using the same
latent input; maximum decoder absolute difference was 0.002908. These checks
do not establish phone inference or every SD1.5 VAE's numerical stability.

### SDXL

For Pony v6, native reconstruction reproduced both QNN weight packs exactly.
Both generated libraries compiled successfully into v75/soc57 HTP contexts on
the host; the compiled context metadata confirms the interfaces listed above.
Small exported-graph fixtures checked through host MNN against PyTorch had
relative RMSE of 0.00456% for encoder mean, 0.00387% for encoder std and 0.00604%
for decoder output. MNN was used only to check the exported graphs; the phone
package continues to use QNN VAE binaries.

A saved full-size decoder latent produced finite Pony output in both CUDA
float32 and float16. Float16 output relative RMSE was 0.0832% against float32,
with maximum absolute difference 0.02243. This is a focused numerical check,
not validation of phone QNN execution or every checkpoint's VAE.

### Precision

**HTP floating execution uses FP16.** Declaring float32 graph tensors does not
provide full-precision internal math (SDK documentation:
`QNN/HTP/relu_fp16_example.html`). Finite weights alone do not guarantee finite
activations. The base SD1.5 and Pony VAEs stayed finite in the tested fixtures;
other VAEs that
require float32 upcasting remain unverified. Generic activation rescaling or
checkpoint-specific quantization is not implemented.

# Checkpoint-owned SDXL components

The app converts SDXL UNet, CLIP-L, CLIP-G, token/position embeddings, and VAE
encoder/decoder weights from the selected single-file checkpoint. Only the
standard tokenizer is shared. [SD1.5 now has the same component ownership](SD15-COMPONENTS.md),
with its own text-encoder graph and 512px VAE templates.

The user-confirmed Pony diagnostic showed that replacing the CLIPs fixed that
model while preserving its compiled UNet and VAE. The user subsequently confirmed
excellent `waiIllustriousSDXL_v170` output from the full app conversion pipeline.
This establishes a successful phone conversion and text-to-image result for that
checkpoint. Other derivatives, image-to-image and other phones still need testing.

## Runtime

1. Read the checkpoint header and validate required component names, shapes and
   F16/F32/BF16 dtypes. The foreground service repeats validation on the imported file.
2. `libcomponentconv.so` reconstructs both MNN graphs and four embedding files
   from `components_sdxl/clip_recipe.bin`. The recipe contains sparse graph bytes
   and tensor mappings, without learned donor payloads. Output weights come from
   the checkpoint, processed one row at a time.
3. `libtplconv.so` writes a weight pack for each VAE graph. Rule 6 writes IEEE
   FP16 with round-to-nearest-even; rule 7 writes FP32. Conv permutations and
   trailing singleton removal preserve the generated graph's tensor layout.
   Non-finite values and FP16 weight overflow fail conversion.
4. The QNN compiler builds each VAE context in a separate process, retaining only
   its output before starting the next graph. The existing UNet pipeline follows.
5. Assembly requires every output file and exports the existing flat ZIP names,
   `SDXL` marker and `231_masked_v1` contract. Temporary packs/checkpoints are
   removed before ZIP assembly to reduce peak storage.

CLIP-L uses 11 transformer layers and penultimate hidden output, with embedded
FP16 weights. CLIP-G uses all 32 layers, penultimate hidden output and per-token
final-layer-normalized projection. The verified MNN 3.6.1 export uses symmetric
per-channel INT8 external weights (`readType=0`), despite the original export
command requesting asymmetric quantization. The recipe writer inspects actual
format metadata rather than assuming the flag was honored.

VAE encoder input is planar float32 `[1,3,1024,1024]`; outputs are posterior mean
and standard deviation `[1,4,128,128]`. Decoder input is planar float32
`[1,4,128,128]`, output `[1,3,1024,1024]`. Latent scaling remains in the generator.

## Limits

- HTP runs floating graphs in FP16 internally. Float32 external tensors do not
  provide float32 VAE arithmetic. The tested Pony VAE matches FP16-fix weights;
  arbitrary VAEs requiring upcasting can still overflow during inference.
- The existing fixed UNet activation ranges and QNN device target are unchanged.
  Using a checkpoint's components does not prove every SDXL derivative works.
- Text-encoder LoRA merging remains unsupported; adapters are applied only to
  supported UNet modules. Text-only adapters are rejected.
- Pruned checkpoints missing required CLIP/VAE weights are rejected. There is no
  automatic donor substitution.
- Legacy SDXL shared-component backup/restore remains available but is not used
  by new SDXL conversions.

## Build assets

Generated assets live under `app/src/main/assets/components_sdxl/` and are
gitignored. A complete local build needs:

```text
clip_recipe.bin
clip_requirements.json
tokenizer.json
vae_encoder/{libqnn_model.so,recipe.bin,tpl_trim.pack,requirements.json,sources.txt,htp_config.json}
vae_decoder/{libqnn_model.so,recipe.bin,tpl_trim.pack,requirements.json,sources.txt,htp_config.json}
```

`tools/export_clip_components.py` exports and checks both CLIPs in separate host
processes. `tools/clip_recipe.py` authors the sparse recipe from those MNN exports and
checks every learned payload against its source checkpoint before removing it.
`tools/vae_template.py` exports standard AutoencoderKL graphs and authors VAE
recipes. Its recipe command requires coverage of all 108 encoder and 140 decoder
parameters and exact equality with the QNN converter's original weight pack.

Use QAIRT 2.50 with preserved float32 I/O and FP16 weights and biases for VAE
authoring. HTP rejects mixed FP16 convolutions with FP32 biases. Patch both
generated graphs with `tools/tpl_patch.py`, then
build their ARM64 model libraries without embedded weights. Runtime graph
configuration uses the existing SDXL v75/soc57 target. See [BUILD.md](BUILD.md)
for SDK placement and native library packaging. Generated SDK/model binaries
and checkpoint weights must remain outside source control; see `NOTICE`.
Detailed reproduction: [CLIP authoring](CLIP-COMPONENTS.md) and
[VAE authoring](VAE-TEMPLATES.md).

## Validation

- Native CLIP conversion reproduces all seven verified Pony component files
  exactly by full-file SHA-256 and size. Host timing was 13.22 seconds with
  137 MiB peak RSS; this is not a phone benchmark.
- Native FP16/FP32 tests cover tie-to-even rounding, subnormals, signed zero,
  convolution layout, attention singleton dimensions and invalid values.
- Existing native and Python LoRA regressions pass after adding the float rules.
- At component integration, 30 Python/native tests passed, including 13 CLIP
  writer cases for quantization, QKV slicing, projection transpose and malformed
  inputs. The current suite and commands are documented in [Testing](TESTING.md).
- Both VAE native packs reproduce the QNN converter's weight bytes exactly.
- Both VAE contexts compile successfully with the host QNN HTP backend. Their
  compiled metadata preserves the float32 planar runtime interfaces above.
- A saved full-resolution Pony decoder fixture is finite in FP32 and FP16;
  FP16 relative RMSE versus FP32 is 0.0832%. This is a host arithmetic check,
  not a QNN device inference comparison.
- Kotlin compilation and full app lint pass with no lint issues.
- Debug APK assembly passes. Its component assets and native converter binaries
  match the local inputs byte for byte; native load segments use 16 KB alignment.
- The user reports excellent quality after full app conversion of `waiIllustriousSDXL_v170`. The supplied screenshot shows 1024 × 1024, 30 steps, CFG 7, seed 418928922 and 45.8 seconds on NPU. This validates conversion and text-to-image for this checkpoint; it does not isolate CLIP versus VAE effects or establish compatibility with every derivative.
  No conversion-duration or peak-memory measurement was supplied for this run.

# Checkpoint-owned SD1.5 components

SD1.5 now follows the same component ownership as SDXL: UNet, CLIP, embeddings
and VAE weights come from the selected checkpoint. The standard tokenizer is
bundled, and conversion does not download or read shared model weights. Legacy
component backup/restore remains available for existing archives.

The app selects `components_sd15` assets for SD1.5 and `components_sdxl` for SDXL.
Both paths check required tensor names, shapes and F16/F32 dtypes before running
native conversion. Missing components are errors; donor substitution is not a
fallback. Native writers reject non-finite component weights and FP16 overflow.

## Runtime contract

- CLIP: `clip_v2.mnn`, `token_emb.bin`, `pos_emb.bin`. The graph preserves the
  existing SD1.5 **clip-skip 2** contract: 11 transformer layers followed by the
  checkpoint's final LayerNorm. This differs from SDXL CLIP-L's unnormalized
  penultimate hidden states. Input and output are float32 `[1,77,768]`.
- VAE encoder: float32 planar `[1,3,512,512]` image input; `mean` and `std`
  outputs `[1,4,64,64]`.
- VAE decoder: float32 planar `[1,4,64,64]` input to `[1,3,512,512]` output.
  The generator continues to apply latent scaling `0.18215` outside the graphs.
- QNN: QAIRT 2.50, v73/soc43, 8 MB VTCM and the existing SD1.5 compiler settings.
  VAE internal floating arithmetic is FP16 despite float32 external tensors.
- ZIP: existing seven flat filenames; no SDXL markers are added to SD1.5 output.

## Authoring and assets

Generated artifacts remain ignored under `app/src/main/assets/components_sd15`:

```text
clip_recipe.bin
clip_requirements.json
tokenizer.json
vae_encoder/{libqnn_model.so,recipe.bin,tpl_trim.pack,requirements.json,sources.txt,htp_config.json}
vae_decoder/{libqnn_model.so,recipe.bin,tpl_trim.pack,requirements.json,sources.txt,htp_config.json}
```

The source tools are shared with SDXL. See [CLIP authoring](CLIP-COMPONENTS.md)
and [VAE authoring](VAE-TEMPLATES.md) for family-specific commands. The native
CLIP recipe format and floating weight rules are unchanged.

Authoring uses the pinned [SD1.5 FP16 archive](https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive).
Only its CLIP and VAE tensor ranges were fetched: 445 source tensors, about
394 MiB. The local reference is deliberately component-only and cannot be used
as a complete checkpoint in the app. Original header, source revision and HTTP
range checks are retained with the local validation artifacts. Generated
recipes remove learned reference weights; each conversion supplies its own.

## Local validation

- Native CLIP output matches all three reference files by full SHA-256 and size.
  MNN prompt/empty fixture relative RMSE is 0.262% / 0.442% against the original
  encoder using clip-skip 2. Host conversion took 1.49 seconds and 82 MiB peak
  RSS; these are not phone measurements.
- All 248 VAE parameters map uniquely; native, Python and QNN converter weight
  packs are byte-identical. Authoring now rejects ambiguous mappings, including
  distinct F32 weights that become identical after FP16 rounding.
- Both v73/soc43 HTP contexts compile successfully. Compiled metadata verifies
  the 512px planar float32 interfaces above: encoder context 75,774,136 bytes,
  decoder context 110,055,552 bytes.
- A 512px host fixture stayed finite in FP32 and FP16. Relative RMSE was
  0.4815% for encoder mean, 0.3982% for encoder std and 0.04780% for decoder
  output. This is a host arithmetic check, not phone QNN inference validation.
- At component integration, Android debug assembly and lint passed with no lint
  findings, alongside 28 existing Python/native tests and two VAE authoring
  cases. See [Testing](TESTING.md) for the current suite and commands.
- SDXL component assets remain byte-identical to the preceding APK.
- All 30 component asset files and both native writers were verified byte for
  byte inside the final debug APK.

## Quality limits

Own-component conversion prevents borrowing another model's trained weights.
It does not repair bad source weights or prove the existing UNet activation
ranges fit every anime model. The earlier MistoonAnime checkpoint contained
non-finite VAE values and separately failed the UNet comparisons; it should now
report the invalid VAE instead of silently substituting another decoder.

Text-encoder LoRA weights are still not merged. Baked clip-skip 2 is preserved;
there is no new clip-skip selector. VAEs requiring full float32 arithmetic are
not supported by these HTP templates. End-to-end phone conversion and rendering
with the expanded SD1.5 pipeline remain unverified.

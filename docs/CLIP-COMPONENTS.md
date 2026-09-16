# Checkpoint-owned SD1.5 and SDXL CLIPs

`native/componentconv.cpp` builds the MNN text encoders and their embedding
tables from the selected checkpoint. The phone reads a small graph recipe and
writes checkpoint weights into the required layouts; it does not run Python,
PyTorch, ONNX or MNNConvert. See [SDXL-COMPONENTS.md](SDXL-COMPONENTS.md) for the
complete app pipeline and current limitations.

## Runtime contract

```text
libcomponentconv.so <components-directory> <checkpoint.safetensors> <output-directory>
```

The component directory contains `clip_recipe.bin`. Android preflight also uses
`clip_requirements.json`, whose format is:

```json
{"tensors":[{"name":"original checkpoint key","shape":[77,768],"dtypes":["F16","F32"]}]}
```

The SDXL recipe requires 567 source tensors and applies 695 output rules.
The native process independently rejects missing tensors, incorrect ranks or
shapes, unsupported dtypes and invalid data ranges before creating outputs.
For SDXL it produces these seven files:

| File | Representation |
| --- | --- |
| `clip.mnn` | CLIP-L graph with embedded FP16 matrix weights |
| `clip_2.mnn` | CLIP-G graph with external weights |
| `clip_2.mnn.weight` | INT8 matrices, FP32 scales, biases, norms and graph constants |
| `token_emb.bin` | Row-major FP16 `[49408,768]` |
| `pos_emb.bin` | Row-major FP32 `[77,768]` |
| `token_emb_2.bin` | Row-major FP16 `[49408,1280]` |
| `pos_emb_2.bin` | Row-major FP32 `[77,1280]` |

CLIP-L accepts FP32 embedded tokens `[1,77,768]` and returns the penultimate
hidden state, using its first 11 transformer layers without final layer norm.
CLIP-G accepts `[1,77,1280]` and returns named outputs `last_hidden_state`
(penultimate hidden state) and `pooled_output`. The latter contains all 77 token
rows after the last layer, final layer norm and text projection; the generator
selects the first EOS token's row. It is not a single preselected pooled vector.

Checkpoint keys use the original SDXL `conditioner.embedders.0.transformer` and
`conditioner.embedders.1.model` naming. Fused OpenCLIP Q/K/V tensors are sliced
by rows; its text projection is transposed. The unused final CLIP-L layer,
CLIP-L final layer norm, legacy `position_ids`, and CLIP-G `logit_scale` are
not required. Every parameter used by the two output graphs is replaced.

Conversion processes one output row at a time and releases consumed checkpoint
mapping pages. FP16 outputs use round-to-nearest-even. Explicit matrix rules
clamp to the MNN converter's finite half range; embedding rules reject FP16
overflow. Nonfinite source values and overflowing quantization scales fail.
A write-time failure can leave partial files in the working directory: callers
must only publish the directory after exit code zero. The app's staging and
cleanup enforce this. Usage errors return 2; conversion errors return 1.

## SD1.5 contract

SD1.5 uses the same native executable and recipe format, with three output files:
`clip_v2.mnn`, `token_emb.bin` and `pos_emb.bin`. Its standard tokenizer is shared.
The graph receives embedded FP32 `[1,77,768]` and has one default output named
`last_hidden_state` with the same shape.

The bundled graph preserves the existing generator's **clip-skip 2** contract:
11 transformer layers followed by the checkpoint's final layer norm. This is
different from SDXL CLIP-L, whose penultimate output omits final layer norm.
Weights use the SD1.5 `cond_stage_model.transformer.text_model` prefix.
The unused twelfth layer and legacy position IDs are not required. Clip-skip is
baked into this graph; it is not a new runtime setting.

The app ships skip 2. Set `SD15_CHECKPOINT` to an actual SD1.5 checkpoint and
`SD15_CONFIG_DIR` to a directory containing a standard CLIP-L
`text_encoder/config.json`. The other variables are described below. Authoring:

```bash
"$CLIP_PYTHON" tools/export_clip_components.py --family sd15 --clip-skip 2 \
  --checkpoint "$SD15_CHECKPOINT" --config "$SD15_CONFIG_DIR" \
  --tokenizer "$CLIP_TOKENIZER" --mnn-converter "$MNN_CONVERTER" \
  --output "$CLIP_EXPORT_DIR"

"$CLIP_PYTHON" tools/clip_recipe.py --family sd15 --clip-skip 2 \
  --clips "$CLIP_EXPORT_DIR/clips" --checkpoint "$SD15_CHECKPOINT" \
  --output app/src/main/assets/components_sd15
```

The native CLI is unchanged; pass `components_sd15` as its first argument.
This recipe has 180 required tensors/rules, 66 convolutions, 23 layer norms and
181,462 bytes. An independent check confirms final layer norm is replaced from
the checkpoint. The host tools also accept skip 1 for a separate 12-layer graph;
the shipped recipe and validation here cover skip 2.

SD1.5 validation used only the CLIP and VAE tensors range-fetched from
[the pinned original FP16 SD1.5 checkpoint](https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/blob/9cfd069101959ca3828bf9c04a4419870832b74f/v1-5-pruned-emaonly-fp16.safetensors).
The reconstructed component-only checkpoint's SHA-256 is
`0d07eebc9ccab07f7db5c48cc817448e5f0e4f28d8470d660bdd9f3194de2e8c`.
The compatible cached SDXL CLIP-L architecture config was reused; its 12 layers,
768 hidden width, 12 heads, QuickGELU and layer-norm settings also describe this
SD1.5 CLIP. No UNet was loaded or exported.

The skip-2 wrapper matched the original checkpoint encoder exactly. MNN CPU
output relative RMSE was 0.262% for the fixed prompt and 0.442% for the empty
prompt, with finite `[1,77,768]` outputs. All three native-generated files then
matched the verified export by full-file SHA-256 and size. Host reconstruction
took 1.49 seconds with 82 MiB peak RSS; this is not phone timing. Re-authoring
SDXL after the shared-tool changes reproduced its recipe and requirements
byte for byte. End-to-end phone SD1.5 generation remains untested.

## Reproduce the graph recipe

Authoring runs on a host once for this architecture. The phone uses its bundled
recipe with each user's checkpoint; it never retains the authoring checkpoint's
learned payloads. Generated assets are ignored by Git.

The validated environment used Python with PyTorch 2.5.1, transformers 4.46.1,
diffusers 0.31.0, accelerate 1.13.0, NumPy, safetensors, tokenizers, ONNX and
MNN 3.6.1. `MNN_CONVERTER` must invoke that converter version. `SDXL_CONFIG_DIR`
contains standard SDXL `text_encoder/config.json` and
`text_encoder_2/config.json`. `CLIP_TOKENIZER` is the standard CLIP tokenizer
JSON used by the target generator. The validated authoring checkpoint was
Pony v6 full text encoders with F16 source weights.

Set the following paths to local inputs, then run from the repository root:

```bash
CLIP_PYTHON=/path/to/venv/bin/python
SDXL_CHECKPOINT=/path/to/checkpoint.safetensors
SDXL_CONFIG_DIR=/path/to/sdxl/config-directory
CLIP_TOKENIZER=/path/to/tokenizer.json
MNN_CONVERTER=/path/to/MNNConvert
CLIP_EXPORT_DIR=/path/to/clip-authoring

"$CLIP_PYTHON" tools/export_clip_components.py \
  --checkpoint "$SDXL_CHECKPOINT" --config "$SDXL_CONFIG_DIR" \
  --tokenizer "$CLIP_TOKENIZER" --mnn-converter "$MNN_CONVERTER" \
  --output "$CLIP_EXPORT_DIR"

"$CLIP_PYTHON" tools/clip_recipe.py \
  --clips "$CLIP_EXPORT_DIR/clips" --checkpoint "$SDXL_CHECKPOINT" \
  --output app/src/main/assets/components_sdxl
```

The exporter loads only the corresponding text-encoder tensors, sequentially in
separate four-thread CPU processes. It strictly loads checkpoint parameters,
checks each wrapper against the original encoder, exports ONNX, converts MNN,
and checks the emitted MNN files against fixed prompt/empty fixtures. Both
EOS-padded and zero-padded CLIP-G fixtures are included. Verification reconstructs
inputs from the emitted embedding files, including FP16 token rounding for F32
source checkpoints. MNN output shapes and finiteness are required; relative
RMSE must be below 10% for every checked output and EOS pooled vector. This
coarse guard catches broken exports; it is not an image-quality guarantee.

The recipe author checks each learned MNN byte range against the checkpoint
before removing it. It rejects unknown parameter types, unexamined subgraphs,
extra tensor payloads and unrecognized Blob constants. Remaining literal
constants are explicitly checked shapes, causal masks, attention scale and
activation scale. Export changes require reviewing these checks instead of
silently keeping new payloads. The SDXL recipe is 1,480,803 bytes and contains
1,393,318 literal graph/constant bytes. It does not contain the roughly 949 MiB
of reconstructed component data. Some converter builds use different FP16
rounding; an authoring payload mismatch is an error, never a reason to skip
validation.

Host smoke test:

```bash
c++ -std=c++17 -O2 -Wall -Wextra -Werror -fno-fast-math \
  native/componentconv.cpp -o /tmp/componentconv
/tmp/componentconv app/src/main/assets/components_sdxl \
  "$SDXL_CHECKPOINT" "$CLIP_EXPORT_DIR/native-clips"
python3 -m unittest discover -s tests -p test_componentconv.py -v
```

Compare all seven native files with the verified files in
`$CLIP_EXPORT_DIR/clips` before using newly authored assets. For the validated
Pony fixture every full-file SHA-256 and byte length matched. Host conversion
took 13.22 seconds with 137 MiB peak RSS. This is not phone timing. The original
MNN comparison measured CLIP-L hidden relative RMSE of 0.080–0.090%, CLIP-G
hidden 1.92–2.92%, and CLIP-G EOS pooled 0.91–1.29% versus original encoders.

The native test suite covers IEEE half ties/subnormals/signed zero, explicit
clamping versus overflow rejection, fused QKV slicing, projection transpose,
symmetric/asymmetric INT8 ties and small-scale behavior, constant rows, denormal
normalization, malformed metadata, truncated files, invalid output ranges and
nonfinite/overflowing values. End-to-end phone inference remains a separate test.

## Binary recipe format

All integers and floats are little-endian; strings are a `u16` byte length
followed by UTF-8 bytes. Supported targets are little-endian ARM64 and x86-64.

```text
"CLIPRCP1"                       8-byte magic
u32 file_count
for each file:
    string filename
    u64 final_size
    u32 literal_count
    for each literal:
        u64 destination_offset
        u32 byte_count
        byte[byte_count] content
u32 rule_count
for each rule:
    u8 file_index
    u8 encoding                  0=FP32, 1=FP16, 2=symmetric8, 3=asymmetric8
    u8 flags                     bit0=transpose, bit1=clampHalf, bit2=alignDenormal
    u8 source_rank               1 or 2
    u64 destination_data_offset
    u64 destination_alpha_offset
    u32 destination_rows
    u32 destination_columns
    u32 source_rows
    u32 source_columns           1 for a rank-1 tensor
    u32 first_source_row
    f32 multiplier
    string checkpoint_tensor_name
```

Literals cover the complement of learned ranges. Authoring rejects overlaps and
checks sizes. All output bytes are covered once by either a literal or a rewrite;
quantized rules rewrite both the encoded matrix and its scale table.

## MNN encoding and provenance

The format follows the upstream MNN 3.6.1
[FlatBuffer schema](https://github.com/alibaba/MNN/tree/3.6.1/schema/default),
[IDST encoder](https://github.com/alibaba/MNN/blob/3.6.1/source/core/IDSTEncoder.hpp),
[weight quantizer](https://github.com/alibaba/MNN/blob/3.6.1/tools/converter/source/common/WeightQuantAndCoding.cpp),
[denormal handling](https://github.com/alibaba/MNN/blob/3.6.1/tools/converter/source/common/AlignDenormalizedValue.cpp),
and [half export](https://github.com/alibaba/MNN/blob/3.6.1/tools/converter/source/common/SaveHalfFloat.cpp).
MNN is [Apache-2.0 licensed](https://github.com/alibaba/MNN/blob/3.6.1/LICENSE.txt).
The native writer implements the required transformations independently; no
MNN library is bundled for conversion. Model/tokenizer redistribution remains
subject to their licenses; see [NOTICE](../NOTICE).

The author reads actual graph metadata. Although the original command supplied
`--weightQuantBits 8 --weightQuantAsymmetric --saveExternalData`, the tested
CLIP-G export contains 193 **symmetric** dense INT8 matrices (`readType=0`),
not asymmetric matrices. CLIP-L matrices use embedded IDST type 3 FP16 buffers.
CLIP-G uses type 1 dense data with a two-dimensional 16-bit shape header and
the full signed-byte codebook. These header/codebook bytes remain literals.

For a symmetric row, `scale=max(abs(weight))/127` is computed in FP32. Above
`1e-6`, divide in FP32, round ties away from zero, clamp to `[-128,127]`, then
store the codebook index `q+128`; otherwise store index 128. The scale is still
written for a small or zero row. The recipe also supports inspected asymmetric
metadata: FP32 row minimum and `(max-min)/255`, rounded and clamped indices
`[0,255]`, or index zero when scale is at most `1e-6`. It rejects scale overflow.
Matrix rules match MNN's normalization of F32 denormals and signed zero to zero;
biases, layer norms and embedding rules preserve their original finite values.

The current recipe fixes architecture and numerical formats. It does not
support arbitrary CLIP architectures, tokenizer changes, text-encoder LoRA
merging, or conversion of incomplete checkpoints through donor substitution.

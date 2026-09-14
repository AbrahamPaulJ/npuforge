# Authoring a new template

`docs/PIPELINE.md` says **why** the template/recipe split works and carries every
measurement. This file is the **runbook**: what to run, in what order, with which
flags, and which gates must pass before you spend the next two hours.

You need this if you want to change any of:

| | why you would |
|---|---|
| the **calibration checkpoint** | `ROADMAP.md` 1b — base SD1.5 is probably a better centroid than DreamShaper |
| the **QAIRT SDK version** | `ROADMAP.md` P1 — 2.28 drops the fp16 stamp |
| the **resolution** | `ROADMAP.md` 7 — the graph is 512² by construction |
| the **activation ranges** | `ROADMAP.md` P2 / 1c — the anime failure |
| the **licence** | ⛔ a clean-room rebuild is what makes the bundle commercially usable |

⚠ **Nothing downstream of the template changes.** `native/tplconv.cpp` is
SDK-agnostic and resolution-agnostic; `tools/tpl_*.py` rediscover the mapping
numerically. A new template is a new `recipe.bin` + `tpl_trim.pack` +
`libqnn_model.so`, and the app takes them as data.

## 1. ⛔ What is not in this repo, and why

The authoring path — unlike everything else here — cannot be shipped under MIT.

| What | Status | Note |
|---|---|---|
| A modified `diffusers` UNet (`redefined_modules/`) | **CC BY-NC 4.0**, from xororz's Local Dream | the ONNX export imports `UNet2DConditionModel` and `CrossAttention` from it, because the stock `diffusers` UNet does not export to a QNN-friendly graph |
| `prepare_data.py` — prompt → calibration rows | same upstream | produces `data.pkl` from a prompt set |
| `gen_quant_data.py` — `data.pkl` → `.raw` files + input list | same upstream | writes one `.raw` per tensor per row |
| `export_onnx_unet_only.py` | a thin wrapper, but it imports the above | ~2.6 KB |
| QAIRT SDK (`qnn-onnx-converter`, `qnn-model-lib-generator`, `qnn-context-binary-generator`) | Qualcomm's, account required | `NOTICE` §1 |
| Render + score harness | never written for a public audience | §5 says what it must do |

**This is the reason `template/` is marked non-commercial** (`NOTICE` §2). To
make it clean, replace the first four rows with your own export path — a UNet
export that produces the same graph topology — and rebuild. Everything from
Phase 1 onwards is this repo's own code and is MIT.

## 2. Phase 0 — quantize one graph properly

Runs on a Linux host (WSL is fine) with the QAIRT SDK on `PATH`. Budget
**~3 h 30 m** wall clock and **>11 GB RAM plus swap**; step 5 is the long one.

1. **Get the calibration checkpoint.** Any SD1.5 single-file `.safetensors`.
   ⚠ Assert its byte size before proceeding — the whole build keys off it, and a
   partial download fails three hours later instead of now.
2. **Calibration rows** — `prepare_data.py --model_path <ckpt> --realistic`.
   About **52 min** on CPU, and it is the step whose *prompt set* decides what the
   template covers. 20 prompts → 942 candidate rows → **400** kept.
   ⚠ This is the input to change for an anime or mixed template. Nothing else
   about the pipeline is style-specific.
3. **Raw rows** — `gen_quant_data.py`, writing `input_list_unet.txt`.
   ⚠ **Check dtypes here.** A timestep written as int32 and read as float32 cost
   **2 h 39 m** and produced a model that passed four checks and computed nothing
   (`PIPELINE.md`, "the dead first template").
4. **Export the UNet to ONNX** — `export_onnx_unet_only.py --model_path <ckpt>`.
   Delete the 4 GB diffusers copy it leaves behind; only `unet/model.onnx` matters.
5. **Quantize and convert** — the long step, **~2 h 20 m at ~20.7 s/row**:

   ```sh
   qnn-onnx-converter -n --input_network ./unet/model.onnx \
       --preserve_io layout \
       --input_list ./input_list_unet.txt \
       --use_per_channel_quantization \
       --bias_bitwidth 32 \
       --act_bitwidth 16
   ```

   ⚠ **Keep `model.cpp` and `model.bin`.** Phases 1 and 2 are derived from them,
   and regenerating them is this entire step again.
6. **`libmodel.so`** — `qnn-model-lib-generator -c model.cpp -b model.bin -t x86_64-linux-clang`.
7. **Context binary** — `qnn-context-binary-generator --model … --backend libQnnHtp.so --config_file htp_backend_<tier>.json`.
   ⚠ `config_file_path` inside the backend-extension JSON must be **absolute** on
   device; a relative path is read against the process CWD and fails confusingly.

### ⭐ The gate that goes between step 5 and step 6

Spend seconds here, not the compile. Read the encodings and assert:

- **0 quantized (`u16`, dtype 1046) tensors with `scale == 0`.** The dead
  template had 3. ⚠ Check dtype 1046 and *nothing else* — an earlier version of
  this gate failed a healthy build by asserting on int32 and float aliases.
- The time-embedding path has a real range: `_time_proj_Cast_output_0` and
  `_time_proj_Mul_output_0` should reach **[0, 999]**, `_time_proj_Sin/Cos_output_0`
  **[−1, 1]**. All-zero ranges mean the UNet will denoise blind.

An 8-row probe build (~2 min) reproduces these ranges closely enough to gate on
— the probe reads [0, 929] where 400 rows read [0, 999] — so you can validate the
whole pipeline cheaply before committing to the full quantize.

## 3. Phases 1 and 2 — this repo's part

**~9 minutes**, all MIT, all in `tools/`.

```sh
tools/tpl_patch.py      # model.cpp -> model_tpl.cpp: 2,383 static sites become tpl:: calls
tools/tpl_pack.py       # model.bin -> an identity TPLPACK1 pack (863 MB)
tools/tpl_recipe.py     # numeric tensor mapping + rule discovery -> recipe.json
tools/tpl_recipe_bin.py # recipe.json -> recipe.bin (397 KB)
tools/tpl_pack_trim.py  # 863 MB pack -> tpl_trim.pack (43 KB)
```

Then `qnn-model-lib-generator` on `model_tpl.cpp` **without** `model.bin` gives
the ~9.7 MB `libqnn_model.so` that loads its weights from a pack at build time.

**Gates, in order of what they catch:**

| gate | expected | catches |
|---|---|---|
| `tpl_patch.py` site count | 2,383 rewritten, **0 raw references left** | a tensor still reading baked data |
| identity-pack context md5 | equal to the stock build's md5 | the pack path not actually driving the build |
| negative control | XOR 4 KB in the pack → the context binary changes | a pack that is being ignored |
| `tpl_recipe.py` discovery | matched **1358** / unmatched **1025** / **ambiguous 0** | a mapping that silently guessed |
| same-checkpoint round trip | **all 2,383 scales exact**, differences in bias bytes only | a wrong quantization rule |

⚠ The mapping is found **numerically** — sorted-value fingerprint plus
permutation search — never by name, because the exporter emits norm gamma/beta as
anonymous `onnx__Mul_*` / `onnx__Add_*`. If you change the export, expect the
*names* to move and the mapping to still work.

## 4. Phase 3 — prove it generalises

Run a **different** checkpoint's weights through the new template and compare
packs, not pictures:

```sh
./tplconv template/recipe.bin template/tpl_trim.pack <other>.safetensors out.pack
```

Expect every entry with a checkpoint source to get new weights *and* recomputed
per-channel scales (1,358 of 2,383), and all 1,025 template constants untouched.
The same-checkpoint control must reproduce all 2,383 scales exactly.

⚠ **Correctness is decided by `cmp` on the weight pack, never by the context
binary's md5** — the compile is not byte-reproducible. Two known-good pack md5s
are in `docs/BUILD.md`.

## 5. What the render harness has to do

Not in the repo, but it is only a driver around a generator. To judge a template
you need, per prompt, with everything else fixed (seed, steps, sampler, and
**only `unet.bin` varying**):

- **PSNR and SSIM against the right arm.** ⚠ A model built from the *recipe* pack
  is compared against a PC build of the *recipe* pack. Comparing against the
  stock template arm reads as total failure when nothing is wrong.
- **`extreme_frac`** — the fraction of saturated pixels. A healthy render scores
  **0.00–0.05**; the dead template scored **0.317** and, the tell, scored it
  *near-identically across different prompts*.
- **A step sweep** when two arms disagree at 20 steps. After one UNet call a
  correct rule agrees to ~46 dB; diffusion is chaotic, so small float noise
  amplifies by step 20 while staying the same image. A wrong rule disagrees at
  step 1.

⚠ **Latency proves a graph compiled, never that it computes anything.** The dead
template ran at 4.0 s against a known-good 3.9 s, exited 0, matched the IO
contract exactly, and produced pure noise. Never accept "it ran" as evidence.

## 6. Clean-room notes

If the goal is a commercially usable bundle, the only thing you must replace is
the **export path** (§1's first four rows). Requirements on a replacement:

- It must produce a UNet ONNX graph that QAIRT quantizes to the same topology —
  that is what makes the numeric mapping find 1,358 matches with 0 ambiguous.
- The calibration rows must carry the timestep as the dtype the graph reads
  (§2 step 3).
- Nothing else in this repository needs to change, and the gates in §3 will tell
  you quickly if the graph moved.

Record what you did in `docs/PIPELINE.md` — it is the file that keeps the next
person from paying for the same dead end twice.

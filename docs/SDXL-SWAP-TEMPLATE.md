# SDXL Swap template: LoRA, ControlNet, IP-Adapter and inpaint as graph inputs

The SD1.5 Swap v3 template ([SD15-LORA-CN-TEMPLATE.md](SD15-LORA-CN-TEMPLATE.md) §9) ported to
SDXL's `231_masked_v1` UNet ([SDXL.md](SDXL.md) §Model contract): one converted model takes any LoRA
(rank ≤ 64 exact), the residuals of an SDXL ControlNet and IP-Adapter Plus K/V per render, and each
of the four features is kept or dropped per conversion. Authored on a Linux host with ≥ 128 GB RAM;
every script was first proven end to end on a tiny SDXL-shaped model (§4).

## Contents

1. Graph contract
2. Decisions and their evidence
3. Authoring (the build script's stages)
4. What the tiny-model runs proved
5. Running it on a rented host
6. Scoring it on the phone
7. Open work: the v2 template

## 1. Graph contract

| input | shape at 1024² | encoding (u16) | meaning |
|---|---|---|---|
| `sample` … `encoder_attention_mask` | as `231_masked_v1` | calibrated (`sample` ±8) | unchanged; first six inputs |
| `lora_S` | [700] | [0, 0.05] | per-target scale (SD1.5: [0, 0.25], §2) |
| `la_i`, `lb_i` | [din, 64], [64, dout] | ±1 | one pair per target: 70 transformer blocks × attn1 q/k/v/out, attn2 q/k/v/out, ff proj/out |
| `res_0..9` | NHWC, [1,128,128,320] … [1,32,32,1280] | ±2 × the largest seen per index | ControlNet: 9 skip residuals + mid (SD1.5 has 13) |
| `ipk_j`, `ipv_j` | [1, inner, 16], [1, 16, inner] | ±1.25 / ±1.65 × max | IP-Adapter K and V (IP scale folded into V), 70 attn2 layers |
| `mask`, `masked_latent` | [1,1,128,128], [1,4,128,128] | [0,1], ±8 | inpaint branch (split `conv_in`), last |

1,559 inputs, 377 MiB of them (LoRA 319 MiB, written once per LoRA change). Zero feature inputs are
exactly the base model. The text mask applies to the text scores only. Bundle: `recipe.bin`,
`tpl_trim.pack`, feature-gated `libqnn_model.so`, `lora_targets.json`, `ip_targets.json`,
`sources.txt`, `swap_features.json` (`sdxl_swap_v1`), the `template_sdxl` HTP configs.

## 2. Decisions and their evidence

| decision | evidence |
|---|---|
| Swap v3 shape (features chosen per conversion) | `tpl_features.py` unchanged: on the tiny model every drop subset leaves exactly its inputs (391 / 52 / 383 / 359 / 8 / 6) |
| LoRA targets = SD1.5's set (attn + ff), rank 64 | seven public SDXL LoRAs: style LoRAs keep 96–99.4 % of their UNet ΔW² in the targets (the rest is proj_in/out, ≤ 1.3 %); rank 64 is exact up to 64 (SD1.5 measured R = 32 losing 10–17 %) |
| Speed LoRAs are baked, not swapped | LCM keeps 78 %, DMD2 91 %, SDXL-Lightning 54.5 % — they train ResNet layers the targets do not cover |
| `lora_S` window [0, 0.05] | SDXL's per-target S is 6–500× smaller than SD1.5's (max 0.0115 at strength 1): median quantisation error falls 5× against [0, 0.25], worst case 67 % → 3.6 %, headroom stays > 4× strength |
| Calibration on SDXL base 1.0, mixed photo/anime prompts | base is the parent of every fine-tune; Animagine XL 3.1 is the other-family Phase 3 checkpoint. Changeable (one config line) |
| Prompt rows match the runtime | unused chunks are zeros (masked), pooled from the first chunk — what Nightmare's backend feeds. The `231_masked_v1` row generator encoded padding chunks and pooled the last |
| Trajectories run the PACKED LoRA | a rank > 64 LoRA reaches the graph SVD-truncated; merging the full LoRA into the trajectory UNet made the row verify fail at rel 1.1e-2 on a rank-128 test LoRA (merging the packed delta: the rows are what the template computes) |
| Timestep written as float32 | the legacy `qnn-onnx-converter` reads every input-list raw as float32: an int32 timestep quantised the time path to [0, 1e-4] on the tiny model (PIPELINE.md, "the dead first template") |
| Legacy converter, not `qairt-converter` | it writes `model.cpp`/`model.bin` directly; the DLC → `model.cpp` step of the other route is not available |

## 3. Authoring

`run_sdxl_swap.sh <config.env>`, resumable per stage, timing and peak RSS logged per stage:

| stage | what |
|---|---|
| D, A | inpaint difference (official SDXL inpaint UNet − SDXL base, LDM keys); add-difference template, discover and base-wide checkpoints |
| E | export (`sdxl_swap.py`: the `231_masked_v1` UNet + Swap inputs; the redefined modules need the ControlNet residual patch) |
| R | trajectory rows from stock diffusers with the LoRA merged, ControlNet and IP-Adapter active; `--verify` checks the Swap graph against that UNet on one row |
| IS, C | IP K/V windows over every reference image; calibration list and overrides (branch ranges by ORT fp32) |
| QA | quantize on the template weights; encoding gate. `PROBE=N` stops here with time and RSS |
| UB, UN, Q | base-wide quantize (txt2img rows), union of ranges, final quantize (SD1.5 v3b) |
| S, P2, P1 | stock vs identity-pack context; discover `--head-dim 64`, finalize (the inpaint half of `conv_in` as template constants), round trip |
| C2, FT, B, REF | second checkpoint packs (txt2img and inpaint); feature-gated source; bundle; fp32 references of the held-out rows |

## 4. What the tiny-model runs proved

A 32 M-parameter random UNet with SDXL's structure (64-channel heads, 2048-wide text, `text_time`
conditioning; a 9-channel twin), saved as an LDM single file that diffusers' own converter reads back
exactly.

| check | result |
|---|---|
| redefined modules vs stock diffusers | rel 1.8e-6 at 1/2/3 active chunks |
| Swap graph vs stock diffusers, each feature and all (inpaint too) | rel 1.3–2.0e-6; all inputs zero = the base export exactly |
| ONNX in ORT vs torch | rel 1.6e-6; input order and `targets.json` order match |
| LoRA packer vs diffusers' own kohya/SGM key reading | rel 1.9e-6, both naming forms, negative strength |
| IP-Adapter Plus SDXL core vs diffusers' loader | rel 1.3e-6 (IP effect 35 %) |
| real trajectory row: Swap graph vs stock UNet | rel 1.56e-6 |
| add-difference chain | reproduces the inpaint model exactly |
| quantize gate | time path [0, 999], 0 dead tensors, inputs exact, overrides 1,064 / 1,070 (the misses: renamed attn1 tensors, as SD1.5's 16) |
| identity pack vs stock context | byte-identical; the negative control differs |
| discover | 794 matched, 0 ambiguous, every source once — the Swap inputs add no weight sites |
| round trip | 0 scale mismatches, bias-only residue (the SD1.5 control) |
| second checkpoint | exactly the mapped sites change; its context compiles |
| the whole chain on the phone (S25, v79): `phone_eval.sh` | contexts composed on the phone from the gated lib in 2–7 s; held-out rows vs fp32: base 40.6 / 37.9 dB (txt2img / inpaint), LoRA gain 1.006 / cosine 0.986 (`lora_S` [0, 0.05]), ControlNet 1.000 / 1.000, IP 1.001 / 0.999, inpaint 1.003 / 0.999; all-dropped context = the same base |

**VTCM at real shapes** (minimal graphs compiled for v79, 8 MB): the 10 MiB residual input `res_0`
(an Add), the 5 MiB per-head LoRA q delta and the 1.25 MiB `ff` LoRA B all compile; the control —
a Slice of one packed 10 MB LoRA input, SD1.5's first failure — fails with the same
`InputSlice … not sufficiently tiled` error, so the probe tests what it claims.

## 5. Running it on a rented host

A VM, not a container (swap is the safety net): ≥ 128 GB RAM, 200 GB disk, Ubuntu 22.04, a 24 GB GPU
optional (it speeds the trajectories; quantize is CPU). `setup_vm.sh` installs the proven environment
(Python 3.10, torch 2.5.1, diffusers 0.31.0, transformers 4.46.1, QAIRT 2.50.0.260828 by plain `curl`,
NDK r27c) and downloads every model with a size check. Then `PROBE=2` — read the probe's peak RSS and
time per row, stop if the projection passes ~80 % of RAM — then `PROBE=0` in the same work directory.

**The real build** (Vast.ai KVM VM: Ryzen 9950X 30 vCPU, 197 GB RAM, RTX 4090 48 GB; SDXL base 1.0,
inpaint, union, rank 64; ~8 h of VM time including the traps below, ~5 h without them):

| stage | wall | peak RSS |
|---|---|---|
| D, A, E (export) | 0.3 / 0.4 / 5 min | 19 / 26 / 24 GiB |
| R (10 rows, verify rel 7.4e-4 in fp16 on the GPU) | 4.5 min | 27 GiB |
| C (calibration list, 9.7 GB of raws) | 3.5 min | 22 GiB |
| QA probe (2 rows) / QA (8 rows) | 30 / 39 min | 53 GiB — flat, set by the graph, not the rows (SD1.5: 400 rows in ~32 GB) |
| EB + UB / UN / Q | 5 + 32 min / 7 s / 41 min | 53 GiB |
| x86 libs, stock and template (in parallel) | 27–33 min each | 24 GiB |
| host context (v79), each | 11 min | 18 GiB; `unet.bin` 2.69 GB |
| P1 discover | 1 min | 9,060 matched, 0 ambiguous, every source once; round trip: 3 biases differ by 1 byte |
| C2 (Animagine XL 3.1) | 1.5 min per pack | exactly the 9,060 mapped sites change |
| ARM lib at `-O0` | 3.6 min | 8 GiB; 190 MB |

**Traps that only SDXL's size exposed** (all fixed in the scripts):

- `clang` 14 selects GCC 12 when it is installed: without `libstdc++-12-dev`, the lib generator fails on
  `<algorithm>`.
- The stock lib links one object per weight: 20,766 objects, 2.35 MB of arguments, over the 2 MB
  default `ARG_MAX` ("Argument list too long"). `ulimit -s 65536` raises it to the kernel's 6 MB.
- The stock lib's 2.6 GB of `objcopy`'d weights in `.data` push `.bss` out of `crtbeginS.o`'s ±2 GB PC32
  reach ("relocation truncated to fit"). `stock_lib` renames the weight sections to `.lrodata` (x86-64
  large data, laid out after `.bss`) and builds in a directory that survives a failed link.
- **`qnn-context-binary-generator` is not deterministic at this size**: one lib compiled twice differs in
  12.9 M of 2.69 G bytes and in `opDataSize` (±256). The md5 identity gate cannot pass; the gate compares
  the context metadata (`qnn-context-binary-utility`, all but `opDataSize`) and requires a byte
  difference under 1 %. Measured: template vs a stock build — metadata identical, 24–29 K bytes apart.
  SD1.5 and the tiny model stay byte-identical.
- The ARM lib at the generator's `-O3 -g` ran over 2.5 h (`-O1` over 1 h): clang inlines all 93,277
  guarded `add*` calls into one `QnnModel_composeGraphs`. The lib only builds the graph, so `arm_lib`
  builds at `-O0 -g0` (3.6 min). Its `composeGraphs` frame is 3 MB: fine on the generator process's
  main thread (8 MB), which is how npuforge and `phone_eval.sh` run it — never call it on a 1 MB thread.
- Ubuntu's unattended upgrades replaced the NVIDIA libraries under the running driver mid-session
  ("Driver/library version mismatch"): any GPU step after that needs a reboot or the CPU.

## 6. Scoring it on the phone

From the authoring PC (phone access stays off the rented host): pull `bundle/`, `identity.pack`,
`ref/` and the held-out rows, then `phone_eval.sh`. It composes three contexts from the gated lib on
the phone (txt2img, inpaint, all features dropped), runs every held-out case through `qnn-net-run`,
and `cmp_sdxl.py` scores base SNR and each feature's gain / cosine against fp32 — SD1.5 v3 shipped at
base 35.2 dB, LoRA 1.027 / 0.986, canny 1.018 / 0.997, IP 1.006 / 0.974. `decode_preview.py` turns each
case's single step (t = 519) into a predicted image (x0 through the SDXL VAE), fp32 beside phone.

On the phone (S25 Ultra, v79):

| | measured |
|---|---|
| compile, inpaint context (every feature kept) | **52 min wall** (31 min CPU), storage-backed heap peak **15.1 GB**, 2,693,166,144 B — awake, plugged in, apps closed |
| base (held-out inpaint row, t = 519) vs fp32 | **46.7 dB** (SD1.5 v3 shipped at 35.2) |
| ControlNet | gain 1.007, cosine 0.999 (effect 1.0 %) |
| inpaint, all features | gain 1.000, cosine 1.000 (effect 24 %) |
| IP-Adapter | gain **1.183**, cosine 0.988 (effect 0.4 %) |
| LoRA (held-out toyface, strength 1) | gain **0.893**, cosine 0.988 (effect 0.1 %) |
| `tplconv` (npuforge 1.0.9 APK) on Juggernaut XL Ragnarok, on the phone | **103 s**, byte-identical to the host's `tpl_apply` pack (md5 `ea4cf45c…`) |

The LoRA and IP effects at this step are tiny, so their gains rest on small absolute deltas — the open
question for a render with a LoRA / reference picture. The txt2img and all-dropped contexts were not
compiled (~50 min each).

What it took to get there:

- Without the storage-backed allocator the compile fails in graph finalize (`error 1002`, "mprotect
  failed … Out of memory"); `phone_eval.sh` preloads npuforge's `libcompiler_heap.so` into the generator.
- A dozing phone with 2.7 GB available lost the compile 3 min in, with no kill logged (lmkd was reclaiming
  apps at the same second); awake, plugged in and with apps closed (4.7 GB available) it completed.
- An `adbd` restart (a USB event) kills every adb shell: the on-phone script runs detached (`nohup setsid`),
  its launch is retried until its log exists, and the PC only polls.
- `adb.exe` reads stdin (it swallowed a `while read` loop's file list): every adb call gets `< /dev/null`;
  the 3,268 held-out files go as one directory push (1.7 GB in 87 s on Wi-Fi).
- Over mobile Tailscale, plain `adb push` of GB files fails and adb reports success on corrupt data (23 of
  78 chunks): `chunkpush.sh`/`chunkfix.sh`/`vm2phone.sh` move 32 MiB chunks, each accepted on its md5.

## 7. Open work: the v2 template

v1 (this doc) is built and shipping in previews: npuforge 1.0.10 converts it, Nightmare 1.6.076 renders
it with **LoRA per render** (`backend-patches/018` there). Its ControlNet / IP-Adapter inputs are bound to
zeros until the backend feeds them on `PipelineSdxl`; inpaint is not offered in the app yet. If v1's LoRA,
ControlNet and IP-Adapter hold up on renders, **one v2 template** replaces it, aimed at Illustrious / NoobAI
NSFW-anime workflows (the user, 2026-10-03). Everything optional per conversion, as v1:

| v2 item | why | graph? |
|---|---|---|
| **Calibration on Illustrious + NoobAI v-pred, Danbooru-tag prompts incl. NSFW ones**, no style words (Mr.J) | v1 calibrated on SDXL base, 6 of 8 rows photo-style; anime / NSFW activations may clip. A v-pred UNet outputs v, not eps — late-layer and output ranges differ | no (rows + calibration ckpts) |
| **Conv-layer delta inputs** (resnet convs) | LoCon / LoHa / LoKr LoRAs (much of the Illustrious ecosystem), speed LoRAs (54–91 % in v1's targets) | yes |
| **Attention coupling** (regional prompts, ComfyUI's Attention Couple) | multi-character scenes in one UNet pass; multi-pass area conditioning needs no graph but costs a UNet pass per region | yes: extra prompt tokens + a region mask per cross-attention layer and resolution |
| **Prompt length: 4–6 chunks** (308–462 tokens) | v1 is 231 (3×77); quality + character + NSFW tag prompts overrun it | yes (cross-attention only, cheap) |
| **FreeU** scales; **PAG** (a per-pass self-attention switch) | common SDXL quality knobs | yes (small) |
| T2I-Adapter (intra-block residuals), a second IP-Adapter slot | optional | yes |

**Dropped** (do not re-propose): **native resolutions** — each size is its own ~2.7 GB context per
checkpoint, and several multi-GB files per checkpoint is not practical (the user, 2026-10-03); non-square
stays the 1024² render cropped (`RequestParser.hpp`). **Inpaint as an on/off switch** — measured below.

### 7a. Measured on the VM, 2026-10-03

**The inpaint switch does not work** (`inpaint_switch.py`; SDXL base vs the official inpaint UNet). Energy of
the difference: 63 % in the 700 targets, 29 % in resnets, 2–4 % each in proj_in/out and up/down samplers.
Kept at rank 16 / 32 / 64 / 128 per target: 10 / 15 / 23 / 33 % of the total — it is full-rank. Function
(eps of the true inpaint UNet vs base + inpaint `conv_in` + a delta on the targets, 5 timesteps, an inpaint
input): captured share of the eps gap — targets full rank **−0.10**, rank 32/64/128 −0.00/−0.03/−0.07,
targets + resnets full rank 0.40; the control (base + the whole difference) 1.000. A partial difference is
worse than none. ⇒ inpaint stays v1's per-conversion add-difference.
**Does the inpaint conversion serve txt2img?** (`inp_as_t2i.py`: Juggernaut vs Juggernaut + the difference,
mask all ones, grey masked image, same prompts/seeds, fp16 on the VM; sheets in WSL `~/sdxl_swap/inp_t2i/`):
photo prompts comparable; the anime prompt coherent but plainer (flatter shading, less background).
**Decided (the user, 2026-10-04): the Inpaint chip stays optional per conversion** — "the inpaint model clearly
does worse for t2i", and latent (blend) inpaint is not wanted; whoever wants both converts twice. Detailers
(ADetailer-style) need neither: crop → 1024² → img2img at denoise ~0.3–0.45 with the same checkpoint →
feathered paste. Candidate for high-denoise inpaint on a txt2img conversion: ControlNet Union ProMax's inpaint
mode through the residual inputs (untested). The difference is hosted:
`AbrahamPJ/npuforge-sdxl-inpaint-diff` / `sdxl_inpaint_diff_f16.safetensors`, 5,135,203,248 B, SHA-256
`2cc2aee6a13b3b0b62217f9b2ed722860b09a2acb7d25746e64bd6c698e29c2b` (openrail++, as its sources).

**LoRA with a real character LoRA** (Marin Kitagawa XL, rank 32, kohya, 722 UNet + 264 text-encoder modules,
on Juggernaut XL Ragnarok — the user's phone render did not look like the character):
- packs onto v1: 700/700 targets, 99.4 % of its UNet ΔW² (the 22 proj_in/out modules are 0.6 %), S max 0.024;
- Nightmare's on-phone pack is **byte-identical** to `pack_lora_sdxl.py`'s (1,401 files, md5 `8f06ebc4…`);
- one UNet step on the S25's NPU (held-out inpaint row, t 519, base-weights context): **gain 1.063, cosine
  0.998** at a 0.76 % effect; the toyface control in the same pass reproduced §6 (0.893 / 0.988, 0.13 %) —
  the 0.89 there was a tiny-effect artefact, not a weak LoRA path;
- diffusers fp16 A/B on Juggernaut (4 prompts × 2 seeds; `marin_ab.py`, `ab_identity.py`): CLIP ViT-H
  likeness to the full LoRA's renders (leave-one-prompt-out centroid, minus no-LoRA) — no LoRA 0.013, the
  phone's packed UNet 0.036, packed × 0.893 0.030, packed + text encoder 0.029, full UNet 0.034, full LoRA
  0.029. The packed UNet carries what the full LoRA carries; the image metric sees no gain from the text
  encoder, although the LoRA moves CLIP-G a lot (below). Contact sheets for a human judgement:
  WSL `~/sdxl_swap/marin/ab/sheet_p*.jpg`.

**Text-encoder LoRA survives npuforge's INT8 CLIP-G** (`clipg_int8.py`: merge ΔW, re-quantize symmetric
per row as `CLIP-COMPONENTS.md` does, compare embeddings): Marin's te2 delta changes CLIP-G's penultimate
hidden states by 25–31 % and the pooled vector by 58–83 %; through INT8 merge-and-requantize the delta's
gain is 1.00, cosine 0.998, INT8 noise 4 % of the effect (the deltas are ~34× an INT8 step). CLIP-L (FP16):
exact. ⇒ text-encoder LoRA needs no FP16 CLIP-G: a Nightmare-side merge into a cached CLIP copy per LoRA set.

**SDXL IP-Adapter heads hosted** (`export_ip_sdxl.py`) beside the SD1.5 ones in `AbrahamPJ/nightmare-ip-adapter`:
`ip_plus_sdxl_head_w8qdq.onnx` (SHA-256 `f2aaf1150f81567aa925f9768781592ec4d2983a7b65dd511980e9fc69df02eb`) and
`ip_face_sdxl_head_w8qdq.onnx` (`9cd9c207e47fe2eb024a8c7c5c1e10979a513a31f1697ec719bf4bc6ba3010bb`), 425,119,893 B
each: hidden [1,257,1280] → `ipk_0..69` / `ipv_0..69` at scale 1 in `ip_targets.json` order, int8 weight-only;
worst K/V cosine through the shipped int16 encoder over 12 pictures + the zeros image 0.99978 (Plus), 0.99995
(Face); ~0.35 s per head on the VM's CPU. The encoder is SD1.5's `clip_vit_h_w16qdq.onnx`, unchanged.

Also open: IP-Adapter gain on the phone (1.18 at a 0.4 % effect) — judge on a render; ~50 min per on-phone
compile (drop features, `O=1`, prebuilt contexts); compiling for v75 from a v79+ phone (contexts run forward
only; on-phone compiles are recorded as ignoring the configured arch — untested on purpose); the SDXL
ControlNet contexts (§7b).

### 7b. SDXL ControlNet contexts (canny, depth, openpose built 2026-10-04)

`build_cn.sh <name> <diffusers ControlNet dir>` + `cn_sdxl.py` (WSL `~/sdxl_swap/vm/`, `code/`; on the VM under
`/root/sdxl`): rows (8 trajectories × 6 steps × both CFG sides = 96 calibration rows from SDXL base + the
ControlNet on canny hints of `/root/sdxl/images`, 2 held-out rows) → ONNX → fp32 refs → `qnn-onnx-converter`
w8a16 per-channel → x86 lib → host v79 context. Resumable stages; 30 min wall on the VM (quantize 26 min, 21 GiB).
**Contract** = the template's residual inputs: `sample` NCHW, `timestep` float32 [1], the FIRST 77 tokens of
`encoder_hidden_states`, `text_embeds`, `time_ids`, `controlnet_cond` [1,3,1024,1024] in [0,1]; outputs
`res_0..9` NHWC at conditioning scale 1 (the backend scales). All inputs u16.
- ⚠ Export with a plain attention processor (`PlainAttn` in `cn_sdxl.py`): diffusers' `AttnProcessor` scores
  with `baddbmm` on an empty tensor, which the export froze as a [heads, Nq, Nk] constant per layer — 2.5 GB
  of zeros, a 4.0 GB `model.bin` that cannot link. ⚠ Keep torch's per-tensor external data (gathering it
  doubled 5 GB on a nearly full disk).
- **canny** (`diffusers/controlnet-canny-sdxl-1.0`): `controlnet.bin` 1,286,491,192 B, md5
  `85a77ec6af1375880374ec0932cb1004` (VM `/root/work/cn_canny/ctx/`, WSL `~/sdxl_swap/cn/`, phone
  `/data/local/tmp/probe/sdxl_cn/`). Held-out rows on the S25 vs fp32: overall SNR 29.0 / 29.9 dB, worst
  residual cosine 0.9991, gains 0.996–1.004. Load 0.4 s (accelerator), **execute 0.99 s** (min 0.94 s) — slower
  than the template UNet's 0.74 s although half its size: plain diffusers modules, not the template's HTP
  rewrites. With CFG a step goes ~1.5 → ~3.5 s.
- **depth** (`diffusers/controlnet-depth-sdxl-1.0`, Depth Anything V2 hints): md5 `fcfac21fdd5225f5354248b378ebc8ca`;
  held-out 29.8 / 30.0 dB, worst cosine 0.9993, gains 0.996–1.004. **openpose** (`thibaud/controlnet-openpose-sdxl-1.0`,
  synthetic COCO-18 skeletons as `gen_rows_sdxl.py`'s): md5 `e5b7bf6f715d42a02a2b9392a6632851`; 28.5 / 30.6 dB,
  worst 0.9989, gains 0.995–1.007. Same architecture as canny (~1.29 GB, ~1 s); VM `/root/work/cn_<name>/`, WSL
  `~/sdxl_swap/cn_<name>/`, phone `/data/local/tmp/probe/sdxl_cn_<name>/`. `build_cn.sh` takes `CN_KIND` = its name.
- Next for speed, in order of cost: run the ControlNet on the cond pass only and reuse its residuals for the
  uncond pass (measure the picture); the `-small` / `-mid` SDXL ControlNets (diffusers publishes canny/depth);
  export through the template's redefined modules (convs, per-head attention). Not hosted yet; not wired into
  `PipelineSdxl` yet.

### 7c. The v2 template (built overnight 2026-10-04)

Everything optional per conversion, as v1. Authoring: `sdxl_swap2.py` (export), `verify2.py` (torch graph vs stock
diffusers), `pack_lora_sdxl2.py`, `gen_rows_sdxl2.py` + `mk_rows_config_v2.py`, `mk_calib_sdxl2.py`,
`mk_union_multi.py`, `tpl_features2.py`, `fp32_ref_sdxl2.py`, `run_sdxl_swap2.sh` + `vm2.env`, `prexport.sh`
(WSL `~/sdxl_swap/code`, `vm/`; VM `/root/sdxl`, work dir `/root/work2`).

| v2 input | shape | zeros = | what |
|---|---|---|---|
| text | `encoder_hidden_states` [1,462,2048], mask [1,462] | — | **6 chunks** (v1: 3); marker for the backend still to define (`462_masked`) |
| more linear LoRA targets | targets 700..749 in `lora_targets.json` (`.w` names) | base | resnet `time_emb_proj`, 1×1 `conv_shortcut`, `proj_in` / `proj_out` — v1's 22 untargeted modules |
| `conv_S`, `ca_j` [cin, 9R], `cb_j` [R, cout] | R = 32, 38 convs (`conv_targets.json`) | base | **3×3 conv LoRA** (LoCon; LoHa / LoKr / speed LoRAs through the SVD of their delta): one projection, shift-and-sum of the 9 taps, ×S, `@ cb` |
| `freeu` [4] | (b1−1, b2−1, s1−1, s2−1) | off | diffusers' `apply_freeu` on up_blocks 0/1, **exact**: the Fourier filter is the linear operator Re(A_H x A_Wᵀ), two constant matmuls |
| `pag` [1] | 0 / 1 | off | mid-block self-attention → identity: the backend runs a 3rd pass with 1 (PAG) |
| `regions` [1,128,128,6] | per-chunk spatial weights | off | **attention coupling**: attn2 = joint + Σ_c r_c (softmax_c V_c − joint) — chunk c is region c's prompt |

- **Verified** on the tiny SDXL-shaped model against stock diffusers (each feature alone and all at once, txt2img and
  inpaint): rel ≤ 2.1e-6 with effects 3–45 %; at full size on a real Illustrious row (LoRA + CN + IP + conv):
  rel 5.4e-4 (fp16 trajectory). The tiny graph compiles to a v79 HTP context.
- **Gating** (`tpl_features2.py`): 8 features, 256 subsets; skips exact per subset (256-bit mask per compose call),
  aliases per feature, composed at runtime and verified on every kept input for all 255 subsets; dead-code removal
  (coupling's per-chunk softmaxes read only clean tensors). Tiny: drop nothing == the template byte for byte; on the
  S25's NPU, feature inputs zero: full context 28.9 dB vs fp32, coupling dropped 28.9 dB, all eight dropped 28.7 dB.
- ⚠ Export traps met: traced shapes are tensors (cast to `int`); `avg_pool2d` on the regions misread by the converter
  (constant averaging matmuls instead); a sum started from `0 +` exports an Add with a constant that reads as a
  branch's merge (start from the first term); a chunk never active in the rows calibrates to a dead tensor (one row
  uses 6 regions); NoobAI ships bf16 (numpy readers fail: converted to fp16 in place on the VM); a UNet-only
  checkpoint loads as SD 2 in diffusers (keep the text-encoder keys).
- **Calibration**: template weights = Illustrious-XL v1.0 + the inpaint difference; 10 rows over Illustrious /
  SDXL base / NoobAI-XL v-pred (Danbooru tags incl. NSFW ones, one 6-chunk prompt, photo prompts on SDXL base),
  every feature active in ≥ 2 rows; union of ranges over the template, base-wide Illustrious, SDXL base and NoobAI.
- ⚠ **This build skipped the union passes** (`/root/work2/UNION_SKIPPED.txt`; markers UB, U_sdxl, U_noob, UN, Q;
  `qf` → `qa`): ranges come from the template weights (Illustrious) over all 10 rows, which include SDXL-base and
  NoobAI trajectories. Why: background pre-exports of the union graphs (two 25-GB tracings at ~13 cores each) left
  the quantizer at ~1.3 cores — QA's first row took 65 min, the second > 2 h. **Never run an export beside a
  quantize.** Measured alone: the v2 export 36 min (v1: 5), QA see the build log. The union can be added later
  (resumable: remove those markers and the `qf` link).
- ⚠ **Build the x86 pack-loading lib at -O0** (`libtpl_o0.sh` → `$W/prebuilt/lib_tpl`, which P2 reuses): the
  generator's -O3 on the 627 MB v2 `model_tpl.cpp` is one single-threaded clang for hours; -O0 took 145 s (the lib
  only composes the graph). The stock lib's link (LS) took 1 h 34 min and its context 26 min.
- ⚠ Never `pgrep -f`/`pkill -f` a pattern that a live process of your OWN is running under: an ssh command line
  containing the pattern killed its own session, and a restart's fresh build matched "the old build" and was
  killed. Kill by PID after checking `ps`.
- ⚠ **All eight features do not fit one HTP process on the S25** (v79): the full v2 context (2.807 GB blob,
  394 MB spill-fill) fails to load — `Failed to find available PD ... context size estimate 4022411264`, err 1002.
  **Coupling is the cost**: dropped, the blob is 2.716 GB and spill-fill 107 MB (~ −380 MB); conv + PAG + FreeU
  add almost nothing (all four dropped: 2.703 GB / 114 MB). ⇒ npuforge must refuse feature sets whose estimate
  exceeds the PD budget (coupling + everything else does not fit; v1's four features did). Host subset contexts:
  `subset_ctx.sh <drop sets…>` (x86 feature-gated lib at -O0 from `model_feat.cpp`, ~13–16 min per context).
- **On the S25** (v2.0 = Illustrious-only calibration; coupling dropped; held-out rows never calibrated; vs ORT
  fp32): txt2img row base **36.6 dB**, every feature at once effect 11.1 % gain 1.028 cosine 0.980; FreeU 1.002 /
  0.988 (10 %), conv 1.019 / 0.929 (2.2 %), ControlNet 1.062 / 0.938 (4 %), IP 1.051 / 0.929 (4 %), PAG 1.067 /
  0.824 (1.9 %), LoRA 0.866 / 0.513 (0.6 % — below the base's own noise). Inpaint row: base 42.3 dB, all 1.003 /
  0.999 (25.7 %), inpaint 1.008 / 0.999. v1 on its rows: 40.6 dB txt2img (the union rebuild should close the gap).
- ⚠ **Speed**: `qnn-net-run`, burst, the same held-out input: v2 coupling-dropped **3.0–3.4 s** per pass;
  **v1 with all four features kept (`ctx_inp.bin`) 3.0–3.2 s** — v2's additions cost ~5 %. The 0.74 s per pass
  measured in Nightmare is a LoRA-only conversion. Every kept feature costs latency even at zero input; which one
  dominates is being measured (base-only and LoRA-only v2 contexts).
- **Feature cost attribution** (v2.0 subset contexts, `qnn-net-run` burst, same input, phone dozing — compare
  relatively; absolute times run ~3× the app's): base only 2.20 s · + LoRA 2.89 s (**+31 %**, the dominant cost:
  1,500 matmuls on 16-bit dynamic weights) · + everything but coupling 3.26 s (+48 % total; ControlNet, IP,
  inpaint, conv, FreeU, PAG together +17 %) · v1 with its four 3.12 s. Coupling also costs ~380 MB of PD memory.
  ⇒ conversions should keep only the features the user will use; LoRA is worth a "LoRA off" conversion option.
- **v2.1 union** (2026-10-04): 102,484 activation ranges from the template (Illustrious) ∪ base-wide Illustrious ∪
  SDXL base ∪ NoobAI v-pred; 32,102 widened (base-wide 22,171, SDXL 8,883, NoobAI 11,644), the most in
  `up_blocks.0` attentions (up to 18.8× — those activations would have clipped for SDXL-family and NoobAI
  checkpoints under v2.0). Union passes need the calibration LIST only (`mk_calib_sdxl2.py --kind`; computing ranges
  there wrote a 10 GB model copy and filled the disk); each pass ~15 min export + ~1.5 h quantize.
- **v2.1 built** 2026-10-04 09:55 UTC (union calibration; P2 gate accepted on blob/spill layout noise only, as
  v2.0): bundle in the archive `v21/bundle/`, full context md5 `78c9b25897774628467b6b6513c292b8` (8 features: too big
  for the S25 — ship subsets). Not yet scored on the phone (v2.0 was; score v2.1 with coupling dropped:
  `subset_ctx.sh couple` from `v21/` sources). ⚠ The S stage at -O0 linked in 3 min (1 h 34 at -O3).
- ⭐ **The VM is gone; everything is in the private HF dataset `AbrahamPJ/sdxl-swap-vm-archive`** (18,948 files:
  `v20/`, `v21/`, `cn/`, `exp/`, `scripts/`, `manifest/`, README with the resume paths). Third-party models: by the
  manifest (URL + SHA-256), not re-uploaded.
- **8 Gen 3 and 8 Elite Gen 5** (2026-10-04, host contexts from the archived x86 libs, QAIRT 2.50, 8 MB VTCM):
  canny / depth / openpose for **v75** (SM8650, soc 57) and **v81** (SM8850, soc 87), ~2–4 min each, in the
  archive as `cn/cn_<name>/ctx_v75/`, `ctx_v81/` (+ their `htp_config.json`). Not run on those phones (none here).
  The tiny v2 graph (every v2 feature) also compiles for v75 and v81 — op support proven; the full v2 template on
  v75 (compile memory, PD fit) is unmeasured until an 8 Gen 3 converts one. npuforge's template already defaults
  to v75 (`htp_config.json`), and on-phone compiles target the phone's own arch.

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
7. Open work

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

## 7. Open work

- LoRA and IP-Adapter strength on the phone (gain 0.89 / 1.18 at tiny effects): measure on a render, and
  on more cases, before calling them good. Calibration prompts: 6 of 8 rows are photo-style; Mr.J advises
  Danbooru tags without style words so one template suits anime and realistic fine-tunes — test an
  anime checkpoint first, recalibrate (a `rows_config` edit, ~3–4 h of VM) if it degrades.
- ~50 min per on-phone conversion compile (8 Elite): drop features, `O=1`, or prebuilt contexts.
- Compiling for v75 from a v79+ phone (contexts run forward only, so a v75 build covers 8 Gen 3 and up):
  npuforge records on-phone compiles as ignoring the configured arch — untested on purpose.
- App side: an `SDXL_SWAP` conversion type in npuforge, Nightmare's SDXL pipeline binding the inputs
  (its SD1.5 patches 015–017), an SDXL ControlNet context on the phone (its 77-token text input is an
  assumption of the rows) and the IP-Adapter SDXL head (the ViT-H encoder is SD1.5 Plus's).

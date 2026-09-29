# Probe: LoRA and ControlNet as QNN graph INPUTS (overnight 2026-09-29)

⭐ **RESUME HERE.** State file for unattended work; update the checklist after every step.
Goal (the user's): an SDXL template that is checkpoint-, LoRA- and ControlNet-agnostic,
swappable per render at near-QNN speed. Tonight is the SD1.5 evidence on this laptop
(15.6 GB RAM; SDXL authoring does not fit — needs Mr.J's workstation or a rented box).

## Rules for tonight (the user's, before sleeping)

- Scope: **Probe A and Probe B only.** No nightmare-mobile app/UI changes, no releases.
- Phone: headless or wired only (adb over Tailscale (address in `../LocalDream/CLAUDE.md`), armed 04:50).
  No app reinstall, no screen taps. Work under `/data/local/tmp/probe/`.
- Usage: `/x` about hourly. Near the 5 h session limit: `/bye`, then sleep until just after
  the reset and resume from this file. **No weekly cap.**
- Disk: WSL has room (61 GB used after deleting the approved dirs). C: ~15 GB free —
  WSL writes grow the vhdx on C:, so delete my own build intermediates as I go.
- Never stop: loop skill, self-paced.

## What is already known (do not re-measure)

- QNN "updatable tensors" LoRA: 2.8× slower per pass, a cliff (`LocalDream/docs/LORA-PROBE.md`).
- QNN ControlNet on SD1.5 works: 13 residual inputs cost ~nothing in memory traffic; a branch
  is ~45 ms/step; blocker for CN-agnostic = residual input u16 windows fitted to ONE branch,
  headroom 1.00–1.05× (`LocalDream/docs/CONTROLNET.md` §multi-ControlNet). Suggested fix:
  `--quantization_overrides` on the 13 inputs — **never tested**.
- npuforge: on-phone SDXL checkpoint conversion ~7 min; Illustrious 30 steps cfg 7 = 45.8 s.

## Plan

- [x] **A0: ✅ YES — `--quantization_overrides` sets a GRAPH INPUT's encoding.** Tiny ONNX
  (x → conv, + y → out), calibration y ∈ ±1, override y ∈ ±8. Plain arm: y scale 3.05e-5
  (±1). Override arm: y **min −8, max 8, scale 2.44e-4, offset −32768** in the quantized DLC
  (`qairt-dlc-info`) AND in the compiled context (`qnn-context-binary-utility` JSON). The
  override goes on `qairt-converter` (`--quantization_overrides`), format
  `{"activation_encodings": {"y": [{"bitwidth":16,"min":-8,"max":8,"dtype":"int"}]},
  "param_encodings": {}}`. Workdir WSL `~/probe/a0`. ⇒ CONTROLNET.md route 2 (widen the 13
  residual windows) is available; LoRA input arrays can get fixed known windows.
- [ ] **A1 — ⛔ PAUSED 05:30: the harness killed the build for low system memory** (Windows
  free 1.8 of 15.6 GB). The LoRA arm's `qairt-quantizer` grows the WSL VM toward its 10 GB
  cap; with Windows on top the laptop hits critical. Harness rule: do not restart without the
  user. ⭐ **To resume (user)**: `%USERPROFILE%\.wslconfig` has `memory=11GB` (tuned in
  August for this quantizer); with Claude Code's memory-pressure reaper watching background
  shells, 11 of 15.6 GB leaves Windows too little. Drop it to `memory=8GB` (the VM keeps its
  48 GB swap, so the quantizer pages instead of starving Windows; slower, not fatal),
  `wsl --shutdown`, then re-run
  `wsl -d Ubuntu -- bash ~/probe/scripts/build_a1.sh` — it resumes at "quantize lora"
  (ctl arm done: `out/unet_ctl.bin`; `unet_lora.dlc` converted). Then push `out/unet_lora.bin`
  + `calib/lora_*.raw`, write `list_lora.txt`, `sh prof2.sh /data/local/tmp/probe/r250 r250
  ctl lora lora ctl`, read with `exec.sh`.
- [x] **A1 — ✅ BUILT AND MEASURED 11:32: LoRA as graph inputs costs +33% per UNet pass**
  (160 targets, R=32, one input per target). NPU "Accelerator (execute)" time, qnn-net-run,
  Nightmare's 2.50 runtime, interleaved both orders: ctl 339 / 345 ms, lora 453 / 454 ms.
  Context 888.2 MB vs 881.6 MB (+6.6 MB). ⇒ misses the ≤15% gate, but it is NOT the
  updatable-tensor cliff (+184%). Projected (ratio only): QNN SDXL 20 steps 26.5 s → ~35 s
  with swappable LoRA, vs ggml 125 s+. The branch is ~4% of MACs, so most of the +33% is
  overhead (small-K matmuls, 24 per-head slices per attention, requantize/add ops) —
  `--profiling_level detailed` would name it; shared x·A for q/k/v is the obvious cut.
  Build lessons: 4 calibration rows (20 rows grew the quantizer past 23 GB); detached +
  Windows keepalive + C: guard (`~/probe/scripts` + `relaunch2.sh`).
- [ ] **A1 (original plan text):** SD1.5 512² UNet with rank-R LoRA branches whose A/B/scale are
  graph inputs (packed into a few tensors, sliced in-graph) vs the same UNet without —
  same toolchain, 20 calibration rows, `qnn-net-run` on the phone, per-pass ms.
  Pass = within ~15% of control (the updatable route was +184%).
- [x] **A1+A2 v2 — ✅ 13:22: LoRA as inputs at +15% time, merged-route accuracy.** Per-op
  profile (`--profiling_level detailed`, `op_breakdown.py`) of v1: of +213 M extra cycles, the
  ×S Mul on the dout-wide delta was **153 M (72%)**, Adds +47 M, k-path Transposes +15 M,
  branch MatMuls only 13 M. v2 (`export_a1.py`): S multiplies the rank-R intermediate —
  ((x·A)·S)·B — and the k delta is transposed once per projection and added channel-first
  before the original reshape. Measured (pass D, interleaved): ctl 358 ms, **v2 413 ms =
  +15%** (calibrated-only 414 ms: the range overrides are free). Accuracy (widened ×4 after a
  calibrated build, `chain_v2.sh`): gain 0.985, cosine 0.990, delta SNR 16.8 dB, output vs fp32
  32.8 dB — same as v1-fixed (outputs differ, mean |Δ| 2.5e-4). Contexts on the phone:
  `unet_lora_v2.bin` (use this), `unet_lora_v2cal.bin`.
- [x] **A2 — ✅ SOLVED 12:37: with the branch intermediates widened ×4, LoRA-as-inputs is AS
  ACCURATE AS the merged-weights route (better, on the delta).** Row 168, camera LoRA,
  NPU vs ORT fp32 on the same ONNX:

  | on the NPU | gain | cosine | delta SNR | output vs fp32-with-LoRA |
  |---|---|---|---|---|
  | merged weights (npuforge's route, same toolchain) | 1.007 | 0.953 | 10.3 dB | 32.6 dB |
  | **LoRA as inputs, branch ranges ×4 (`gen_overrides.py`)** | **0.985** | **0.990** | **16.8 dB** | **32.8 dB** |
  | LoRA as inputs, calibrated ranges only | 0.514 | 0.904 | 4.8 dB | 26.2 dB |

  Speed after the fix (pass C, interleaved): ctl 335 ms, lora 415 ms = **+24%** (the first
  build measured +33%; call it +24–33% until repeated). 480 overrides (x·A, ·B, ·S × 160).
  History of the half-strength result below.
- [x] **A2 (first measurement) — ⚠ the LoRA acted in the RIGHT DIRECTION at ~HALF STRENGTH.**
  Row 168 (t=495, not a calibration row), camera-style LoRA (not the calibration LoRA:
  Colorwater), 4-row calibration. ORT fp32 on the same ONNX: S=0 is an exact no-op; the LoRA
  moves the output 7.4%. NPU vs fp32: ctl 39.0 dB, LoRA-graph S=0 33.4 dB, LoRA on 26.2 dB;
  **LoRA delta cosine 0.904, magnitude 0.51×** (noise floor between the two NPU graphs 0.25×
  the delta). Strength sweep, NPU delta ÷ fp32 ×1 delta: ×1 0.51 · ×2 1.30 · ×4 2.98 → gain
  0.51 → 0.65 → 0.75 of ideal = a **dead-zone signature** (a fixed rounding loss that bigger
  deltas outgrow). Falsified: "an A16×A16 MatMul operand drops to 8-bit" — the 199 uFxp_8
  tensors are GroupNorm gammas (ctl has 197). Next: per-layer dump of ONE target
  (`qnn-net-run --debug` vs ORT intermediates) to find the op that rounds the branch away;
  candidates: the ·S multiply's output encoding, the delta add into a per-head slice, the
  4-row calibration of the branch intermediates (calibrated on Colorwater, whose S is 4–10×
  camera's).
- [ ] **A2 (original plan text):** feed a real LoRA's A/B; compare one UNet pass against ORT fp32 on
  the merged-weight ONNX (and the zero-LoRA pass against ORT base) — LoRA delta direction
  and SNR.
- [ ] **B1 — in progress 13:35:** SD1.5 UNet with 13 residual inputs (`export_cn.py`, NHWC
  like DreamUI's), residual windows = **2 × the cross-type max** (`windows.json`), calibration
  rows mixing canny/depth/openpose residuals, then the residual Adds widened ×2
  (`build_cn.sh`, two builds; log `~/probe/b1/build_cn.log`). ⭐ **Measured (GPU fp16,
  `residual_stats.py`, 3 types × 6 timesteps × 3 latents, scale 1, synthetic type-correct
  hints): residual magnitudes differ by up to 7.7× between ControlNet types at the same
  injection point** — res 2: canny 11.7 vs openpose 2.3; res 5: depth 66.9 / openpose 64.3 vs
  canny 10.8; mid: depth 43.0 vs canny 16.5. ⇒ a UNet whose residual windows are fitted to one
  type (DreamUI's openpose) clips another type ~5×: this is WHY it was not agnostic.
  Windows chosen: [2.8, 5.5, 23.4, 19.0, 18.6, 133.8, 28.0, 15.0, 26.7, 25.6, 43.5, 29.3, 85.9].
- [x] **B1+B2 — ✅ 14:08: ONE NPU UNet reproduces three ControlNet types.** Row 168, residuals
  from canny / depth / openpose (GPU fp16, synthetic type-correct hints) through the same
  control-UNet (`unet_cn_cal.bin`, 4 calibration rows mixing the types, residual windows = 2 ×
  cross-type max); control delta = out(type) − out(none), NPU vs ORT fp32:
  canny gain 1.015 / cosine 0.997 / 22.3 dB · depth 0.981 / 0.989 / 16.6 dB · openpose 1.000 /
  0.987 / 15.9 dB; controlled outputs 33.3–34.3 dB vs fp32; no-control 33.5 dB. Widening the
  13 residual Adds ×2 changed nothing (`unet_cn_wide.bin`, identical) — the input windows are
  what matter. (Not built: the one-type-window control that would show the clipping; the
  measured 5–7× cross-type spread already predicts it.)
- ⚠ 14:00 — **C: hit 0.7 GB free**: the B1 quantize swapped heavily (WSL swap vhdx lives on C:)
  and my C: guard only watched `build_a1.sh`. `wsl --shutdown` released ~23 GB; deleted the
  ControlNet HF cache (4 GB, residuals already made). ⇒ **build-agnostic Windows-side guard**
  `cguard.ps1` (hidden process: C: < 6 GB → `wsl --shutdown`), and `wsl --shutdown` between
  builds to release swap. The queued rank-64 chain was killed; folded into the combined build.
- [ ] **B2 — staged (original):** row 168 residuals from canny, depth, openpose (+ zeros) through ONE
  NPU control-UNet; per type, control delta vs ORT fp32 (`b2_ref.py`, `b2_phone.sh`,
  `b2_cmp.py`). Pass = gain ≈ 1 and high cosine for all three types. (ControlNet conversion
  per type is DreamUI-proven for openpose; a phone-side ControlNet template = npuforge recipe
  work, not a probe question.)
- [ ] Write-up: findings into `npuforge/docs` (new doc) + `nightmare-mobile/docs/SDXL-GGML.md`
  pointer; HANDOFF lines in both.

## Log

- 04:50 — WSL cleaned (−42 GB), adb over Tailscale armed, plan written.
- 04:55 — A0 passed (see checklist). User: "DreamUI's ControlNet is not agnostic, plan for a
  template" — B is the ControlNet TEMPLATE (graph + recipe so any CN converts on the phone)
  plus residual windows sized for any branch (A0's mechanism). Starting A1.
- 04:55 — A1 build started (WSL `~/probe/a1/build.log`; sources now in WSL
  `~/probe/scripts/`: `export_a1.py` (torch hooks inject LoRA from packed inputs lora_A/lora_B/lora_S,
  160 targets rank 32), `mk_calib.py`, `build_a1.sh` (resumable: skips finished stages),
  `run_a1.sh` (phone timing), `watch.sh`). Phone runtime pushed: `/data/local/tmp/probe/`
  (qnn-net-run + QAIRT 2.49 HTP libs + v79 skel). ⚠ Git Bash mangles `/mnt/c/...` passed
  to `wsl`: prefix `MSYS_NO_PATHCONV=1`.
- 05:25 — ctl context built (881.6 MB), runs on the phone. ⚠ **Timing method**: wall time of
  qnn-net-run is useless (float→u16 input conversion per inference + ±1.5 s load noise, and
  the LoRA arm feeds 48 MB of inputs); use `--profiling_level basic` + host
  `qnn-profile-viewer`, "Accelerator (execute) time" (`~/probe/scripts/prof2.sh`, `exec.sh`). Android
  `date` has no %N — time with /proc/uptime. ⚠ **qnn-net-run reads ~3× the app's per-pass**
  for ANY context: the shipped npuforge DreamShaper UNet (`~/ctxinfo/out_base_unet.bin`,
  QAIRT 2.50) = 265–288 ms/pass here; our qairt-2.49 ctl = 343–386 ms. Only ctl-vs-lora
  (same toolchain, same runner, interleaved) is the A1 verdict. Nightmare's 2.50 runtime
  pushed to `/data/local/tmp/probe/r250/` (runs 2.49 and 2.50 contexts).

## Design draft — one SDXL template: checkpoint + LoRA + ControlNet + inpaint (not built)

Written 05:40 while A1 waits on memory. Every row cites its evidence; ❓ = unmeasured.

| attach point | graph (authored once, workstation) | phone, per use | evidence |
|---|---|---|---|
| checkpoint | npuforge SDXL template + recipe (exists) | tplconv + on-phone compile, ~7 min, once per checkpoint | npuforge `docs/SDXL.md`, RESULTS |
| LoRA | rank-R branches on attn q,k,v,out (both attns) + FF in/out, A/B/S as 3 packed graph inputs (`a1/export_a1.py` pattern; q/k/v delta computed once, sliced per head) | CPU: kohya `lora_down/up` → pad/SVD to R, normalise to ±1, fold scales into S; write inputs; **no recompile** | A0 ✅ (input windows settable); A1 ❓ latency; A2 ❓ numerics |
| ControlNet | UNet with 13 residual inputs, windows **overridden** wide (A0), zeros = no control; plus a *ControlNet template* (graph + recipe) so any CN `.safetensors` converts like a checkpoint | load that CN's context; ~+45 ms/step on SD1.5 | DreamUI `CONTROLNET.md` (16-input UNet ≈ free; branch cost); A0 |
| inpaint | 9-channel `conv_in` | plain checkpoint: zero extra channels (≡ txt2img); inpaint ckpt: its weights; any ckpt: add-difference at conversion | npuforge SD15-INPAINT (measured), SDXL-INPAINT-TEMPLATE runbook |

Open design choices for the user (not decided):
1. **Rank R and target set.** R=32 on 10 targets/block covers typical character/style LoRAs;
   conv (LoCon) and proj_in/out layers are dropped — a LoRA trained there loses that part.
   Measure what fraction of a real LoRA's delta norm those layers carry before choosing.
2. **LoRA stacking**: two LoRAs = concatenate along rank (R_total ≤ R) — free if R has room.
3. **Residual windows**: set from measured stats of several CN types (canny, depth, pose,
   tile) × 1.5–2 headroom, instead of one branch's calibration (1.00–1.05× today).
4. **Text-encoder LoRA**: CLIP runs in MNN on CPU in npuforge exports — merge at load (cheap)
   rather than on the NPU.
5. **Resolution**: a template is one shape; 1024² first, portrait buckets later.

What only a workstation can do: author the SDXL graph once (export with the LoRA inputs,
residual inputs, 9-ch conv_in; 30-row calibration as npuforge did; overrides for the new
inputs). Mr.J's host or a rented box. Everything after that runs on the phone.

### ✅ Design choice 1 measured (05:50, `~/probe/scripts/lora_energy.py`, CPU only)

Share of each LoRA's UNet ||ΔW||² (ΔW = alpha/r · U·D, via QR cores — exact, no ΔW formed):

| LoRA | attn | ff | proj | resnet | rank-32 keeps of attn+ff |
|---|---|---|---|---|---|
| Ayaka Illustrious (SDXL character, r32) | 36.7% | 62.3% | 1.0% | — | 100% |
| pixel-art-xl (SDXL style, r32) | 20.1% | 79.3% | 0.6% | — | 100% |
| Colorwater v4 (SD1.5, r128) | 31.4% | 62.8% | 5.8% | — | 94.6% |
| camera-style (SD1.5, r64) | 9.3% | 87.9% | 2.8% | — | 99.6% |
| LCM-LoRA SDXL (distillation, r64) | 9.1% | 69.3% | 1.7% | 18.1% | 99.8% |

⇒ attn + FF at rank 32 covers **94–99%** of a character/style LoRA's weight change;
**FF dominates (62–88%)** — an attention-only input scheme would miss most of it. Only a
distillation LoRA (LCM) leans on ResNet convs (18%). proj_in/out ≤ 5.8% — optional. Text
encoder parts exist (Ayaka: 792 tensors) but are CPU-side (MNN), not the NPU's problem.
⚠ Energy share is a proxy for effect, not a render comparison.

### ✅ The LoRA packer, and the input windows it implies (06:00, `~/probe/scripts/pack_lora.py`)

`pack_lora.py <targets.json> <lora> [strength] [S_max] [out_dir]` turns a kohya LoRA into
`lora_A/B/S.raw` for the A1 graph (160/160 SD1.5 targets matched by name; exact SVD
truncation for r > R; negative strength folded into B's sign so S ≥ 0). Measured:

| | camera-style (r64) | Colorwater (r128) |
|---|---|---|
| S range | 0.0012 – 0.0199 | 0.0088 – 0.072 |
| rank-truncation error at **R=32**, median / max per target | 9.9% / 26% | 17% / 50% |
| at **R=64** | **2e-7 (exact)** | 9.1% / 25% |
| u16 quantisation alone, S window ±4 | 0.31% / 2.2% | — |
| u16 quantisation alone, S window **[0, 0.25]** | **0.02% / 0.1%** | — |

⇒ (1) **lora_S window [0, 0.25], not ±4** — 14× less error; A1's build used ±4 (fine for a
latency test, wrong for A2/product). (2) **R=64** reproduces every LoRA up to rank 64
exactly (the common 16/32/64) and halves the r128 loss; energy share understates this —
the per-target error concentrates in few layers. Cost: SD1.5 inputs 10.4 M + 13.6 M floats
(written once per LoRA change: QNN input buffers persist across executes, DreamUI
`CONTROLNET.md`); branch MACs ≈ R·(din+dout) ≈ 6–10% of each Linear at R=64 — A1 measures
what that costs in time. (3) Calibration (A2) must feed REAL packed LoRAs so the branch
activations (x·A, ·B, ·S) get honest encodings.

### SDXL sizes for the template (07:15, `~/probe/scripts/sdxl_sizes.py`, shapes only)

From an SDXL kohya LoRA's tensor shapes: **700 attn+FF targets** (70 transformer blocks × 10).
LoRA inputs as u16: **R=32 → 160 MiB, R=64 → 319 MiB** (written once per LoRA change, not per
step). Branch MACs vs those Linears at 1024²: +4.2% (R=32) / +8.4% (R=64) — an upper bound
(attn2 k/v really see 77 tokens). Memory is affordable next to a ~2.6 GB W8A16 UNet on a 12 GB
phone; whether the MACs cost ~their share in TIME is exactly A1's question.
- 07:15 — session reset; A1 still paused on the user (memory). Waiting.
- 10:04 — user said go: `.wslconfig` memory 11→8 GB, `wsl --shutdown`, build relaunched
  DETACHED inside WSL (`setsid nohup`, survives with no wsl.exe attached). LoRA arm quantized
  10:04–10:14 with no memory trouble.
- 10:16 — ⚠ **LoRA context compile FAILED — packed inputs cannot be sliced on HTP**:
  `q::*InputSlice … not sufficiently tiled to fit in TCM. Requires 10358784 bytes` — a Slice
  of a graph INPUT must hold the whole input in VTCM (8 MB); lora_A (5.18 M × u16 = 10.4 MB)
  and lora_B (13.6 MB) do not fit. ⇒ **Design rule: one graph input per LoRA target**
  (largest single A/B ≈ 330 KB at SD1.5 R=32; SDXL R=64: max 1280×… still ≪ 8 MB), only tiny
  tensors (lora_S, 640 B) may stay packed. Rebuilt with 160 `la_i` + 160 `lb_i` + `lora_S`
  (window now [0, 0.25]), calibrated on a REAL packed LoRA (Colorwater) — A2 tests a
  different one (camera-style) so the result is not in-distribution.
- 10:22 — build killed again: the Claude memory reaper killed the MONITOR's wsl.exe (the last
  WSL session) → WSL stopped the idle distro ~60 s later, taking the setsid'd build with it.
  ⇒ `setsid` does not survive a distro stop; hold WSL up with a keepalive started as a plain
  Windows process: `Start-Process -WindowStyle Hidden wsl.exe "-d Ubuntu -- sleep 10800"`.
- 10:25 — ⛔ stopped it myself: the per-target LoRA quantize needs > 18 GB (8 GB RAM + swap
  10.6 GB and climbing); WSL swap is a vhdx on C:, and C: fell 14 → 3.8 GB (ext4.vhdx also grew
  ~9 GB this rebuild — the 42 GB deleted overnight is not reclaimed until the vhdx is
  compacted). `wsl --shutdown` restored C: to 14.2 GB. **Needs the user (admin):**
  a `diskpart` compact of the WSL disk image (after `wsl --shutdown`), ≈ +40 GB.
  Then relaunch with the keepalive and a C:-free-space guard. State: ctl context done;
  `lora/model.onnx` + `unet_lora.dlc` (per-target inputs) done; quantize is next.
- 11:52 — ✅ **micro-graph: the LoRA branch itself is EXACT on HTP.** One branch
  `y + ((x·A)·B)·S`, target-0 shapes (64×64×320, R=32), calibrated on Colorwater's real
  la_0/lb_0/S, run with the camera LoRA's (S 5× smaller): NPU delta vs numpy **gain 1.000,
  cosine 1.0000**; S=0 reproduces y within one output step. ⇒ A2's 0.51 gain is NOT the
  branch ops or their ranges. Leading hypothesis: the quantized base graph (residual-stream
  adds span ±100s, so a LoRA-sized nudge below one 16-bit step rounds away) — which would hit
  npuforge's shipped MERGED LoRA identically. ⇒ fair control building: camera LoRA merged
  into the weights (`export_a1.py mrg`, MERGE_DIR = the packed camera LoRA, exact same delta),
  same toolchain + 4 base rows (`build_mrg.sh`, log `~/probe/a1/build_mrg.log`). Verdict rule:
  if merged also ≈ 0.5 of fp32, input-LoRA is as good as the shipped route.
- 12:05 — A2 diagnosed, step by step (fp32 per-group runs, least squares on the NPU delta):
  **merged-weights control** (same camera delta baked into the weights, same toolchain):
  gain 1.007, cosine 0.953 ⇒ the quantized base graph is NOT the loss. Per group (NPU inputs
  graph): q 1.01, k 0.91, attn1.v 1.11, **attn2.v 0.42**, out 0.98, ff 0.99 — attn2.v carries
  0.40 of the camera LoRA's effect, hence the overall 0.51.
- 12:25 — **root cause: the start-token outlier clips LoRA-dependent intermediate ranges.**
  Micro-graph of the cross-attn v path (real text embedding, per-head convs): every token
  exact except **token 0 (CLIP BOS, the attention sink) at 0.70**; slicing B per head instead
  changes nothing. Camera LoRA token-0 values vs compiled (Colorwater-calibrated) ranges:
  x·A −72.4 vs [−50.9, 25.2]; (x·A)·B [−62.9, 74.3] vs [−47.1, 50.2] ⇒ clipped. A and B are
  normalised per LoRA, so these intermediate ranges depend on WHICH LoRA is loaded — a
  template must not. ⇒ **Design rule: override every branch intermediate (x·A, ·B, ·S) wide**
  (×4 the calibrated range, 2 of 16 bits); later, a LoRA-agnostic bound (unit-norm A columns ⇒
  |x·a| ≤ ‖x‖). Rebuilding with the ×4 overrides.
- 14:06 — ⭐ **User confirmed the end goal (2026-09-29): an SD1.5 template for on-phone
  checkpoint conversion where LoRA and ControlNet are swappable.** Next build is the COMBINED
  template graph: rank-64 LoRA inputs (320 + lora_S, v2 structure) + 13 ControlNet residual
  inputs, one graph (`export_a1.py combo 64`, `build_combo.sh`, log
  `~/probe/combo/build_combo.log`; calibration = rank-64 Colorwater LoRA + mixed-type
  residuals on 4 rows; then LoRA intermediates ×4). Phone test (`combo_phone.sh`,
  `combo_ref.py`, `combo_cmp.py`), row 168: none / lora (camera R64) / canny / both.
  Remaining after it: (1) the template through npuforge's recipe pipeline (tpl_recipe.py
  discover → recipe.bin/tpl_trim.pack/libqnn_model.so) and a SECOND checkpoint converted on the
  phone with LoRA + ControlNet swapped at render time; (2) runtime: the backend's 3-input check,
  a phone-side LoRA packer (port of pack_lora.py), residual feeding (DreamUI's
  executeControlUnetGraphs pattern); (3) ControlNet conversion per type on the phone (a
  ControlNet template = the same recipe trick).
- 14:15 — **Capstone design (from TEMPLATE-AUTHORING.md §3, PIPELINE.md):** npuforge templates
  come from the LEGACY `qnn-onnx-converter` (model.cpp + model.bin) → `tpl_patch.py` (2,383
  weight sites → pack reads) → `tpl_recipe.py` (numeric fingerprint mapping) →
  `qnn-model-lib-generator` → pack-loading `libqnn_model.so`. The LoRA and residual additions
  are graph INPUTS with NO weights, so the recipe/pack machinery should be untouched (same
  weight sites; only inputs and ops added) — to verify: site count and discovery numbers must
  equal the stock SD1.5 template's (2,383 / 1,358 matched / 0 ambiguous). ⭐ Calibration
  shortcut: PIPELINE Test 1 proves BORROWED activation ranges work ("719 overrides landed") —
  take the shipped SD1.5 template's 400-row encodings for every base tensor + our overrides for
  the new ones (la/lb/S windows, x·A/·S/·B ×4, res windows), so only the new tensors need
  data: production base quality (the 39 dB class, not the 4-row 33 dB) without the 2 h 20 m
  quantize. Then the capstone test: a SECOND checkpoint converted by tplconv on the phone,
  LoRA + ControlNet fed at render time.
- 14:45 — ✅ **COMBINED TEMPLATE PROBE PASSES: rank-64 LoRA + ControlNet in ONE NPU UNet**
  (`unet_combo_wide.bin`, 337 inputs; 4-row calibration with the rank-64 Colorwater LoRA +
  mixed-type residuals, LoRA intermediates ×4, residual windows 2× cross-type max). Row 168,
  camera LoRA (R64, exact) and canny residuals, NPU vs ORT fp32:

  | case | effect | gain | cosine | delta SNR | output vs fp32 |
  |---|---|---|---|---|---|
  | LoRA only | 7.4% | 0.989 | 0.990 | 16.8 dB | 32.7 dB |
  | canny only | 20.2% | 1.015 | 0.997 | 22.1 dB | 34.1 dB |
  | **LoRA + canny** | 22.7% | **1.011** | **0.997** | **21.9 dB** | **33.3 dB** |
  | base | — | — | — | — | 33.3 dB |

  fp32 interaction |both − (lora + canny)| / |both| = 0.367 — the NPU tracks it. Calibrated-only
  build: LoRA gain 0.760 (re-confirms the ×4 widening). **Speed: 474 ms vs ctl 377 ms = +26%**
  (R=32 LoRA alone was +15%; R=64 + residuals adds ~11 points). Base 33.3 dB is the 4-row
  calibration.
- 14:45 — ✗ **Borrowing the 400-row encodings by name is unsafe** (`cap/name_match.py`): of our
  node outputs, 773 match p0's `model_net.json` by name AND shape, 2,299 match by name with a
  DIFFERENT shape (the ONNX export's op counters shift once LoRA/residual ops are inserted, so
  the same name is a different tensor), 8,262 not found. The capstone calibrates itself on 4
  rows; production calibration (more rows) is a later, separate step.
- 14:48 — **Capstone running** (`~/cap/run_cap.sh`, log `~/cap/cap.log`, state `~/cap/state`):
  legacy `qnn-onnx-converter` (QAIRT 2.50) on `~/probe/combo/combo/model.onnx`, 4 rows spanning
  t = 999/4/846/724 (the npuforge time_proj gate needs t≈999) with the Colorwater R64 LoRA and
  zero/canny/depth/openpose residuals, all 814 overrides; gate also checks the overrides landed
  → stock arm vs tpl identity arm (md5) → discover (DreamShaper) + finalize + round trip →
  apply **SD1.5 base (v1-5-pruned-emaonly)** as the second checkpoint → bundle + aarch64 lib →
  fp32 ref for the second checkpoint (re-export with its weights, `cap_ref.py`, held-out row
  168 + camera LoRA + canny).
- 15:35 — ⭐ **CAPSTONE PASSES END TO END: an SD1.5 template with swappable LoRA + ControlNet,
  and the PHONE converts a new checkpoint into it by itself.** Scripts in the session
  scratchpad `cap/` (`run_cap.sh`, `onphone.sh`, `push_cap.ps1`); WSL `~/cap` keeps `keep/`,
  `lib_tpl/`, `identity.pack`, `bundle/`, `ref15_*.raw`.

  | gate | result |
  |---|---|
  | encodings (4 rows, t 999/4/846/724) | time_proj reaches 999, Sin/Cos ±1, 0 dead; **798/814 overrides landed** (16 misses: attn1 `MatMul_3` — the k-branch ·B, renamed by the converter; they keep calibrated ranges) |
  | Phase 2: tpl identity pack vs stock | **2,383 sites, byte-identical contexts** (md5 equal, 0 differing bytes) — the same count as the stock SD1.5 template |
  | Phase 1: discover (DreamShaper) | **matched 1,358, unmatched 1,025, ambiguous 0** — identical to stock; the LoRA/residual inputs add no weight sites |
  | round trip (DreamShaper) | 0 scale mismatches, 243 byte mismatches, all bias (the stock control's signature) |
  | second checkpoint (SD1.5 base) | 1,358 byte mismatches = weight 914, bias 295, MatMul 64, permute 24, other 61 — exactly PIPELINE Phase 3's breakdown |
  | native tplconv (host) from the 44 KB trimmed pack | **byte-identical** to Python (`5ca5cb32…`), 42 s, 3.6 GB RSS |
  | **phone** tplconv | same md5 `5ca5cb32…`, **36 s** |
  | **phone** context compile (2.50 generator + pack-loading lib) | rc 0, **191 s** (the plain template: 93 s), 885,977,216 B |
  | phone-built vs PC-built context, 4 cases | **max abs diff 0.0** |

  Accuracy, row 168 (held out), camera LoRA R64 (held out; calibration used Colorwater), NPU vs
  ORT fp32 of the same checkpoint:

  | build | base | LoRA gain / cos / δSNR | canny gain / cos / δSNR | both gain / cos / δSNR |
  |---|---|---|---|---|
  | DreamShaper (template's own) | 35.3 dB | 0.987 / 0.989 / 16.6 | 1.013 / 0.997 / 22.1 | 1.009 / 0.997 / 21.9 |
  | **SD1.5 base via recipe (phone-built)** | **38.1 dB** | **1.004 / 0.992 / 18.0** | **1.006 / 0.998 / 24.7** | **1.006 / 0.998 / 24.9** |
  | (DLC combo probe, for scale) | 33.3 dB | 0.989 / 0.990 / 16.8 | 1.015 / 0.997 / 22.1 | 1.011 / 0.997 / 21.9 |

  ⚠ The legacy converter re-quantizes the LoRA A/B inputs to **8-bit** before their MatMul
  (`…_la_i_reshape_converted_UFIXED_POINT_8`, range ±1). It costs nothing measurable (LoRA δSNR
  16.6 vs 16.8 dB for the 16-bit DLC build) and probably explains the speed below.
  **Speed** (qnn-net-run accelerator time, 21 inferences, burst, noisy ±40 ms between passes):
  min 303–343 ms vs npuforge's reference template 272–277 ms (≈ +10–15%) — the DLC combo build
  was min 397 ms. IPS is lower (2.1 vs 3.1) because qnn-net-run converts the 24M LoRA floats on
  the CPU every inference; a runtime quantizes them ONCE per LoRA and reuses the buffers.
- 16:00 — ⭐ **Real renders through Nightmare's backend** (patch `015` in nightmare-mobile
  `backend-patches/`: ≥ 3 UNet inputs, extras bound by name, re-quantized only when changed;
  per request: `lora_dir`, `lora_strength`, `controlnet`, `control_image` — one launch swapped
  them over six requests with 6/6 pixel hashes exact). Model folder =
  `DreamShaper_8_pruned_cs1`'s CLIP/VAE + the DreamShaper capstone UNet (identity pack), 20 steps
  512², seed 42, S25 Ultra:

  | run | 20-step sample | UNet call |
  |---|---|---|
  | stock DreamShaper (3 inputs) | 3.2 s | 70 ms |
  | template, no LoRA | 3.7 s | 82 ms |
  | + Colorwater LoRA 0.5 / 1.0 | 3.8–4.0 s | 86–96 ms |
  | + AI Hub canny ControlNet | 6.1 s | — |
  | + both | 6.7 s | — |

  Stock vs template-no-LoRA: same composition, 24.1 dB (the 4-row calibration). LoRA strength
  visibly scales (0.5 painterly, 1.0 watercolor). Canny follows a hand-drawn house exactly and
  stacks with the LoRA. A no-control render after a controlled one is byte-identical to the
  first (zero residuals = identity); the stock model's pixel hash is unchanged by the patch.
  Images: session scratchpad `cap/png/grid.png`, `grid_cn.png`.
- 16:20 — ✅ **Phone-side LoRA packer**: nightmare-mobile `TemplateLora.kt` — byte-identical to
  `pack_lora.py` on the camera LoRA (321 files); rank > 64 (Colorwater is rank 128, so `cw64`
  was always a truncation) via thin QR + Jacobi SVD, delta within 1e-3 of torch's; 16 s on the
  PC, sequential. Needs `targets.json` shipped beside every template UNet (as
  `lora_targets.json`): the target order is the export's hook order.
  Remaining: the app wiring, ControlNet contexts per type (only AI Hub
  canny is on hand; the template accepts any SD1.5 ControlNet's residuals), the ControlNet itself
  as a template, CLIP/VAE for a phone-converted checkpoint, production calibration.

# Checkpoint families — how many templates does SD1.5 actually need?

`ROADMAP.md` P2 assumes the answer is **two**: the photoreal template that exists
and an anime one to build, with the app choosing between them by weight-span
ratio. That assumes the SD1.5 population is two clusters. This file tests the
assumption, and answers the two questions that hang off it: which checkpoint's
CLIP and VAE should be the generic default, and what a second **resolution**
would cost.

Measured 2026-09-14 on the host PC. Nothing here needed the phone.

## Summary

| question | answer |
|---|---|
| Do photoreal checkpoints share a root? | Yes — base SD1.5 itself. CyberRealistic is a **7.7% median** perturbation of it and its weight spans never exceed **×1.22** |
| Do anime checkpoints share a root? | Yes, but **not Pony** — Pony v6 is SDXL. The SD1.5 anime root is the leaked **NovelAI** model, via Anything V3 |
| Are two templates enough? | **Unknown, and two samples cannot say.** But the population is a *continuum* between the two roots, not two clusters, so a 2-template design needs a fallback, not just a switch |
| Is "anime" even the failing category? | ⚠ **Doubtful.** ReV Animated — anime-lineage, 2.5D — profiles like a photoreal model (max span ratio **1.27**). MistoonAnime's 48 may be one merge, not a family |
| Cheapest way to find out | **0.9 MB of a checkpoint decides it** (`tools/span_probe.py`), and it can be read over HTTP ranges without downloading the file — a 50-checkpoint survey is an afternoon, request-bound |
| Generic CLIP | **Stock SD1.5's** (CLIP ViT-L/14). Both checkpoints measured sit within **0.2–0.4%** of it |
| Generic VAE | **`vae-ft-mse-840000-ema-pruned`**. CyberRealistic's baked VAE *is* it, to 0.021% |
| Another resolution | ~10 MB more bundle, a **~3 h PC rebuild**, and 1.6×–2.7× the arithmetic. The blocker is the **compile's 4.8 GB peak**, not the graph |

⭐ The largest finding is none of those: **activation ranges are the only thing a
template really carries, they are 16-bit, and headroom in a 16-bit range is
nearly free.** That points at a cheaper fix than more templates — §6.

## 1. Method

`tools/span_probe.py`, added by this work, and the reference table it builds.

```sh
# the reference: base SD1.5, v1-5-pruned-emaonly-fp16.safetensors, 2,132,696,762 B
#   huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive
python tools/span_probe.py dump v1-5-pruned-emaonly-fp16.safetensors sd15.json
python tools/span_probe.py probe sd15.json <ckpt.safetensors|URL> "<name>"
```

Two quantities are used throughout:

- **span ratio** — per tensor, the checkpoint's `max|w|` over the reference's.
  This is the number `docs/LIMITS.md` reports as 1.00 / 1.02 / 1.117 / 1.061 for
  the checkpoints that convert and **49.4** for the one that does not.
- **relative difference** — per tensor, `‖a−b‖₂ / ‖a‖₂`, reported as a median
  over a component (UNet, VAE, text encoder).

⚠ **The reference is a choice, and the numbers only mean something against it.**
`LIMITS.md`'s ratios are against *DreamShaper 8*, the template's own checkpoint.
This file uses **base SD1.5**, because every SD1.5 checkpoint is a perturbation
of it, so the ratio reads as distance from the common ancestor instead of
distance from whichever finetune the template happened to be built on. The two
references agree closely where both exist:

| checkpoint | max span ratio vs **DreamShaper 8** | max span ratio vs **base SD1.5** |
|---|---|---|
| CyberRealistic | 1.117 | **1.22** |
| MistoonAnime v3.0 | 49.4 | **48.26** |

⚠ Only **two** checkpoints were available locally. Every conclusion below that
rests on them is labelled as resting on two points.

## 2. The population: two roots, and a populated space between them

This section is the **community record, not a measurement here** — it is what
decides which checkpoints are worth probing, and the probing is what settles
anything. Civitai, where the download counts live, returns HTTP 451 from this
region, so there is no authoritative popularity ranking in this file.

**The photoreal branch has no separate root: it *is* base SD1.5.** The early
finetunes everyone merged (F222, Analog Diffusion, Protogen, HassanBlend) were
trained directly on SD1.5, and the models people name today — Realistic Vision,
epiCRealism, CyberRealistic, AbsoluteReality, Photon, DreamShaper — are merges
and re-finetunes of that pool. That is why they stay numerically close to base
(§3): nothing in their history ever left it.

**The anime branch has a different root, and it is not Pony.** Pony Diffusion v6
is **SDXL**; only v4 and earlier were SD1.5, via an early Waifu Diffusion
checkpoint. The SD1.5 anime population descends instead from **NAI Diffusion**,
the NovelAI model leaked in October 2022, through **Anything V3**, and from there
into the OrangeMix / Counterfeit / MeinaMix / AnyLoRA families. Waifu Diffusion
(SD1.4 + Danbooru) is a second, much less influential root.

**The two branches are merged together constantly, which is the part that breaks
a two-cluster picture.** ChilloutMix — the model that made the "realistic Asian
portrait" style — is basil_mix (anime lineage) merged with photoreal models, and
majicMIX, BRA and the kawaiiRealistic family follow it. A model like ReV Animated
is anime-lineage weights aimed at a 2.5D look. MistoonAnime's own author
describes it as "created using a lot of different models" plus custom LoRAs.
Nothing enforces a gap in the middle, and models are deliberately placed there.

Two consequences for the template plan:

1. A photoreal/anime switch has to decide what to do with a checkpoint that is
   genuinely between the two. Reporting the measured number and letting the user
   proceed is honest; silently picking the nearer template is a guess.
2. Distillations (LCM, Hyper-SD, Turbo variants of DreamShaper and Realistic
   Vision) are *not* a third family for this purpose — they are finetunes of
   models already in the population. What they change is the sampler and CFG at
   generation time, which is the generator's problem, not the converter's.

## 3. Where the two measured checkpoints actually sit

Median relative difference against **base SD1.5**, per component:

| | UNet | text encoder | VAE |
|---|---|---|---|
| CyberRealistic | **7.68%** (p90 23.1%, max 52%) | **0.204%** (max 1.07%) | 9.47% |
| MistoonAnime v3.0 | **123.5%** (p90 203.5%, max 2354%) | **0.363%** (p90 3.70%, max 4.55%) | 207.4% |

⭐ A photoreal finetune is a small perturbation of base SD1.5. An anime merge is
**a different UNet** — a 123% median relative difference means the difference is
larger than the thing itself, tensor by tensor. That is the quantitative form of
"two roots".

⭐ **The text encoder barely moves in either case**, including the anime one.
That is what makes a borrowed CLIP safe, and it is measured against base rather
than against the template — see §5.

Span ratios against base SD1.5, over all 686 UNet tensors:

| | p50 | p90 | p99 | max | >2× | >5× | >10× |
|---|---|---|---|---|---|---|---|
| CyberRealistic | 0.994 | 1.043 | 1.094 | **1.22** | 0 | 0 | 0 |
| MistoonAnime v3.0 | 0.579 | 2.846 | 9.879 | **48.26** | 91 | 30 | 7 |

⚠ **The max is a one-tensor statistic and it is fragile.** MistoonAnime's 48.26
is driven by a single tensor, `output_blocks.2.0.skip_connection.weight` (base
0.110 → 5.414). A checkpoint with one odd tensor and an otherwise ordinary
profile would score the same, and a checkpoint that is broadly 3× out would score
lower. The distribution separates the two far more robustly than the max does:
**91 of 686 tensors above 2×** against CyberRealistic's **0**. If a gate is ever
built, gate on a count or a percentile, not on the maximum.

⚠ Note also that MistoonAnime's median span is **0.58** — most of its weights are
*smaller* than base, not larger. "Anime checkpoints have bigger weights" is the
wrong summary; "anime checkpoints have a different distribution, with a long
upper tail" is the right one.

⚠ **This is not about absolute magnitude.** MistoonAnime's largest UNet weight is
**7.20** against base SD1.5's **5.30** — the same order. Its *median* span is
**0.20** against base's **0.41**, so its weights are globally about half the
scale, with a long upper tail. Whatever breaks it, it is not an overflowed UNet:
its UNet contains **zero** non-finite values (the 516 are all in the VAE, §5).

## 3a. ⭐ The space between the two roots is smoothly populated

The obvious objection to a 2-template design is that community merges put models
*between* the roots. That is testable without finding such a model, because a
merge **is** a weighted sum of weights — so the sum can just be computed. Linear
blends of the two local checkpoints, `(1−a)·CyberRealistic + a·MistoonAnime`,
span ratio against base SD1.5 over the 404-tensor probe set:

| a | 0.00 | 0.10 | 0.25 | 0.40 | 0.50 | 0.60 | 0.75 | 0.90 | 1.00 |
|---|---|---|---|---|---|---|---|---|---|
| **max ratio** | 1.22 | 2.20 | 5.16 | 8.24 | **10.30** | 12.36 | 15.44 | 18.53 | 20.58 |
| p50 | 0.996 | 0.900 | 0.759 | 0.623 | 0.532 | 0.449 | 0.363 | 0.329 | 0.338 |
| tensors > 2× | 0 | 1 | 1 | 1 | 1 | 2 | 3 | 4 | 4 |

⭐ **There is no gap.** A 10% anime blend already sits at 2.20; a 50/50 merge sits
at 10.30. `LIMITS.md` calls the region between 1.2 and 49 "untested", which is
true of our sample — but it is not empty in the world, and it is exactly where
the ChilloutMix / majicMIX / 2.5D families live by construction.

**So "which of two templates" is a cut through a continuum, not a classification.**
Any threshold is arbitrary until something measures where conversion actually
breaks along this axis — and this sweep is also the cheapest way to find out,
since each blend is a real checkpoint that can be converted and rendered.

⚠ These are synthesised blends, not downloaded models. They are made by the same
operation community merges use, but no one has rendered one.

## 4. The survey is cheap: 0.9 MB per checkpoint

The span ratio needs weights, not just the header — so `ROADMAP.md` item 1
("compute it during the header inspection") cannot be done from the header. But
it needs far less than the whole UNet. Candidate subsets, scored on whether they
preserve the separation:

| subset | tensors | bytes | CyberRealistic max | MistoonAnime max |
|---|---|---|---|---|
| all 686 (whole UNet) | 686 | 1,719 MB | 1.22 | 48.26 |
| ≤ 1 MB per tensor | 521 | 67.5 MB | 1.22 | 22.22 |
| ≤ 256 KB per tensor | 446 | 9.1 MB | 1.22 | 20.58 |
| **1-dimensional only (biases, norm gammas)** | **404** | **0.89 MB** | **1.22** | **20.58** |

⭐ **0.89 MB keeps the separation** — 1.22 against 20.58, a factor of 17. It also
keeps the distributional signal (p50 0.996 vs 0.338). The absolute maximum drops
from 48 to 21 because the one worst tensor is a 4-D conv weight, which is exactly
the fragility noted above; the *decision* does not change.

Two things follow:

- **On device**, the existing header inspection can become a real prediction for
  under a megabyte of extra reading and a ~1.6 KB reference table (404 floats)
  added to the bundle. That is roadmap item 1, now costed.
- **Off device**, a checkpoint can be profiled **without downloading it**.
  Safetensors puts a JSON header first, Hugging Face serves HTTP ranges (verified:
  `206 Partial Content`), and `span_probe.py probe <URL>` reads the header and
  then only the tensors it needs.

⚠ Ranged reads are cheap in bytes but not in requests, and the two trade off
sharply. The 404 tensors are scattered, so a coalescing threshold decides how
many requests and how much waste. Measured on `anyloraCheckpoint` (0.89 MB
actually needed):

| merge holes smaller than | requests | fetched |
|---|---|---|
| 0 (no coalescing) | 404 | 0.9 MB |
| **256 KB** (the tool's default) | **105** | **1.1 MB** |
| 2 MB | 72 | **42.7 MB** |

Hugging Face redirects every range to a CDN host, so a session gets no keep-alive
and each request costs ~1–2.5 s: ~2 min per checkpoint, request-bound. A real
survey should resolve the redirect once and range-read the final URL, which is
where the remaining time is.

## 4a. ⭐ "Anime" is not the failing category

Profiled over HTTP ranges, never downloaded (except the two local files):

| checkpoint | lineage | p50 | p90 | p99 | **max** | >2× | cost |
|---|---|---|---|---|---|---|---|
| CyberRealistic (local) | photoreal | 0.996 | 1.050 | 1.134 | **1.22** | 0 | 0.89 MB |
| ReV Animated v2 Rebirth | anime-lineage 2.5D | 1.004 | 1.063 | 1.141 | **1.27** | 0 | 15.5 MB, 80 req |
| anyloraCheckpoint (bakedvae) | anime, NAI lineage | 1.013 | 1.082 | 1.196 | **1.30** | 0 | 1.30 MB, 107 req |
| aamAnyloraAnimeMix v1 | ⚠ an AnyLoRA derivative | 1.013 | 1.081 | 1.196 | **1.30** | 0 | 1.30 MB, 107 req |
| meinahentai v4 | anime, Meina family | 1.013 | 1.083 | 1.196 | **1.31** | 0 | 2.11 MB, 181 req |
| **MistoonAnime v3.0** (local) | anime merge | 0.338 | 1.206 | 1.970 | **20.58** | 4 | 0.89 MB |

⭐ **Every anime checkpoint here except MistoonAnime profiles like the photoreal
one.** Their worst tensors are the same two that CyberRealistic's are
(`output_blocks.8.1.…norm3.bias`, `input_blocks.7.1.…attn1.to_out.0.bias`) —
ordinary variation, not a family signature. All of them would be sent to an
"anime template" by any rule based on lineage, tags or style.

**So the failing category is not a style. It is "merged far enough from base",
and MistoonAnime is so far the only member.** That is the single result most
likely to change `ROADMAP.md` P2, and it cost about four megabytes.

### 🐞 The crossover result that was a lie

The one checkpoint aimed at the photoreal×anime middle, `kawaiiRealisticAsian_v01`,
came back **p50 0.588, p90 1.617, MAX 5.76, 9 tensors above 2×** — squarely
between the two clusters, exactly the shape of result §2 predicts, and it is
**worthless**. The file is **SDXL**: `conditioner.*` instead of
`cond_stage_model.*`, 1,680 UNet tensors instead of 686. Only **251 of the 404**
probe names exist in it at all, and **93 of those have the wrong shape**, so the
number was computed across a partial name collision between two architectures.

The tell was in the output all along — `251/404 probe tensors` — and it read as a
detail rather than a refusal.

**`span_probe.py` now refuses** unless every probe tensor is present *and* the
right shape, and prints the header's prefixes so the architecture is visible.
`CheckpointInfo.kt` already refuses conversions on the same grounds; a profiling
tool has to refuse on them too.

⚠ **The photoreal×anime crossover family is therefore still unprobed** — it is
the most interesting cell in §8's survey and nothing here measures it.

⚠ **Four rows, three independent samples.** `aamAnyloraAnimeMix` matches
`anyloraCheckpoint` to three decimals in every percentile — it is a derivative,
not corroboration.

⚠ **And the three NAI-lineage models agree with each other almost exactly**
(p50 1.013, p99 1.196 for all three). Two readings, and they matter differently:
the benign one is that biases and norm gammas barely move across a shared
lineage, which is what a common ancestor looks like. The other is that the 1-dim
probe subset is **insensitive between near relatives**. It plainly still catches
gross deviation — MistoonAnime separates by a factor of 16 — so it is sound as a
*screen*, but do not read small differences between two probe results as
meaningful. ⚠ If the survey needs to resolve fine structure, re-profile the
interesting checkpoints on the full 686.

## 5. The borrowed CLIP and VAE: what the generic default should be

The template currently carries **DreamShaper 8's** compiled CLIP and VAE. They are
compiled binaries, so "use the checkpoint's own instead" is not a runtime switch —
it is the rejected work in `ROADMAP.md`. The live question is only *whose* parts
the template should be built from.

**Text encoder — use stock SD1.5's (CLIP ViT-L/14).** Measured above: both
checkpoints sit within 0.204% / 0.363% median of it, anime included. It is the
common ancestor of every SD1.5 text encoder, it is the most licence-clean
artefact available, and choosing it removes a finetune's name from the pipeline
at no measured cost.

**VAE — use `vae-ft-mse-840000-ema-pruned`.** Median relative difference of each
baked VAE against each candidate:

| baked VAE of | vs base SD1.5's VAE | vs **ft-mse-840000** | vs kl-f8-anime2 |
|---|---|---|---|
| CyberRealistic | 9.47% | **0.021%** | 8.64% |
| base SD1.5 | — | 9.71% | 3.00% |
| MistoonAnime v3.0 | 207.4% | 213.4% | 212.4% |

⭐ CyberRealistic does not merely resemble ft-mse — **it is ft-mse**, to fp16
rounding. That is the norm: baking ft-mse is what checkpoint authors do, and
Realistic Vision's own model card tells users to fetch that exact file. The three
candidate VAEs sit within ~10% of each other, so the choice is not delicate; the
argument for ft-mse is that it is what the population already ships.

⚠ **`LIMITS.md` overstates one thing and this is the correction.** It says the
text encoder and VAE "barely move — median 0.13–0.39%". That holds for the
**text encoder** and for **photoreal checkpoints' VAEs**, which are all the same
ft-mse file. It does **not** hold across the population: MistoonAnime's VAE is
207% from every candidate, with **0 of 248** tensors within 1% of ft-mse and only
22 within 20%.

⭐ And MistoonAnime's VAE is not "the anime VAE" either — it is not kl-f8-anime2
(212%) and it is not base SD1.5's (207%). Its median tensor norm is **0.53×**
ft-mse's, and `decoder.up.3.block.0.conv1.weight` has norm **2,858,648** against
ft-mse's 78.4, with **516 non-finite values** and a span of 59,200 — just under
fp16's 65,504. That is fp16 overflow in a merge, not a style choice. It makes the
borrowed VAE a *feature* for such a checkpoint rather than a compromise: the
converted model gets a working decoder that the source file no longer has.

**The recommendation, then, is stronger than "swap the VAE".** When a template is
next built (`ROADMAP.md` P1's 2.28 rebuild is the natural occasion, since it is a
full Phase 0 anyway), build it from **base SD1.5 + stock CLIP + ft-mse VAE**
rather than from DreamShaper 8. Same cost, and it removes a specific finetune
from the middle of the pipeline.

⚠ **One honest caveat against that.** Base SD1.5 is the *ancestral* centroid, not
necessarily the *activation-range* centroid. The one comparison available points
very slightly the other way: CyberRealistic's max span ratio is 1.117 against
DreamShaper and 1.22 against base. A broad merge like DreamShaper may genuinely
sit more centrally in range terms than the base model does. One checkpoint is not
evidence — the §4 survey is what would settle it, and it should be run **before**
the template is rebuilt, not after.

## 6. ⭐ The cheaper alternative to more templates

A template carries exactly one thing that a checkpoint cannot supply: the
**activation ranges**. Everything else — weights, their scales, their biases — is
already recomputed per checkpoint by `tplconv`. So "how many templates" is really
"how many sets of activation ranges", and there are two ways to need fewer.

**Where the ranges live, measured.** In `~/tpl/p0/model_tpl.cpp`:

| | count |
|---|---|
| NATIVE (activation) tensors | 3,623 |
| STATIC (weight) tensors | 5,343 |
| tensors with a defined encoding | 6,007 |
| already redirected to the pack by `tpl_patch.py` | **2,383** |
| activation dtype `UFIXED_POINT_16` | 3,369 |
| activation/weight dtype `UFIXED_POINT_8` | 709 |

Activation encodings are `scaleOffsetEncoding` **literals compiled into
`libqnn_model.so`**. The phone cannot touch them today. But `tpl_patch.py`
already rewrites 2,383 static sites into `tpl::data/len/axis/scalar(...)` calls
that read from the pack — extending the same rewrite to NATIVE encodings is the
same edit, and would add roughly **29 KB** (3,623 × 8 bytes) to a 43 KB pack.

**Hypothesis A — stretch the ranges on device.** With encodings pack-driven,
`tplconv` could scale each activation range by a function of the per-tensor span
ratio it already computes. For `y = Wx + b`, a 10× wider weight span is a
first-order reason to expect a wider `y`. If it works, one template covers the
anime case with no PC calibration at all.
**Test:** one `tpl_patch.py` change, one library rebuild, one 93 s device compile,
render MistoonAnime. **Falsified by:** noise that does not improve, or photoreal
checkpoints regressing.

**Hypothesis B — calibrate wider on purpose.** Activations are **16-bit**
(3,369 of them), so range headroom is almost free: doubling a range costs one bit
of sixteen, while a range that is too narrow **clips**, which is the catastrophic
failure. A template calibrated on a *mixture* of checkpoints — or simply with a
deliberate headroom factor — may cover both families at a precision cost too
small to see. Calibration on a mixture costs the same 50 min + 2 h 20 m as
calibration on one checkpoint.
⚠ **Where this could hurt:** the 709 `UFIXED_POINT_8` tensors. The graph converts
some attention inputs down to 8 bits, and there a 4× wider range really does cost
two bits of eight. Widen the u16 majority, leave the u8-converted attention path
alone, and check the attention path explicitly.
⚠ And per `PIPELINE.md`, 48 overrides were correctly overruled on Sigmoid outputs
— ops with semantically fixed ranges must not be widened at all.

**Both are cheaper than a second template, and B is cheaper than A is risky.**
Neither has been tried. Try A first: it costs an afternoon and it is the one that
would change the roadmap.

## 7. Resolution — what a second one costs

The graph is built for one resolution. `~/tpl/p0/model_tpl.cpp` line 126:
`uint32_t dimensions_sample[] = {1, 4, 64, 64}` — a 64×64 latent, i.e. 512×512 —
and every intermediate tensor's dimensions are literals beside it.

**Arithmetic.** MACs for one UNet pass, counted from the checkpoint's own tensor
shapes (`scratchpad/macs.py` in this session; conv at output grid, attention
matmuls at the token count of each level, 77 context tokens):

| pixels | latent | conv + linear | attention | total | vs 512² | attention share |
|---|---|---|---|---|---|---|
| **512×512** | 64×64 | 335.8 G | 63.0 G | **398.8 G** | 1.00× | 15.8% |
| 512×768 | 64×96 | 502.9 G | 140.5 G | 643.4 G | **1.61×** | 21.8% |
| 640×640 | 80×80 | 523.8 G | 152.3 G | 676.1 G | 1.70× | 22.5% |
| 512×896 | 64×112 | 586.5 G | 190.7 G | 777.2 G | 1.95× | 24.5% |
| 768×768 | 96×96 | 753.6 G | 314.1 G | 1,067.7 G | **2.68×** | 29.4% |
| 896×896 | 112×112 | 1,025.2 G | 579.9 G | 1,605.1 G | 4.02× | 36.1% |

⚠ **MACs are work, not latency.** They say how much bigger the arithmetic gets,
not how the HTP schedules it, and the attention share growing from 16% to 36% is
precisely where a scheduler's behaviour stops being proportional. Treat 2.68× as
a lower bound on 768²'s cost, not an estimate. Nothing here was run on the phone.

**What each extra resolution costs to build and ship:**

| | |
|---|---|
| PC | a full Phase 0 at that resolution — ONNX export, ~50 min calibration, ~2 h 20 m quantize — plus the ~9 min Phase 1/2 re-derivation |
| bundle | **+~10.1 MB** (a second `libqnn_model.so` at 9.7 MB and a second 397 KB recipe) |
| weight pack | ⚠ *mostly* shared, not shared. Weight quantization is resolution-independent, but the recipe's `in_scale` for bias quantization comes from that resolution's calibration, so the packs differ |
| output model | **unchanged, ~1.3 GB** — the context binary is mostly weights |
| device compile | scales with the activation footprint. The 512² compile already peaks at **~4.8 GB** on a 12 GB phone, so this is the binding constraint and it is **unmeasured** above 512² |

⭐ **The compile peak, not the graph, is what decides this.** A 768² compile has
2.25× the activation memory to plan, against a ceiling that a 12 GB phone is
already close to. Measure one 768² compile before planning anything else — the
same discipline `PIPELINE.md` applies to SDXL, and for the same reason.

**Which resolutions would actually be worth it.** SD1.5 is trained at 512² and
that is still the safe shape, but the population has moved: Realistic Vision V6's
own model card recommends 896×896, 768×1024 and 640×1152, and warns about
duplication artefacts at the top of that range. Portrait 512×768 is the most-used
non-square SD1.5 shape and is the cheapest of the options above at 1.61×.

⚠ **The alternative is to build none of them.** Hires-fix — generate at 512²,
upscale, denoise lightly — is what generators already do, needs no second
template, and keeps the compile at its measured peak. A second *resolution* buys
composition at large sizes; it does not buy detail that upscaling cannot.

**A shortcut worth 30 minutes of investigation, not more.** The spatial dimensions
are literals in `model_tpl.cpp` (`{1, 4, 64, 64}`, `{1, 64, 64, 320}`, …) and the
attention token counts appear as dimensions too (320 arrays contain `4096`). A
mechanical rewrite of those to a different grid, reusing the 512² activation
encodings, would be the *resolution* analogue of the checkpoint transplant this
whole project is built on — no requantize, just a rebuild. ⚠ It is also exactly
the kind of change that produces a graph which compiles, runs at a plausible
speed and computes nothing (`LIMITS.md`, the dead template). Gate it on renders,
never on latency, and note that softmax statistics genuinely change with token
count so the borrowed ranges have a real reason to be wrong.

## 8. Open questions

1. **Run the survey.** ~50 checkpoints spanning both roots and the merged middle,
   profiled with `span_probe.py`. The histogram of span ratios against base SD1.5
   is what decides whether the population is two clusters or a continuum, and
   therefore whether two templates is a design or a guess.
2. **Is base SD1.5 or a broad merge the better template centroid?** §5's caveat.
   The same survey answers it, and it should run before P1's rebuild.
3. **Hypothesis A** (§6) — pack-driven activation encodings. The cheapest thing
   here that could change the roadmap.
4. **Where between 1.2 and 20 does conversion actually break?** Still untested,
   as `LIMITS.md` says. The survey finds checkpoints in that gap; only a device
   render settles what happens to them.
5. **One 768² compile on the phone**, for its peak RSS and nothing else.

## What was *not* established

- **No popularity ranking.** Civitai returns HTTP 451 to this region on both its
  API and its pages, so "the most popular SD1.5 checkpoints" is taken from
  Hugging Face download counts and secondary write-ups, and the lineage in §2 is
  the community record rather than anything measured here.
- **Two local checkpoints.** Every measured row above is CyberRealistic or
  MistoonAnime. They are a photoreal finetune and an anime merge, which is the
  right pair for a first cut and is not a sample.
- **Nothing in this file was run on the phone.**

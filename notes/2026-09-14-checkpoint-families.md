# 2026-09-14 (later) — what the SD1.5 population actually looks like

Session history. The findings live in `docs/CHECKPOINT-FAMILIES.md`; this is the
order things happened, what cost time, and the assumptions that did not survive.

## The question that started it

"Two templates" (photoreal + anime) had been written into `ROADMAP.md` as if the
population were two clusters. The user asked whether that is true — and, in the
same breath, whether DreamShaper's CLIP and VAE should really be hardcoded into
every converted model, and what other resolutions would cost.

## Sequence

1. **Span tables for everything on disk.** CyberRealistic, MistoonAnime, then
   base SD1.5 downloaded as a neutral reference
   (`Comfy-Org/stable-diffusion-v1-5-archive`, LDM keys, so no diffusers key
   mapping was needed). Base SD1.5 reproduces `LIMITS.md`'s ratios closely —
   48.26 where DreamShaper gave 49.4 — so the reference can be the ancestor
   instead of a finetune.
2. **The probe.** Subset sweep showed the 404 one-dimensional UNet tensors keep
   the whole separation for **0.89 MB**. Verified that Hugging Face answers HTTP
   ranges (206), so a checkpoint can be profiled without downloading it →
   `tools/span_probe.py`.
3. **ReV Animated, profiled remotely, came back photoreal** (1.27). It is
   anime-lineage. That is the result that put P2's premise in doubt.
4. **The merge sweep.** Blending the two local checkpoints linearly — which is
   what community merges literally are — moves the max ratio smoothly 1.22 →
   20.58. No gap, so no threshold.
5. **CLIP/VAE measured against candidate defaults**, which is where the VAE claim
   in `LIMITS.md` turned out to be narrower than written.
6. **Where the activation ranges live** — checked directly in
   `~/tpl/p0/model_tpl.cpp` over `wsl.exe` rather than assumed. They are literals
   in the library, which is what makes item 1c a real proposal and not a guess.
7. **Resolution costed** by counting MACs from the checkpoint's own shapes, and
   by reading the graph's literal `{1, 4, 64, 64}`.

## Assumptions that did not survive

- **"The anime family fails."** One anime-lineage model profiles like a photoreal
  one. The failing category may be "merged far from base", which is not a style.
- **"The region between 1.2 and 49 is untested."** True of our sample, but it is
  not empty — a 50/50 merge lands at 10.3, and the ChilloutMix / majicMIX / 2.5D
  families live there by construction.
- **"The text encoder and VAE barely move."** The text encoder does not move. The
  VAE claim was measured only across photoreal checkpoints, which all bake the
  same ft-mse file. `LIMITS.md` has been narrowed.
- **"Compute the span ratio during the header inspection."** The header does not
  contain weights. It is still cheap — 0.89 MB — but it is a read, not a parse.

## Things that cost time

- **Civitai is HTTP 451 from this region**, on both the API and the pages, via
  curl *and* WebFetch. There is no download-ranked list of SD1.5 checkpoints in
  this record because of it; popularity came from Hugging Face counts and
  secondary write-ups. Do not spend the time again.
- **The coalescing threshold in the probe matters more than it looks.** Merging
  ranges across holes smaller than 2 MB turned a 0.89 MB read into a 42.7 MB one
  for a handful of saved requests. 256 KB is the measured default; the tool's
  docstring carries the numbers.
- **Remote probing is request-bound, not byte-bound.** Hugging Face redirects
  every range to a CDN host, so a session gets no keep-alive: ~1–2.5 s per
  request, ~2–4 min per checkpoint. Resolving the redirect once is the obvious
  fix and was not done.

## 🐞 The mistake of this session

`kawaiiRealisticAsian_v01` was picked as the photoreal×anime crossover — the one
model expected to land in the middle — and the probe reported **MAX 5.76, p50
0.588, 9 tensors above 2×**. It is in the middle. It is also **SDXL**
(`conditioner.*`, 1,680 UNet tensors), and only 251 of 404 probe names exist in
it, 93 of those with the wrong shape.

The number was believable *because* it matched the prediction, which is the whole
danger. The tell — `251/404 probe tensors` — was printed and read as a detail.
The tool now refuses on missing or mis-shaped tensors and prints the header's
prefixes. **When a result confirms what you expected, suspect the check** — the
same rule as `CLAUDE.md`'s, pointed the other way.

## Not done

- The survey itself (`ROADMAP.md` item 1a), and in particular the
  **photoreal×anime crossover family** — the one cell that would settle whether
  the middle is really occupied by models people use. The candidate picked for it
  turned out to be SDXL, so it is still unmeasured.
- Nothing in this session touched the phone, and nothing was rendered. Every
  claim is about weights.

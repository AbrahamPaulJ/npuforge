# Original SD1.5 template bundle

This directory contains the original SD1.5 recipe and compact constants pack.
**These artifacts are excluded from the repository's MIT source license and
are treated as non-commercial.** They were derived using the modified diffusers
exporter distributed with Local Dream under CC BY-NC 4.0. See [NOTICE](../NOTICE)
for provenance and distribution boundaries.

| File | Size | Purpose |
|---|---|---|
| `recipe.bin` | Approximately 397 KB | 2,383 entries describing checkpoint sources, layouts and quantization rules |
| `tpl_trim.pack` | 43,328 bytes | 257 graph constants containing 832 bytes of payload |

A complete conversion also needs the matching pack-loading model library and
QAIRT runtime. Those files are supplied separately. The current Android app
additionally requires family-specific UNet and CLIP/VAE assets described in
[Build](../docs/BUILD.md).

The full original template pack is approximately 863 MB. Each learned weight is
replaced from the selected checkpoint; only constant entries survive. The
original comparison produced byte-identical packs from the full and trimmed
templates. See [the derivation](../docs/PIPELINE.md).

## Historical configuration

DreamShaper 8, SD1.5 text-to-image, 512 × 512, QAIRT 2.49, W8A16,
400-row calibration, v73 target and 8 MB VTCM. The activation ranges are reused
for converted checkpoints and do not guarantee quality across all fine-tunes.

## Regeneration

See [template authoring](../docs/TEMPLATE-AUTHORING.md),
[`tpl_recipe_bin.py`](../tools/tpl_recipe_bin.py) and
[`tpl_pack_trim.py`](../tools/tpl_pack_trim.py). Commercial use requires resolving
the artifact rights or regenerating templates from an independently licensed
export path, as described in [NOTICE](../NOTICE).

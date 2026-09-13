# The template bundle

This is everything a device needs besides the QAIRT runtime and the 9.7 MB
`libqnn_model.so` (gitignored; see `../NOTICE` and `../docs/BUILD.md`).

| file | size | what |
|---|---|---|
| `recipe.bin` | 397 KB | 2,383 entries: for each graph tensor, its checkpoint source, layout transform and quantization rule |
| `tpl_trim.pack` | 43 KB | the 257 graph constants with no checkpoint source — 832 bytes of payload |

The full template pack is ~863 MB. Trimming is safe because the recipe overwrites
every template weight; only entries whose rule is `template` survive conversion.
Verified: a pack built from this 43 KB file is byte-identical to one built from
the full 863 MB pack.

Template: **DreamShaper 8**, realistic SD1.5 txt2img, 512×512, QAIRT 2.49, w8a16,
400-row calibration, `_8gen2` tier (v73, 8 MB VTCM). Its activation ranges are
what every converted checkpoint borrows.

Regenerate with `../tools/tpl_recipe_bin.py` and `../tools/tpl_pack_trim.py`.

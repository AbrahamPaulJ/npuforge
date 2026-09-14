# The template bundle

> ## ⛔ These two files are NOT under this repository's MIT licence
>
> `recipe.bin` and `tpl_trim.pack` are **non-commercial**. They contain no
> third-party source, but they were derived from a UNet graph exported with a
> modified `diffusers` UNet redistributed under **CC BY-NC 4.0**, so the
> derivation chain is non-commercial and these artefacts inherit that until
> someone qualified says otherwise. The MIT licence in `../LICENSE` covers the
> repository's own source — `native/`, `tools/`, `app/`, the documentation — and
> **not this directory**.
>
> **Commercial use requires regenerating this bundle from a clean-room export.**
> `../docs/TEMPLATE-AUTHORING.md` says how. The rest of the pipeline is generic
> over templates, so a clean-room bundle drops straight in.
>
> Full detail and the rest of the third-party picture: `../NOTICE` §2.

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

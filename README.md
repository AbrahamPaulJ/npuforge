# npuforge

**Convert a Stable Diffusion 1.5 checkpoint into a Qualcomm NPU model on the phone
itself, in under two minutes, with no PC.**

Today, putting a new SD1.5 checkpoint on an NPU means a developer running a
multi-hour conversion on a workstation: ~50 min of calibration, ~20 min of ONNX
export and a **2 h 18 m** quantize. npuforge replaces all of that, at conversion
time, with a template plus a weight recipe — so the phone only has to re-quantize
the weights and compile.

Measured on a Samsung SM-S938B (SM8750, HTP v79), 2026-09-14:

| step | time |
|---|---|
| `tplconv`: `.safetensors` → weight pack (863,626,496 B) | **24 s** |
| `qnn-context-binary-generator`: pack → `unet.bin` (882,780,736 B) | **93 s** |
| **total, on the phone** | **117 s** |

The result renders **byte-identical** to the same pack compiled on a PC.

## Why it is small enough to ship

A naive "ship the template" approach means a 863 MB pack. But the recipe
**overwrites every weight in the template** — the only payloads that survive are
257 graph constants with no checkpoint source, totalling **832 bytes**.

| what the app ships | size |
|---|---|
| `template/tpl_trim.pack` | 43 KB |
| `template/recipe.bin` | 397 KB |
| `libqnn_model.so` (aarch64, pack-loading) | 9.7 MB |

Verified: a pack built from the 43 KB trimmed template is byte-identical to one
built from the full 863 MB pack.

## How it works

1. **Template (PC, once per architecture).** One fixed SD1.5 graph, quantized with
   one checkpoint's *activation* ranges. Activation ranges are what needs
   calibration, and they transfer between checkpoints — two photoreal SD1.5
   finetunes agree to ×1.033 at p50 over 4,638 tensors.
2. **Recipe (PC, once per template).** A numeric map from checkpoint tensors to
   graph tensors — 686 ↔ 1,358 entries, found by value fingerprint and permutation
   search, never by name, because the exporter names norms anonymously. Plus the
   exact quantization rule for each.
3. **Convert (phone).** Read the user's `.safetensors`, apply the recipe's
   transforms and per-channel quantization, write a weight pack.
4. **Compile (phone).** `qnn-context-binary-generator` links the pack into a
   context binary for *this* chip.

## Correctness is decided by `cmp`, not by looking at pictures

The PC pipeline produces reference packs, so the on-device converter has an exact
target. `tplconv` reproduces them byte-for-byte:

| built by | checkpoint | md5 |
|---|---|---|
| Python (PC) | DreamShaper 8 | `042b99cf38b37cecc15483f37118bb46` |
| C++ (PC) | DreamShaper 8 | `042b99cf…` ✅ |
| C++ (**phone**) | DreamShaper 8 | `042b99cf…` ✅ |
| Python (PC) | AbsoluteReality | `52ca4f492e9f43634d08698ea9a5360b` |
| C++ (PC) | AbsoluteReality | `52ca4f49…` ✅ |

Two different checkpoints, first run, no tolerance.

⚠ **Arithmetic order is the specification.** Each rule was measured against the
reference converter's own output, and the obvious-looking alternative is wrong:

- `i8`: `q = rha(w * 127 / max)` — *not* `w / (max/127)`. A weight at exactly
  `max/2` must land on 63.5.
- `u8`: `x = f32(w*255/(hi-lo) - o)` then `floor(f64(x + f32(0.5)))` — the `+0.5`
  happens in **float32**.
- `i32`: divide in **float64** by the **float32** scale. Doing it in float32 moves
  results in 128-unit steps.

`rha` is round-half-away-from-zero. The header of `native/tplconv.cpp` repeats
this, because getting one wrong is silent — the model still builds and still runs.

## Memory

`tplconv` peaks at **1.98 GB** where the Python reference peaks at 5.07 GB. It
makes two passes: compute every scale first (a few MB — payload lengths follow
from the dims), then stream each entry's bytes. That is what makes it run on a
phone at all.

⚠ The **compile** step is the heavy one: ~4.8 GB peak. 8 GB phones are unproven.

## Status

✅ **Working end to end as an Android app** (2026-09-14, one device). Pick a
`.safetensors`, get a loadable model directory. The app reproduces the
adb-driven pipeline: its model renders **byte-identical PNGs** to both the
adb-built and the PC-built versions. Output goes to
**`Download/npuforge/<name>/`** via MediaStore — no storage permission, and
somewhere a person can actually find it.

Before picking a file it reads the safetensors **header** (a short read, not a
2 GB copy) and shows what the checkpoint contains — UNet, VAE, text encoder —
which of those conversion keeps, and whether all 686 tensors the recipe needs
are present. An SDXL, SD2 or diffusers-layout file is refused immediately
instead of failing two minutes in.

⚠ **Getting QNN to run inside an app took four separate fixes**, each producing
the same "Device Creation failure". If you touch the packaging or the library
paths, read `docs/ANDROID.md` first.

What is not done:

- **CLIP/VAE come from the template**, so a converted checkpoint uses the
  template's text encoder. The UNet carries the style, so this is fine for now,
  but it is the next fidelity step. They are downloaded once (~1.0 GB fetched,
  395 MB kept) rather than bundled.
- ⚠ **Everything produced is a QAIRT 2.49 build, so it carries the fp16 stamp.**
  Chips that reject 2.49 models will reject these too. `tplconv` itself is
  SDK-agnostic — the stamp comes from the template library and the generator —
  so a 2.28 variant is a rebuild, not a redesign, and needs the 2.28 SDK.
- **Moving a converted model into a generator is still manual** — the app writes
  to `Download/npuforge/<name>/`, which any file manager can see, but it does not
  install the model anywhere for you.
- **The target tier is hardcoded** to `_8gen2` (v73, 8 MB VTCM). An on-device
  converter should compile for the chip it is running on; that is one of the
  original motivations and is not implemented.
- **One template**: realistic SD1.5 txt2img, 512×512. An anime checkpoint
  borrowing photoreal activation ranges is untested and is the known hard case.
- ⛔ **Shipping is gated on the QAIRT redistribution question.** See `NOTICE`.

## Build and run

See `docs/BUILD.md`.

## Licence

MIT (`LICENSE`) for this repository's own source. ⚠ **`NOTICE` is not optional
reading** — the QAIRT runtime is redistribution-restricted, and the template's
derivation chain passes through a non-commercial licence.

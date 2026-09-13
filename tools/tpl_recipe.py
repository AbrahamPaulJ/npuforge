#!/usr/bin/env python
"""Phase 1 (docs/ON-DEVICE-CONVERT.md): derive the weight RECIPE for a template.

    tpl_recipe.py discover <model.cpp> <model.bin> <checkpoint.safetensors> <recipe.json>

For every STATIC pack entry, dequantize the template's own bytes and find, numerically,
the checkpoint tensor it came from plus the transform:
  * source   : an LDM key (model.diffusion_model.*), optionally a row slice
               [h*D:(h+1)*D] (the export's 8-way per-head attention split)
  * layout   : trailing singleton dims + an axis permutation
  * quant    : which formula reproduces the template's encoding (checked, not assumed)
Entries with no checkpoint source are recorded as TEMPLATE constants (bytes kept).

Nothing here guesses names: the ONNX/converter names for norms are anonymous
(onnx__Mul_*), so matching is by value. A match needs elementwise agreement within
one quantization step, and the chosen source must be unique.
"""
import itertools
import json
import sys
import tarfile
from collections import Counter, defaultdict

import numpy as np
from safetensors.numpy import load_file

sys.path.insert(0, __import__("os").path.dirname(__file__))
from tpl_parse import parse  # noqa: E402

NP = {"QNN_DATATYPE_SFIXED_POINT_8": np.int8, "QNN_DATATYPE_UFIXED_POINT_8": np.uint8,
      "QNN_DATATYPE_SFIXED_POINT_32": np.int32, "QNN_DATATYPE_UFIXED_POINT_16": np.uint16,
      "QNN_DATATYPE_INT_32": np.int32, "QNN_DATATYPE_FLOAT_32": np.float32,
      "QNN_DATATYPE_UINT_32": np.uint32}
HEADS = 8


def dequant(t, raw):
    q = np.frombuffer(raw, dtype=NP[t["dtype"]]).astype(np.float64).reshape(t["dims"])
    if t["enc"] == "none":
        return q
    s = np.array([p[0] for p in t["pairs"]]); o = np.array([p[1] for p in t["pairs"]])
    if t["enc"] == "axis":
        shape = [1] * q.ndim; shape[t["axis"]] = -1
        return (q + o.reshape(shape)) * s.reshape(shape)
    return (q + o[0]) * s[0]


def fp(a, k=129):
    v = np.sort(a.ravel())
    return v[np.linspace(0, v.size - 1, k).astype(np.int64)]


def load_sources(path):
    sd = load_file(path)
    src = {}
    for k, v in sd.items():
        if k.startswith("model.diffusion_model."):
            src[k] = v.astype(np.float32)
    del sd
    cands = defaultdict(list)            # size -> [(key, slice h or None)]
    for k, v in src.items():
        cands[v.size].append((k, None))
        if v.ndim == 2 and k.split(".")[-2] in ("to_q", "to_k", "to_v") and v.shape[0] % HEADS == 0:
            for h in range(HEADS):
                cands[v.size // HEADS].append((k, h))
    return src, cands


def get(src, key, h):
    v = src[key]
    if h is None:
        return v
    d = v.shape[0] // HEADS
    return v[h * d:(h + 1) * d]


def layouts(s, dims):
    """All (perm, reshaped) of s padded with trailing 1s whose shape == dims."""
    base = s.reshape(s.shape + (1,) * (len(dims) - s.ndim)) if s.ndim < len(dims) else s
    if base.ndim != len(dims):
        if base.size == int(np.prod(dims)):
            yield ("reshape",), base.reshape(dims)
        return
    for perm in itertools.permutations(range(base.ndim)):
        if tuple(base.shape[i] for i in perm) == tuple(dims):
            yield perm, np.transpose(base, perm)


def step(t):
    if t["enc"] == "none":
        return 0.0
    return max(p[0] for p in t["pairs"])


def discover(cpp, binf, ckpt, out):
    tensors = parse(cpp)
    raws = {}
    with tarfile.open(binf) as tar:
        for m in tar.getmembers():
            if m.isfile() and m.name.endswith(".raw"):
                raws[m.name.rsplit("/", 1)[-1][:-4]] = tar.extractfile(m).read()
    src, cands = load_sources(ckpt)
    print("pack entries %d, checkpoint UNet tensors %d" % (len(tensors), len(src)))
    fps = {}
    recipe, unmatched, ambiguous = [], [], []
    used = Counter()
    for i, t in enumerate(tensors):
        T = dequant(t, raws[t["binvar"]])
        tol = 0.51 * step(t) + 1e-7 * (np.abs(T).max() + 1e-12)
        best = []
        ft = fp(T)
        for key, h in cands.get(T.size, []):
            ck = (key, h)
            if ck not in fps:
                fps[ck] = fp(get(src, key, h))
            if np.abs(fps[ck] - ft).max() > 2 * tol + 1e-6:
                continue
            for perm, S in layouts(get(src, key, h), t["dims"]):
                err = float(np.abs(S.astype(np.float64) - T).max())
                if err <= tol:
                    best.append((err, key, h, perm))
        entry = {"binvar": t["binvar"], "dtype": t["dtype"], "dims": t["dims"], "enc": t["enc"]}
        if t["enc"] == "axis":
            entry["axis"] = t["axis"]
        if not best:
            entry["source"] = None
            unmatched.append(t)
        else:
            best.sort()
            keys = {(b[1], b[2]) for b in best}
            if len(keys) > 1:
                ambiguous.append((t["binvar"], sorted(keys)[:4]))
            err, key, h, perm = best[0]
            entry.update(source=key, head=h, perm=list(perm) if perm != ("reshape",) else "reshape")
            used[(key, h)] += 1
        recipe.append(entry)
        if (i + 1) % 400 == 0:
            print("  %d/%d matched so far: %d" % (i + 1, len(tensors), sum(1 for r in recipe if r["source"])), flush=True)

    print("\nmatched %d, unmatched %d, ambiguous %d" % (len(tensors) - len(unmatched), len(unmatched), len(ambiguous)))
    print("unmatched by dtype/enc:", Counter((t["dtype"][14:], t["enc"]) for t in unmatched))
    for t in unmatched[:12]:
        print("  UNMATCHED %-60s %s %s" % (t["binvar"][:60], t["dims"], t["dtype"][14:]))
    for b, ks in ambiguous[:8]:
        print("  AMBIGUOUS %-50s %s" % (b[:50], ks))
    reused = [k for k, n in used.items() if n > 1]
    print("sources used more than once: %d %s" % (len(reused), reused[:4]))
    unused = [k for k in src if (k, None) not in used and not any((k, h) in used for h in range(HEADS))]
    print("checkpoint tensors never used: %d %s" % (len(unused), unused[:6]))
    print("perms:", Counter(str(r.get("perm")) + " " + r["dtype"][14:] for r in recipe if r["source"]).most_common(8))
    json.dump({"entries": recipe, "pairs": {t["binvar"]: t["pairs"] for t in tensors}}, open(out, "w"))
    print("wrote", out)


if __name__ == "__main__":
    if sys.argv[1] == "discover":
        discover(*sys.argv[2:6])
    else:
        sys.exit(__doc__)

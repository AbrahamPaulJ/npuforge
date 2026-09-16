#!/usr/bin/env python
"""Phase 1 (docs/ON-DEVICE-CONVERT.md): derive the weight RECIPE for a template.

    tpl_recipe.py discover <model.cpp> <model.bin> <checkpoint.safetensors> <recipe.json> [--head-dim 64]

For every STATIC pack entry, dequantize the template's own bytes and find, numerically,
the checkpoint tensor it came from plus the transform:
  * source   : an LDM key (model.diffusion_model.*), optionally a row slice
               [h*D:(h+1)*D] (eight heads for SD1.5; --head-dim 64 for SDXL)
  * layout   : trailing singleton dims + an axis permutation
  * quant    : which formula reproduces the template's encoding (checked, not assumed)
Entries with no checkpoint source are recorded as TEMPLATE constants (bytes kept).

Nothing here guesses names: the ONNX/converter names for norms are anonymous
(onnx__Mul_*), so matching is by value. A match needs elementwise agreement within
one quantization step, and the chosen source must be unique.
"""
import argparse
import itertools
import json
import sys
import tarfile
from collections import Counter, defaultdict

import numpy as np
from safetensors import safe_open

sys.path.insert(0, __import__("os").path.dirname(__file__))
from tpl_parse import parse  # noqa: E402

NP = {"QNN_DATATYPE_SFIXED_POINT_8": np.int8, "QNN_DATATYPE_UFIXED_POINT_8": np.uint8,
      "QNN_DATATYPE_SFIXED_POINT_32": np.int32, "QNN_DATATYPE_UFIXED_POINT_16": np.uint16,
      "QNN_DATATYPE_INT_32": np.int32, "QNN_DATATYPE_FLOAT_16": np.float16,
      "QNN_DATATYPE_FLOAT_32": np.float32,
      "QNN_DATATYPE_UINT_32": np.uint32}


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


def load_sources(path, head_dim, source_prefix="model.diffusion_model."):
    with safe_open(path, framework="np") as sd:
        src = {k: sd.get_tensor(k).astype(np.float32) for k in sd.keys()
               if k.startswith(source_prefix)}
    cands = defaultdict(list)            # size -> [(key, slice h or None)]
    for k, v in src.items():
        cands[v.size].append((k, None))
        if v.ndim == 2 and k.split(".")[-2] in ("to_q", "to_k", "to_v"):
            d = v.shape[0] // 8 if head_dim is None else head_dim
            for h in range(v.shape[0] // d):
                cands[d * v.shape[1]].append((k, h))
    return src, cands


def get(src, key, h, size):
    v = src[key]
    if h is None:
        return v
    d = size // (v.size // v.shape[0])
    return v[h * d:(h + 1) * d]


def layouts(s, dims):
    """All (perm, reshaped) of s padded with trailing 1s whose shape == dims."""
    while s.ndim > len(dims) and s.shape[-1] == 1:
        s = s.reshape(s.shape[:-1])
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


def discover(cpp, binf, ckpt, out, head_dim=None, source_prefix="model.diffusion_model.", reject_ambiguous=False):
    tensors = parse(cpp)
    raws = {}
    with tarfile.open(binf) as tar:
        for m in tar.getmembers():
            if m.isfile() and m.name.endswith(".raw"):
                raws[m.name.rsplit("/", 1)[-1][:-4]] = tar.extractfile(m).read()
    src, cands = load_sources(ckpt, head_dim, source_prefix)
    print("pack entries %d, checkpoint source tensors %d" % (len(tensors), len(src)))
    fps = {}
    recipe, unmatched, ambiguous = [], [], []
    used = Counter()
    for i, t in enumerate(tensors):
        T = dequant(t, raws[t["binvar"]])
        tol = 0.51 * step(t) + 1e-7 * (np.abs(T).max() + 1e-12)
        best = []
        ft = fp(T)
        candidates = cands.get(T.size, [])
        fp_key = (T.size, t["dtype"])
        if candidates and fp_key not in fps:
            fps[fp_key] = np.array([
                fp(get(src, key, h, T.size).astype(np.float16) if t["dtype"] == "QNN_DATATYPE_FLOAT_16"
                   else get(src, key, h, T.size)) for key, h in candidates])
        indices = np.flatnonzero(
            ~(np.abs(fps[fp_key] - ft).max(axis=1) > 2 * tol + 1e-6)
        ) if candidates else []
        for index in indices:
            key, h = candidates[index]
            for perm, S in layouts(get(src, key, h, T.size), t["dims"]):
                if t["dtype"] == "QNN_DATATYPE_FLOAT_16":
                    S = S.astype(np.float16)
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
    used_sources = {key for key, head in used}
    unused = [k for k in src if k not in used_sources]
    print("checkpoint tensors never used: %d %s" % (len(unused), unused[:6]))
    print("perms:", Counter(str(r.get("perm")) + " " + r["dtype"][14:] for r in recipe if r["source"]).most_common(8))
    if reject_ambiguous and ambiguous:
        raise ValueError(f"Ambiguous checkpoint sources for {len(ambiguous)} tensors; resolve graph provenance before authoring")
    with open(out, "w") as output:
        json.dump({"entries": recipe, "pairs": {t["binvar"]: t.get("pairs", []) for t in tensors}}, output)
    print("wrote", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["discover"])
    for name in ("cpp", "binf", "ckpt", "out"):
        parser.add_argument(name)
    parser.add_argument("--head-dim", type=int, help="Fixed attention head width; default: SD1.5's eight heads")
    parser.add_argument("--source-prefix", default="model.diffusion_model.",
                        help="Checkpoint component prefix; use first_stage_model. for a VAE")
    parser.add_argument("--reject-ambiguous", action="store_true",
                        help="Refuse recipes whose source mapping is not unique")
    args = parser.parse_args()
    discover(args.cpp, args.binf, args.ckpt, args.out, args.head_dim, args.source_prefix, args.reject_ambiguous)

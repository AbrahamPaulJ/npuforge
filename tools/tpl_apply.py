#!/usr/bin/env python
"""Phase 1 (docs/ON-DEVICE-CONVERT.md): checkpoint + recipe -> TPLPACK1, and its gate.

    tpl_apply.py finalize <model.cpp> <recipe_raw.json> <recipe.json>
        Adds what the phone needs and cannot derive: per-bias input-activation scale
        and weight partner. Template constants stay as template bytes.
    tpl_apply.py apply <recipe.json> <template.pack> <checkpoint.safetensors> <out.pack>
    tpl_apply.py compare <a.pack> <b.pack>

Quantization rules, each verified against the converter's output (tpl_rules.py/diag):
  int8  per-axis (conv weights):  s_c = f32(max|w_c| / 127);     q = rha(f32 w / f32 s_c)
  uint8 scalar (linear, norms):   lo=min(0,w) hi=max(0,w); s = f32((hi-lo)/255);
                                  o = round(lo/s);               q = rha(f32 w / f32 s) - o
  int32 per-axis (conv biases):   s_c = f32(in_scale * s_w_c) using the NEW weight scale;
                                  q = rha(f32 b / f32 s_c)  (injected q/k/v biases are 0)
  int32 scalar (linear/norm bias):s = f32(max|b| / 2^31);        q = rha(f32 b / f32 s)
where rha = round half AWAY from zero applied to the float32 quotient, then clipped.
This is the whole algorithm the on-device port must reproduce.
"""
import json
import re
import struct
import sys

import numpy as np
from pathlib import Path
from safetensors.numpy import load_file

sys.path.insert(0, __import__("os").path.dirname(__file__))
from tpl_pack import write_pack  # noqa: E402

NP = {"QNN_DATATYPE_SFIXED_POINT_8": np.int8, "QNN_DATATYPE_UFIXED_POINT_8": np.uint8,
      "QNN_DATATYPE_SFIXED_POINT_32": np.int32, "QNN_DATATYPE_UFIXED_POINT_16": np.uint16}


def read_pack(path):
    buf = Path(path).read_bytes()
    assert buf[:8] == b"TPLPACK1"
    c = 8
    (n,) = struct.unpack_from("<I", buf, c); c += 4
    out = []
    for _ in range(n):
        (nl,) = struct.unpack_from("<H", buf, c); c += 2
        name = buf[c:c + nl].decode(); c += nl
        (np_,) = struct.unpack_from("<I", buf, c); c += 4
        pairs = [struct.unpack_from("<fi", buf, c + 8 * k) for k in range(np_)]; c += 8 * np_
        off, ln = struct.unpack_from("<QQ", buf, c); c += 16
        out.append((name, pairs, buf[off:off + ln]))
    return out


def rha(x32, lo, hi, dt):
    x = x32.astype(np.float64)
    return np.clip(np.sign(x) * np.floor(np.abs(x) + 0.5), lo, hi).astype(dt)


def finalize(cpp, raw_json, out):
    R = json.load(open(raw_json))
    src = open(cpp, errors="replace").read()
    consumer = {}
    for m in re.finditer(r'const char\*\s+inputs_\w+\[\] = \{\s*([^}]*)\};', src):
        names = re.findall(r'"([^"]+)"', m.group(1))
        if len(names) == 3:
            consumer[names[2]] = (names[0], names[1])
    act = {}
    for m in re.finditer(r'\.name= "([^"]+)",\s*\.type= QNN_TENSOR_TYPE_(?:NATIVE|APP_WRITE|APP_READ),.*?\.scaleOffsetEncoding= \{\.scale= ([-0-9.eE+]+)f', src, re.S):
        act.setdefault(m.group(1), float(m.group(2)))
    rules = {}
    for e in R["entries"]:
        d, enc, s = e["dtype"], e["enc"], e["source"]
        if d.endswith("SFIXED_POINT_8") and enc == "axis":
            e["rule"] = "i8_axis"
        elif d.endswith("UFIXED_POINT_8") and enc == "scalar":
            e["rule"] = "u8_asym"
        elif d.endswith("SFIXED_POINT_32") and enc == "axis":
            inp, w = consumer[e["binvar"]]
            e.update(rule="i32_axis_bias" if s else "i32_axis_zero", in_scale=act[inp], weight=w)
        elif d.endswith("SFIXED_POINT_32") and enc == "scalar":
            e["rule"] = "i32_scalar"
        else:
            e["rule"] = "template"
        assert e["rule"] == "template" or s or e["rule"] == "i32_axis_zero", e["binvar"]
        rules[e["rule"]] = rules.get(e["rule"], 0) + 1
    del R["pairs"]
    json.dump(R, open(out, "w"))
    print("rules:", rules, "->", out)


def src_of(sd, e):
    v = sd[e["source"]].astype(np.float32)
    if e["head"] is not None:
        d = int(np.prod(e["dims"])) // (v.size // v.shape[0])
        v = v[e["head"] * d:(e["head"] + 1) * d]
    if e["perm"] == "reshape":
        return np.ascontiguousarray(v.reshape(e["dims"]))
    v = v.reshape(v.shape + (1,) * (len(e["dims"]) - v.ndim))
    return np.ascontiguousarray(np.transpose(v, e["perm"]))


def apply(recipe, template_pack, ckpt, out):
    R = json.loads(Path(recipe).read_text())
    tpl = {n: (p, b) for n, p, b in read_pack(template_pack)}
    sd = load_file(ckpt)
    wscale = {}          # weight binvar -> new per-channel scales (float32)
    entries = []
    order = sorted(R["entries"], key=lambda e: e["rule"] in ("i32_axis_bias", "i32_axis_zero"))
    built = {}
    for e in order:
        b, rule = e["binvar"], e["rule"]
        if rule == "template":
            built[b] = tpl[b]
            continue
        # ⚠ Arithmetic ORDER matters for exact bytes (measured, tpl_round*.py):
        #   i8   x = w*127/max, not w/(max/127): a weight at exactly max/2 must give 63.5
        #   i32  divide in float64 by the float32 scale, never in float32 (128-unit steps)
        if rule == "i8_axis":
            W = src_of(sd, e); ax = e["axis"]
            red = tuple(i for i in range(W.ndim) if i != ax)
            mx = np.abs(W).max(axis=red).astype(np.float64)
            s = (mx / 127.0).astype(np.float32)
            shape = [1] * W.ndim; shape[ax] = -1
            q = rha(W.astype(np.float64) * 127.0 / mx.reshape(shape), -128, 127, np.int8)
            wscale[b] = s
            built[b] = ([(float(x), 0) for x in s], q.tobytes())
        elif rule == "u8_asym":
            W = src_of(sd, e).astype(np.float64)
            lo, hi = min(0.0, float(W.min())), max(0.0, float(W.max()))
            s = np.float32((hi - lo) / 255.0)
            o = int(np.round(lo / float(s)))
            # exact on all 197 (tpl_round5.py): x in float64, then +0.5 in FLOAT32
            x = (W * 255.0 / (hi - lo) - o).astype(np.float32)
            q = np.clip(np.floor((x + np.float32(0.5)).astype(np.float64)), 0, 255).astype(np.uint8)
            wscale[b] = np.array([s], dtype=np.float32)
            built[b] = ([(float(s), o)], q.tobytes())
        elif rule == "i32_scalar":
            B = src_of(sd, e).astype(np.float64)
            s = np.float32(float(np.abs(B).max()) / 2**31)
            q = rha(B / np.float64(s), -2**31, 2**31 - 1, np.int32)
            built[b] = ([(float(s), 0)], q.tobytes())
        elif rule in ("i32_axis_bias", "i32_axis_zero"):
            ws = wscale[e["weight"]]
            s = (np.float64(np.float32(e["in_scale"])) * ws.astype(np.float64)).astype(np.float32)
            n = int(np.prod(e["dims"]))
            if rule == "i32_axis_zero":
                q = np.zeros(n, dtype=np.int32)
            else:
                q = rha(src_of(sd, e).astype(np.float64) / s.astype(np.float64), -2**31, 2**31 - 1, np.int32)
            built[b] = ([(float(x), 0) for x in s], q.tobytes())
    for e in R["entries"]:                                   # template order
        p, by = built[e["binvar"]]
        entries.append((e["binvar"], p, by))
    size = write_pack(entries, out)
    print("wrote %s (%d bytes, %d entries)" % (out, size, len(entries)))


def compare(a, b, recipe=None):
    """Byte gate. With a recipe: i32_scalar entries may differ by <=128 units -- the
    converter's own float bias carries 1-ulp noise we cannot reproduce from the
    checkpoint (tpl_round4-6.py); 128 units is below float32 resolution of the bias."""
    A, B = read_pack(a), read_pack(b)
    assert [x[0] for x in A] == [x[0] for x in B], "entry order/names differ"
    rule = {e["binvar"]: e["rule"] for e in json.load(open(recipe))["entries"]} if recipe else {}
    bad_s, bad_b, tol_ok, worst = [], [], 0, 0
    for (n, pa, ba), (_, pb, bb) in zip(A, B):
        if pa != pb:
            bad_s.append(n)
        if ba != bb:
            if rule.get(n) == "i32_scalar":
                d = int(np.abs(np.frombuffer(ba, np.int32).astype(np.int64) - np.frombuffer(bb, np.int32).astype(np.int64)).max())
                worst = max(worst, d)
                if d <= 128:
                    tol_ok += 1
                    continue
            bad_b.append((n, sum(1 for x, y in zip(ba, bb) if x != y)))
    if recipe:
        print("i32_scalar entries within tolerance: %d (worst %d units)" % (tol_ok, worst))
    print("entries %d: scale mismatches %d, byte mismatches %d" % (len(A), len(bad_s), len(bad_b)))
    for n in bad_s[:8]:
        print("  SCALE", n)
    kinds = {}
    for n, k in bad_b:
        kind = "weight_permute" if "permute" in n else ("MatMul" if "MatMul" in n else ("bias" if "bias" in n or "beta" in n else ("weight" if "weight" in n else "other")))
        kinds[kind] = kinds.get(kind, 0) + 1
    print("  byte mismatches by kind:", kinds)
    for n, k in bad_b[:8]:
        print("  BYTES", n, k, "bytes differ")
    same = open(a, "rb").read() == open(b, "rb").read()
    print("files byte-identical:", same)


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"finalize": finalize, "apply": apply, "compare": compare}[cmd](*sys.argv[2:])

#!/usr/bin/env python
"""Phase 1b: which quantization formula reproduces each entry class EXACTLY?

    tpl_rules.py <model.cpp> <model.bin> <checkpoint> <recipe_raw.json>

Classes (dtype, enc, matched?) are tested against candidate formulas; a formula counts
only if it reproduces the template's float32 scales AND the stored bytes for every
tensor in the class. Prints a compact table plus the first counterexample per class.
"""
import json
import re
import sys
import tarfile
from collections import defaultdict

import numpy as np
from safetensors.numpy import load_file

sys.path.insert(0, __import__("os").path.dirname(__file__))
from tpl_recipe import NP, HEADS  # noqa: E402

cpp, binf, ckpt, rj = sys.argv[1:5]
R = json.load(open(rj))
entries, pairs = R["entries"], R["pairs"]
raws = {}
with tarfile.open(binf) as tar:
    for m in tar.getmembers():
        if m.isfile() and m.name.endswith(".raw"):
            raws[m.name.rsplit("/", 1)[-1][:-4]] = tar.extractfile(m).read()
sd = load_file(ckpt)


def f32(x):
    return np.float32(x)


def src_of(e):
    v = sd[e["source"]].astype(np.float32)
    if e["head"] is not None:
        d = v.shape[0] // HEADS
        v = v[e["head"] * d:(e["head"] + 1) * d]
    if e["perm"] == "reshape":
        return v.reshape(e["dims"])
    v = v.reshape(v.shape + (1,) * (len(e["dims"]) - v.ndim))
    return np.transpose(v, e["perm"])


# ---- find each bias's consumer: parse addNode blocks for (input, weight, bias) triples
src = open(cpp, errors="replace").read()
consumer = {}       # bias binvar-name -> (input tensor name, weight tensor name)
for m in re.finditer(r'const char\*\s+inputs_\w+\[\] = \{\s*([^}]*)\};', src):
    names = re.findall(r'"([^"]+)"', m.group(1))
    if len(names) == 3:
        consumer[names[2]] = (names[0], names[1])
act = {}            # activation tensor name -> scale (NATIVE / APP_WRITE tensors)
for m in re.finditer(r'\.name= "([^"]+)",\s*\.type= QNN_TENSOR_TYPE_(?:NATIVE|APP_WRITE|APP_READ),.*?\.scaleOffsetEncoding= \{\.scale= ([-0-9.eE+]+)f', src, re.S):
    act.setdefault(m.group(1), float(m.group(2)))
byname = {e["binvar"]: e for e in entries}
# tensor names in cpp vs binvar: binvar == tensor name here (checked below)
print("bias triples found: %d, activation encodings: %d" % (len(consumer), len(act)))

results = defaultdict(lambda: defaultdict(lambda: [0, 0]))
first_bad = {}


def check(cls, rule, ok, detail=""):
    results[cls][rule][0 if ok else 1] += 1
    if not ok and (cls, rule) not in first_bad:
        first_bad[(cls, rule)] = detail


for e in entries:
    b = e["binvar"]
    P = pairs[b]
    q = np.frombuffer(raws[b], dtype=NP[e["dtype"]]).reshape(e["dims"])
    cls = "%s/%s/%s" % (e["dtype"][14:], e["enc"], "src" if e["source"] else "nosrc")
    if e["dtype"].endswith("SFIXED_POINT_8") and e["enc"] == "axis":
        W = src_of(e).astype(np.float64)
        ax = e["axis"]
        red = tuple(i for i in range(W.ndim) if i != ax)
        mx = np.abs(W).max(axis=red)
        s_t = np.array([f32(p[0]) for p in P])
        for div in (127.0, 127.5, 128.0):
            s = (mx / div).astype(np.float32)
            ok = np.array_equal(s, s_t)
            check(cls, "max|w|/%g" % div, ok, "%s ch0 got %r want %r" % (b, s[0], s_t[0]))
        shape = [1] * W.ndim; shape[ax] = -1
        for name, fn in (("round", np.round), ("rint-half-even", np.rint), ("floor+.5", lambda x: np.floor(x + 0.5))):
            qq = np.clip(fn(W / s_t.astype(np.float64).reshape(shape)), -128, 127).astype(np.int8)
            check(cls, "bytes " + name, np.array_equal(qq, q), b)
    elif e["dtype"].endswith("UFIXED_POINT_8") and e["enc"] == "scalar" and e["source"]:
        W = src_of(e).astype(np.float64)
        s_t, o_t = f32(P[0][0]), P[0][1]
        lo, hi = min(0.0, W.min()), max(0.0, W.max())
        s = f32((hi - lo) / 255.0)
        o = int(np.round(lo / float(s)))
        check(cls, "tf asym scale", s == s_t, "%s got %r want %r (lo %r hi %r)" % (b, s, s_t, lo, hi))
        check(cls, "tf asym offset", o == o_t, "%s got %d want %d" % (b, o, o_t))
        qq = np.clip(np.round(W / float(s_t)) - o_t, 0, 255).astype(np.uint8)
        check(cls, "bytes round(w/s)-off", np.array_equal(qq, q), b)
    elif e["dtype"].endswith("SFIXED_POINT_32"):
        c = consumer.get(b)
        if not c:
            check(cls, "has consumer", False, b); continue
        inp, wn = c
        we = byname.get(wn)
        if inp not in act or not we:
            check(cls, "input act + weight entry", False, "%s inp=%s w=%s" % (b, inp in act, bool(we))); continue
        in_s = act[inp]
        ws = np.array([p[0] for p in pairs[wn]], dtype=np.float64)
        s_t = np.array([f32(p[0]) for p in P])
        s = (np.float64(f32(in_s)) * ws.astype(np.float32).astype(np.float64)).astype(np.float32)
        if s.size != s_t.size and ws.size == 1:
            s = np.repeat(s, s_t.size)
        check(cls, "in_scale*w_scale", np.array_equal(s, s_t), "%s got %r want %r" % (b, s[:1], s_t[:1]))
        if e["source"]:
            B = src_of(e).astype(np.float64)
            qq = np.clip(np.round(B / s_t.astype(np.float64)), -2**31, 2**31 - 1).astype(np.int32)
            check(cls, "bytes round(b/s)", np.array_equal(qq, q), b)
        else:
            check(cls, "bytes all zero", not q.any(), b)
    else:
        check(cls, "no rule (template constant)", True)

for cls in sorted(results):
    print(cls)
    for rule, (ok, bad) in results[cls].items():
        print("   %-28s ok %5d  bad %5d" % (rule, ok, bad))
for k, v in list(first_bad.items())[:12]:
    print("  first bad", k, "->", v)

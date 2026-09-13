#!/usr/bin/env python
"""Parse a qnn-onnx-converter model.cpp into its STATIC (weight-bearing) tensors.

On-device converter MVP, Phase 2 groundwork (docs/ON-DEVICE-CONVERT.md). The
generated model.cpp defines every tensor in its own `addTensor_<name>` function.
A checkpoint swap changes exactly two things per STATIC tensor:
  * the bytes behind `.data=BINVARSTART(<n>)` / `.dataSize=BINLEN(<n>)`
  * its quantizeParams -- either a per-axis array `scaleOffset_<n>[]`
    (AXIS_SCALE_OFFSET) or an inline `{.scaleOffsetEncoding= {.scale=, .offset=}}`
Everything else (topology, activation encodings) stays the template's.

    tpl_parse.py model.cpp [--json out.json]

The JSON lists, per static tensor: name, dtype, rank, dims, encoding kind, axis,
per-axis scale/offset pairs or the scalar pair. It is the template's weight
manifest, and tpl_patch.py rewrites exactly the sites this finds.
"""
import json
import re
import sys

FN = re.compile(r"^static ModelError_t addTensor_(\w+)\(QnnModel& model\)\{$", re.M)
SO_ARR = re.compile(r"Qnn_ScaleOffset_t (scaleOffset_\w+)\[\] = \{(.*?)\};", re.S)
PAIR = re.compile(r"\{\.scale= ([-0-9.eE+]+)f, \.offset= (-?\d+)\}")
DIMS = re.compile(r"uint32_t dimensions_\w+\[\] = \{([^}]*)\};")


def parse(path):
    src = open(path, errors="replace").read()
    starts = [(m.start(), m.group(1)) for m in FN.finditer(src)]
    out = []
    for i, (pos, fname) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(src)
        body = src[pos:end]
        if "QNN_TENSOR_TYPE_STATIC" not in body or "BINVARSTART(" not in body:
            continue
        name = re.search(r'model\.addTensor\("([^"]+)"', body).group(1)
        binvar = re.search(r"BINVARSTART\((\w+)\)", body).group(1)
        dtype = re.search(r"\.dataType= (QNN_DATATYPE_\w+)", body).group(1)
        dims = [int(x) for x in DIMS.search(body).group(1).split(",") if x.strip()]
        t = {"name": name, "binvar": binvar, "dtype": dtype, "dims": dims}
        if "QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET" in body:
            ax = re.search(r"\.axis= (\d+), \.numScaleOffsets= (\d+), \.scaleOffset=(scaleOffset_\w+)", body)
            arr = SO_ARR.search(body)
            pairs = [(float(s), int(o)) for s, o in PAIR.findall(arr.group(2))]
            assert arr.group(1) == ax.group(3) and len(pairs) == int(ax.group(2)), name
            t.update(enc="axis", axis=int(ax.group(1)), pairs=pairs)
        elif "QNN_QUANTIZATION_ENCODING_SCALE_OFFSET" in body:
            s, o = PAIR.search(body.split(".quantizeParams=")[1]).groups()
            t.update(enc="scalar", pairs=[(float(s), int(o))])
        else:
            t.update(enc="none")
        out.append(t)
    return out


if __name__ == "__main__":
    tensors = parse(sys.argv[1])
    from collections import Counter
    print("static tensors with data:", len(tensors))
    for (d, e), n in sorted(Counter((t["dtype"], t["enc"]) for t in tensors).items()):
        print("  %-28s %-7s %5d" % (d, e, n))
    sym = sum(1 for t in tensors if t["enc"] == "axis" and all(o == 0 for _, o in t["pairs"]))
    print("axis-encoded with all offsets 0 (symmetric):", sym)
    for kind in ("axis", "scalar", "none"):
        ex = next((t for t in tensors if t["enc"] == kind), None)
        if ex:
            print("example %-6s %s %s %s %s" % (kind, ex["name"], ex["dtype"], ex["dims"], ex["pairs"][:2] if kind != "none" else ""))
    u8 = [t for t in tensors if t["dtype"] == "QNN_DATATYPE_UFIXED_POINT_8"][:5]
    for t in u8:
        print("  u8:", t["name"], t["dims"], t["enc"], t["pairs"][:1] if t["enc"] != "none" else "")
    if "--json" in sys.argv:
        json.dump(tensors, open(sys.argv[sys.argv.index("--json") + 1], "w"))

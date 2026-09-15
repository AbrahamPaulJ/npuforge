#!/usr/bin/env python
"""recipe.json -> recipe.bin, so the on-device converter needs no JSON parser.

Phase 4 (docs/ON-DEVICE-CONVERT.md). The recipe is authored on the PC as part of a
template bundle, so emitting a compact binary form costs nothing and removes the
most error-prone dependency from the C++ side. The C++ still parses one JSON blob
-- the safetensors header -- but that is a flat, regular structure.

    tpl_recipe_bin.py recipe.json recipe.bin

Format (little-endian throughout):

    "TPLRCP1\\0"            8 bytes
    u32 count
    count x entry:
        u8  rule            0 template, 1 i8_axis, 2 u8_asym,
                            3 i32_scalar, 4 i32_axis_bias, 5 i32_axis_zero
        u8  ndims           1, 2 or 4
        i8  axis            -1 when absent
        i8  head            -1 when absent, else the head index; dims imply its width
        u8  nperm           0 when absent
        u8  perm[4]         only the first nperm are meaningful
        u32 dims[4]         only the first ndims are meaningful
        f32 in_scale        0 when absent (i32_axis_* only)
        u16 + bytes         binvar
        u16 + bytes         source LDM key ("" when absent)
        u16 + bytes         weight binvar ("" when absent)

Entry ORDER is the template's order and must be preserved: the pack writer emits
entries in this order and the context-binary generator matches them up by name,
but a stable order keeps the output byte-comparable against the PC's pack.
"""
import json
import struct
import sys
from pathlib import Path

RULES = {
    "template": 0,
    "i8_axis": 1,
    "u8_asym": 2,
    "i32_scalar": 3,
    "i32_axis_bias": 4,
    "i32_axis_zero": 5,
}


def s(v):
    b = (v or "").encode()
    assert len(b) < 65536
    return struct.pack("<H", len(b)) + b


def main(src, dst):
    entries = json.loads(Path(src).read_text())["entries"]
    out = [b"TPLRCP1\0", struct.pack("<I", len(entries))]
    for e in entries:
        dims = e["dims"]
        perm = e.get("perm")
        # src_of() has a "reshape" branch; no recipe built so far uses it, and the
        # C++ does not implement it. Fail loudly rather than emit something the
        # device would silently get wrong.
        assert perm != "reshape", f"{e['binvar']}: reshape perm is not supported"
        perm = perm or []
        assert len(dims) <= 4 and len(perm) <= 4, e["binvar"]
        head = e.get("head")
        out.append(
            struct.pack(
                "<BBbbB4B4If",
                RULES[e["rule"]],
                len(dims),
                e.get("axis", -1) if e.get("axis") is not None else -1,
                -1 if head is None else head,
                len(perm),
                *(list(perm) + [0] * (4 - len(perm))),
                *(list(dims) + [0] * (4 - len(dims))),
                float(e.get("in_scale") or 0.0),
            )
        )
        out.append(s(e["binvar"]))
        out.append(s(e.get("source")))
        out.append(s(e.get("weight")))
    blob = b"".join(out)
    Path(dst).write_bytes(blob)
    counts = {}
    for e in entries:
        counts[e["rule"]] = counts.get(e["rule"], 0) + 1
    print("wrote %s (%d bytes, %d entries) %s" % (dst, len(blob), len(entries), counts))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])

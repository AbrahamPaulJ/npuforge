#!/usr/bin/env python
"""Write a TPLPACK1 weight pack (format: tpl_runtime.hpp).

On-device converter MVP (docs/ON-DEVICE-CONVERT.md).

    tpl_pack.py identity model.cpp model.bin out.pack
        The template's OWN weights and encodings, straight from the converter's
        output. A libmodel.so built from tpl_patch.py's cpp + this pack must give
        the same context binary / renders as the stock build -- Phase 2's gate.

`write_pack(entries, path)` is the shared writer; the weight tool (Phase 1) calls
it with a new checkpoint's bytes and encodings.
"""
import io
import struct
import sys
import tarfile

sys.path.insert(0, __import__("os").path.dirname(__file__))
from tpl_parse import parse  # noqa: E402

ALIGN = 64


def write_pack(entries, path):
    """entries: list of (binvar, [(scale, offset), ...], bytes)."""
    head = io.BytesIO()
    head.write(b"TPLPACK1")
    head.write(struct.pack("<I", len(entries)))
    # header size is known before offsets: compute it first
    hsize = 12 + sum(2 + len(n.encode()) + 4 + 8 * len(p) + 16 for n, p, _ in entries)
    off = (hsize + ALIGN - 1) // ALIGN * ALIGN
    layout = []
    for n, p, b in entries:
        layout.append(off)
        off = (off + len(b) + ALIGN - 1) // ALIGN * ALIGN
    for (n, p, b), o in zip(entries, layout):
        nb = n.encode()
        head.write(struct.pack("<H", len(nb)) + nb + struct.pack("<I", len(p)))
        for s, z in p:
            head.write(struct.pack("<fi", s, z))
        head.write(struct.pack("<QQ", o, len(b)))
    assert head.tell() == hsize
    with open(path, "wb") as f:
        f.write(head.getvalue())
        for (n, p, b), o in zip(entries, layout):
            f.seek(o)
            f.write(b)
        f.truncate(off)
    return off


def identity(cpp, binf, out):
    tensors = parse(cpp)
    raws = {}
    with tarfile.open(binf) as tar:
        for m in tar.getmembers():
            if m.isfile() and m.name.endswith(".raw"):
                raws[m.name.rsplit("/", 1)[-1][:-4]] = tar.extractfile(m).read()
    entries = []
    for t in tensors:
        b = raws.pop(t["binvar"])
        entries.append((t["binvar"], t["pairs"] if t["enc"] != "none" else [], b))
    print("pack entries: %d; model.bin members unused: %d" % (len(entries), len(raws)))
    size = write_pack(entries, out)
    print("wrote %s (%d bytes)" % (out, size))


if __name__ == "__main__":
    if sys.argv[1] == "identity":
        identity(*sys.argv[2:5])
    else:
        sys.exit(__doc__)

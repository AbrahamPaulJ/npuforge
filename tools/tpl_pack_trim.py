#!/usr/bin/env python
"""Strip a template pack down to what a converter actually needs.

Phase 4 (docs/ON-DEVICE-CONVERT.md). The full template pack is ~863 MB because it
carries the template checkpoint's own quantized weights -- and the recipe
overwrites every one of them. The only payloads that survive conversion are the
`template`-rule entries: u16 graph constants the checkpoint has no source for.

Measured on the DreamShaper template: 257 entries, **~1 KB** of the 860.9 MB.

    tpl_pack_trim.py recipe.json full.pack trimmed.pack

So a converter ships the trimmed pack, not the full one. Verify with tplconv: the
output pack must stay byte-identical to the one built from the full template.
"""
import json
import struct
import sys

sys.path.insert(0, __import__("os").path.dirname(__file__))
from tpl_pack import write_pack  # noqa: E402


def read_pack(path):
    buf = open(path, "rb").read()
    assert buf[:8] == b"TPLPACK1", "not a TPLPACK1 pack"
    (n,) = struct.unpack_from("<I", buf, 8)
    c = 12
    out = []
    for _ in range(n):
        (nl,) = struct.unpack_from("<H", buf, c); c += 2
        name = buf[c:c + nl].decode(); c += nl
        (np_,) = struct.unpack_from("<I", buf, c); c += 4
        pairs = [struct.unpack_from("<fi", buf, c + 8 * k) for k in range(np_)]
        c += 8 * np_
        off, ln = struct.unpack_from("<QQ", buf, c); c += 16
        out.append((name, pairs, buf[off:off + ln]))
    return out


def main(recipe, full, trimmed):
    keep = {e["binvar"] for e in json.load(open(recipe))["entries"] if e["rule"] == "template"}
    entries = [e for e in read_pack(full) if e[0] in keep]
    missing = keep - {e[0] for e in entries}
    assert not missing, f"template pack lacks {len(missing)} needed entries: {sorted(missing)[:3]}"
    size = write_pack(entries, trimmed)
    payload = sum(len(b) for _, _, b in entries)
    print(
        "kept %d of %d entries, %d bytes of payload -> %s (%d bytes)"
        % (len(entries), len(keep), payload, trimmed, size)
    )


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:4])

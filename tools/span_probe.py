#!/usr/bin/env python
"""Weight-span profile of an SD1.5 checkpoint, local or remote.

WHY THIS EXISTS.  A checkpoint converts cleanly only if its weights stay inside
the activation ranges the template was calibrated on, and the statistic that has
predicted that so far is the per-tensor ratio of the checkpoint's weight span
(max|w|) to a reference checkpoint's (docs/LIMITS.md).  Computing it needs the
weights, not just the header -- but NOT all of them: the 404 one-dimensional
UNet tensors (biases and norm gammas) carry the same separation as all 686, for
0.9 MB instead of 1,719 MB.  So a checkpoint can be profiled from a ~1 MB
ranged read, without downloading it (docs/CHECKPOINT-FAMILIES.md).

    span_probe.py dump  <ckpt.safetensors> <out.json>     # full reference table
    span_probe.py probe <ref.json> <ckpt.safetensors|URL> [name]

⚠ `dump` reads every tensor and is for building the reference table once.
`probe` reads only the 1-dim UNet tensors and is the cheap path.

⚠ The reference is a CHOICE and the numbers only mean something against it.
Base SD1.5 is the reasonable one: every SD1.5 checkpoint is a perturbation of it,
so the ratio reads as "distance from the common ancestor" rather than "distance
from whichever finetune we happened to build the template on".

Remote note: Hugging Face redirects every ranged GET to a CDN host, so a session
does not get keep-alive and each range costs ~2 s.  Resolve the redirect once and
range-read the final URL when profiling many checkpoints.
"""
import json
import mmap
import struct
import sys

import numpy as np

UNET = "model.diffusion_model."
FLOAT = {"F16": np.float16, "F32": np.float32, "F64": np.float64}


def as_f32(buf, dtype):
    if dtype == "BF16":
        return (np.frombuffer(buf, np.uint16).astype(np.uint32) << 16).view(np.float32)
    if dtype not in FLOAT:
        return None
    return np.frombuffer(buf, FLOAT[dtype]).astype(np.float32)


def span_of(x):
    """max|w| over the finite values, plus how many were not finite.

    ⚠ Non-finite values are dropped rather than poisoning the max: NaNs in an
    fp16 SD1.5 merge's VAE are common and authentic (docs/LIMITS.md), and a
    single inf would otherwise make every span read as inf.
    """
    fin = np.isfinite(x)
    nbad = int(x.size - fin.sum())
    x = x[fin]
    if not x.size:
        return 0.0, nbad
    return float(np.abs(x).max()), nbad


# ------------------------------------------------------------------ local ---

def read_header(fh):
    (hlen,) = struct.unpack("<Q", fh[:8])
    hdr = json.loads(fh[8:8 + hlen].decode())
    hdr.pop("__metadata__", None)
    return hdr, 8 + hlen


def dump(path, out):
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    hdr, base = read_header(mm)
    res = {}
    for k, v in hdr.items():
        a, b = v["data_offsets"]
        x = as_f32(mm[base + a:base + b], v["dtype"])
        if x is None:
            continue
        s, nbad = span_of(x)
        res[k] = dict(shape=v["shape"], dtype=v["dtype"], n=int(x.size),
                      span=s, nbad=nbad)
    json.dump(res, open(out, "w"))
    print(f"{path}: {len(res)} tensors -> {out}")


# ----------------------------------------------------------------- remote ---

def coalesce(runs, gap):
    out = []
    for s, e in runs:
        if out and s - out[-1][1] < gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def admit(hdr, keys, ref, label):
    """Refuse anything that is not the architecture the reference describes.

    🐞 This guard exists because its absence produced a believable lie.
    `kawaiiRealisticAsian_v01` is an **SDXL** checkpoint — `conditioner.*`
    instead of `cond_stage_model.*`, 1,680 UNet tensors instead of 686 — but
    251 of the 404 probe names collide with SD1.5's, so the probe happily
    reported "p50 0.588, MAX 5.76" and it read exactly like the photoreal×anime
    crossover we were hoping to find. A partial name match across architectures
    is not a measurement. `CheckpointInfo.kt` refuses on the same grounds before
    a conversion; a profiling tool has to refuse on them too.
    """
    missing = [k for k in keys if k not in hdr]
    wrong = [k for k in keys
             if k in hdr and list(hdr[k]["shape"]) != list(ref[k]["shape"])]
    if missing or wrong:
        print(f"{label}: REFUSED -- {len(missing)} of {len(keys)} probe tensors "
              f"absent, {len(wrong)} the wrong shape. Not the reference "
              f"architecture (header has {len(hdr)} tensors, prefixes "
              f"{sorted({k.split('.')[0] for k in hdr})}).")
        return None
    return {k: hdr[k] for k in keys}


class Remote:
    """Just enough of a file to range-read a safetensors over HTTP."""

    def __init__(self, url):
        import requests
        self.url = url
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": "npuforge-span-probe/1"})
        self.bytes = 0
        self.reqs = 0

    def get(self, a, b):
        self.reqs += 1
        r = self.s.get(self.url, headers={"Range": f"bytes={a}-{b}"}, timeout=180)
        r.raise_for_status()
        self.bytes += len(r.content)
        return r.content


def probe(refpath, target, name=None, gap=1 << 18):
    """gap: merge two ranges when the hole between them is smaller than this.

    ⚠ The default 256 KB is measured, not arbitrary. The probe tensors are
    scattered through the file, so the threshold trades requests against wasted
    bytes and the curve is sharp -- for anyloraCheckpoint (0.89 MB actually
    needed): 404 requests / 0.9 MB at gap 0, **105 requests / 1.1 MB at 256 KB**,
    72 requests / **42.7 MB** at 2 MB. Coalescing harder than this buys a handful
    of requests for forty megabytes.
    """
    ref = json.load(open(refpath))
    keys = [k for k in ref
            if k.startswith(UNET) and len(ref[k]["shape"]) == 1 and ref[k]["span"] > 0]

    if target.startswith("http"):
        io_ = Remote(target)
        hlen = struct.unpack("<Q", io_.get(0, 7))[0]
        hdr = json.loads(io_.get(8, 8 + hlen - 1).decode())
        hdr.pop("__metadata__", None)
        base = 8 + hlen
        want = admit(hdr, keys, ref, name or target)
        if want is None:
            return None
        order = sorted(want.items(), key=lambda kv: kv[1]["data_offsets"][0])
        blobs = [(s, io_.get(base + s, base + e - 1)) for s, e in
                 coalesce([list(v["data_offsets"]) for _, v in order], gap)]

        def read(a, b):
            for s, blob in blobs:
                if s <= a and b <= s + len(blob):
                    return blob[a - s:b - s]
            raise KeyError(a)
        cost = f"fetched {io_.bytes/1e6:.2f} MB in {io_.reqs} requests"
    else:
        f = open(target, "rb")
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        hdr, base = read_header(mm)
        want = admit(hdr, keys, ref, name or target)
        if want is None:
            return None
        order = sorted(want.items(), key=lambda kv: kv[1]["data_offsets"][0])

        def read(a, b):
            return mm[base + a:base + b]
        cost = f"read {sum(v['data_offsets'][1]-v['data_offsets'][0] for _,v in order)/1e6:.2f} MB"

    ratios, nbad = {}, 0
    for k, v in order:
        a, b = v["data_offsets"]
        x = as_f32(read(a, b), v["dtype"])
        if x is None:
            continue
        s, bad = span_of(x)
        nbad += bad
        if s:
            ratios[k] = s / ref[k]["span"]

    v = np.array(list(ratios.values()))
    print(f"{name or target}")
    print(f"  {len(v)}/{len(keys)} probe tensors | {cost} | non-finite {nbad}")
    print(f"  span ratio  p50 {np.percentile(v,50):.3f}  p90 {np.percentile(v,90):.3f}"
          f"  p99 {np.percentile(v,99):.3f}  MAX {v.max():.2f}"
          f"  (>2: {(v>2).sum()}, >5: {(v>5).sum()})")
    for k, r in sorted(ratios.items(), key=lambda kv: -kv[1])[:3]:
        print(f"    {r:8.2f}  {k[len(UNET):]}")
    sys.stdout.flush()
    return v


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("dump", "probe"):
        sys.exit(__doc__)
    (dump if sys.argv[1] == "dump" else probe)(*sys.argv[2:])

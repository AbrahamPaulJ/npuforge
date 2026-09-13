#!/usr/bin/env python
"""Rewrite a qnn-onnx-converter model.cpp to load weights from a pack at runtime.

On-device converter MVP, Phase 2 (docs/ON-DEVICE-CONVERT.md).

    tpl_patch.py model.cpp model_tpl.cpp

For every STATIC tensor that has binary data (the 2,127 sites tpl_parse.py
finds), inside its own addTensor_<n> function only:
    .data=BINVARSTART(X)            -> .data=tpl::data("X")
    .dataSize=BINLEN(X)             -> .dataSize=tpl::len("X")
    .scaleOffset=scaleOffset_X      -> .scaleOffset=tpl::axis("X", N)
    {.scaleOffsetEncoding= {..}}    -> {.scaleOffsetEncoding= tpl::scalar("X")}
Activation tensors, params and op configs are untouched, so the graph and its
activation encodings stay exactly the template's.

The pack key is the BINVARSTART symbol (== the <X>.raw member name in model.bin),
because that is what ties bytes to a tensor unambiguously.

⚠ Counts are asserted: every rewrite kind must hit exactly the number of sites the
parser found. A regex that silently misses one tensor would leave a dangling
BINVARSTART, which then fails at link time rather than at runtime -- but a missed
ENCODING would keep the template's literal scale against new weights, which
builds and runs and is wrong. Hence the asserts.
"""
import re
import sys

sys.path.insert(0, __import__("os").path.dirname(__file__))
from tpl_parse import FN  # noqa: E402

src_path, out_path = sys.argv[1], sys.argv[2]
src = open(src_path, errors="replace").read()

starts = [m.start() for m in FN.finditer(src)] + [len(src)]
pieces = [src[:starts[0]]]
n_data = n_len = n_axis = n_scalar = 0
for i in range(len(starts) - 1):
    body = src[starts[i]:starts[i + 1]]
    if "QNN_TENSOR_TYPE_STATIC" in body and "BINVARSTART(" in body:
        x = re.search(r"BINVARSTART\((\w+)\)", body).group(1)
        body, k = re.subn(r"\.data=BINVARSTART\(%s\)" % x, '.data=tpl::data("%s")' % x, body); n_data += k
        body, k = re.subn(r"\.dataSize=BINLEN\(%s\)" % x, '.dataSize=tpl::len("%s")' % x, body); n_len += k
        if "QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET" in body:
            body, k = re.subn(r"\.numScaleOffsets= (\d+), \.scaleOffset=scaleOffset_\w+",
                              lambda m: '.numScaleOffsets= %s, .scaleOffset=tpl::axis("%s", %s)' % (m.group(1), x, m.group(1)),
                              body)
            n_axis += k
        elif "QNN_QUANTIZATION_ENCODING_SCALE_OFFSET" in body:
            head, tail = body.split(".quantizeParams=", 1)
            tail, k = re.subn(r"\{\.scaleOffsetEncoding= \{\.scale= [-0-9.eE+]+f, \.offset= -?\d+\}\}",
                              '{.scaleOffsetEncoding= tpl::scalar("%s")}' % x, tail, count=1)
            body = head + ".quantizeParams=" + tail
            n_scalar += k
    pieces.append(body)
out = "".join(pieces)
# Inlined, not #included: qnn-model-lib-generator copies model.cpp into its own
# build tree, where a sibling header would not be on the include path.
rt = open(__import__("os").path.join(__import__("os").path.dirname(__file__), "tpl_runtime.hpp")).read()
out = out.replace('#include "QnnModel.hpp"', '#include "QnnModel.hpp"\n' + rt.replace("#pragma once", ""), 1)

left = len(re.findall(r"BINVARSTART\(|BINLEN\(", out))
print("rewrote data=%d len=%d axis=%d scalar=%d; BINVARSTART/BINLEN left=%d" % (n_data, n_len, n_axis, n_scalar, left))
assert n_data == n_len and left == 0, "unrewritten binary references remain"
open(out_path, "w").write(out)
print("wrote", out_path)

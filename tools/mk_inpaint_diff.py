#!/usr/bin/env python
"""Build the SD1.5 inpaint difference file from the official pair.

    tools/mk_inpaint_diff.py sd-v1-5-inpainting.ckpt v1-5-pruned-emaonly.safetensors out_prefix

For every UNet tensor: diff = inpaint - base. conv_in is 9-wide in the inpaint
model and 4-wide in the base: channels 0..3 hold inpaint - base, channels 4..8
hold the inpaint model's own mask/masked-latent weights. So applying is always
    out = zero_pad_channels(custom) + diff
Writes <prefix>_f32.safetensors and <prefix>_f16.safetensors (UNet keys only).
"""
import sys
import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import save_file

inp_p, base_p, prefix = sys.argv[1:4]
U = "model.diffusion_model."
sd = torch.load(inp_p, map_location="cpu", weights_only=True)
sd = sd.get("state_dict", sd)
inp = {k: v for k, v in sd.items() if k.startswith(U)}
print("inpaint UNet tensors:", len(inp), "conv_in", tuple(inp[U + "input_blocks.0.0.weight"].shape))
diff = {}
with safe_open(base_p, "np") as B:
    base_keys = {k for k in B.keys() if k.startswith(U)}
    assert base_keys == set(inp), (len(base_keys), len(inp), sorted(base_keys ^ set(inp))[:5])
    for k, v in inp.items():
        a = v.float().numpy()
        b = B.get_tensor(k).astype(np.float32)
        if a.shape != b.shape:
            assert k == U + "input_blocks.0.0.weight" and a.shape[1] == 9 and b.shape[1] == 4
            d = a.copy()
            d[:, :4] -= b
        else:
            d = a - b
        diff[k] = np.ascontiguousarray(d)
absmax = max(float(np.abs(v).max()) for v in diff.values())
n = sum(v.size for v in diff.values())
print(f"diff tensors {len(diff)}, params {n:,}, max|d| {absmax:.4g}")
meta = {"format": "npuforge-sd15-inpaint-diff-v1",
        "recipe": "out = zero_pad_channels(custom) + diff; conv_in ch4-8 are the inpaint model's own"}
save_file(diff, prefix + "_f32.safetensors", metadata=meta)
save_file({k: v.astype(np.float16) for k, v in diff.items()}, prefix + "_f16.safetensors", metadata=meta)
f16err = max(float(np.abs(v.astype(np.float16).astype(np.float32) - v).max()) for v in diff.values())
print(f"max fp16 storage error {f16err:.3g}")

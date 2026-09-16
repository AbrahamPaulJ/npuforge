#!/usr/bin/env python
"""Merge a kohya UNet LoRA into an SD1.5/SDXL checkpoint, in checkpoint space.

    lora_merge.py <base.safetensors> <lora.safetensors> <strength> [out.safetensors]

Why merge before converting rather than at runtime: runtime LoRA needs
`UPDATEABLE_STATIC` tensors, and marking even one costs **2.8x inference** on
this hardware -- a cliff, not a slope (LORA-PROBE.md). A merged checkpoint is an
ordinary model and runs at full speed. The route was rejected before only
because re-converting cost 1-1.5 h on a PC; on-device conversion makes it ~2 min.

Merging in **checkpoint space** also sidesteps the per-head attention split:
`to_q`/`to_k`/`to_v` are merged as plain weights and the recipe then splits them
exactly as it splits any base weight. No new mapping, no new failure mode.

    W' = W + strength * (alpha / rank) * (up @ down)

⚠ Key mapping is done by BUILDING the kohya name from each checkpoint key, never
by reversing it. kohya forms its names by replacing "." with "_" in the LDM path,
and that is not reversible -- `transformer_blocks`, `to_q`, `skip_connection` and
`emb_layers` all contain legitimate underscores. Going forwards is exact.

⚠ The text-encoder half of the LoRA (`lora_te_*`) is DROPPED: this merger covers
the UNet only. For a style LoRA that loses part of the effect. It is reported, not silently ignored.
"""
import re
import sys

import numpy as np
from safetensors.numpy import load_file, save_file


def merge(base_path, lora_path, strength, out_path=None):
    if not np.isfinite(strength):
        raise SystemExit("LoRA strength must be finite")
    base = load_file(base_path)
    lora = load_file(lora_path)

    # Exact FORWARD map: for every checkpoint weight, compute the kohya name it
    # would carry, and index by that. Reversing a kohya name is ambiguous --
    # "." -> "_" is lossy and `to_out_0`, `ff_net_0_proj` and `transformer_blocks_0`
    # all collide with it -- but going forwards is deterministic.
    #
    # ⚠ kohya names SD1.5 LoRAs with **diffusers** block names while the
    # checkpoint uses **LDM** ones, so the block prefix needs translating.
    # Attention blocks, which is all a standard style LoRA touches:
    #     input_blocks.N.1   N in 1,2,4,5,7,8 -> down_blocks.(N-1)//3.attentions.(N-1)%3
    #     middle_block.1                      -> mid_block.attentions.0
    #     output_blocks.N.1  N in 3..11       -> up_blocks.N//3.attentions.N%3
    # Everything inside an attention block is named identically in both.
    def diffusers_prefix(body):
        m = re.match(r"input_blocks\.(\d+)\.1$", body)
        if m:
            n = int(m.group(1))
            return f"down_blocks_{(n - 1) // 3}_attentions_{(n - 1) % 3}"
        if body == "middle_block.1":
            return "mid_block_attentions_0"
        m = re.match(r"output_blocks\.(\d+)\.1$", body)
        if m:
            n = int(m.group(1))
            return f"up_blocks_{n // 3}_attentions_{n % 3}"
        return None

    kohya_of = {}
    for k in base:
        if not (k.startswith("model.diffusion_model.") and k.endswith(".weight")):
            continue
        body = k[len("model.diffusion_model."):-len(".weight")]
        # LDM-native naming, used by some LoRAs
        kohya_of.setdefault("lora_unet_" + body.replace(".", "_"), k)
        # diffusers naming, used by kohya for SD1.5
        for cut in range(len(body)):
            if body[cut] != ".":
                continue
            pre, rest = body[:cut], body[cut + 1:]
            dp = diffusers_prefix(pre)
            if dp:
                kohya_of.setdefault("lora_unet_" + dp + "_" + rest.replace(".", "_"), k)

    SUF = ".lora_down.weight"
    modules = sorted(k[: -len(SUF)] for k in lora
                     if k.startswith("lora_unet_") and k.endswith(SUF))
    text_keys = {k for k in lora if k.startswith(("lora_te", "text_encoder.", "text_encoder_2."))}
    te = sum(k.endswith((SUF, ".lora_A.weight")) for k in text_keys)
    used = set(text_keys)
    for module in modules:
        if module in kohya_of and module + ".lora_up.weight" in lora:
            used.update(module + suffix for suffix in (SUF, ".lora_up.weight", ".alpha")
                        if module + suffix in lora)
    unmatched = sorted(set(lora) - used)
    print(f"LoRA modules: {len(modules)} unet, {te} text-encoder (DROPPED -- text-encoder LoRA merging unsupported)")
    if unmatched:
        raise SystemExit(f"{len(unmatched)} unsupported or unmatched adapter tensors, first: {unmatched[0]}; "
                         "requires standard kohya UNet lora_down/lora_up/alpha (no DoRA, LyCORIS or PEFT)")
    if not modules:
        raise SystemExit("LoRA matched no UNet tensors")

    merged = dict(base)
    applied = skipped = 0
    deltas = []
    for m in modules:
        target = kohya_of.get(m)
        if target is None:
            skipped += 1
            continue
        down = lora[m + ".lora_down.weight"].astype(np.float32)
        up = lora[m + ".lora_up.weight"].astype(np.float32)
        if (down.ndim not in (2, 4) or up.ndim not in (2, 4)
                or min(down.shape) <= 0 or min(up.shape) <= 0 or down.shape[0] != up.shape[1]):
            raise SystemExit(f"{m}: invalid LoRA down/up shapes or rank")
        if up.ndim == 4 and up.shape[-2:] != (1, 1):
            raise SystemExit(f"{m}: spatial LoRA up kernels are unsupported")
        rank = down.shape[0]
        alpha_values = lora.get(m + ".alpha", np.array(rank)).reshape(-1)
        if alpha_values.size != 1 or not np.isfinite(alpha_values[0]):
            raise SystemExit(f"{m}: LoRA alpha must be one finite value")
        alpha = float(alpha_values[0])
        scale = strength * alpha / rank

        W = base[target].astype(np.float32)
        if down.ndim == 2:                      # linear
            delta = up @ down
        elif down.shape[-2:] == (1, 1):         # 1x1 conv
            delta = (up.reshape(up.shape[0], -1) @ down.reshape(down.shape[0], -1))
            delta = delta.reshape(W.shape)
        else:                                   # kxk conv: up is (out, r, 1, 1)
            delta = np.einsum("orij,rckl->ockl", up.reshape(up.shape[0], rank, 1, 1), down)
        if delta.shape != W.shape:
            raise SystemExit(f"{m}: delta {delta.shape} != weight {W.shape}")
        new = W + scale * delta
        deltas.append(float(np.abs(scale * delta).max() / max(float(np.abs(W).max()), 1e-9)))
        merged[target] = new.astype(base[target].dtype)
        applied += 1

    print(f"merged {applied}, unmatched {skipped}, strength {strength}")
    if deltas:
        d = np.array(deltas)
        print("relative weight change per tensor: "
              + "  ".join(f"p{p}={np.percentile(d, p):.3f}" for p in (50, 90, 99))
              + f"  max={d.max():.3f}")
    if out_path:
        save_file(merged, out_path)
        print(f"wrote {out_path}")
    return merged


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    merge(sys.argv[1], sys.argv[2], float(sys.argv[3]),
          sys.argv[4] if len(sys.argv) > 4 else None)

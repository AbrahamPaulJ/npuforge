#!/usr/bin/env python3
"""Author checkpoint-replaceable QNN VAE templates, without exporting a UNet.

Run export with a local standard AutoencoderKL config and checkpoint, then run
qnn-onnx-converter with --float_bitwidth 16 --float_bias_bitwidth 16 --preserve_io
and a fixed --input_dim: SD1.5 uses encoder 1,3,512,512 / decoder 1,4,64,64;
SDXL uses encoder 1,3,1024,1024 / decoder 1,4,128,128.
Run recipe against each emitted model.cpp/model.bin pair, then tpl_patch.py and
qnn-model-lib-generator as for the UNet. SDK-generated artifacts stay untracked.

The export traces 64px examples with dynamic spatial axes, avoiding a full-size
attention allocation during authoring. Runtime dimensions are fixed by QNN's
converter. No image/latent scaling is included: the generator owns that math.
"""
import argparse
import json
import math
from pathlib import Path


def export(args):
    import numpy as np
    import torch
    from accelerate import init_empty_weights
    from diffusers import AutoencoderKL
    from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
    from safetensors import safe_open

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    config = AutoencoderKL.load_config(args.config)
    with init_empty_weights():
        vae = AutoencoderKL.from_config(config)
    with safe_open(args.checkpoint, framework="pt", device="cpu") as source:
        state = {key: source.get_tensor(key) for key in source.keys()
                 if key.startswith("first_stage_model.")}
    mapped = convert_ldm_vae_checkpoint(state, config)
    vae.load_state_dict(mapped, strict=True, assign=True)
    vae = vae.float().eval()
    del state, mapped

    class Encoder(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.vae = model

        def forward(self, image):
            posterior = self.vae.encode(image).latent_dist
            return posterior.mean, posterior.std

    class Decoder(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.vae = model

        def forward(self, latent):
            return self.vae.decode(latent, return_dict=False)[0]

    for name, model, shape, outputs in (
        ("vae_encoder", Encoder(vae), (1, 3, 64, 64), ["mean", "std"]),
        ("vae_decoder", Decoder(vae), (1, 4, 8, 8), ["output"]),
    ):
        destination = Path(args.output) / name
        destination.mkdir(parents=True, exist_ok=True)
        sample = torch.randn(shape)
        with torch.no_grad():
            expected = model(sample)
            values = expected if isinstance(expected, tuple) else (expected,)
            np.savez(destination / "reference.npz", input=sample.numpy(),
                     **{key: value.numpy() for key, value in zip(outputs, values)})
            axes = {"input": {2: "input_height", 3: "input_width"}}
            axes.update({key: {2: "output_height", 3: "output_width"} for key in outputs})
            torch.onnx.export(model, sample, str(destination / "model.onnx"),
                              input_names=["input"], output_names=outputs,
                              dynamic_axes=axes, opset_version=17)
        print(f"Exported {name}; strict checkpoint load passed; reference {shape}", flush=True)


def recipe(args):
    from safetensors import safe_open
    from tpl_apply import finalize, apply as apply_recipe
    from tpl_pack import identity
    from tpl_pack_trim import main as trim
    from tpl_recipe import discover
    from tpl_recipe_bin import main as binary_recipe

    root = Path(args.model)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    discover(str(root / "model.cpp"), str(root / "model.bin"), args.checkpoint,
             str(root / "recipe_raw.json"), source_prefix="first_stage_model.", reject_ambiguous=True)
    finalize(str(root / "model.cpp"), str(root / "recipe_raw.json"), str(root / "recipe.json"))
    data = json.loads((root / "recipe.json").read_text())
    entries = data["entries"]
    used = {entry["source"] for entry in entries if entry.get("source")}
    prefixes = ("first_stage_model.encoder.", "first_stage_model.quant_conv.") if args.component == "vae_encoder" else (
        "first_stage_model.decoder.", "first_stage_model.post_quant_conv.")
    with safe_open(args.checkpoint, framework="np") as source:
        expected = {key for key in source.keys() if key.startswith(prefixes)}
        if used != expected:
            raise ValueError(f"Incomplete VAE mapping: missing={sorted(expected-used)}, extra={sorted(used-expected)}")
        requirements = {"tensors": [{"name": key, "shape": list(source.get_slice(key).get_shape()),
                                      "dtypes": ["F16", "F32"]} for key in sorted(used)]}
    # A template constant must never conceal an unmapped checkpoint-sized tensor.
    constants = [entry for entry in entries if entry["rule"] == "template"]
    if any(math.prod(entry["dims"]) > 4 for entry in constants):
        raise ValueError(f"Unexpected non-scalar VAE constants: {constants}")
    binary_recipe(str(root / "recipe.json"), str(destination / "recipe.bin"))
    identity(str(root / "model.cpp"), str(root / "model.bin"), str(root / "identity.pack"))
    trim(str(root / "recipe.json"), str(root / "identity.pack"), str(destination / "tpl_trim.pack"))
    apply_recipe(str(root / "recipe.json"), str(destination / "tpl_trim.pack"), args.checkpoint,
                 str(root / "reference.pack"))
    if (root / "reference.pack").read_bytes() != (root / "identity.pack").read_bytes():
        raise ValueError("Checkpoint-reconstructed VAE pack differs from QNN converter weights")
    (destination / "sources.txt").write_text("\n".join(sorted(used)) + "\n")
    (destination / "requirements.json").write_text(json.dumps(requirements, indent=2) + "\n")
    print(f"Verified {len(used)} checkpoint tensors and {len(constants)} graph constants byte-for-byte")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export_parser = commands.add_parser("export")
    export_parser.add_argument("--checkpoint", required=True)
    export_parser.add_argument("--config", required=True)
    export_parser.add_argument("--output", required=True)
    recipe_parser = commands.add_parser("recipe")
    recipe_parser.add_argument("--checkpoint", required=True)
    recipe_parser.add_argument("--model", required=True)
    recipe_parser.add_argument("--component", choices=["vae_encoder", "vae_decoder"], required=True)
    recipe_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    {"export": export, "recipe": recipe}[args.command](args)


if __name__ == "__main__":
    main()

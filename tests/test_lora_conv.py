"""DMD2-style convolution/ResNet LoRA mappings and checkpoint-space pack parity."""
import contextlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from lora_merge import merge
from tpl_recipe_bin import main as write_recipe


# Literal LDM/Diffusers correspondences, independent of the implementation's
# block arithmetic. SD1.5's first upsampler is .1; SDXL's is .2.
COMMON = [
    ("input_blocks.3.0.op", "down_blocks_0_downsamplers_0_conv", "conv"),
    ("input_blocks.6.0.op", "down_blocks_1_downsamplers_0_conv", "conv"),
    ("input_blocks.0.0", "conv_in", "conv"),
    ("out.2", "conv_out", "conv"),
    ("time_embed.0", "time_embedding_linear_1", "linear"),
    ("time_embed.2", "time_embedding_linear_2", "linear"),
    ("input_blocks.1.0.in_layers.2", "down_blocks_0_resnets_0_conv1", "conv"),
    ("input_blocks.2.0.out_layers.3", "down_blocks_0_resnets_1_conv2", "conv"),
    ("input_blocks.4.0.emb_layers.1", "down_blocks_1_resnets_0_time_emb_proj", "linear"),
    ("input_blocks.4.0.skip_connection", "down_blocks_1_resnets_0_conv_shortcut", "point"),
    ("middle_block.0.in_layers.2", "mid_block_resnets_0_conv1", "conv"),
    ("middle_block.0.emb_layers.1", "mid_block_resnets_0_time_emb_proj", "linear"),
    ("middle_block.2.out_layers.3", "mid_block_resnets_1_conv2", "conv"),
    ("output_blocks.0.0.in_layers.2", "up_blocks_0_resnets_0_conv1", "conv"),
    ("output_blocks.1.0.out_layers.3", "up_blocks_0_resnets_1_conv2", "conv"),
    ("output_blocks.2.0.emb_layers.1", "up_blocks_0_resnets_2_time_emb_proj", "linear"),
    ("output_blocks.3.0.skip_connection", "up_blocks_1_resnets_0_conv_shortcut", "point"),
    ("input_blocks.4.1.transformer_blocks.0.attn1.to_q",
     "down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q", "linear"),
]
FAMILIES = {
    "sd15": COMMON + [
        ("input_blocks.9.0.op", "down_blocks_2_downsamplers_0_conv", "conv"),
        ("input_blocks.10.0.in_layers.2", "down_blocks_3_resnets_0_conv1", "conv"),
        ("input_blocks.11.0.emb_layers.1", "down_blocks_3_resnets_1_time_emb_proj", "linear"),
        ("output_blocks.11.0.out_layers.3", "up_blocks_3_resnets_2_conv2", "conv"),
        ("output_blocks.2.1.conv", "up_blocks_0_upsamplers_0_conv", "conv"),
        ("output_blocks.5.2.conv", "up_blocks_1_upsamplers_0_conv", "conv"),
        ("output_blocks.8.2.conv", "up_blocks_2_upsamplers_0_conv", "conv"),
    ],
    "sdxl": COMMON + [
        ("label_emb.0.0", "add_embedding_linear_1", "linear"),
        ("label_emb.0.2", "add_embedding_linear_2", "linear"),
        ("input_blocks.8.0.out_layers.3", "down_blocks_2_resnets_1_conv2", "conv"),
        ("output_blocks.8.0.out_layers.3", "up_blocks_2_resnets_2_conv2", "conv"),
        ("output_blocks.2.2.conv", "up_blocks_0_upsamplers_0_conv", "conv"),
        ("output_blocks.5.2.conv", "up_blocks_1_upsamplers_0_conv", "conv"),
    ],
}


def fixtures(cases, adapter_dtype, *, native_names=False):
    base, adapter, expected, recipes = {}, {}, {}, []
    for index, (body, diffusers, kind) in enumerate(cases):
        key = "model.diffusion_model." + body + ".weight"
        module = "lora_unet_" + (body.replace(".", "_") if native_names else diffusers)
        shape = (3, 2) if kind == "linear" else (3, 2, 1, 1) if kind == "point" else (3, 2, 3, 3)
        weight = ((np.arange(np.prod(shape), dtype=np.float32) % 13 - 6) / 7 + index / 16)
        weight = weight.reshape(shape).astype(np.float16)
        down_shape = (2,) + shape[1:]
        down = ((np.arange(np.prod(down_shape), dtype=np.float32) % 11 - 5) / 31)
        down = down.reshape(down_shape).astype(adapter_dtype)
        up = np.array([[0.37, -0.21], [0.13, 0.41], [-0.29, 0.17]], dtype=adapter_dtype)
        if kind != "linear":
            up = up.reshape(3, 2, 1, 1)
        base[key] = weight
        adapter[module + ".lora_down.weight"] = down
        adapter[module + ".lora_up.weight"] = up
        adapter[module + ".alpha"] = np.array(1.5, dtype=adapter_dtype)

        # Explicit rank-2 contraction, followed by the saved-checkpoint dtype.
        # Conv spatial positions stay in OIHW until conversion applies HWIO.
        flat_down = down.astype(np.float32).reshape(2, -1)
        flat_up = up.astype(np.float32).reshape(3, 2)
        delta = np.empty((3, flat_down.shape[1]), dtype=np.float32)
        for output in range(3):
            for column in range(flat_down.shape[1]):
                delta[output, column] = np.float32(
                    np.float32(flat_up[output, 0] * flat_down[0, column])
                    + np.float32(flat_up[output, 1] * flat_down[1, column]))
        expected[key] = (weight.astype(np.float32)
                         + np.float32(0.75 * 1.5 / 2) * delta.reshape(shape)).astype(np.float16)
        perm = [1, 0] if kind == "linear" else [2, 3, 1, 0]
        dims = [shape[axis] for axis in perm]
        for rule, prefix in (("float32", "float"), ("i8_axis", "quant")):
            recipes.append(dict(binvar=f"{prefix}_{index}", source=key, rule=rule,
                                dims=dims, perm=perm, axis=len(dims) - 1))
    base["untouched.weight"] = np.array([0.25, -0.75], dtype=np.float16)
    expected["untouched.weight"] = base["untouched.weight"]
    return base, adapter, expected, recipes


def read_pack(path):
    data = path.read_bytes()
    if data[:8] != b"TPLPACK1":
        raise AssertionError("bad pack magic")
    count = struct.unpack_from("<I", data, 8)[0]
    position = 12
    entries = {}
    for _ in range(count):
        size = struct.unpack_from("<H", data, position)[0]
        position += 2
        name = data[position:position + size].decode()
        position += size
        pairs = struct.unpack_from("<I", data, position)[0]
        position += 4
        scales = [struct.unpack_from("<fi", data, position + i * 8) for i in range(pairs)]
        position += 8 * pairs
        offset, size = struct.unpack_from("<QQ", data, position)
        position += 16
        entries[name] = (scales, data[offset:offset + size])
    return entries


def save_typed_tensors(path, tensors, dtypes=None):
    if dtypes is None:
        save_file(tensors, str(path))
        return
    header, payload = {}, bytearray()
    for name, values in tensors.items():
        dtype = dtypes[name]
        if dtype == "BF16":
            words = np.asarray(values, dtype="<f4").view("<u4")
            if np.any(words & 0xffff):
                raise AssertionError("BF16 fixtures must be exactly representable")
            data = (words >> 16).astype("<u2").tobytes()
        else:
            data = np.asarray(values, dtype="<f2" if dtype == "F16" else "<f4").tobytes()
        start = len(payload)
        payload.extend(data)
        header[name] = dict(dtype=dtype, shape=list(values.shape),
                            data_offsets=[start, len(payload)])
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


class LoraConvolutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory(prefix="lora-conv-test-")
        cls.addClassCleanup(cls.build.cleanup)
        cls.binary = Path(cls.build.name) / "tplconv"
        subprocess.run(["c++", "-O2", "-std=c++17", "-ffp-contract=off",
                        str(ROOT / "native/tplconv.cpp"), "-o", str(cls.binary)],
                       check=True, capture_output=True, text=True, timeout=45)

    def native(self, directory, base, adapter, recipes, name="merged", *,
               base_dtypes=None, adapter_dtypes=None):
        checkpoint = directory / f"{name}.safetensors"
        save_typed_tensors(checkpoint, base, base_dtypes)
        recipe_json = directory / "recipe.json"
        recipe_json.write_text(json.dumps({"entries": recipes}))
        recipe = directory / "recipe.bin"
        with contextlib.redirect_stdout(io.StringIO()):
            write_recipe(str(recipe_json), str(recipe))
        template = directory / "template.pack"
        template.write_bytes(b"TPLPACK1" + struct.pack("<I", 0))
        output = directory / f"{name}.pack"
        command = [str(self.binary), str(recipe), str(template), str(checkpoint), str(output)]
        if adapter is not None:
            lora = directory / "adapter.safetensors"
            save_typed_tensors(lora, adapter, adapter_dtypes)
            command += ["--lora", str(lora) + ":0.75"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        return result, output

    def test_mixed_bf16_unet_and_lora_match_expanded_fp32_packs(self):
        def mixed_storage(tensors):
            expanded, dtypes = {}, {}
            for index, (name, values) in enumerate(tensors.items()):
                # Rotate each module too, so down, up and scalar alpha each
                # exercise BF16, F16 and F32 rather than one fixed dtype.
                dtype = ("BF16", "F16", "F32")[(index + index // 3) % 3]
                values = values.astype(np.float32)
                if dtype == "BF16":
                    words = values.view(np.uint32) & np.uint32(0xffff0000)
                    values = np.asarray(words).view(np.float32)
                elif dtype == "F16":
                    values = values.astype(np.float16).astype(np.float32)
                expanded[name], dtypes[name] = values, dtype
            return expanded, dtypes

        for family, cases in FAMILIES.items():
            for with_lora in (False, True):
                with self.subTest(family=family, lora=with_lora), tempfile.TemporaryDirectory() as tmp:
                    directory = Path(tmp)
                    base, adapter, _, recipes = fixtures(cases, np.float32)
                    base, base_dtypes = mixed_storage(base)
                    adapter, adapter_dtypes = mixed_storage(adapter)
                    if not with_lora:
                        adapter = None
                    result, actual = self.native(directory, base, adapter, recipes,
                                                  base_dtypes=base_dtypes,
                                                  adapter_dtypes=adapter_dtypes)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    result, reference = self.native(directory, base, adapter, recipes, "reference")
                    self.assertEqual(result.returncode, 0, result.stderr)
                    # Includes transformed float32 weights and quantized UNet
                    # weights/scales, with the existing FP16 LoRA merge rounding.
                    self.assertEqual(actual.read_bytes(), reference.read_bytes())

    def test_native_all_conv_resnet_and_embedding_mappings(self):
        for family, cases in FAMILIES.items():
            for dtype in (np.float16, np.float32):
                with self.subTest(family=family, dtype=dtype), tempfile.TemporaryDirectory() as tmp:
                    directory = Path(tmp)
                    base, adapter, expected, recipes = fixtures(cases, dtype)
                    result, actual = self.native(directory, base, adapter, recipes)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn(f"{len(cases)} UNet modules matched", result.stderr)
                    self.assertIn("0 unmatched tensors", result.stderr)
                    result, reference = self.native(directory, expected, None, recipes, "reference")
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(actual.read_bytes(), reference.read_bytes())

                    # Check both full-precision transformed bytes and quantized
                    # bytes/scales independently, not just two converter paths.
                    packed = read_pack(actual)
                    for index, (body, _, kind) in enumerate(cases):
                        weight = expected["model.diffusion_model." + body + ".weight"]
                        perm = (1, 0) if kind == "linear" else (2, 3, 1, 0)
                        transformed = weight.astype(np.float32).transpose(perm)
                        self.assertEqual(packed[f"float_{index}"], ([], transformed.tobytes()))
                        channels = transformed.reshape(-1, 3).astype(np.float64)
                        scales = (np.max(np.abs(channels), axis=0) / 127).astype(np.float32)
                        quotient = channels / scales.astype(np.float64)
                        quantized = (np.sign(quotient) * np.floor(np.abs(quotient) + 0.5))
                        quantized = np.clip(quantized, -127, 127).astype(np.int8)
                        self.assertEqual(packed[f"quant_{index}"],
                                         ([(float(scale), 0) for scale in scales], quantized.tobytes()))

    def test_python_all_mappings_have_correct_merged_weights(self):
        for family, cases in FAMILIES.items():
            for dtype in (np.float16, np.float32):
                with self.subTest(family=family, dtype=dtype), tempfile.TemporaryDirectory() as tmp:
                    directory = Path(tmp)
                    base, adapter, expected, _ = fixtures(cases, dtype)
                    save_file(base, str(directory / "base.safetensors"))
                    save_file(adapter, str(directory / "adapter.safetensors"))
                    with contextlib.redirect_stdout(io.StringIO()):
                        actual = merge(str(directory / "base.safetensors"),
                                       str(directory / "adapter.safetensors"), 0.75)
                    self.assertEqual(actual.keys(), expected.keys())
                    for key, weight in expected.items():
                        np.testing.assert_array_equal(actual[key], weight, err_msg=key)

    def test_ldm_native_aliases_still_merge_spatial_convolutions(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            base, adapter, expected, recipes = fixtures(FAMILIES["sdxl"], np.float32, native_names=True)
            result, output = self.native(directory, base, adapter, recipes)
            self.assertEqual(result.returncode, 0, result.stderr)
            result, reference = self.native(directory, expected, None, recipes, "reference")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(output.read_bytes(), reference.read_bytes())
            with contextlib.redirect_stdout(io.StringIO()):
                actual = merge(str(directory / "merged.safetensors"),
                               str(directory / "adapter.safetensors"), 0.75)
            for key, weight in expected.items():
                np.testing.assert_array_equal(actual[key], weight, err_msg=key)

    def test_unknown_module_warns_and_known_convolution_still_merges(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            base, adapter, expected, recipes = fixtures([COMMON[0]], np.float16)
            unknown = "lora_unet_down_blocks_0_resnets_0_unknown_conv"
            adapter[unknown + ".lora_down.weight"] = np.ones((1, 2, 3, 3), dtype=np.float16)
            adapter[unknown + ".lora_up.weight"] = np.ones((3, 1, 1, 1), dtype=np.float16)
            adapter[unknown + ".alpha"] = np.array(1, dtype=np.float32)
            result, output = self.native(directory, base, adapter, recipes)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("1 UNet modules matched", result.stderr)
            self.assertIn("3 unmatched tensors", result.stderr)
            self.assertIn("unsupported or unmatched adapter tensor", result.stderr)
            self.assertIn("warning:", result.stderr)
            result, reference = self.native(directory, expected, None, recipes, "reference")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(output.read_bytes(), reference.read_bytes())
            warning = io.StringIO()
            with contextlib.redirect_stdout(warning):
                actual = merge(str(directory / "merged.safetensors"),
                               str(directory / "adapter.safetensors"), 0.75)
            self.assertIn("warning:", warning.getvalue())
            self.assertIn("3 unsupported or unmatched adapter tensors", warning.getvalue())
            for key, weight in expected.items():
                np.testing.assert_array_equal(actual[key], weight, err_msg=key)

    def test_no_matched_convolution_still_fails_without_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            base, _, _, recipes = fixtures([COMMON[0]], np.float16)
            unknown = "lora_unet_down_blocks_0_resnets_0_unknown_conv"
            adapter = {
                unknown + ".lora_down.weight": np.ones((1, 2, 3, 3), dtype=np.float16),
                unknown + ".lora_up.weight": np.ones((3, 1, 1, 1), dtype=np.float16),
            }
            result, output = self.native(directory, base, adapter, recipes)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("0 UNet modules matched", result.stderr)
            self.assertIn("matched no tensors", result.stderr)
            self.assertFalse(output.exists())
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(
                    SystemExit, "matched no UNet tensors"):
                merge(str(directory / "merged.safetensors"),
                      str(directory / "adapter.safetensors"), 0.75)


if __name__ == "__main__":
    unittest.main()

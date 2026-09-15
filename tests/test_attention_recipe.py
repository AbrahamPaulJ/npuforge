"""Native/Python pack parity for SD1.5 and SDXL attention head layouts.

Run with the authoring environment: python -m unittest discover -s tests
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from tpl_apply import apply, read_pack, src_of
from tpl_pack import write_pack
from tpl_recipe import get, load_sources
from tpl_recipe_bin import main as write_recipe


class AttentionRecipeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory()
        cls.binary = Path(cls.workspace.name) / "tplconv"
        subprocess.run([
            "c++", "-O2", "-std=c++17", "-ffp-contract=off",
            str(ROOT / "native/tplconv.cpp"), "-o", str(cls.binary),
        ], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.workspace.cleanup()

    def test_attention_heads(self):
        rng = np.random.default_rng(87)
        for heads, width, input_channels in (
            (8, 40, 320), (8, 80, 768), (8, 160, 1280), (10, 64, 640), (20, 64, 2048),
        ):
            with self.subTest(heads=heads, width=width), tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                block_index = {320: 1, 640: 4, 1280: 7}[heads * width]
                attention = "attn1" if input_channels == heads * width else "attn2"
                key = (f"model.diffusion_model.input_blocks.{block_index}.1."
                       f"transformer_blocks.0.{attention}.to_k.weight")
                weight = rng.normal(0, 0.05, (heads * width, input_channels)).astype(np.float16)
                checkpoint = work / "checkpoint.safetensors"
                save_file({key: weight}, str(checkpoint))
                source, candidates = load_sources(checkpoint, None if heads == 8 else width)
                self.assertEqual(candidates[width * input_channels], [(key, h) for h in range(heads)])
                entries = []
                expected = {}
                for h in (0, heads // 2, heads - 1):
                    entry = dict(binvar=f"head_{h}", source=key, head=h,
                                 dims=[1, 1, input_channels, width], perm=[2, 3, 1, 0],
                                 axis=3, enc="axis", dtype="QNN_DATATYPE_SFIXED_POINT_8",
                                 rule="i8_axis")
                    block = weight[h * width:(h + 1) * width].astype(np.float32)
                    np.testing.assert_array_equal(get(source, key, h, block.size), block)
                    np.testing.assert_array_equal(src_of(source, entry), block.T.reshape(entry["dims"]))
                    # Independent expected bytes: quantize checkpoint rows before layout conversion.
                    values = block.astype(np.float64)
                    maxima = np.abs(values).max(axis=1, keepdims=True)
                    scaled = values * 127 / maxima
                    quantized = (np.sign(scaled) * np.floor(np.abs(scaled) + 0.5)).astype(np.int8)
                    expected[entry["binvar"]] = (quantized.T.tobytes(), (maxima[:, 0] / 127).astype(np.float32))
                    entries.append(entry)
                recipe = work / "recipe.json"
                recipe.write_text(json.dumps({"entries": entries}))
                write_recipe(recipe, work / "recipe.bin")
                write_pack([], work / "template.pack")
                apply(recipe, work / "template.pack", checkpoint, work / "python.pack")
                subprocess.run([str(self.binary), str(work / "recipe.bin"),
                                str(work / "template.pack"), str(checkpoint),
                                str(work / "native.pack")], check=True)
                self.assertEqual((work / "native.pack").read_bytes(), (work / "python.pack").read_bytes())
                for name, pairs, payload in read_pack(work / "native.pack"):
                    data, scales = expected[name]
                    self.assertEqual(payload, data)
                    np.testing.assert_array_equal(np.array([s for s, _ in pairs], dtype=np.float32), scales)
                    self.assertEqual([offset for _, offset in pairs], [0] * width)

                # A LoRA must merge the full weight before selecting any head.
                module = (f"lora_unet_down_blocks_{(block_index - 1) // 3}_attentions_0_"
                          f"transformer_blocks_0_{attention}_to_k")
                down = (rng.integers(-8, 9, (1, input_channels)) / 64).astype(np.float16)
                up = (rng.integers(-8, 9, (heads * width, 1)) / 64).astype(np.float16)
                save_file({module + ".lora_down.weight": down,
                           module + ".lora_up.weight": up,
                           module + ".alpha": np.array(0.5, dtype=np.float32)},
                          str(work / "lora.safetensors"))
                merged = (weight.astype(np.float32) + np.float32(0.75 * 0.5)
                          * (up.astype(np.float32) @ down.astype(np.float32))).astype(np.float16)
                save_file({key: merged}, str(work / "merged.safetensors"))
                apply(recipe, work / "template.pack", work / "merged.safetensors", work / "merged.pack")
                subprocess.run([str(self.binary), str(work / "recipe.bin"),
                                str(work / "template.pack"), str(checkpoint),
                                str(work / "lora.pack"), "--lora",
                                str(work / "lora.safetensors") + ":0.75"], check=True)
                self.assertEqual((work / "lora.pack").read_bytes(), (work / "merged.pack").read_bytes())


if __name__ == "__main__":
    unittest.main()

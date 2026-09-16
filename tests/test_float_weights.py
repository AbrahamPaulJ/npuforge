"""Exercise actual native VAE pack bytes, including IEEE half rounding and layout."""
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FloatWeightsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.work = Path(cls.directory.name)
        cls.native = cls.work / "tplconv"
        subprocess.run(["c++", "-O2", "-std=c++17", "-ffp-contract=off",
                        str(ROOT / "native/tplconv.cpp"), "-o", str(cls.native)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def convert(self, values, shape, dtype, rule, dims, perm, truncate=0):
        data = struct.pack("<" + ("e" if dtype == "F16" else "f") * len(values), *values)
        header = json.dumps({"first_stage_model.weight": {
            "dtype": dtype, "shape": shape, "data_offsets": [0, len(data)]}}).encode()
        checkpoint = self.work / "input.safetensors"
        checkpoint.write_bytes(struct.pack("<Q", len(header)) + header + data)
        if truncate:
            checkpoint.write_bytes(checkpoint.read_bytes()[:-truncate])
        recipe = self.work / "recipe.json"
        recipe.write_text(json.dumps({"entries": [{
            "binvar": "vae_weight", "source": "first_stage_model.weight",
            "rule": rule, "dims": dims, "perm": perm}]}))
        subprocess.run(["python3", str(ROOT / "tools/tpl_recipe_bin.py"),
                        str(recipe), str(self.work / "recipe.bin")], check=True,
                       capture_output=True)
        template = self.work / "template.pack"
        template.write_bytes(b"TPLPACK1" + struct.pack("<I", 0))
        output = self.work / "output.pack"
        result = subprocess.run([str(self.native), str(self.work / "recipe.bin"),
                                 str(template), str(checkpoint), str(output)],
                                capture_output=True, text=True)
        if result.returncode:
            return result, None
        blob = output.read_bytes()
        self.assertEqual(blob[:12], b"TPLPACK1" + struct.pack("<I", 1))
        length = struct.unpack_from("<H", blob, 12)[0]
        pos = 14 + length
        pairs, offset, size = struct.unpack_from("<IQQ", blob, pos)
        self.assertEqual(pairs, 0, "floating weights must not carry quantization scales")
        return result, blob[offset:offset + size]

    def test_fp32_to_fp16_ieee_rounding(self):
        values = [0.0, -0.0, 2**-25, 3*2**-25, 2**-14, 65504.0,
                  1 + 2**-11, 1 + 3*2**-11, -1 - 2**-11, -1 - 3*2**-11]
        result, actual = self.convert(values, [10], "F32", "float16", [10], [0])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(actual, struct.pack("<10e", *values))

    def test_fp16_conv_weight_to_fp32_hwio(self):
        values = list(range(-8, 8))
        result, actual = self.convert(values, [2, 2, 2, 2], "F16", "float32",
                                      [2, 2, 2, 2], [2, 3, 1, 0])
        self.assertEqual(result.returncode, 0, result.stderr)
        expected = [values[((o*2 + i)*2 + h)*2 + w]
                    for h in range(2) for w in range(2)
                    for i in range(2) for o in range(2)]
        self.assertEqual(actual, struct.pack("<16f", *expected))

    def test_attention_squeezes_only_trailing_singletons(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        result, actual = self.convert(values, [2, 3, 1, 1], "F32", "float16",
                                      [3, 2], [1, 0])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(actual, struct.pack("<6e", 1, 4, 2, 5, 3, 6))
        result, _ = self.convert(values, [2, 1, 1, 3], "F32", "float16",
                                 [3, 2], [1, 0])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("rank", result.stderr)

    def test_invalid_floating_weights_fail_conversion(self):
        for value, rule in [(float("nan"), "float16"), (65536.0, "float16"),
                            (float("inf"), "float32")]:
            with self.subTest(value=value, rule=rule):
                result, _ = self.convert([value], [1], "F32", rule, [1], [0])
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("non-finite", result.stderr)

    def test_truncated_checkpoint_fails_before_reading_tensor(self):
        result, _ = self.convert([1.0, 2.0], [2], "F32", "float16", [2], [0], truncate=4)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("tensor data exceeds checkpoint", result.stderr)


if __name__ == "__main__":
    unittest.main()

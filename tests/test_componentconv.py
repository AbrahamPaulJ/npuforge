"""Exercise native CLIP output bytes and malformed checkpoint handling, without models."""
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def string(value):
    data = value.encode()
    return struct.pack("<H", len(data)) + data


def rule(*, encoding=0, flags=0, rows=1, cols=1, source_shape=None,
         source_row=0, offset=4, alpha=128, name="weight", multiplier=1.0):
    source_shape = source_shape or [rows, cols]
    source_rows, source_cols = (source_shape[0], 1) if len(source_shape) == 1 else source_shape
    return struct.pack("<BBBBQQIIIIIf", 0, encoding, flags, len(source_shape),
                       offset, alpha, rows, cols, source_rows, source_cols,
                       source_row, multiplier) + string(name)


class ComponentConversionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory()
        cls.binary = Path(cls.build.name) / "componentconv"
        subprocess.run(["c++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                        "-fno-fast-math", str(ROOT / "native/componentconv.cpp"),
                        "-o", str(cls.binary)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.build.cleanup()

    def convert(self, values, shape, rules, *, dtype="F32", header_change=None,
                truncate=0, recipe_truncate=0):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            payload = struct.pack("<" + ("e" if dtype == "F16" else "f") * len(values), *values)
            header = {"weight": {"dtype": dtype, "shape": shape,
                                  "data_offsets": [0, len(payload)]}}
            if header_change:
                header_change(header)
            raw = json.dumps(header).encode()
            checkpoint = struct.pack("<Q", len(raw)) + raw + payload
            (directory / "input.safetensors").write_bytes(checkpoint[:-truncate] if truncate else checkpoint)
            recipe = (b"CLIPRCP1" + struct.pack("<I", 1) + string("output.bin")
                      + struct.pack("<QI", 256, 2)
                      + struct.pack("<QI", 0, 4) + b"HEAD"
                      + struct.pack("<QI", 252, 4) + b"TAIL"
                      + struct.pack("<I", len(rules)) + b"".join(rules))
            (directory / "clip_recipe.bin").write_bytes(recipe[:-recipe_truncate] if recipe_truncate else recipe)
            output = directory / "out"
            result = subprocess.run([str(self.binary), str(directory),
                                     str(directory / "input.safetensors"), str(output)],
                                    capture_output=True, text=True)
            data = (output / "output.bin").read_bytes() if (output / "output.bin").exists() else None
            return result, data, output.exists()

    def successful(self, *args, **kwargs):
        result, data, _ = self.convert(*args, **kwargs)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(data[:4], b"HEAD")
        self.assertEqual(data[-4:], b"TAIL")
        return data

    def test_ieee_half_rounding_and_preserved_sign(self):
        values = [0.0, -0.0, 2**-25, 3*2**-25, 2**-14, 65504.0,
                  1 + 2**-11, 1 + 3*2**-11, -1 - 2**-11, -1 - 3*2**-11]
        data = self.successful(values, [1, 10], [rule(encoding=1, cols=10)])
        self.assertEqual(data[4:24], struct.pack("<10e", *values))

    def test_half_subnormals_decode_to_f32(self):
        values = [0.0, -0.0, 2**-24, -2**-24, 2**-14, 65504.0]
        data = self.successful(values, [6], [rule(rows=6, source_shape=[6])], dtype="F16")
        self.assertEqual(data[4:28], struct.pack("<6f", *values))

    def test_fused_qkv_slice(self):
        data = self.successful(list(range(18)), [6, 3],
                               [rule(rows=2, cols=3, source_shape=[6, 3], source_row=2)])
        self.assertEqual(data[4:28], struct.pack("<6f", 6, 7, 8, 9, 10, 11))

    def test_projection_transpose(self):
        data = self.successful([1, 2, 3, 4, 5, 6], [2, 3],
                               [rule(flags=1, rows=3, cols=2, source_shape=[2, 3])])
        self.assertEqual(data[4:28], struct.pack("<6f", 1, 4, 2, 5, 3, 6))

    def test_symmetric_rounding_and_zero_scale(self):
        # absmax 127 => exact scale 1. MNN rounds ties away from zero.
        values = [-127, -1.5, -0.5, 0.5, 1.5, 127, 0, 0, 0, 0, 0, 0]
        data = self.successful(values, [2, 6], [rule(encoding=2, rows=2, cols=6)])
        self.assertEqual(data[4:16], bytes([1, 126, 127, 129, 130, 255] + [128]*6))
        self.assertEqual(data[128:136], struct.pack("<2f", 1, 0))

    def test_asymmetric_rounding_and_constant_row(self):
        # range 255 => exact scale 1, zero-based table indices.
        values = [-10, -9.5, -8.5, 245, 7, 7, 7, 7]
        data = self.successful(values, [2, 4], [rule(encoding=3, rows=2, cols=4)])
        self.assertEqual(data[4:12], bytes([0, 1, 2, 255, 0, 0, 0, 0]))
        self.assertEqual(data[128:144], struct.pack("<4f", -10, 1, 7, 0))

    def test_quantizer_small_scale_threshold(self):
        for encoding, expected in ((2, 128), (3, 0)):
            with self.subTest(encoding=encoding):
                data = self.successful([-1e-8, 1e-8], [1, 2], [rule(encoding=encoding, cols=2)])
                self.assertEqual(data[4:6], bytes([expected, expected]))

    def test_denormal_normalization_only_when_flagged(self):
        for flags, expected in ((0, [-0.0, 2**-149, -2**-149]), (4, [0.0]*3)):
            with self.subTest(flags=flags):
                data = self.successful([-0.0, 2**-149, -2**-149], [1, 3], [rule(flags=flags, cols=3)])
                self.assertEqual(data[4:16], struct.pack("<3f", *expected))

    def test_half_overflow_clamp_requires_explicit_rule_flag(self):
        result, _, _ = self.convert([65536, -65536], [1, 2], [rule(encoding=1, cols=2)])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exceeds FP16 range", result.stderr)
        data = self.successful([65536, -65536], [1, 2], [rule(encoding=1, flags=2, cols=2)])
        self.assertEqual(data[4:8], struct.pack("<2e", 65504, -65504))

    def test_missing_or_bad_metadata_creates_no_output(self):
        variants = {
            "missing": lambda h: h.clear(),
            "shape": lambda h: h["weight"].update(shape=[2, 1]),
            "rank": lambda h: h["weight"].update(shape=[2]),
            "dtype": lambda h: h["weight"].update(dtype="BF16"),
            "bytes": lambda h: h["weight"].update(data_offsets=[0, 4]),
            "range": lambda h: h["weight"].update(data_offsets=[8, 16]),
        }
        for case, change in variants.items():
            with self.subTest(case=case):
                result, data, exists = self.convert([1, 2], [1, 2], [rule(cols=2)], header_change=change)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(exists)
                self.assertIsNone(data)

    def test_truncated_files_create_no_output(self):
        for options in ({"truncate": 1}, {"recipe_truncate": 1}):
            with self.subTest(options=options):
                result, _, exists = self.convert([1], [1, 1], [rule()], **options)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(exists)

    def test_invalid_rule_range_or_slice_creates_no_output(self):
        for invalid in (rule(offset=254), rule(source_row=1), rule(encoding=2, alpha=254)):
            result, _, exists = self.convert([1], [1, 1], [invalid])
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(exists)

    def test_nonfinite_source_and_quantization_overflow_fail(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                result, _, _ = self.convert([value], [1, 1], [rule()])
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("nonfinite", result.stderr)
        result, _, _ = self.convert([-3.4e38, 3.4e38], [1, 2], [rule(encoding=3, cols=2)])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("scale overflow", result.stderr)


if __name__ == "__main__":
    unittest.main()

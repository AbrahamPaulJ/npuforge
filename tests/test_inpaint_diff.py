"""Add-difference inpainting: tplconv --inpaint-diff against its definition and
against the Python reference (tools/tpl_apply.py apply --inpaint-diff)."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[1]
U = "model.diffusion_model."
CONV = U + "input_blocks.0.0.weight"
LIN = U + "out.2.weight"


class InpaintDiffTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.work = Path(cls.directory.name)
        cls.native = cls.work / "tplconv"
        subprocess.run(["c++", "-O2", "-std=c++17", "-ffp-contract=off",
                        str(ROOT / "native/tplconv.cpp"), "-o", str(cls.native)], check=True)
        rng = np.random.default_rng(7)
        cls.custom = {CONV: rng.standard_normal((2, 4, 3, 3)).astype(np.float16),
                      LIN: rng.standard_normal((3, 5)).astype(np.float16)}
        cls.diff = {CONV: (rng.standard_normal((2, 9, 3, 3)) * 0.1).astype(np.float16),
                    LIN: (rng.standard_normal((3, 5)) * 0.1).astype(np.float16)}
        save_file(cls.custom, str(cls.work / "custom.safetensors"))
        save_file(cls.diff, str(cls.work / "diff.safetensors"))
        # conv_in is emitted HWIO like the real template; the linear weight as is.
        recipe = {"entries": [
            {"binvar": "conv_in", "source": CONV, "rule": "float32", "dims": [3, 3, 9, 2],
             "perm": [2, 3, 1, 0], "head": None},
            {"binvar": "linear", "source": LIN, "rule": "float32", "dims": [3, 5],
             "perm": [0, 1], "head": None}]}
        (cls.work / "recipe.json").write_text(json.dumps(recipe))
        subprocess.run(["python3", str(ROOT / "tools/tpl_recipe_bin.py"), str(cls.work / "recipe.json"),
                        str(cls.work / "recipe.bin")], check=True, capture_output=True)
        (cls.work / "template.pack").write_bytes(b"TPLPACK1" + (0).to_bytes(4, "little"))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def native_run(self, diff="diff.safetensors", out="native.pack"):
        return subprocess.run([str(self.native), str(self.work / "recipe.bin"),
                               str(self.work / "template.pack"), str(self.work / "custom.safetensors"),
                               str(self.work / out), "--inpaint-diff", str(self.work / diff)],
                              capture_output=True, text=True)

    def payloads(self, pack):
        blob = (self.work / pack).read_bytes()
        count = int.from_bytes(blob[8:12], "little")
        pos, out = 12, {}
        for _ in range(count):
            n = int.from_bytes(blob[pos:pos + 2], "little"); pos += 2
            name = blob[pos:pos + n].decode(); pos += n
            pairs = int.from_bytes(blob[pos:pos + 4], "little"); pos += 4 + 8 * pairs
            off = int.from_bytes(blob[pos:pos + 8], "little"); size = int.from_bytes(blob[pos + 8:pos + 16], "little")
            pos += 16
            out[name] = np.frombuffer(blob[off:off + size], np.float32)
        return out

    def test_conv_in_is_zero_padded_then_added(self):
        result = self.native_run()
        self.assertEqual(result.returncode, 0, result.stderr)
        got = self.payloads("native.pack")
        wide = np.zeros((2, 9, 3, 3), np.float32)
        wide[:, :4] = self.custom[CONV].astype(np.float32)
        expected = np.transpose(wide + self.diff[CONV].astype(np.float32), (2, 3, 1, 0)).ravel()
        np.testing.assert_array_equal(got["conv_in"], expected)
        # channels 4..8 are the diff's own values exactly
        self.assertTrue(np.array_equal(np.transpose(got["conv_in"].reshape(3, 3, 9, 2), (3, 2, 0, 1))[:, 4:],
                                       self.diff[CONV][:, 4:].astype(np.float32)))

    def test_same_shape_tensor_is_added_in_float32(self):
        self.assertEqual(self.native_run().returncode, 0)
        expected = self.custom[LIN].astype(np.float32) + self.diff[LIN].astype(np.float32)
        np.testing.assert_array_equal(self.payloads("native.pack")["linear"], expected.ravel())

    def test_native_matches_python_reference_bytes(self):
        self.assertEqual(self.native_run().returncode, 0)
        subprocess.run(["python3", str(ROOT / "tools/tpl_apply.py"), "apply", str(self.work / "recipe.json"),
                        str(self.work / "template.pack"), str(self.work / "custom.safetensors"),
                        str(self.work / "python.pack"), "--inpaint-diff", str(self.work / "diff.safetensors")],
                       check=True, capture_output=True)
        self.assertEqual((self.work / "native.pack").read_bytes(), (self.work / "python.pack").read_bytes())

    def test_missing_diff_tensor_is_an_error(self):
        save_file({CONV: self.diff[CONV]}, str(self.work / "partial.safetensors"))
        result = self.native_run("partial.safetensors", "partial.pack")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("lacks", result.stderr)

    def test_incompatible_shape_is_an_error(self):
        bad = dict(self.diff)
        bad[LIN] = np.zeros((3, 6), np.float16)
        save_file(bad, str(self.work / "bad.safetensors"))
        result = self.native_run("bad.safetensors", "bad.pack")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not fit", result.stderr)


if __name__ == "__main__":
    unittest.main()

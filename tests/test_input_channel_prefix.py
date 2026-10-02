"""--input-channel-prefix (Swap v3): a sample-side conv_in entry that maps 4 input channels reads the
LEADING 4 of the 9-wide add-differenced conv_in. tplconv against its definition and against the
Python reference (tools/tpl_apply.py apply --inpaint-diff ... --input-channel-prefix)."""
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


class InputChannelPrefixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.work = Path(cls.directory.name)
        cls.native = cls.work / "tplconv"
        subprocess.run(["c++", "-O2", "-std=c++17", "-ffp-contract=off",
                        str(ROOT / "native/tplconv.cpp"), "-o", str(cls.native)], check=True)
        rng = np.random.default_rng(11)
        cls.custom = {CONV: rng.standard_normal((2, 4, 3, 3)).astype(np.float16)}
        cls.diff = {CONV: (rng.standard_normal((2, 9, 3, 3)) * 0.1).astype(np.float16)}
        save_file(cls.custom, str(cls.work / "custom.safetensors"))
        save_file(cls.diff, str(cls.work / "diff.safetensors"))
        # The v3 template's sample-side conv: 4 input channels, emitted HWIO.
        recipe = {"entries": [
            {"binvar": "conv_in_a", "source": CONV, "rule": "float32", "dims": [3, 3, 4, 2],
             "perm": [2, 3, 1, 0], "head": None}]}
        (cls.work / "recipe.json").write_text(json.dumps(recipe))
        subprocess.run(["python3", str(ROOT / "tools/tpl_recipe_bin.py"), str(cls.work / "recipe.json"),
                        str(cls.work / "recipe.bin")], check=True, capture_output=True)
        (cls.work / "template.pack").write_bytes(b"TPLPACK1" + (0).to_bytes(4, "little"))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def native_run(self, out, *extra):
        return subprocess.run([str(self.native), str(self.work / "recipe.bin"), str(self.work / "template.pack"),
                               str(self.work / "custom.safetensors"), str(self.work / out), *extra],
                              capture_output=True, text=True)

    def payload(self, pack):
        blob = (self.work / pack).read_bytes()
        pos = 12
        n = int.from_bytes(blob[pos:pos + 2], "little"); pos += 2 + n
        pairs = int.from_bytes(blob[pos:pos + 4], "little"); pos += 4 + 8 * pairs
        off = int.from_bytes(blob[pos:pos + 8], "little"); size = int.from_bytes(blob[pos + 8:pos + 16], "little")
        return np.frombuffer(blob[off:off + size], np.float32)

    def test_leading_channels_of_the_add_differenced_conv(self):
        r = self.native_run("inp.pack", "--inpaint-diff", str(self.work / "diff.safetensors"), "--input-channel-prefix")
        self.assertEqual(r.returncode, 0, r.stderr)
        wide = np.zeros((2, 9, 3, 3), np.float32)
        wide[:, :4] = self.custom[CONV].astype(np.float32)
        wide += self.diff[CONV].astype(np.float32)
        np.testing.assert_array_equal(self.payload("inp.pack"), np.transpose(wide[:, :4], (2, 3, 1, 0)).ravel())

    def test_native_matches_python_reference_bytes(self):
        args = ["--inpaint-diff", str(self.work / "diff.safetensors"), "--input-channel-prefix"]
        self.assertEqual(self.native_run("native.pack", *args).returncode, 0)
        subprocess.run(["python3", str(ROOT / "tools/tpl_apply.py"), "apply", str(self.work / "recipe.json"),
                        str(self.work / "template.pack"), str(self.work / "custom.safetensors"),
                        str(self.work / "python.pack"), *args], check=True, capture_output=True)
        self.assertEqual((self.work / "native.pack").read_bytes(), (self.work / "python.pack").read_bytes())

    def test_plain_checkpoint_is_unchanged_by_the_flag(self):
        self.assertEqual(self.native_run("plain.pack").returncode, 0)
        self.assertEqual(self.native_run("plain_flag.pack", "--input-channel-prefix").returncode, 0)
        self.assertEqual((self.work / "plain.pack").read_bytes(), (self.work / "plain_flag.pack").read_bytes())

    def test_without_the_flag_a_wider_source_stays_an_error(self):
        r = self.native_run("bad.pack", "--inpaint-diff", str(self.work / "diff.safetensors"))
        self.assertNotEqual(r.returncode, 0)


if __name__ == "__main__":
    unittest.main()

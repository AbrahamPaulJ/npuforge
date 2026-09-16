"""Python/native agreement on supported LoRA math and unsupported adapters."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from safetensors.numpy import save_file

from test_lora_merge import KEYS, MODULES, save_tensors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from lora_merge import merge


class LoraReferenceTest(unittest.TestCase):
    def test_supported_sdxl_modules_and_unsupported_adapters(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "base.safetensors"
            lora = Path(directory) / "lora.safetensors"
            save_file({key: np.array([[1, 2], [3, 4]], dtype=np.float16) for key in KEYS}, str(base))
            tensors = {}
            for module in MODULES:
                tensors[module + ".lora_down.weight"] = ([2, 2], [0.5, 0.25, 0.125, -0.25])
                tensors[module + ".lora_up.weight"] = ([2, 2], [0.25, 0.5, -0.25, 0.125])
                tensors[module + ".alpha"] = ([], [1.0])
            save_tensors(lora, tensors)
            with contextlib.redirect_stdout(io.StringIO()):
                result = merge(str(base), str(lora), 0.75)
            expected = np.array([[1.0703125, 1.9765625], [2.958984375, 3.96484375]], dtype=np.float16)
            for weight in result.values():
                np.testing.assert_array_equal(weight, expected)
            for suffix, values, message in (
                (".dora_scale", ([2], [1, 1]), "unsupported or unmatched"),
                (".lora_up.weight", ([2, 1], [1, 1]), "invalid LoRA down/up shapes"),
                (".alpha", ([2], [1, 1]), "alpha must be one finite value"),
            ):
                with self.subTest(suffix=suffix):
                    altered = dict(tensors)
                    altered[MODULES[0] + suffix] = values
                    save_tensors(lora, altered)
                    with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(SystemExit, message):
                        merge(str(base), str(lora), 0.75)


if __name__ == "__main__":
    unittest.main()

"""VAE authoring must distinguish component sources at the emitted precision."""
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from tpl_recipe import discover


class VaeRecipeTest(unittest.TestCase):
    def author(self, root, sources):
        checkpoint = root / "source.safetensors"
        save_file(sources, str(checkpoint))
        raw = np.array([1.0, 2.0], dtype="<f2").tobytes()
        archive = root / "model.bin"
        with tarfile.open(archive, "w") as output:
            entry = tarfile.TarInfo("vae_weight.raw")
            entry.size = len(raw)
            output.addfile(entry, io.BytesIO(raw))
        tensor = {"name": "vae_weight", "binvar": "vae_weight", "dims": [2],
                  "dtype": "QNN_DATATYPE_FLOAT_16", "enc": "none"}
        destination = root / "recipe.json"
        with patch("tpl_recipe.parse", return_value=[tensor]):
            discover("unused.cpp", str(archive), str(checkpoint), str(destination),
                     source_prefix="first_stage_model.", reject_ambiguous=True)
        return json.loads(destination.read_text())

    def test_float32_source_matches_after_half_rounding_with_vae_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            source = np.array([1.0001, 2.0002], dtype=np.float32)
            result = self.author(Path(directory), {
                "first_stage_model.decoder.weight": source,
                "model.diffusion_model.unrelated.weight": source,
            })
            self.assertEqual(result["entries"][0]["source"],
                             "first_stage_model.decoder.weight")

    def test_half_precision_collision_requires_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "Ambiguous checkpoint sources"):
                self.author(root, {
                    "first_stage_model.decoder.first.weight": np.array([1.0001, 2.0002], np.float32),
                    "first_stage_model.decoder.second.weight": np.array([1.0002, 2.0003], np.float32),
                })
            self.assertFalse((root / "recipe.json").exists())


if __name__ == "__main__":
    unittest.main()

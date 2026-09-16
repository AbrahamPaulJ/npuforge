"""Small native LoRA regressions; only Python's standard library and c++ needed."""
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
KEYS = [
    "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_q.weight",
    "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_k.weight",
    "model.diffusion_model.output_blocks.0.1.transformer_blocks.9.attn2.to_v.weight",
    "model.diffusion_model.middle_block.1.transformer_blocks.9.ff.net.2.weight",
]
MODULES = [
    "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q",
    "lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_k",
    "lora_unet_up_blocks_0_attentions_0_transformer_blocks_9_attn2_to_v",
    "lora_unet_mid_block_attentions_0_transformer_blocks_9_ff_net_2",
]


def save_tensors(path, tensors):
    header, payload = {}, bytearray()
    for name, (shape, values) in tensors.items():
        start = len(payload)
        payload.extend(struct.pack("<" + "f" * len(values), *values))
        header[name] = dict(dtype="F32", shape=shape,
                            data_offsets=[start, len(payload)])
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


class LoraMergeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory()
        cls.work = Path(cls.workspace.name)
        cls.binary = cls.work / "lora_probe"
        driver = cls.work / "lora_probe.cpp"
        key_literals = ",\n".join(json.dumps(key) for key in KEYS)
        driver.write_text(
            '#define main tplconv_main\n'
            f'#include {json.dumps(str(ROOT / "native/tplconv.cpp"))}\n'
            '#undef main\n'
            'int main(int argc, char** argv) {\n'
            f'  std::vector<std::string> keys = {{{key_literals}}};\n'
            '  LoraSet loras(std::stoull(argv[1]));\n'
            '  for (int i = 2; i < argc; ++i) loras.add(argv[i], keys);\n'
            '  for (int i : {0, 1, 0, 2, 1, 3, 0}) {\n'
            '    std::vector<float> weight = {1, 2, 3, 4};\n'
            '    loras.apply(keys[i], weight);\n'
            '    unsigned mask = 0;\n'
            '    for (int j = 0; j < 4; ++j)\n'
            '      if (loras.cache.count(keys[j])) mask |= (1u << j);\n'
            '    printf("%d %zu %u", i, loras.cache_bytes, mask);\n'
            '    for (float value : weight) printf(" %.17g", value);\n'
            '    puts("");\n'
            '  }\n'
            '}\n'
        )
        subprocess.run(["c++", "-O2", "-std=c++17", "-ffp-contract=off",
                        str(driver), "-o", str(cls.binary)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.workspace.cleanup()

    def adapter(self, name="lora.safetensors", alpha=True, extra=None):
        tensors = {}
        for module in MODULES:
            tensors[module + ".lora_down.weight"] = ([2, 2], [0.5, 0.25, 0.125, -0.25])
            tensors[module + ".lora_up.weight"] = ([2, 2], [0.25, 0.5, -0.25, 0.125])
            if alpha:
                tensors[module + ".alpha"] = ([], [1.0])
        tensors.update(extra or {})
        path = self.work / name
        save_tensors(path, tensors)
        return path

    def run_probe(self, budget, *adapters):
        return subprocess.run([str(self.binary), str(budget), *map(str, adapters)],
                              capture_output=True, text=True)

    @staticmethod
    def expected(scale):
        # Independent rank-2 product, with exactly representable binary values.
        delta = [0.1875, -0.0625, -0.109375, -0.09375]
        return [struct.unpack("<e", struct.pack("<e", base + scale * change))[0]
                for base, change in zip([1, 2, 3, 4], delta)]

    def test_cache_evicts_and_recomputes_without_changing_merge(self):
        result = self.run_probe(32, str(self.adapter()) + ":0.75")
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [line.split() for line in result.stdout.splitlines()]
        self.assertEqual([int(row[1]) for row in rows], [16, 32, 32, 32, 32, 32, 32])
        self.assertEqual([int(row[2]) for row in rows], [1, 3, 3, 5, 6, 10, 9])
        for row in rows:
            self.assertEqual(list(map(float, row[3:])), self.expected(0.75 / 2))

    def test_oversized_tensor_is_merged_without_retaining_it(self):
        result = self.run_probe(8, str(self.adapter()) + ":0.75")
        self.assertEqual(result.returncode, 0, result.stderr)
        for line in result.stdout.splitlines():
            row = line.split()
            self.assertEqual(row[1:3], ["0", "0"])
            self.assertEqual(list(map(float, row[3:])), self.expected(0.75 / 2))

    def test_stacked_adapters_and_default_alpha_survive_eviction(self):
        result = self.run_probe(32, str(self.adapter()) + ":0.75",
                                str(self.adapter("second.safetensors", alpha=False)) + ":-0.25")
        self.assertEqual(result.returncode, 0, result.stderr)
        for line in result.stdout.splitlines():
            self.assertEqual(list(map(float, line.split()[3:])), self.expected(0.75 / 2 - 0.25))

    def test_partial_adapter_is_rejected_before_merging(self):
        unknown = "lora_unet_down_blocks_1_resnets_0_conv1"
        result = self.run_probe(32, self.adapter(extra={
            unknown + ".lora_down.weight": ([1, 1], [1]),
            unknown + ".lora_up.weight": ([1, 1], [1]),
        }))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("4 UNet modules matched", result.stderr)
        self.assertIn("2 unmatched tensors", result.stderr)
        self.assertIn("unsupported or unmatched adapter tensor", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_dora_is_rejected_even_when_low_rank_weights_match(self):
        result = self.run_probe(32, self.adapter(extra={
            MODULES[0] + ".dora_scale": ([2], [1, 1]),
        }))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("dora_scale", result.stderr)

    def test_both_sdxl_text_encoders_are_reported_as_dropped(self):
        extra = {}
        for prefix in ("lora_te1_", "lora_te2_"):
            module = prefix + "text_model_encoder_layers_0_self_attn_q_proj"
            extra[module + ".lora_down.weight"] = ([1, 1], [1])
            extra[module + ".lora_up.weight"] = ([1, 1], [1])
        result = self.run_probe(32, self.adapter(extra=extra))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("2 text-encoder modules DROPPED", result.stderr)

    def test_invalid_rank_is_rejected(self):
        result = self.run_probe(32, self.adapter(extra={
            MODULES[0] + ".lora_up.weight": ([2, 1], [1, 1]),
        }))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid LoRA down/up shapes or rank", result.stderr)

    def test_spatial_up_kernel_is_rejected(self):
        result = self.run_probe(32, self.adapter(extra={
            MODULES[0] + ".lora_up.weight": ([1, 2, 1, 2], [1, 1, 1, 1]),
        }))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("spatial LoRA up kernels are unsupported", result.stderr)


if __name__ == "__main__":
    unittest.main()

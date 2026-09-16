#!/usr/bin/env python3
"""Author a weight-free SD1.5/SDXL MNN CLIP recipe from a verified MNN 3.6.1 export.

The FlatBuffer fields and dense IDST encoding are defined by upstream MNN:
https://github.com/alibaba/MNN/tree/3.6.1/schema/default
https://github.com/alibaba/MNN/blob/3.6.1/source/core/IDSTEncoder.hpp
https://github.com/alibaba/MNN/blob/3.6.1/tools/converter/source/common/WeightQuantAndCoding.cpp

Every learned payload is compared with the source checkpoint before removal.
Run `--help` for inputs. This performs no graph export or model inference.
"""
import argparse
import hashlib
import json
import mmap
from pathlib import Path
import re
import struct

import numpy as np

NAMES = ("clip.mnn", "clip_2.mnn", "clip_2.mnn.weight", "token_emb.bin", "pos_emb.bin", "token_emb_2.bin", "pos_emb_2.bin")
SD15_NAMES = ("clip_v2.mnn", "token_emb.bin", "pos_emb.bin")
F32, F16, Q8_SYMMETRIC, Q8_ASYMMETRIC = range(4)


class FlatBuffer:
    def __init__(self, path):
        self.file = path.open("rb")
        self.data = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)

    def scalar(self, at, kind):
        return struct.unpack_from("<" + kind, self.data, at)[0]

    def field(self, table, index):
        if not table:
            return 0
        vtable = table - self.scalar(table, "i")
        entry = 4 + index * 2
        offset = self.scalar(vtable + entry, "H") if entry < self.scalar(vtable, "H") else 0
        return table + offset if offset else 0

    def pointer(self, at):
        return at + self.scalar(at, "I") if at else 0

    def value(self, table, index, kind, default=0):
        at = self.field(table, index)
        return self.scalar(at, kind) if at else default

    def vector(self, table, index, kind):
        at = self.pointer(self.field(table, index))
        count = self.scalar(at, "I") if at else 0
        return at + 4 if at else 0, count, np.frombuffer(self.data, dtype="<" + kind, count=count, offset=at + 4) if at else np.array([])

    def string(self, table, index):
        at = self.pointer(self.field(table, index))
        return self.data[at + 4:at + 4 + self.scalar(at, "I")].decode()

    def operations(self):
        root = self.scalar(0, "I")
        at, count, _ = self.vector(root, 3, "u4")
        for index in range(count):
            op = self.pointer(at + index * 4)
            yield self.value(op, 1, "B"), self.string(op, 3), self.pointer(self.field(op, 2))


class Checkpoint:
    def __init__(self, path):
        self.file = path.open("rb")
        length = struct.unpack("<Q", self.file.read(8))[0]
        raw = self.file.read(length)
        self.header = json.loads(raw)
        self.header_hash = hashlib.sha256(raw).hexdigest()
        self.base = 8 + length
        self.path = path

    def tensor(self, key):
        info = self.header[key]
        dtype = {"F16": "<f2", "F32": "<f4", "BF16": "<u2"}[info["dtype"]]
        values = np.memmap(self.path, dtype=dtype, mode="r", shape=tuple(info["shape"]), offset=self.base + info["data_offsets"][0])
        if info["dtype"] == "BF16":
            return (values.astype("<u4") << 16).view("<f4")
        return values


def source_for(encoder, name, suffix, family="sdxl"):
    """Explicit original-SDXL checkpoint naming and OpenCLIP QKV/projection layout."""
    if family == "sd15" and (name.startswith("/final_layer_norm/") or name == "last_hidden_state"):
        return "cond_stage_model.transformer.text_model.final_layer_norm." + suffix, 0, False
    if name == "pooled_output__matmul_converted":
        assert encoder == 2 and suffix == "weight"
        return "conditioner.embedders.1.model.text_projection", 0, True
    if name.startswith("/final_layer_norm/"):
        assert encoder == 2
        return "conditioner.embedders.1.model.ln_final." + suffix, 0, False
    match = re.match(r"/encoder/layers\.(\d+)/(self_attn/(?:q|k|v|out)_proj|mlp/fc[12]|layer_norm[12])/", name)
    if not match:
        raise ValueError(f"Unmapped learned MNN operation: {name}")
    layer, part = match.groups()
    if encoder == 1:
        prefix = "cond_stage_model.transformer" if family == "sd15" else "conditioner.embedders.0.transformer"
        return f"{prefix}.text_model.encoder.layers.{layer}.{part.replace('/', '.')}.{suffix}", 0, False
    base = f"conditioner.embedders.1.model.transformer.resblocks.{layer}."
    if part in ("self_attn/q_proj", "self_attn/k_proj", "self_attn/v_proj"):
        row = ("self_attn/q_proj", "self_attn/k_proj", "self_attn/v_proj").index(part) * 1280
        return base + "attn.in_proj_" + suffix, row, False
    mapped = {"self_attn/out_proj": "attn.out_proj", "mlp/fc1": "mlp.c_fc", "mlp/fc2": "mlp.c_proj", "layer_norm1": "ln_1", "layer_norm2": "ln_2"}[part]
    return base + mapped + "." + suffix, 0, False


def dense_quantize(values, encoding):
    """MNN 3.6.1: FP32 row scales, FP32 division, round-half-away-from-zero."""
    if encoding == Q8_SYMMETRIC:
        scale = np.max(np.abs(values), axis=1).astype(np.float32) / np.float32(127)
        quotient = np.divide(values, scale[:, None], out=np.zeros_like(values), where=scale[:, None] > np.float32(1e-6))
        integers = np.copysign(np.floor(np.abs(quotient).astype(np.float64) + 0.5), quotient)
        data = (np.clip(integers, -128, 127) + 128).astype(np.uint8)
        return data.tobytes(), scale.astype("<f4").tobytes()
    minimum = np.min(values, axis=1).astype(np.float32)
    scale = (np.max(values, axis=1).astype(np.float32) - minimum) / np.float32(255)
    quotient = np.divide(values - minimum[:, None], scale[:, None], out=np.zeros_like(values), where=scale[:, None] > np.float32(1e-6))
    data = np.clip(np.floor(quotient.astype(np.float64) + 0.5), 0, 255).astype(np.uint8)
    return data.tobytes(), np.stack((minimum, scale), axis=1).astype("<f4").tobytes()


def validate_constant(graph, external_file, encoder, name, main):
    """Only known shape, causal-mask and activation constants may remain literal."""
    width, heads = (768, 12) if encoder == 1 else (1280, 20)
    attention = "/encoder/layers.0/self_attn/"
    integers = {
        "Concat10": [-1, width, 1, 1], "Unsqueeze16": [0], "Const18": [2],
        "Unsqueeze23": [1], "Const5": [width], "Const156": [width * 4],
        "Concat170" if encoder == 1 else "Concat167": [-1, width * 4, 1, 1],
        attention + "Constant_3_output_0": [1, 77, heads, 64],
        attention + "Constant_4_output_0": [heads, -1, 64],
        attention + "Constant_1_output_0": [1, -1, heads, 64],
        attention + "Constant_7_output_0": [1, heads, 77, 77],
        attention + "Constant_9_output_0": [heads, 77, 77],
        attention + "Constant_10_output_0": [1, heads, 77, 64],
        attention + "Constant_11_output_0": [1, 77, width],
    }
    _, _, shape = graph.vector(main, 0, "i4")
    dtype = graph.value(main, 2, "i", 1)
    _, _, external = graph.vector(main, 9, "i8")
    if name in integers:
        assert dtype == 3 and not len(external), name
        _, _, values = graph.vector(main, 5, "i4")
        assert list(values) == integers[name], (name, values)
        payload_field = 5
    else:
        assert dtype == (19 if encoder == 1 else 1), (name, dtype)
        element_type = "<f2" if dtype == 19 else "<f4"
        payload_field = 3 if dtype == 19 else 7
        if len(external):
            assert encoder == 2 and len(external) == 2, name
            at, length = map(int, external)
            values = np.frombuffer(external_file.data, dtype=element_type,
                                   count=length // 4, offset=at)
            assert graph.vector(main, payload_field, "u1")[1] == 0, name
        else:
            at, count, _ = graph.vector(main, payload_field, "u1" if dtype == 19 else "f4")
            values = np.frombuffer(graph.data, dtype=element_type,
                                   count=count // 2 if dtype == 19 else count, offset=at)
        mask = re.fullmatch(r"/encoder/layers\.(\d+)/self_attn/Constant_8_output_0", name)
        if mask:
            assert int(mask.group(1)) < (1 if encoder == 1 else 32), name
            assert list(shape) == [1, 1, 77, 77], name
            expected = np.zeros((77, 77), dtype=element_type)
            expected[np.triu_indices(77, 1)] = -np.finfo(expected.dtype).max
            assert np.array_equal(values, expected.ravel()), name
        else:
            assert not len(shape) and not len(external), name
            allowed = {attention + "Constant_output_0": 0.125}
            if encoder == 1:
                # This export's MNN half constant uses truncation toward zero.
                allowed["/encoder/layers.0/mlp/activation_fn/Constant_output_0"] = 1.701171875
            assert name in allowed and list(values) == [allowed[name]], name
    for field in range(3, 9):
        if field != payload_field:
            assert graph.vector(main, field, "u1")[1] == 0, (name, field)


def author(clips, checkpoint_path, output, family="sdxl", clip_skip=1):
    checkpoint = Checkpoint(checkpoint_path)
    names = SD15_NAMES if family == "sd15" else NAMES
    files = [FlatBuffer(clips / name) for name in names]
    rules, ranges, requirements = [], [[] for _ in names], {}

    def add(file_index, at, rows, cols, encoding, key, source_row=0, transpose=False, alpha=0, clamp=False, align=False):
        source = checkpoint.tensor(key)
        source_shape = list(source.shape)
        source_rows, source_cols = (source_shape[0], 1) if source.ndim == 1 else source_shape
        matrix = source.reshape(source_rows, source_cols)[source_row:source_row + (cols if transpose else rows)]
        if transpose:
            matrix = matrix.T
        assert matrix.shape == (rows, cols), (key, matrix.shape, rows, cols)
        values = matrix.astype(np.float32)
        actual_size = rows * cols * (4 if encoding == F32 else 2 if encoding == F16 else 1)
        actual = files[file_index].data[at:at + actual_size]
        multiplier = None
        for factor in ([1.0, 0.125] if "q_proj/" in current_name else [1.0]):
            transformed = values * np.float32(factor)
            if align:
                transformed[np.abs(transformed) < np.finfo(np.float32).tiny] = 0
            if encoding == F32:
                expected, scales = transformed.astype("<f4").tobytes(), b""
            elif encoding == F16:
                if clamp:
                    transformed = np.clip(transformed, -65504, 65504)
                expected, scales = transformed.astype("<f2").tobytes(), b""
            else:
                expected, scales = dense_quantize(transformed, encoding)
            if actual == expected and (not scales or files[file_index].data[alpha:alpha + len(scales)] == scales):
                multiplier = factor
                break
        if multiplier is None:
            differing = sum(a != b for a, b in zip(actual, expected))
            raise ValueError(f"Payload mismatch {current_name} {key}: {differing}/{actual_size} differing bytes")
        flags = int(transpose) | (2 if clamp else 0) | (4 if align else 0)
        rules.append(dict(file=file_index, encoding=encoding, flags=flags, offset=at, alpha=alpha, rows=rows, cols=cols,
                          source_rank=len(source_shape), source_rows=source_rows, source_cols=source_cols,
                          source_row=source_row, multiplier=multiplier, source=key))
        ranges[file_index].append((at, at + actual_size))
        if scales:
            ranges[file_index].append((alpha, alpha + len(scales)))
        requirements[key] = source_shape

    counts = {}
    for encoder, file_index in (((1, 0),) if family == "sd15" else ((1, 0), (2, 1))):
        graph = files[file_index]
        root = graph.scalar(0, "I")
        assert graph.vector(root, 1, "u4")[1] == 0  # no TensorDescribe payloads
        assert graph.vector(root, 10, "u4")[1] == 0  # no unexamined subgraphs
        convs, norms = 0, 0
        for kind, current_name, main in graph.operations():
            assert kind in {0, 4, 6, 7, 9, 21, 25, 29, 55, 64, 66, 70, 88}, (kind, current_name)
            if kind == 7:
                validate_constant(graph, files[2], encoder, current_name, main)
            if kind == 9:  # schema/default/MNN.fbs OpParameter_Convolution2D
                convs += 1
                common = graph.pointer(graph.field(main, 0))
                rows, cols = graph.value(common, 10, "i"), graph.value(common, 11, "i")
                assert graph.value(common, 2, "i", 1) == graph.value(common, 3, "i", 1) == 1
                quant = graph.pointer(graph.field(main, 3))
                key, row, transpose = source_for(encoder, current_name, "weight", family)
                if encoder == 1:
                    assert graph.value(quant, 2, "i") == 3
                    at, size, _ = graph.vector(quant, 0, "i1")
                    assert size == rows * cols * 2
                    add(0, at, rows, cols, F16, key, row, transpose, clamp=True, align=True)
                    bias, size, _ = graph.vector(main, 2, "f4")
                else:
                    _, _, external = graph.vector(main, 6, "i8")
                    external = list(map(int, external))
                    assert len(external) == 5 and external[4] == 0
                    assert graph.value(quant, 2, "i") == 1 and graph.value(quant, 7, "i") == 8
                    encoding = Q8_ASYMMETRIC if graph.value(quant, 9, "i") else Q8_SYMMETRIC
                    assert graph.value(quant, 11, "B") == 0  # 16-bit dense shape header
                    header = struct.pack("<BHHB", 2, rows, cols, 0) + bytes(range(128, 256)) + bytes(range(128))
                    start, buffer_size, alpha_size, bias_size, _ = external
                    assert files[2].data[start:start + len(header)] == header
                    assert buffer_size == rows * cols + len(header)
                    assert alpha_size == rows * (8 if encoding == Q8_ASYMMETRIC else 4)
                    add(2, start + len(header), rows, cols, encoding, key, row, transpose, start + buffer_size, align=True)
                    bias, size = start + buffer_size + alpha_size, bias_size // 4
                assert size == rows
                if current_name == "pooled_output__matmul_converted":
                    assert files[2].data[bias:bias + size * 4] == bytes(size * 4)
                else:
                    key, row, transpose = source_for(encoder, current_name, "bias", family)
                    add(0 if encoder == 1 else 2, bias, rows, 1, F32, key, row, transpose)
            elif kind == 88:  # OpParameter_LayerNorm
                norms += 1
                if encoder == 1:
                    gamma, count, _ = graph.vector(main, 2, "f4")
                    beta, beta_count, _ = graph.vector(main, 3, "f4")
                    assert count == beta_count
                else:
                    _, _, external = graph.vector(main, 5, "i8")
                    gamma, gamma_bytes, beta_bytes = map(int, external)
                    assert gamma_bytes == beta_bytes
                    count, beta = gamma_bytes // 4, gamma + gamma_bytes
                for at, suffix in ((gamma, "weight"), (beta, "bias")):
                    key, row, transpose = source_for(encoder, current_name, suffix, family)
                    add(0 if encoder == 1 else 2, at, count, 1, F32, key, row, transpose)
        expected_counts = ((13 - clip_skip) * 6, (13 - clip_skip) * 2 + 1) if family == "sd15" else ((66, 22) if encoder == 1 else (193, 65))
        assert (convs, norms) == expected_counts, (convs, norms, expected_counts)
        counts[f"clip{encoder}"] = dict(convolutions=convs, layer_norms=norms)
    current_name = "embeddings"
    embeddings = (
        (3, "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight", 49408, 768, F16),
        (4, "conditioner.embedders.0.transformer.text_model.embeddings.position_embedding.weight", 77, 768, F32),
        (5, "conditioner.embedders.1.model.token_embedding.weight", 49408, 1280, F16),
        (6, "conditioner.embedders.1.model.positional_embedding", 77, 1280, F32),
    ) if family == "sdxl" else (
        (1, "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight", 49408, 768, F16),
        (2, "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight", 77, 768, F32),
    )
    for index, key, rows, cols, encoding in embeddings:
        add(index, 0, rows, cols, encoding, key)

    output.mkdir(parents=True, exist_ok=True)
    literal_bytes = 0
    with (output / "clip_recipe.bin").open("wb") as stream:
        def write_string(text):
            value = text.encode()
            stream.write(struct.pack("<H", len(value)) + value)
        stream.write(b"CLIPRCP1" + struct.pack("<I", len(files)))
        for index, file in enumerate(files):
            last, literals = 0, []
            for begin, end in sorted(ranges[index]):
                assert last <= begin < end <= len(file.data), (names[index], last, begin, end)
                if last < begin:
                    literals.append((last, begin))
                last = end
            if last < len(file.data):
                literals.append((last, len(file.data)))
            write_string(names[index])
            stream.write(struct.pack("<QI", len(file.data), len(literals)))
            for begin, end in literals:
                stream.write(struct.pack("<QI", begin, end - begin))
                stream.write(file.data[begin:end])
                literal_bytes += end - begin
        stream.write(struct.pack("<I", len(rules)))
        for rule in rules:
            stream.write(struct.pack("<BBBBQQIIIIIf", rule["file"], rule["encoding"], rule["flags"], rule["source_rank"],
                                     rule["offset"], rule["alpha"], rule["rows"], rule["cols"], rule["source_rows"],
                                     rule["source_cols"], rule["source_row"], rule["multiplier"]))
            write_string(rule["source"])
    (output / "clip_requirements.json").write_text(json.dumps({"tensors": [
        {"name": name, "shape": shape, "dtypes": ["F16", "F32", "BF16"]}
        for name, shape in sorted(requirements.items())
    ]}, indent=2) + "\n")
    report = {"checkpoint": str(checkpoint_path), "checkpoint_header_sha256": checkpoint.header_hash,
              "family": family, "clip_skip": clip_skip if family == "sd15" else None,
              "graphs": counts, "rules": len(rules), "source_tensors": len(requirements), "literal_bytes": literal_bytes,
              "recipe_bytes": (output / "clip_recipe.bin").stat().st_size,
              "encodings": {str(kind): sum(rule["encoding"] == kind for rule in rules) for kind in range(4)},
              "multipliers": sorted(set(rule["multiplier"] for rule in rules)), "verification": "All removed weight, bias, norm, scale and embedding payloads matched checkpoint-derived bytes."}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--family", choices=["sd15", "sdxl"], default="sdxl")
    parser.add_argument("--clip-skip", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    author(args.clips, args.checkpoint, args.output, args.family, args.clip_skip)

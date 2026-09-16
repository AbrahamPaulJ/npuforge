#!/usr/bin/env python3
"""Export and verify checkpoint-owned SD1.5/SDXL CLIPs, one CPU model/process at a time.

Requires torch, accelerate, diffusers, transformers, safetensors, tokenizers, ONNX
and MNN. The config directory must contain text_encoder/config.json and,
for SDXL, text_encoder_2/config.json. No UNet, VAE, pipeline load or image render occurs.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = CHECKPOINT = CONFIG = TOKENIZER = CONVERTER = CLIPS = None
FAMILY, CLIP_SKIP = "sdxl", 1
PROMPT = "a photo of a red fox sitting in a forest, detailed fur, soft daylight"


def metrics(actual, expected):
    import numpy as np
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(expected, dtype=np.float64)
    difference = a - b
    return {
        "shape": list(a.shape), "finite": bool(np.isfinite(a).all()),
        "max_abs_difference": float(np.abs(difference).max()),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "reference_rms": float(np.sqrt(np.mean(b * b))),
        "relative_rmse": float(np.linalg.norm(difference) / np.linalg.norm(b)),
        "cosine_similarity": float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b))),
    }


def export(encoder):
    import numpy as np
    import torch
    from accelerate import init_empty_weights
    from diffusers.loaders.single_file_utils import convert_ldm_clip_checkpoint, convert_open_clip_checkpoint
    from safetensors import safe_open
    from tokenizers import Tokenizer
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    destination = ROOT / f"clip{encoder}"
    destination.mkdir(exist_ok=True)
    config_path = CONFIG / ("text_encoder" if encoder == 1 else "text_encoder_2")
    config = CLIPTextConfig.from_pretrained(str(config_path), local_files_only=True)
    config._attn_implementation = "eager"
    cls = CLIPTextModel if encoder == 1 else CLIPTextModelWithProjection
    with init_empty_weights():
        model = cls(config)
    prefix = "cond_stage_model.transformer." if FAMILY == "sd15" else f"conditioner.embedders.{encoder - 1}."
    print(f"Reading only {prefix} tensors", flush=True)
    with safe_open(str(CHECKPOINT), framework="pt", device="cpu") as checkpoint:
        source = {key: checkpoint.get_tensor(key) for key in checkpoint.keys() if key.startswith(prefix)}
    source_count = len(source)
    mapped = convert_ldm_clip_checkpoint(source) if encoder == 1 else convert_open_clip_checkpoint(
        model, source, prefix="conditioner.embedders.1.model.")
    discarded = []
    position_ids = "text_model.embeddings.position_ids"
    if position_ids in mapped and position_ids not in model.state_dict():
        del mapped[position_ids]
        discarded.append(position_ids)
    model.load_state_dict(mapped, strict=True, assign=True)
    del mapped, source
    gc.collect()
    model = model.float().eval()
    print(f"Loaded CLIP-{encoder}; {source_count} source tensors; strict state load passed", flush=True)

    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    fixture_ids = {}
    for name, text in (("prompt", PROMPT), ("empty", "")):
        body = tokenizer.encode(text, add_special_tokens=False).ids
        assert len(body) <= 75
        ids = [49406, *body, 49407] + [49407] * (75 - len(body))
        fixture_ids[name + "_eos"] = ids
        if encoder == 2:
            eos = ids.index(49407)
            fixture_ids[name + "_zero"] = ids[:eos + 1] + [0] * (76 - eos)
    saved = {}
    references = {}
    with torch.no_grad():
        for name, ids in fixture_ids.items():
            input_ids = torch.tensor([ids], dtype=torch.long)
            original = model(input_ids, output_hidden_states=True)
            embedding = model.text_model.embeddings(input_ids)
            hidden = (model.text_model.final_layer_norm(original.hidden_states[-CLIP_SKIP])
                      if FAMILY == "sd15" else original.hidden_states[-2])
            reference = {"last_hidden_state": hidden.clone()}
            if encoder == 2:
                normed = model.text_model.final_layer_norm(original.hidden_states[-1])
                reference["pooled_output"] = model.text_projection(normed)
                eos = ids.index(49407)
                torch.testing.assert_close(reference["pooled_output"][:, eos], original.text_embeds, atol=1e-5, rtol=1e-5)
            references[name] = (embedding, reference)
            saved[name + "__ids"] = np.asarray(ids, dtype=np.int32)
            saved[name + "__input_embedding"] = embedding.numpy()
            stored_tokens = model.text_model.embeddings.token_embedding(input_ids).half().float()
            stored_positions = model.text_model.embeddings.position_embedding.weight[None]
            saved[name + "__stored_input_embedding"] = (stored_tokens + stored_positions).numpy()
            for output, value in reference.items():
                saved[name + "__" + output] = value.numpy()
    np.savez(destination / "references.npz", **saved)

    class Wrapper(torch.nn.Module):
        def __init__(self, clip):
            super().__init__()
            if encoder == 1:
                layers = 13 - CLIP_SKIP if FAMILY == "sd15" else 11
                clip.text_model.encoder.layers = clip.text_model.encoder.layers[:layers]
            self.text_model = clip.text_model
            if encoder == 2:
                self.text_projection = clip.text_projection

        def forward(self, input_embedding):
            mask = torch.tril(torch.ones(77, 77, dtype=torch.bool, device=input_embedding.device))
            causal = torch.zeros(77, 77, dtype=input_embedding.dtype, device=input_embedding.device)
            causal.masked_fill_(~mask, torch.finfo(input_embedding.dtype).min)
            output = self.text_model.encoder(
                input_embedding, causal_attention_mask=causal.unsqueeze(0).unsqueeze(0),
                output_hidden_states=encoder == 2,
            )
            if encoder == 1:
                return (self.text_model.final_layer_norm(output.last_hidden_state)
                        if FAMILY == "sd15" else output.last_hidden_state)
            return output.hidden_states[-2], self.text_projection(self.text_model.final_layer_norm(output.last_hidden_state))

    wrapper = Wrapper(model).eval()
    outputs = ["last_hidden_state"] if encoder == 1 else ["last_hidden_state", "pooled_output"]
    wrapper_results = {}
    with torch.no_grad():
        for name, (embedding, expected) in references.items():
            result = wrapper(embedding)
            result = (result,) if encoder == 1 else result
            wrapper_results[name] = {}
            for output_name, actual in zip(outputs, result):
                torch.testing.assert_close(actual, expected[output_name], atol=1e-4, rtol=1e-5)
                wrapper_results[name][output_name] = metrics(actual.numpy(), expected[output_name].numpy())
    suffix = "" if encoder == 1 else "_2"
    with torch.no_grad():
        model.text_model.embeddings.token_embedding.weight.to(torch.float16).numpy().tofile(CLIPS / f"token_emb{suffix}.bin")
        model.text_model.embeddings.position_embedding.weight.numpy().tofile(CLIPS / f"pos_emb{suffix}.bin")
    print("Wrapper references passed; exporting ONNX", flush=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper, torch.zeros(1, 77, config.hidden_size), str(destination / "model.onnx"),
            input_names=["input_embedding"], output_names=outputs, opset_version=17,
        )
    report = {
        "checkpoint": str(CHECKPOINT), "config": str(config_path / "config.json"),
        "family": FAMILY, "clip_skip": CLIP_SKIP if FAMILY == "sd15" else None,
        "tokenizer": str(TOKENIZER), "prompt": PROMPT, "source_tensor_count": source_count,
        "discarded_legacy_nonparameter_keys": discarded, "strict_state_load": True,
        "reference_dtype": "float32 from original checkpoint weights; stored token embeddings round to float16", "cpu_threads": 4,
        "fixture_ids": fixture_ids, "wrapper_comparisons": wrapper_results,
        "onnx": str(destination / "model.onnx"),
    }
    (destination / "export-results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"CLIP-{encoder} ONNX export complete", flush=True)


def convert(encoder):
    name = "clip_v2.mnn" if FAMILY == "sd15" else "clip.mnn" if encoder == 1 else "clip_2.mnn"
    flags = ["--fp16"] if encoder == 1 else ["--weightQuantBits", "8", "--weightQuantAsymmetric", "--saveExternalData"]
    subprocess.run([
        str(CONVERTER), "-f", "ONNX", "--modelFile", str(ROOT / f"clip{encoder}" / "model.onnx"),
        "--MNNModel", str(CLIPS / name), *flags,
    ], check=True)


def verify(encoder):
    import numpy as np
    import MNN
    destination = ROOT / f"clip{encoder}"
    references = np.load(destination / "references.npz")
    name = "clip_v2.mnn" if FAMILY == "sd15" else "clip.mnn" if encoder == 1 else "clip_2.mnn"
    interpreter = MNN.Interpreter(str(CLIPS / name))
    session = interpreter.createSession({"backend": "CPU", "numThread": 4, "precision": "high"})
    output_names = ["last_hidden_state"] if encoder == 1 else ["last_hidden_state", "pooled_output"]
    width = 768 if encoder == 1 else 1280
    suffix = "" if encoder == 1 else "_2"
    token_embeddings = np.memmap(CLIPS / f"token_emb{suffix}.bin", dtype=np.float16, mode="r", shape=(49408, width))
    positions = np.fromfile(CLIPS / f"pos_emb{suffix}.bin", dtype=np.float32).reshape(77, width)
    results = {}
    actual_outputs = {}
    for key in references.files:
        if not key.endswith("__input_embedding"):
            continue
        fixture = key.removesuffix("__input_embedding")
        ids = references[fixture + "__ids"]
        value = np.ascontiguousarray((token_embeddings[ids].astype(np.float32) + positions)[None])
        stored_key = fixture + "__stored_input_embedding"
        # Old F16-only diagnostics predate the separate stored-input fixture.
        np.testing.assert_array_equal(value, references[stored_key] if stored_key in references else references[key])
        target = interpreter.getSessionInput(session, "input_embedding")
        assert tuple(target.getShape()) == value.shape
        target.copyFrom(MNN.Tensor(value.shape, MNN.Halide_Type_Float, value, MNN.Tensor_DimensionType_Caffe))
        status = interpreter.runSession(session)
        assert status in (0, None), status
        results[fixture] = {"emitted_embeddings_exactly_match_reference": True}
        actual_outputs[fixture] = {}
        for output in output_names:
            tensor = interpreter.getSessionOutput(session, output)
            host = MNN.Tensor(tensor.getShape(), MNN.Halide_Type_Float, np.zeros(tensor.getShape(), np.float32), MNN.Tensor_DimensionType_Caffe)
            tensor.copyToHostTensor(host)
            actual = np.array(host.getNumpyData(), copy=True)
            actual_outputs[fixture][output] = actual
            expected = references[fixture + "__" + output]
            assert actual.shape == expected.shape, (output, actual.shape, expected.shape)
            result = metrics(actual, expected)
            assert result["finite"], (fixture, output)
            assert result["relative_rmse"] < 0.10, (fixture, output, result)
            results[fixture][output] = result
            if output == "pooled_output":
                eos = list(references[fixture + "__ids"]).index(49407)
                results[fixture]["pooled_eos"] = metrics(actual[:, eos], expected[:, eos])
                assert results[fixture]["pooled_eos"]["relative_rmse"] < 0.10, (fixture, "pooled_eos")
            print(f"{fixture} {output}: relative RMSE={result['relative_rmse']:.6g}; cosine={result['cosine_similarity']:.6g}", flush=True)
    padding_comparisons = {}
    if encoder == 2:
        for fixture in ("prompt", "empty"):
            eos = list(references[fixture + "_eos__ids"]).index(49407)
            eos_outputs = actual_outputs[fixture + "_eos"]
            zero_outputs = actual_outputs[fixture + "_zero"]
            padding_comparisons[fixture] = {
                "eos_index": eos,
                "pooled_eos": metrics(zero_outputs["pooled_output"][:, eos], eos_outputs["pooled_output"][:, eos]),
                "hidden_through_eos": metrics(zero_outputs["last_hidden_state"][:, :eos + 1], eos_outputs["last_hidden_state"][:, :eos + 1]),
                "hidden_after_eos": metrics(zero_outputs["last_hidden_state"][:, eos + 1:], eos_outputs["last_hidden_state"][:, eos + 1:]),
            }
    report = {
        "model": str(CLIPS / name), "cpu_threads": 4, "backend": "CPU", "reference": str(destination / "references.npz"),
        "format": "FP16" if encoder == 1 else "INT8 weight-only, external data; inspect actual IDST metadata when authoring recipe",
        "metric_definition": "relative_rmse = L2(actual - reference) / L2(reference); metrics calculated in float64",
        "maximum_relative_rmse": 0.10,
        "results": results, "padding_comparisons": padding_comparisons,
        "limitation": "Host conditioning comparison only; no image generation or phone execution. Quantization errors are reported, not treated as proof of equivalent image quality.",
    }
    (destination / "mnn-results.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    global ROOT, CHECKPOINT, CONFIG, TOKENIZER, CONVERTER, CLIPS, FAMILY, CLIP_SKIP
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--mnn-converter", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--family", choices=["sd15", "sdxl"], default="sdxl")
    parser.add_argument("--clip-skip", type=int, choices=[1, 2], default=1,
                        help="SD1.5 graph depth: 1=12 layers, 2=11 layers; both apply final layer norm")
    parser.add_argument("--phase", choices=["export", "convert", "verify"])
    parser.add_argument("--encoder", type=int, choices=[1, 2])
    args = parser.parse_args()
    FAMILY, CLIP_SKIP = args.family, args.clip_skip
    if FAMILY == "sd15" and args.encoder == 2:
        parser.error("SD1.5 has one CLIP encoder")
    CHECKPOINT, CONFIG, TOKENIZER, CONVERTER, ROOT = (
        args.checkpoint.resolve(), args.config.resolve(), args.tokenizer.resolve(),
        args.mnn_converter.resolve(), args.output.resolve(),
    )
    CLIPS = ROOT / "clips"
    CLIPS.mkdir(parents=True, exist_ok=True)
    if args.phase:
        if args.encoder is None:
            parser.error("--encoder is required with --phase")
        return {"export": export, "convert": convert, "verify": verify}[args.phase](args.encoder)
    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false"})
    base_arguments = [
        "--checkpoint", str(CHECKPOINT), "--config", str(CONFIG),
        "--tokenizer", str(TOKENIZER), "--mnn-converter", str(CONVERTER), "--output", str(ROOT),
        "--family", FAMILY, "--clip-skip", str(CLIP_SKIP),
    ]
    timings = {}
    for encoder in ((1,) if FAMILY == "sd15" else (1, 2)):
        for phase in ("export", "convert", "verify"):
            log = ROOT / f"clip{encoder}-{phase}.log"
            print(f"Starting CLIP-{encoder} {phase}; log={log}", flush=True)
            start = time.monotonic()
            with log.open("w") as output:
                result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--phase", phase, "--encoder", str(encoder), *base_arguments], stdout=output, stderr=subprocess.STDOUT, env=environment)
            timings[f"clip{encoder}_{phase}_seconds"] = round(time.monotonic() - start, 3)
            (ROOT / "timings.json").write_text(json.dumps(timings, indent=2) + "\n")
            if result.returncode:
                print(log.read_text()[-8000:], flush=True)
                raise SystemExit(result.returncode)
            print(f"Completed CLIP-{encoder} {phase} in {timings[f'clip{encoder}_{phase}_seconds']} seconds", flush=True)
    print(f"Finished {3 if FAMILY == 'sd15' else 7} replacement component files in {CLIPS}", flush=True)


if __name__ == "__main__":
    main()

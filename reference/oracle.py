"""Capture deterministic stage-boundary fixtures from the pinned Kokoro oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any

from assets import ARTIFACTS, MODEL_DIR, ROOT, verify_assets

os.environ.setdefault("HF_HOME", str(MODEL_DIR / ".huggingface"))

import numpy as np
import torch
from kokoro.model import KModel


REFERENCE_DIR = Path(__file__).resolve().parent
INPUT_PATH = REFERENCE_DIR / "oracle_input.json"
PINS_PATH = REFERENCE_DIR / "pins.json"
FIXTURE_PATH = REFERENCE_DIR / "fixtures/af_heart_hello.npz"
MANIFEST_PATH = REFERENCE_DIR / "fixtures/manifest.json"
MODEL_NAME = "hexgrad/Kokoro-82M"
MODEL_REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"
SOURCE_REVISION = "dfb907a02bba8152ca444717ca5d78747ccb4bec"


def _copy_tensor(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().contiguous().numpy().copy()


def _input_data(model: KModel) -> dict[str, Any]:
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    phoneme_ids = [model.vocab.get(character) for character in data["phonemes"]]
    if phoneme_ids != data["phoneme_ids"]:
        raise RuntimeError(f"prepared phoneme IDs no longer match pinned config: {phoneme_ids}")
    if data["input_ids"] != [0, *data["phoneme_ids"], 0]:
        raise RuntimeError("input_ids must contain the pinned phoneme IDs between Kokoro BOS/EOS IDs")
    if data["voice_style_index"] != len(data["phonemes"]) - 1:
        raise RuntimeError("voice row must use the official len(phonemes) - 1 selection rule")
    return data


def _check_toolchain() -> None:
    pins = json.loads(PINS_PATH.read_text(encoding="utf-8"))["tools"]
    observed = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    package_version = __import__("importlib.metadata", fromlist=["version"]).version("kokoro")
    observed["kokoro"] = package_version
    expected = {
        "python": pins["python"],
        "torch": pins["torch"],
        "numpy": pins["numpy"],
        "kokoro": "0.9.4",
    }
    if observed != expected:
        raise RuntimeError(f"reference toolchain mismatch: observed {observed}, expected {expected}")


def _capture(model: KModel, input_data: dict[str, Any], voice_table: torch.Tensor) -> dict[str, np.ndarray]:
    traces: dict[str, np.ndarray] = {}

    def save(name: str, value: torch.Tensor) -> None:
        traces[name] = _copy_tensor(value)

    def capture_output(name: str, transform=lambda output: output):
        def hook(_module, _inputs, output):
            save(name, transform(output))

        return hook

    def capture_source(_module, _inputs, output):
        save("excitation_harmonic", output[0])
        save("excitation_noise", output[1])
        save("excitation_voiced", output[2])

    handles = [
        model.bert.register_forward_hook(capture_output("bert_hidden")),
        model.bert_encoder.register_forward_hook(
            capture_output("contextual_features", lambda output: output.transpose(-1, -2))
        ),
        model.predictor.text_encoder.register_forward_hook(capture_output("duration_features")),
        model.predictor.lstm.register_forward_hook(capture_output("duration_rnn", lambda output: output[0])),
        model.predictor.duration_proj.register_forward_hook(capture_output("duration_logits")),
        model.text_encoder.register_forward_hook(capture_output("text_features")),
        model.decoder.register_forward_pre_hook(lambda _module, inputs: save("aligned_text", inputs[0])),
        model.decoder.generator.register_forward_pre_hook(
            lambda _module, inputs: save("decoder_features", inputs[0])
        ),
        model.decoder.generator.m_source.register_forward_hook(capture_source),
    ]

    stft = model.decoder.generator.stft
    original_inverse = stft.inverse
    original_f0n = model.predictor.F0Ntrain

    def capture_f0n(features: torch.Tensor, style: torch.Tensor):
        output = original_f0n(features, style)
        save("f0", output[0])
        save("noise", output[1])
        return output

    def capture_inverse(magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        save("spectral_magnitude", magnitude)
        save("spectral_phase", phase)
        return original_inverse(magnitude, phase)

    stft.inverse = capture_inverse
    model.predictor.F0Ntrain = capture_f0n
    input_ids = torch.tensor([input_data["input_ids"]], dtype=torch.long)
    input_lengths = torch.full((input_ids.shape[0],), input_ids.shape[-1], dtype=torch.long)
    text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(input_ids.shape[0], -1)
    text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))
    attention_mask = (~text_mask).int()
    ref_s = voice_table[input_data["voice_style_index"]]
    save("input_ids", input_ids)
    save("input_lengths", input_lengths)
    save("attention_mask", attention_mask)
    save("text_mask", text_mask)
    try:
        torch.manual_seed(input_data["seed"])
        waveform, predicted_durations = model.forward_with_tokens(
            input_ids,
            ref_s,
            speed=input_data["speed"],
        )
    finally:
        stft.inverse = original_inverse
        model.predictor.F0Ntrain = original_f0n
        for handle in handles:
            handle.remove()

    duration_values = torch.sigmoid(torch.from_numpy(traces["duration_logits"])).sum(axis=-1) / input_data["speed"]
    durations = predicted_durations.to(dtype=torch.long).reshape(-1)
    if not torch.equal(torch.round(duration_values).clamp(min=1).long().reshape(-1), durations):
        raise RuntimeError("traced duration values do not reproduce upstream rounding/clamping")
    token_indices = torch.repeat_interleave(torch.arange(input_ids.shape[1]), durations)
    alignment = torch.zeros((input_ids.shape[1], token_indices.shape[0]), dtype=torch.float32)
    alignment[token_indices, torch.arange(token_indices.shape[0])] = 1.0
    traces["duration_values"] = _copy_tensor(duration_values)
    traces["durations"] = _copy_tensor(durations)
    traces["alignment"] = _copy_tensor(alignment.unsqueeze(0))

    for speed_tag, speed in (("0_5", 0.5), ("2_0", 2.0)):
        # The pinned forward path applies speed after the shared duration projection, so these
        # endpoint cases reuse its captured, speed-independent logits without rerunning the decoder.
        speed_values = torch.sigmoid(torch.from_numpy(traces["duration_logits"])).sum(axis=-1) / speed
        speed_durations = torch.round(speed_values).clamp(min=1).long().reshape(-1)
        speed_indices = torch.repeat_interleave(torch.arange(input_ids.shape[1]), speed_durations)
        speed_alignment = torch.zeros((input_ids.shape[1], speed_indices.shape[0]), dtype=torch.float32)
        speed_alignment[speed_indices, torch.arange(speed_indices.shape[0])] = 1.0
        traces[f"duration_values_speed_{speed_tag}"] = _copy_tensor(speed_values)
        traces[f"durations_speed_{speed_tag}"] = _copy_tensor(speed_durations)
        traces[f"alignment_speed_{speed_tag}"] = _copy_tensor(speed_alignment.unsqueeze(0))

    traces["waveform"] = _copy_tensor(waveform)
    return traces


def generate_twice() -> tuple[dict[str, np.ndarray], dict[str, dict[str, float | bool]]]:
    """Run the official forward path twice with the same captured excitation seed."""
    verify_assets()
    _check_toolchain()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    config_path = MODEL_DIR / "config.json"
    model_path = MODEL_DIR / "kokoro-v1_0.pth"
    voice_path = MODEL_DIR / "voices/af_heart.pt"
    model = KModel(
        repo_id=MODEL_NAME,
        config=str(config_path),
        model=str(model_path),
        disable_complex=False,
    ).cpu().eval()
    voice_table = torch.load(voice_path, map_location="cpu", weights_only=True)
    input_data = _input_data(model)
    first = _capture(model, input_data, voice_table)
    second = _capture(model, input_data, voice_table)
    if first.keys() != second.keys():
        raise RuntimeError("reference runs produced different stage output sets")

    metrics: dict[str, dict[str, float | bool]] = {}
    for name in first:
        left, right = first[name], second[name]
        equal = left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
        difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
        rmse = float(np.sqrt(np.mean(difference * difference))) if difference.size else 0.0
        reference_rms = float(np.sqrt(np.mean(right.astype(np.float64) ** 2))) if right.size else 0.0
        metrics[name] = {
            "bitwise_equal": bool(equal),
            "max_abs_difference": float(difference.max()) if difference.size else 0.0,
            "rmse": rmse,
            "normalized_rmse": rmse / reference_rms if reference_rms else 0.0,
        }
        if not equal:
            raise RuntimeError(f"reference is not repeatable for {name}: {metrics[name]}")
    return first, metrics


def _array_manifest(arrays: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    result = {}
    for name, array in sorted(arrays.items()):
        data = np.ascontiguousarray(array).tobytes()
        result[name] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    return result


def write_fixture(arrays: dict[str, np.ndarray], repeatability: dict[str, dict[str, float | bool]]) -> None:
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FIXTURE_PATH, **arrays)
    pins = json.loads(PINS_PATH.read_text(encoding="utf-8"))
    input_data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    fixture_data = FIXTURE_PATH.read_bytes()
    budgets_data = (REFERENCE_DIR / "budgets.json").read_bytes()
    manifest = {
        "schema_version": 1,
        "fixture": {
            "path": FIXTURE_PATH.name,
            "size_bytes": len(fixture_data),
            "sha256": hashlib.sha256(fixture_data).hexdigest(),
        },
        "input": input_data,
        "model": pins["model"],
        "inference_source": pins["inference_source"],
        "tools": pins["tools"],
        "repeatability": {"runs": 2, "per_array": repeatability},
        "arrays": _array_manifest(arrays),
        "budgets": {
            "path": "../budgets.json",
            "sha256": hashlib.sha256(budgets_data).hexdigest(),
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_fixture(
    arrays: dict[str, np.ndarray], repeatability: dict[str, dict[str, float | bool]]
) -> None:
    if not FIXTURE_PATH.is_file() or not MANIFEST_PATH.is_file():
        raise RuntimeError(
            "checked-in oracle fixture is missing; run "
            "`uv run --project reference --locked python reference/oracle.py --write`."
        )
    with np.load(FIXTURE_PATH, allow_pickle=False) as expected:
        if set(expected.files) != set(arrays):
            raise RuntimeError(f"fixture stage names differ: {sorted(expected.files)} != {sorted(arrays)}")
        for name, actual in arrays.items():
            target = expected[name]
            if actual.dtype != target.dtype or actual.shape != target.shape or not np.array_equal(actual, target):
                raise RuntimeError(f"reference fixture changed at {name}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    fixture_data = FIXTURE_PATH.read_bytes()
    expected_hash = hashlib.sha256(fixture_data).hexdigest()
    if manifest["fixture"]["sha256"] != expected_hash:
        raise RuntimeError("fixture file SHA-256 does not match its manifest")
    if manifest["arrays"] != _array_manifest(arrays):
        raise RuntimeError("fixture array metadata or hashes do not match the manifest")
    if manifest["repeatability"]["per_array"] != repeatability:
        raise RuntimeError("reference repeatability metrics do not match the manifest")
    budgets_data = (REFERENCE_DIR / "budgets.json").read_bytes()
    if manifest["budgets"]["sha256"] != hashlib.sha256(budgets_data).hexdigest():
        raise RuntimeError("numerical budget file SHA-256 does not match the manifest")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="replace the compact checked-in fixture and manifest")
    group.add_argument("--verify", action="store_true", help="compare a fresh repeated run with the checked-in fixture")
    arguments = parser.parse_args()
    output, repeatability = generate_twice()
    if arguments.write:
        write_fixture(output, repeatability)
        print(f"Wrote {FIXTURE_PATH.relative_to(ROOT)} and {MANIFEST_PATH.relative_to(ROOT)}")
    else:
        verify_fixture(output, repeatability)
        print(f"Verified {len(output)} arrays against {FIXTURE_PATH.relative_to(ROOT)}")

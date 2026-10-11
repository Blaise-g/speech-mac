"""Export compact files consumed by the native C tests."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from assets import ARTIFACTS, HF_REPOSITORY, HF_REVISION
from pack import SOURCE_REVISION, VOICE_NAME, write_pack


ROOT = Path(__file__).resolve().parent.parent
REFERENCE = ROOT / "reference"
FIXTURES = REFERENCE / "fixtures"


def export_context_features() -> None:
    with np.load(FIXTURES / "af_heart_hello.npz", allow_pickle=False) as fixture:
        features = np.asarray(fixture["contextual_features"], dtype="<f4")
    if features.shape != (1, 512, 8):
        raise ValueError(f"unexpected contextual feature shape: {features.shape}")
    manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
    raw = features.tobytes(order="C")
    expected = manifest["arrays"]["contextual_features"]["sha256"]
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("raw contextual fixture does not match its manifest digest")
    (FIXTURES / "contextual_features.f32le").write_bytes(raw)


def export_duration_fixtures() -> None:
    specs = {
        "duration_features": ("<f4", (1, 8, 640)),
        "duration_rnn": ("<f4", (1, 8, 512)),
        "duration_logits": ("<f4", (1, 8, 50)),
        "duration_values": ("<f4", (1, 8)),
        "durations": ("<i8", (8,)),
        "alignment": ("<f4", (1, 8, 55)),
    }
    for speed_tag in ("0_5", "2_0"):
        specs[f"duration_values_speed_{speed_tag}"] = ("<f4", (1, 8))
        specs[f"durations_speed_{speed_tag}"] = ("<i8", (8,))
        specs[f"alignment_speed_{speed_tag}"] = ("<f4", (1, 8, None))

    manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
    with np.load(FIXTURES / "af_heart_hello.npz", allow_pickle=False) as fixture:
        for name, (dtype, shape) in specs.items():
            array = np.ascontiguousarray(fixture[name], dtype=dtype)
            shape_matches = array.shape == shape
            if shape[-1] is None:
                shape_matches = array.ndim == 3 and array.shape[:2] == shape[:2] and array.shape[2] > 0
            if not shape_matches:
                raise ValueError(f"unexpected {name} fixture shape: {array.shape}")
            raw = array.tobytes(order="C")
            if hashlib.sha256(raw).hexdigest() != manifest["arrays"][name]["sha256"]:
                raise ValueError(f"raw {name} fixture does not match its manifest digest")
            suffix = "i64le" if dtype == "<i8" else "f32le"
            (FIXTURES / f"{name}.{suffix}").write_bytes(raw)


def export_minimal_pack() -> None:
    metadata = {
        "model": HF_REPOSITORY,
        "model_revision": HF_REVISION,
        "source_revision": SOURCE_REVISION,
        "voice": VOICE_NAME,
        "checkpoint_sha256": ARTIFACTS["kokoro-v1_0.pth"][1],
        "config_sha256": ARTIFACTS["config.json"][1],
        "voice_sha256": ARTIFACTS["voices/af_heart.pt"][1],
    }
    tensors = {
        "bert.embeddings.word_embeddings.weight": ("f32", (1,), struct.pack("<f", 1.0)),
        "bert_encoder.weight": ("f32", (1,), struct.pack("<f", 1.0)),
        "predictor.projection.weight": ("f32", (1,), struct.pack("<f", 1.0)),
        "text_encoder.projection.weight": ("f32", (1,), struct.pack("<f", 1.0)),
        "decoder.projection.weight": ("f32", (1,), struct.pack("<f", 1.0)),
        "voice.af_heart.style": ("f32", (1, 1, 256), struct.pack("<256f", *([0.25] * 256))),
    }
    target = ROOT / "tests/fixtures/minimal_valid_pack.kpack"
    write_pack(target, tensors, metadata)


if __name__ == "__main__":
    export_context_features()
    export_duration_fixtures()
    export_minimal_pack()
    print("Exported contextual feature and minimal pack fixtures for the C tests.")

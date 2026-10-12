"""Versioned compact style artifact for a separately installed Kokoro voice."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from assets import AM_ADAM_SOURCE_SHA256, HF_REVISION


MAGIC = b"KVOICE1\0"
VERSION = 1
STYLE_ROWS = 510
STYLE_WIDTH = 256
HEADER = struct.Struct("<8sIIII40s64s64s")
PAYLOAD_SIZE = STYLE_ROWS * STYLE_WIDTH * 4
PINS_PATH = Path(__file__).with_name("pins.json")


class VoiceArtifactError(ValueError):
    """A voice style artifact is malformed or incompatible with the pinned model."""


def serialize_voice(dtype: str, shape: tuple[int, ...], data: bytes,
                    source_sha256: str = AM_ADAM_SOURCE_SHA256,
                    model_revision: str = HF_REVISION) -> bytes:
    if dtype != "f32" or shape != (STYLE_ROWS, 1, STYLE_WIDTH) or len(data) != PAYLOAD_SIZE:
        raise VoiceArtifactError("am_adam style must be float32 with shape [510, 1, 256]")
    if source_sha256 != AM_ADAM_SOURCE_SHA256:
        raise VoiceArtifactError("am_adam source SHA-256 does not match the pinned snapshot")
    if model_revision != HF_REVISION:
        raise VoiceArtifactError("voice model revision does not match the pinned snapshot")
    values = np.frombuffer(data, dtype="<f4")
    if not np.isfinite(values).all():
        raise VoiceArtifactError("am_adam style contains a non-finite value")
    payload_sha256 = hashlib.sha256(data).hexdigest()
    header = HEADER.pack(
        MAGIC,
        VERSION,
        STYLE_ROWS,
        STYLE_WIDTH,
        PAYLOAD_SIZE,
        model_revision.encode("ascii"),
        source_sha256.encode("ascii"),
        payload_sha256.encode("ascii"),
    )
    return header + data


def write_voice(path: str | Path, dtype: str, shape: tuple[int, ...], data: bytes,
                source_sha256: str = AM_ADAM_SOURCE_SHA256,
                model_revision: str = HF_REVISION) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(serialize_voice(dtype, shape, data, source_sha256, model_revision))
    return target


def read_voice(path: str | Path) -> tuple[bytes, str]:
    raw = Path(path).read_bytes()
    if len(raw) != HEADER.size + PAYLOAD_SIZE:
        raise VoiceArtifactError("voice artifact has the wrong size")
    magic, version, rows, width, payload_size, revision, source_sha, payload_sha = HEADER.unpack_from(raw)
    payload = raw[HEADER.size:]
    if magic != MAGIC:
        raise VoiceArtifactError("invalid voice artifact magic")
    if version != VERSION or rows != STYLE_ROWS or width != STYLE_WIDTH or payload_size != PAYLOAD_SIZE:
        raise VoiceArtifactError("incompatible voice artifact format or style shape")
    if revision != HF_REVISION.encode("ascii") or source_sha != AM_ADAM_SOURCE_SHA256.encode("ascii"):
        raise VoiceArtifactError("voice provenance does not match the pinned snapshot")
    actual_payload_sha = hashlib.sha256(payload).hexdigest()
    if actual_payload_sha.encode("ascii") != payload_sha:
        raise VoiceArtifactError("voice style payload SHA-256 mismatch")
    if not np.isfinite(np.frombuffer(payload, dtype="<f4")).all():
        raise VoiceArtifactError("voice style contains a non-finite value")
    artifact_sha = hashlib.sha256(raw).hexdigest()
    pins = json.loads(PINS_PATH.read_text(encoding="utf-8"))
    voice_pin = pins["model"]["additional_voices"]["am_adam"]
    if actual_payload_sha != voice_pin["style_payload_sha256"]:
        raise VoiceArtifactError("voice style payload SHA-256 does not match the pin")
    if artifact_sha != voice_pin["artifact_sha256"]:
        raise VoiceArtifactError("voice artifact SHA-256 does not match the pin")
    return payload, artifact_sha

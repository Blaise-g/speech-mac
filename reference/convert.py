"""Convert the pinned checkpoint and af_heart style into a C-readable pack."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import torch

from assets import ARTIFACTS, MODEL_DIR, ROOT, verify_assets
from pack import MODEL_NAME, MODEL_REVISION, SOURCE_REVISION, VOICE_NAME, read_pack, write_pack
from voice_artifact import read_voice, write_voice


PACK_PATH = MODEL_DIR / "kokoro-v1_0-af_heart.kpack"
ADAM_VOICE_PATH = MODEL_DIR / "voices/am_adam.kvoice"
REQUIRED_GROUPS = {"bert", "bert_encoder", "predictor", "text_encoder", "decoder"}


def _tensor_bytes(tensor: torch.Tensor) -> tuple[str, tuple[int, ...], bytes]:
    contiguous = tensor.detach().cpu().contiguous()
    if contiguous.dtype == torch.float32:
        dtype, numpy_dtype = "f32", np.dtype("<f4")
    elif contiguous.dtype == torch.int64:
        dtype, numpy_dtype = "i64", np.dtype("<i8")
    else:
        raise ValueError(f"unsupported checkpoint dtype: {contiguous.dtype}")
    values = contiguous.numpy().astype(numpy_dtype, copy=False)
    return dtype, tuple(contiguous.shape), values.tobytes(order="C")


def convert(output: Path = PACK_PATH) -> Path:
    asset_hashes = verify_assets()
    config_path = MODEL_DIR / "config.json"
    checkpoint_path = MODEL_DIR / "kokoro-v1_0.pth"
    voice_path = MODEL_DIR / "voices/af_heart.pt"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["hidden_dim"] != 512 or config["style_dim"] != 128 or config["n_token"] != 178:
        raise ValueError("config does not match the pinned Kokoro-82M v1.0 contract")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if set(checkpoint) != REQUIRED_GROUPS:
        raise ValueError(f"unexpected checkpoint groups: {sorted(checkpoint)}")
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
    for group in sorted(REQUIRED_GROUPS):
        for name, tensor in sorted(checkpoint[group].items()):
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"checkpoint value is not a tensor: {group}.{name}")
            tensors[f"{group}.{name}"] = _tensor_bytes(tensor)

    voice = torch.load(voice_path, map_location="cpu", weights_only=True)
    if not isinstance(voice, torch.Tensor) or tuple(voice.shape) != (510, 1, 256):
        raise ValueError(f"unexpected af_heart style shape: {getattr(voice, 'shape', None)}")
    tensors[f"voice.{VOICE_NAME}.style"] = _tensor_bytes(voice)

    metadata = {
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "source_repository": "https://github.com/hexgrad/kokoro",
        "source_revision": SOURCE_REVISION,
        "checkpoint_sha256": asset_hashes["kokoro-v1_0.pth"],
        "config_sha256": asset_hashes["config.json"],
        "voice": VOICE_NAME,
        "voice_sha256": asset_hashes["voices/af_heart.pt"],
        "voice_shape": list(voice.shape),
        "tensor_layout": "little-endian contiguous; float32 and int64 only",
        "toolchain": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    write_pack(output, tensors, metadata)
    pack = read_pack(output)
    if set(pack.tensors) != set(tensors):
        raise RuntimeError("pack tensor names changed during serialization")
    for name, (_, shape, _) in tensors.items():
        if pack.tensors[name].shape != shape:
            raise RuntimeError(f"pack tensor shape changed during serialization: {name}")
    adam_voice_path = MODEL_DIR / "voices/am_adam.pt"
    adam_voice = torch.load(adam_voice_path, map_location="cpu", weights_only=True)
    if not isinstance(adam_voice, torch.Tensor) or tuple(adam_voice.shape) != (510, 1, 256):
        raise ValueError(f"unexpected am_adam style shape: {getattr(adam_voice, 'shape', None)}")
    write_voice(ADAM_VOICE_PATH, *_tensor_bytes(adam_voice))
    _, artifact_sha256 = read_voice(ADAM_VOICE_PATH)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PACK_PATH)
    args = parser.parse_args()
    path = convert(args.output)
    print(f"Wrote {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")
    print(f"Verified {len(read_pack(path).tensors)} named tensors.")
    _, artifact_sha256 = read_voice(ADAM_VOICE_PATH)
    print(f"Wrote {ADAM_VOICE_PATH.relative_to(ROOT)} ({ADAM_VOICE_PATH.stat().st_size} bytes)")
    print(f"am_adam artifact SHA-256: {artifact_sha256}")

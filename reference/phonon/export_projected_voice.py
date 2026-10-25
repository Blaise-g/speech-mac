"""Derive a model-bound projected voice from pinned GGUF BF16 weights, independently of C."""
from pathlib import Path
import json

import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from verify_assets import ROOT as ASSETS, PINS, verify
from quantization import sha256 as sha

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / ".build/phonon-cli-projected.safetensors"
WEIGHT = "flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight"


def project(raw: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Source latents [1,512,T] become Flow-LM embeddings [1,T,512]."""
    if (raw.dtype != torch.float32 or raw.ndim != 3 or raw.shape[:2] != (1, 512)
            or not 1 <= raw.shape[2] <= 4096 or weight.shape != (512, 512)
            or weight.dtype != torch.float32 or not torch.isfinite(raw).all()
            or not torch.isfinite(weight).all()):
        raise ValueError("invalid source voice or speaker projection")
    with torch.inference_mode():
        result = torch.nn.functional.linear(raw.transpose(1, 2).contiguous(), weight)
    if not torch.isfinite(result).all():
        raise ValueError("nonfinite projected voice")
    return result


def create() -> Path:
    verify()
    torch.set_num_threads(1)
    pins = json.loads(PINS.read_text())
    config = json.loads((ASSETS / "config.json").read_text())
    if config["embeddings_ext"] != ["ba7a5a45@300"]:
        raise ValueError("unsupported voice compatibility identity")
    source = ASSETS / "model.q8.gguf"
    voice = ASSETS / "default-voice.safetensors"
    reader = GGUFReader(str(source))
    tensor = next(item for item in reader.tensors if item.name == WEIGHT)
    if tensor.tensor_type.name != "BF16":
        raise ValueError("speaker projection is not published BF16")
    shape = tuple(reversed(tuple(int(dim) for dim in tensor.shape)))
    weight = torch.from_numpy(np.ascontiguousarray(
        dequantize(tensor.data, tensor.tensor_type).reshape(shape), dtype=np.float32))
    with safe_open(str(voice), framework="pt") as file:
        metadata = file.metadata()
        if metadata.get("sig") != "ba7a5a45" or metadata.get("epoch") != "300":
            raise ValueError("source raw voice compatibility mismatch")
    projected = project(load_file(str(voice))["speaker_wavs"], weight)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_file({"emb": projected}, str(OUTPUT), metadata={"model_ext": "ba7a5a45@300"})
    provenance = {
        "source_revision": pins["model"]["revision"],
        "source_model_sha256": sha(source),
        "source_config_sha256": sha(ASSETS / "config.json"),
        "source_voice_sha256": sha(voice),
        "model_id": config["model_id"],
        "embeddings_ext": config["embeddings_ext"],
        "weight": WEIGHT,
        "weight_dtype": "BF16",
        "shape": list(projected.shape),
        "projected_sha256": sha(OUTPUT),
        "operation": "independent Torch F.linear from published GGUF BF16 speaker projection; no native outputs",
    }
    OUTPUT.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    return OUTPUT


if __name__ == "__main__":
    print(create())

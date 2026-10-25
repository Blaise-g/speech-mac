"""Write the compact native pack while preserving the published GGUF encodings."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile

from gguf import GGUFReader
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(os.environ.get("PHONON_MODEL_DIR", "/Volumes/Code/Personal/speech-mac-models/phonon-40m-en"))
HERE = Path(__file__).resolve().parent
PINS = json.loads((HERE / "pins.json").read_text())
PIN = next(item for item in PINS["files"] if item["path"] == "model.q8.gguf")
OUT = ROOT / ".build" / f"phonon-native-{PIN['sha256'][:16]}-v1.kpack"
MAGIC = b"PHNQ8P1\0"
HEADER = struct.Struct("<8sIIQQ32s32s32s32s32s")
DTYPES = {"Q8_0": 1, "BF16": 2}


def sha(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def create(path: Path = OUT) -> Path:
    from verify_assets import verify
    verify()
    source = ASSETS / "model.q8.gguf"
    config = ASSETS / "config.json"
    reader = GGUFReader(str(source))
    observed = {kind: sum(t.tensor_type.name == kind for t in reader.tensors) for kind in DTYPES}
    if len(reader.tensors) != 209 or observed != {"Q8_0": 36, "BF16": 173}:
        raise ValueError(f"unexpected published tensor inventory: {observed}")
    with safe_open(str(ASSETS / "model.safetensors"), framework="pt") as original:
        original_names = set(original.keys())
        packed_names = {tensor.name for tensor in reader.tensors}
        if original_names != packed_names:
            raise ValueError("GGUF tensor names differ from the pinned source checkpoint")
        for tensor in reader.tensors:
            source_shape = tuple(original.get_slice(tensor.name).get_shape())
            if source_shape != tuple(reversed(tuple(int(dim) for dim in tensor.shape))):
                raise ValueError(f"GGUF shape differs from source checkpoint: {tensor.name}")
    entries = bytearray()
    payload = bytearray()
    for tensor in sorted(reader.tensors, key=lambda item: item.name):
        name = tensor.name.encode("utf-8")
        dtype = DTYPES.get(tensor.tensor_type.name)
        shape = tuple(reversed(tuple(int(dim) for dim in tensor.shape)))
        if dtype is None or not name or len(name) > 1024 or not shape or len(shape) > 8 or any(dim <= 0 for dim in shape):
            raise ValueError(f"invalid tensor descriptor: {tensor.name}")
        payload.extend(b"\0" * (-len(payload) % 8))
        raw = memoryview(tensor.data).cast("B").tobytes()
        expected = 1
        for dimension in shape:
            expected *= dimension
        if dtype == 1:
            if expected % 32 or len(raw) != expected // 32 * 34:
                raise ValueError(f"invalid Q8_0 size for {tensor.name}: {len(raw)}")
        elif len(raw) != expected * (2 if dtype == 2 else 4):
            raise ValueError(f"invalid BF16 size for {tensor.name}: {len(raw)}")
        entries.extend(struct.pack("<HBB", len(name), dtype, len(shape)))
        entries.extend(struct.pack("<" + "Q" * len(shape), *shape))
        entries.extend(struct.pack("<QQ", len(payload), len(raw)))
        entries.extend(name)
        payload.extend(raw)
    metadata = json.dumps({"model": "gradium/phonon-40m-en", "revision": PINS["model"]["revision"], "source": "model.q8.gguf", "source_sha256": PIN["sha256"], "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(), "tensors": 209}, sort_keys=True, separators=(",", ":")).encode()
    metadata += b" " * (-(HEADER.size + 4 + len(metadata) + len(entries)) % 8)
    header = HEADER.pack(MAGIC, 1, 209, len(entries), len(payload), bytes.fromhex(PIN["sha256"]), sha(config.read_bytes()), sha(entries), sha(payload), sha(metadata))
    blob = header + struct.pack("<I", len(metadata)) + metadata + entries
    blob += payload
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(blob)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return path


if __name__ == "__main__":
    print(create())

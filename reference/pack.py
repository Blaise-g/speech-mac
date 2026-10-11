"""Version 1 C-readable tensor pack writer and validator."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path

from assets import ARTIFACTS, HF_REPOSITORY, HF_REVISION


MAGIC = b"KOKORO1\0"
VERSION = 1
HEADER = struct.Struct("<8sIIIIQQ")
ENTRY = struct.Struct("<HBBI")
RANGE = struct.Struct("<QQ")
DIMENSION = struct.Struct("<Q")
MAX_TENSORS = 4096
MAX_NAME_BYTES = 1024
MAX_RANK = 8
MODEL_REVISION = HF_REVISION
SOURCE_REVISION = "dfb907a02bba8152ca444717ca5d78747ccb4bec"
MODEL_NAME = HF_REPOSITORY
VOICE_NAME = "af_heart"
FORMAT_NAME = "kokoro-native-tensor-pack"
DTYPE_CODES = {"f32": (1, 4), "i64": (2, 8)}
CODE_DTYPES = {code: (name, size) for name, (code, size) in DTYPE_CODES.items()}
MODEL_TENSOR_NAME = re.compile(r"^(bert|bert_encoder|predictor|text_encoder|decoder)\.[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*$")
MODEL_GROUPS = {"bert", "bert_encoder", "predictor", "text_encoder", "decoder"}
VOICE_TENSOR_NAME = f"voice.{VOICE_NAME}.style"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
PINNED_DIGESTS = {
    "checkpoint_sha256": ARTIFACTS["kokoro-v1_0.pth"][1],
    "config_sha256": ARTIFACTS["config.json"][1],
    "voice_sha256": ARTIFACTS["voices/af_heart.pt"][1],
}


class PackError(ValueError):
    """A pack is malformed, incompatible, or contains invalid tensor data."""


@dataclass(frozen=True)
class Tensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data: bytes


@dataclass(frozen=True)
class TensorPack:
    metadata: dict[str, object]
    tensors: dict[str, Tensor]


def _element_count(shape: tuple[int, ...]) -> int:
    if len(shape) > MAX_RANK or any(not isinstance(size, int) or size <= 0 for size in shape):
        raise PackError("invalid tensor shape")
    return math.prod(shape)


def _validate_name(name: str) -> None:
    if name == VOICE_TENSOR_NAME:
        return
    if not MODEL_TENSOR_NAME.fullmatch(name):
        raise PackError(f"invalid tensor name: {name!r}")


def write_pack(path: str | Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]], metadata: dict[str, object]) -> None:
    """Write sorted little-endian tensor data with an index and JSON provenance."""
    if len(tensors) > MAX_TENSORS:
        raise PackError("too many tensors")

    index = bytearray()
    payload = bytearray()
    for name in sorted(tensors):
        _validate_name(name)
        dtype, shape, data = tensors[name]
        if dtype not in DTYPE_CODES:
            raise PackError(f"unsupported tensor dtype: {dtype}")
        code, element_size = DTYPE_CODES[dtype]
        shape = tuple(shape)
        expected_size = _element_count(shape) * element_size
        if len(data) != expected_size:
            raise PackError(f"tensor byte size does not match shape: {name}")
        name_data = name.encode("utf-8")
        if not name_data or len(name_data) > MAX_NAME_BYTES:
            raise PackError(f"invalid tensor name length: {name!r}")
        index.extend(ENTRY.pack(len(name_data), code, len(shape), 0))
        for dimension in shape:
            index.extend(DIMENSION.pack(dimension))
        index.extend(RANGE.pack(len(payload), len(data)))
        index.extend(name_data)
        payload.extend(data)

    full_metadata = dict(metadata)
    full_metadata["format"] = full_metadata.get("format", FORMAT_NAME)
    full_metadata["format_version"] = full_metadata.get("format_version", VERSION)
    full_metadata["tensor_count"] = len(tensors)
    full_metadata["payload_sha256"] = hashlib.sha256(payload).hexdigest()
    metadata_data = json.dumps(full_metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    header = HEADER.pack(MAGIC, VERSION, len(tensors), len(index), len(metadata_data), len(payload), 0)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(header + index + metadata_data + payload)


def read_pack(path: str | Path) -> TensorPack:
    """Read and validate a version 1 Kokoro pack without loading tensor values."""
    try:
        data = Path(path).read_bytes()
    except OSError as error:
        raise PackError(f"cannot read pack: {error}") from error
    if len(data) < HEADER.size:
        raise PackError("truncated pack header")

    magic, version, tensor_count, index_size, metadata_size, payload_size, reserved = HEADER.unpack_from(data)
    if magic != MAGIC:
        raise PackError("invalid pack magic")
    if version != VERSION:
        raise PackError(f"unsupported pack version: {version}")
    if reserved != 0:
        raise PackError("nonzero reserved header field")
    if tensor_count > MAX_TENSORS:
        raise PackError("too many tensors")

    index_start = HEADER.size
    metadata_start = index_start + index_size
    payload_start = metadata_start + metadata_size
    payload_end = payload_start + payload_size
    if payload_end > len(data):
        raise PackError("truncated pack body")
    if payload_end < len(data):
        raise PackError("trailing bytes after pack payload")

    try:
        metadata = json.loads(data[metadata_start:payload_start].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PackError("invalid metadata JSON") from error
    _validate_metadata(metadata, tensor_count, data[payload_start:payload_end])

    tensors: dict[str, Tensor] = {}
    ranges: list[tuple[int, int]] = []
    cursor = index_start
    for _ in range(tensor_count):
        if cursor + ENTRY.size > metadata_start:
            raise PackError("truncated tensor index")
        name_size, dtype_code, rank, entry_reserved = ENTRY.unpack_from(data, cursor)
        cursor += ENTRY.size
        if entry_reserved != 0:
            raise PackError("nonzero reserved tensor field")
        if not name_size or name_size > MAX_NAME_BYTES or rank > MAX_RANK:
            raise PackError("invalid tensor index entry")
        dimension_bytes = rank * DIMENSION.size
        entry_tail = dimension_bytes + RANGE.size + name_size
        if cursor + entry_tail > metadata_start:
            raise PackError("truncated tensor index entry")
        shape = tuple(DIMENSION.unpack_from(data, cursor + offset)[0] for offset in range(0, dimension_bytes, DIMENSION.size))
        cursor += dimension_bytes
        offset, size = RANGE.unpack_from(data, cursor)
        cursor += RANGE.size
        try:
            name = data[cursor : cursor + name_size].decode("utf-8")
        except UnicodeDecodeError as error:
            raise PackError("invalid tensor name encoding") from error
        cursor += name_size
        _validate_name(name)
        if name in tensors:
            raise PackError(f"duplicate tensor name: {name}")
        if dtype_code not in CODE_DTYPES:
            raise PackError(f"unsupported tensor dtype code: {dtype_code}")
        dtype, element_size = CODE_DTYPES[dtype_code]
        expected_size = _element_count(shape) * element_size
        if size != expected_size:
            raise PackError(f"tensor byte size does not match shape: {name}")
        if offset > payload_size or size > payload_size - offset:
            raise PackError(f"tensor data outside payload: {name}")
        ranges.append((offset, offset + size))
        tensors[name] = Tensor(name, dtype, shape, data[payload_start + offset : payload_start + offset + size])

    if cursor != metadata_start:
        raise PackError("tensor index size mismatch")
    for previous, current in zip(sorted(ranges), sorted(ranges)[1:]):
        if current[0] < previous[1]:
            raise PackError("overlapping tensor data")
    if VOICE_TENSOR_NAME not in tensors:
        raise PackError(f"missing required voice tensor: {VOICE_TENSOR_NAME}")
    if len(tensors[VOICE_TENSOR_NAME].shape) != 3 or tensors[VOICE_TENSOR_NAME].shape[1:] != (1, 256):
        raise PackError("voice tensor shape must be [styles, 1, 256]")
    present_groups = {name.split(".", 1)[0] for name in tensors if name != VOICE_TENSOR_NAME}
    missing_groups = MODEL_GROUPS - present_groups
    if missing_groups:
        raise PackError(f"missing required model tensor groups: {', '.join(sorted(missing_groups))}")
    return TensorPack(metadata, tensors)


def _validate_metadata(metadata: object, tensor_count: int, payload: bytes) -> None:
    if not isinstance(metadata, dict):
        raise PackError("metadata must be a JSON object")
    if metadata.get("format") != FORMAT_NAME:
        raise PackError("incompatible pack metadata format")
    if metadata.get("format_version") != VERSION:
        raise PackError("incompatible pack metadata version")
    if metadata.get("model") != MODEL_NAME:
        raise PackError("incompatible model metadata")
    if metadata.get("model_revision") != MODEL_REVISION:
        raise PackError("incompatible model revision metadata")
    if metadata.get("source_revision") != SOURCE_REVISION:
        raise PackError("incompatible source revision metadata")
    if metadata.get("voice") != VOICE_NAME:
        raise PackError("incompatible voice metadata")
    if metadata.get("tensor_count") != tensor_count:
        raise PackError("tensor count does not match metadata")
    for key in ("checkpoint_sha256", "config_sha256", "voice_sha256", "payload_sha256"):
        if not isinstance(metadata.get(key), str) or not SHA256.fullmatch(metadata[key]):
            raise PackError(f"invalid metadata digest: {key}")
    for key, expected in PINNED_DIGESTS.items():
        if metadata[key] != expected:
            raise PackError(f"metadata digest does not match pinned {key}: {metadata[key]}")
    if hashlib.sha256(payload).hexdigest() != metadata["payload_sha256"]:
        raise PackError("payload SHA-256 mismatch")

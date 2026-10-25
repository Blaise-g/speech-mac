"""Export independent GGUF/PyTorch outputs for the native packed-Q8 census."""
from pathlib import Path
import struct
import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize
from native_pack import ASSETS, ROOT

OUTFILE = ROOT / ".build" / "phonon-native-q8-projections-v1.bin"
BF16_OUTFILE = ROOT / ".build" / "phonon-native-bf16-oracle-v1.bin"

def create():
    torch.set_num_threads(1)
    reader = GGUFReader(str(ASSETS / "model.q8.gguf"))
    tensors = sorted((tensor for tensor in reader.tensors if tensor.tensor_type.name == "Q8_0"), key=lambda item: item.name)
    with OUTFILE.open("wb") as stream:
        stream.write(b"PHNPROJ1")
        stream.write(struct.pack("<I", len(tensors)))
        for tensor in tensors:
            matrix = np.ascontiguousarray(dequantize(tensor.data, tensor.tensor_type).reshape(tuple(reversed(tensor.shape))), dtype=np.float32)
            if matrix.ndim != 2:
                raise ValueError(f"Q8 projection is not a matrix: {tensor.name}: {matrix.shape}")
            rows, columns = matrix.shape
            vector = np.asarray([((index * 17 % 101) - 50) / 37.0 for index in range(columns)], dtype=np.float32)
            with torch.inference_mode():
                projected = torch.mv(torch.from_numpy(matrix), torch.from_numpy(vector)).numpy()
            if not np.isfinite(projected).all():
                raise ValueError(f"non-finite Q8 projection oracle result: {tensor.name}")
            name = tensor.name.encode()
            stream.write(struct.pack("<H", len(name)))
            stream.write(name)
            stream.write(struct.pack("<II", rows, columns))
            stream.write(projected.tobytes())
    bf16_tensors = sorted((tensor for tensor in reader.tensors if tensor.tensor_type.name == "BF16"), key=lambda item: item.name)
    with BF16_OUTFILE.open("wb") as stream:
        stream.write(b"PHNBF161")
        stream.write(struct.pack("<I", len(bf16_tensors)))
        for tensor in bf16_tensors:
            values = np.ascontiguousarray(dequantize(tensor.data, tensor.tensor_type).reshape(tuple(reversed(tensor.shape))), dtype=np.float32).reshape(-1)
            name = tensor.name.encode()
            count = min(16, values.size)
            stream.write(struct.pack("<H", len(name)))
            stream.write(name)
            stream.write(struct.pack("<I", count))
            stream.write(values[:count].tobytes())
    return OUTFILE

if __name__ == "__main__": print(create())

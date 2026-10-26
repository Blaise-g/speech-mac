"""Independent identical-input diagnostics; does not replace the frozen generation gate.

Run with locked reference/phonon uv. --transformer-source permits comparison with
an archived source file compatible with the current headers and guarded
PHONON_PACK_TEST_API attention primitives. Older private-seam archives must use
the diagnostic from their own revision. Builds a temporary library for inspection.
Source anchors: PyTorch v2.7.1 aten/src/ATen/native/{Linear.cpp,
cpu/layer_norm_kernel.cpp,cpu/moments_utils.h,cpu/SumKernel.cpp}; pinned HF
modeling_gradium_phonon_tts.py RoPE; quantization.py's published Q8 expansion.
"""
from pathlib import Path
import argparse
import json
import ctypes as C
import subprocess
import tempfile
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
STAGE_BUDGET = json.loads((ROOT / "reference/phonon/budgets.json").read_text())["native_parity_gate"]["native_q8_stage_targets"]["backbone_output"]


def assert_component(actual, expected):
    # Use the existing frozen backbone stage budget; this focused component
    # regression check never replaces the independent autonomous gate.
    if actual.shape != expected.shape:
        raise AssertionError(f"native component shape mismatch: {actual.shape} != {expected.shape}")
    error = np.abs(actual - expected)
    limit = STAGE_BUDGET["atol"] + STAGE_BUDGET["rtol"] * np.abs(expected)
    if not np.isfinite(actual).all() or not np.all(error <= limit):
        raise AssertionError(f"native component exceeds frozen stage budget: max_abs={error.max():.9g}; "
                             f"over={np.count_nonzero(error > limit)}/{actual.size}")


class TensorView(C.Structure):
    _fields_ = [("dtype", C.c_int), ("rank", C.c_uint), ("shape", C.c_uint64 * 8),
                ("data", C.c_void_p), ("byte_count", C.c_size_t), ("linear_data", C.c_void_p)]


def build_library(source: Path, scratch: Path):
    """Compile ABI checks before allowing any ctypes tensor calls."""
    source = source.resolve()
    text = source.read_text()
    if not all(name in text for name in ("PHONON_PACK_TEST_API", "phonon_attention_test(",
                                         "phonon_prefill_attention_test(")):
        raise ValueError("unsupported transformer source: requires current headers and guarded "
                         "PHONON_PACK_TEST_API attention primitives; use the diagnostic from "
                         "the archived source revision for older private-seam sources")
    wrapper = scratch / "operators.c"
    checks = [f'_Static_assert(sizeof(PhononTensorView) == {C.sizeof(TensorView)}, "TensorView size ABI drift");']
    for name, _ in TensorView._fields_:
        checks.append(f'_Static_assert(offsetof(PhononTensorView, {name}) == '
                      f'{getattr(TensorView, name).offset}, "TensorView {name} ABI drift");')
    wrapper.write_text('#include <stddef.h>\n#include "phonon_pack.h"\n' +
                       "\n".join(checks) + '\n#include "' + str(source) + '"\n')
    library = scratch / "operators.dylib"
    subprocess.run(["clang", "-std=c17", "-O0", "-dynamiclib", "-DPHONON_PACK_TEST_API",
                    "-I" + str(ROOT / "src"), str(wrapper),
                    *(str(ROOT / "src" / name) for name in
                      ("phonon_q8.c", "phonon_pack.c", "sha256.c", "phonon_math.c")),
                    "-framework", "Accelerate", "-o", str(library)], check=True)
    return C.CDLL(str(library))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformer-source", type=Path, default=ROOT / "src/phonon_transformer.c")
    args = parser.parse_args()
    scratch = Path(tempfile.mkdtemp(prefix="phonon-operators-"))
    library = scratch / "operators.dylib"
    try:
        lib = build_library(args.transformer_source, scratch)
    except ValueError as error:
        parser.error(str(error))
    lib.phonon_linear.argtypes = [C.POINTER(TensorView), C.c_void_p, C.c_size_t,
                                 C.c_size_t, C.c_void_p, C.c_void_p]
    lib.phonon_layer_norm.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p,
                                     C.c_size_t, C.c_float, C.c_void_p]
    lib.phonon_rope_interleaved.argtypes = [C.c_void_p, C.c_size_t, C.c_size_t,
                                           C.c_size_t, C.c_float]
    torch.set_num_threads(1)
    rng = np.random.default_rng(2402)
    for dtype in ("BF16", "Q8"):
        unequal = 0
        maximum = 0.0
        count = 0
        for rows, cols in ((2, 32), (512, 32), (512, 256), (512, 512),
                           (1536, 512), (1024, 512), (32, 512), (1, 512)):
            for iteration in range(8):
                if dtype == "BF16":
                    weights = torch.from_numpy(rng.normal(size=(rows, cols)).astype("float32")).bfloat16().float()
                    packed = (weights.numpy().view("uint32") >> 16).astype("uint16")
                    tag = 2
                else:
                    packed = np.empty(rows * cols // 32, dtype=np.dtype([("scale", "<f2"), ("q", "i1", 32)]))
                    packed["scale"] = rng.uniform(0.001, 0.1, len(packed)).astype("float16")
                    packed["q"] = rng.integers(-127, 128, size=(len(packed), 32), dtype="int8")
                    expanded = (packed["scale"].astype("float32")[:, None] * packed["q"].astype("float32")).reshape(rows, cols).copy()
                    weights = torch.from_numpy(expanded)
                    tag = 1
                x = rng.normal(size=cols).astype("float32")
                bias = rng.normal(size=rows).astype("float32") if iteration % 2 else None
                actual = np.empty(rows, dtype="float32")
                view = TensorView(tag, 2, (C.c_uint64 * 8)(rows, cols), packed.ctypes.data, packed.nbytes)
                assert lib.phonon_linear(C.byref(view), x.ctypes.data, rows, cols,
                                         None if bias is None else bias.ctypes.data, actual.ctypes.data)
                expected = torch.nn.functional.linear(torch.from_numpy(x).reshape(1, 1, cols), weights,
                                                       None if bias is None else torch.from_numpy(bias)).numpy().ravel()
                maximum = max(maximum, float(np.abs(actual - expected).max()))
                unequal += int(np.count_nonzero(actual != expected))
                count += 1
        print(f"{dtype}: {count} independent linear inputs; unequal={unequal}; max_abs={maximum:.9g}")
    maximum = 0.0
    unequal = 0
    patterns = [np.ones(512, dtype="float32"),
                (1 + np.arange(512, dtype="float32") * np.float32(1e-7)),
                np.tile(np.array([1e8, 1, -1e8, 1], dtype="float32"), 128)]
    patterns += [rng.normal(size=512).astype("float32") * scale
                 for scale in (1e-5, 1, 1e4) for _ in range(16)]
    for x in patterns:
        weight = rng.normal(size=512).astype("float32")
        bias = rng.normal(size=512).astype("float32")
        actual = np.empty(512, dtype="float32")
        assert lib.phonon_layer_norm(x.ctypes.data, weight.ctypes.data, bias.ctypes.data,
                                    512, 1e-5, actual.ctypes.data)
        expected = torch.nn.functional.layer_norm(torch.from_numpy(x).reshape(1, 1, 512), (512,),
                                                  torch.from_numpy(weight), torch.from_numpy(bias), 1e-5).numpy().ravel()
        maximum = max(maximum, float(np.abs(actual - expected).max()))
        unequal += int(np.count_nonzero(actual != expected))
    print(f"LayerNorm: {len(patterns)} independent inputs; unequal={unequal}; max_abs={maximum:.9g}")
    inv = 1.0 / (10000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    maximum = 0.0
    unequal = 0
    positions = (0, 1, 7, 31, 139, 140, 171, 250, 1023, 4095, 8191)
    for position in positions:
        x = rng.normal(size=512).astype("float32")
        actual = x.copy()
        assert lib.phonon_rope_interleaved(actual.ctypes.data, 512, 8, position, 10000.0)
        pairs = torch.from_numpy(x.reshape(8, 32, 2).copy())
        angle = position * inv
        expected = torch.stack((pairs[:, :, 0] * angle.cos() - pairs[:, :, 1] * angle.sin(),
                                pairs[:, :, 1] * angle.cos() + pairs[:, :, 0] * angle.sin()), -1).numpy().ravel()
        maximum = max(maximum, float(np.abs(actual - expected).max()))
        unequal += int(np.count_nonzero(actual != expected))
    print(f"RoPE: {len(positions)} positions through8191; unequal={unequal}; max_abs={maximum:.9g}")
    lib.phonon_erf_gelu.argtypes = [C.c_void_p, C.c_size_t]
    x = np.linspace(-8, 8, 65537, dtype="float32")
    actual = x.copy()
    assert lib.phonon_erf_gelu(actual.ctypes.data, len(actual))
    expected = torch.nn.functional.gelu(torch.from_numpy(x), approximate="none").numpy()
    print(f"GELU:65537 independent grid values+tail; unequal={np.count_nonzero(actual != expected)}; "
          f"max_abs={np.abs(actual - expected).max():.9g}")
    if hasattr(lib, "phonon_attention_test"):
        lib.phonon_attention_test.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p,
                                       C.c_size_t, C.c_size_t, C.c_void_p]
        maximum = 0.0
        unequal = 0
        lengths = (1, 2, 3, 7, 139, 171, 511, 512, 513, 1023, 1024, 1025, 8192)
        for length in lengths:
            # Preserve the actual cache stride, including unrelated head slots.
            keys = rng.normal(size=(length, 512)).astype("float32")
            values = rng.normal(size=(length, 512)).astype("float32")
            query = rng.normal(size=64).astype("float32")
            actual = np.empty(64, dtype="float32")
            assert lib.phonon_attention_test(query.ctypes.data, keys.ctypes.data, values.ctypes.data,
                                       length, 512, actual.ctypes.data)
            expected = torch.nn.functional.scaled_dot_product_attention(
                torch.from_numpy(np.concatenate((query[::2], query[1::2]))).reshape(1, 1, 1, 64),
                torch.from_numpy(np.concatenate((keys[:, :64:2], keys[:, 1:64:2]), axis=-1)).reshape(1, 1, length, 64),
                torch.from_numpy(values[:, :64]).reshape(1, 1, length, 64)).numpy().ravel()
            assert_component(actual, expected)
            maximum = max(maximum, float(np.abs(actual - expected).max()))
            unequal += int(np.count_nonzero(actual != expected))
        print(f"SDPA: {len(lengths)} cache lengths through8192; unequal={unequal}; max_abs={maximum:.9g}")
    if hasattr(lib, "phonon_prefill_attention_test"):
        lib.phonon_prefill_attention_test.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p,
                                               C.c_size_t, C.c_void_p]
        maximum = 0.0
        unequal = 0
        lengths = (1, 31, 32, 33, 191, 192, 193, 511, 512, 513, 767, 768, 769, 1025)
        for length in lengths:
            query = rng.normal(size=(length, 1536)).astype("float32")
            keys = rng.normal(size=(length, 512)).astype("float32")
            values = rng.normal(size=(length, 512)).astype("float32")
            actual = np.empty((length, 512), dtype="float32")
            assert lib.phonon_prefill_attention_test(query.ctypes.data, keys.ctypes.data, values.ctypes.data,
                                               length, actual.ctypes.data)
            def half_order(x):
                pairs = torch.from_numpy(x.copy()).reshape(length, 8, 32, 2)
                return torch.cat((pairs[..., 0], pairs[..., 1]), dim=-1).transpose(0, 1).unsqueeze(0)
            expected = torch.nn.functional.scaled_dot_product_attention(
                half_order(query[:, :512]), half_order(keys),
                torch.from_numpy(values).reshape(length, 8, 64).transpose(0, 1).unsqueeze(0),
                is_causal=True)[0].transpose(0, 1).reshape(length, 512).numpy()
            assert_component(actual, expected)
            maximum = max(maximum, float(np.abs(actual - expected).max()))
            unequal += int(np.count_nonzero(actual != expected))
        print(f"Prefill SDPA: {len(lengths)} query/key boundary lengths; unequal={unequal}; max_abs={maximum:.9g}")
    print(f"Temporary library: {library}; component checks pass; autonomous acceptance remains a separate gate.")


if __name__ == "__main__":
    main()

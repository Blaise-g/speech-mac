# Kokoro v1.0 reference and weight pack

This directory contains development-only Python tools for the official Kokoro-82M v1.0 oracle, compact module-boundary fixtures, and conversion to the C-readable weight pack. Python is not part of the shipped inference path.

The immutable source and artifact pins are in [pins.json](pins.json). The Hugging Face model snapshot is `f3ff3571791e39611d31c381e3a41a3af07b4987`; its checkpoint SHA-256 is the v1.0 release hash `496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4`. The official inference code is `hexgrad/kokoro` commit `dfb907a02bba8152ca444717ca5d78747ccb4bec`, package version 0.9.4. Core source file hashes are recorded alongside the repository revision. Python dependencies are pinned in [pyproject.toml](pyproject.toml) and [uv.lock](uv.lock).

Use Python 3.12.12 and uv 0.9.5 for the pinned toolchain. The model checkpoint and voice files go under the root `models/` directory, which is ignored by Git.

```sh
uv python install 3.12.12
uv sync --project reference --locked
uv run --project reference --locked python reference/assets.py download
uv run --project reference --locked python reference/convert.py
uv run --project reference --locked python reference/oracle.py --write
uv run --project reference --locked python reference/oracle.py --verify
uv run --project reference --locked python -m unittest discover -s reference/tests
```

`assets.py verify` checks file sizes and SHA-256 before conversion or inference. The converter accepts only the pinned checkpoint/config/voice, requires the expected checkpoint modules and `af_heart` tensor shape, and writes `models/kokoro-v1_0-af_heart.kpack`. Re-running conversion writes the same pack bytes for the same lock and inputs.

The current checked-in fixture set covers one hello-only input. The fixed prepared input is in [oracle_input.json](oracle_input.json). It spells `həˈloʊ` and supplies its exact vocabulary IDs between the model's zero BOS/EOS IDs. The voice row follows the official `len(phonemes) - 1` selection rule. The reference wrapper calls the pinned `KModel.forward_with_tokens` and captures module outputs with hooks. It reconstructs the explicit monotonic alignment from the official returned durations using the upstream repeat-interleave construction. It seeds the official random source, captures harmonic/noise/voicing excitation, and runs twice before writing a fixture. Both reference runs must be bitwise equal. Later stage tickets can add boundary-case inputs while these initial numerical budgets stay fixed.

The checked-in [fixture](fixtures/af_heart_hello.npz) contains fixed IDs and compact outputs at each planned seam: contextual encoder, duration predictor and alignment, F0/noise, text encoder and alignment, decoder input, generator excitation and spectrum, and the final 24 kHz waveform. [Its manifest](fixtures/manifest.json) records provenance, each array's shape, dtype, and SHA-256, plus measured repeatability.

## C-readable tensor pack, version 1

All integers and tensor elements use little-endian encoding. The pack starts with a 40-byte header:

| Offset | Type | Meaning |
| --- | --- | --- |
| 0 | 8 bytes | `KOKORO1\0` magic |
| 8 | `u32` | format version (`1`) |
| 12 | `u32` | tensor count |
| 16 | `u32` | index byte length |
| 20 | `u32` | UTF-8 JSON metadata byte length |
| 24 | `u64` | tensor payload byte length |
| 32 | `u64` | reserved, must be zero |

The index immediately follows the header. Each sorted tensor entry is `u16 name_length`, `u8 dtype_code`, `u8 rank`, `u32 reserved`, `rank` `u64` dimensions, `u64 payload_offset`, `u64 payload_size`, and `name_length` UTF-8 bytes. Dtype code 1 is float32; code 2 is int64. Tensor offsets are relative to the payload start. The JSON metadata and then the contiguous payload follow the index. The payload is ordered by tensor name and its SHA-256 is in the metadata. V1 supports at most 4096 tensors, rank 8, and 1024-byte names. Only `bert.*`, `bert_encoder.*`, `predictor.*`, `text_encoder.*`, `decoder.*`, and `voice.af_heart.style` names are accepted.

The reference reader validates magic/version, metadata compatibility and hashes, tensor names and uniqueness, rank/shape/byte-size agreement, offset bounds and overlap, required model and voice tensors, truncation, and trailing bytes. The C-side implementation in later tickets can parse the fixed header and index without a Python runtime or JSON dependency; JSON metadata is skippable by its recorded length.

## Numerical budgets

Budgets in [budgets.json](budgets.json) are set before native inference. Two identical oracle runs in the locked CPU toolchain establish bitwise repeatability and a measured maximum absolute difference of zero. Float32 has unit roundoff `u = 2^-24`; the largest configured dot-product reduction is the decoder convolution with 1090 input channels and a 3-tap kernel, for `gamma_3270 = 3270u/(1 - 3270u) ≈ 1.949452e-4`. Module-boundary float outputs therefore use `rtol=1e-3` and `atol=1e-5`, about 5.13 times that accumulation bound plus a near-zero floor. The final waveform uses normalized RMSE at most `1e-3` (about 60 dB signal-to-error ratio) and maximum absolute error at most `2e-3`. Integer IDs, masks, durations, alignment, and captured generator excitation compare exactly. A later native mismatch is diagnosed against these fixed limits; native output does not set or widen them.

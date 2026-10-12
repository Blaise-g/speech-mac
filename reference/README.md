# Kokoro v1.0 reference and weight pack

This directory contains development-only Python tools for the official Kokoro-82M v1.0 oracle, compact module-boundary fixtures, and conversion to the C-readable weight pack. Python is not part of the shipped inference path.

The immutable source and artifact pins are in [pins.json](pins.json). The Hugging Face model snapshot is `f3ff3571791e39611d31c381e3a41a3af07b4987`; its checkpoint SHA-256 is the v1.0 release hash `496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4`. Both `af_heart` and `am_adam` voice files come from this snapshot. The official inference code is `hexgrad/kokoro` commit `dfb907a02bba8152ca444717ca5d78747ccb4bec`, package version 0.9.4. Core source file hashes are recorded alongside the repository revision. Python dependencies are pinned in [pyproject.toml](pyproject.toml) and [uv.lock](uv.lock).

Use Python 3.12.12 and uv 0.9.5 for the pinned toolchain. The model checkpoint and voice files go under the root `models/` directory, which is ignored by Git.

```sh
UV_CACHE_DIR=/private/tmp/kokoro-uv-cache uv python install 3.12.12
UV_CACHE_DIR=/private/tmp/kokoro-uv-cache uv sync --project reference --locked
UV_CACHE_DIR=/private/tmp/kokoro-uv-cache uv run --project reference --locked python reference/assets.py download
UV_CACHE_DIR=/private/tmp/kokoro-uv-cache uv run --project reference --locked python reference/convert.py
UV_CACHE_DIR=/private/tmp/kokoro-uv-cache uv run --project reference --locked python reference/oracle.py --write
UV_CACHE_DIR=/private/tmp/kokoro-uv-cache uv run --project reference --locked python reference/oracle.py --verify
(cd reference && UV_CACHE_DIR=/private/tmp/kokoro-uv-cache uv run --project . --offline --no-sync --locked python -m unittest discover -s tests)
```

`assets.py verify` checks file sizes and SHA-256 before conversion or inference. The converter accepts only the pinned checkpoint/config/voice files, requires the expected checkpoint modules and both `[510, 1, 256]` style shapes, and writes `models/kokoro-v1_0-af_heart.kpack` plus `models/voices/am_adam.kvoice`. Re-running conversion writes the same artifacts for the same lock and inputs. The model pack stays single-copy; the separate artifact contains only `am_adam` style rows.

The current checked-in fixture set covers one hello-only input. The fixed prepared input is in [oracle_input.json](oracle_input.json). It spells `həˈloʊ` and supplies its exact vocabulary IDs between the model's zero BOS/EOS IDs. The voice row follows the official `len(phonemes) - 1` selection rule. The reference wrapper calls the pinned `KModel.forward_with_tokens` and captures module outputs with hooks. It reconstructs the explicit monotonic alignment from the official returned durations using the upstream repeat-interleave construction. It seeds the official random source, captures harmonic/noise/voicing excitation, and runs twice before writing a fixture. Both reference runs must be bitwise equal. Later stage tickets can add boundary-case inputs while these initial numerical budgets stay fixed.

The checked-in [fixture](fixtures/af_heart_hello.npz) contains fixed IDs and compact outputs at each planned seam: contextual encoder, duration predictor and alignment, F0/noise, text encoder and alignment, decoder input, generator excitation and source STFT, generator spectrum, and the final 24 kHz waveform. [Its manifest](fixtures/manifest.json) records provenance, each array's shape, dtype, and SHA-256, plus measured repeatability.

The neural generator accepts an optional captured source spectrogram so tests can compare its learned convolution/residual stages on exactly the same conditioning as the pinned oracle. This seam isolates the learned network from the numerical phase branch chosen by Torch's source FFT when an imaginary coefficient is near zero: angles differing by `2π` represent the same complex coefficient, but the following learned convolution can react to the raw phase value. The native harmonic-to-STFT path remains the production default and is checked separately against oracle magnitudes and reconstructed complex coefficients under the fixed float budget. This conditioning replay verifies learned-stage parity; it does not establish free-running waveform sample parity.

The pinned `SineGen` adds random initial phase offsets to its eight overtone channels at the first sample of its 300x-upsampled F0 input. It immediately downsamples those phase increments by 300 with linear interpolation, whose first sample is centered around input position 149.5, so none of those one-sample offsets contributes to the frame-rate increments or the resulting phase. Native C excitation accumulates those same frame-rate increments and interpolates the expanded phase; the seed controls its Gaussian source noise.

The native C tests compare the context encoder with the raw little-endian export in `fixtures/contextual_features.f32le`; its digest matches the `contextual_features` array in the fixture manifest. Regenerate that export and the small structural pack used by the loader tests with `(cd reference && UV_CACHE_DIR=/private/tmp/kokoro-uv-cache uv run --project . --offline --no-sync --locked python export_c_test_fixtures.py)`.

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

The index immediately follows the header. Each sorted tensor entry is `u16 name_length`, `u8 dtype_code`, `u8 rank`, `u32 reserved`, `rank` `u64` dimensions, `u64 payload_offset`, `u64 payload_size`, and `name_length` UTF-8 bytes. Dtype code 1 is float32; code 2 is int64. Tensor offsets are relative to the payload start and aligned to the dtype's element size. The JSON metadata and then the payload follow the index; the metadata may end in JSON whitespace so the payload begins on an 8-byte boundary. The payload is ordered by tensor name, with zero padding between entries where needed for dtype alignment, and its SHA-256 is in the metadata. V1 supports at most 4096 tensors, rank 8, and 1024-byte names. Only `bert.*`, `bert_encoder.*`, `predictor.*`, `text_encoder.*`, `decoder.*`, and `voice.af_heart.style` names are accepted.

The reference reader validates magic/version, metadata compatibility and hashes, tensor names and uniqueness, rank/shape/byte-size agreement, offset bounds and overlap, required model and voice tensors, truncation, and trailing bytes. The native C loader applies the same structural checks and alignment requirements without a Python runtime or JSON library; it parses the required provenance fields from the recorded metadata bytes.

## Separate voice artifact, version 1

`am_adam.kvoice` keeps a second voice from duplicating the model checkpoint in the tensor pack. Its 192-byte header contains the `KVOICE1\0` magic, format version, style row count and width, payload byte count, pinned model revision, original official voice-file SHA-256, and converted style-payload SHA-256. The payload is contiguous little-endian float32 data with shape `[510, 1, 256]`. The C loader checks the header, pinned hashes, exact artifact digest, shape, and finite values before exposing the mapped style rows to the resident engine.

## Numerical budgets

Budgets in [budgets.json](budgets.json) are set before native inference. Two identical oracle runs in the locked CPU toolchain establish bitwise repeatability and a measured maximum absolute difference of zero. Float32 has unit roundoff `u = 2^-24`; the largest configured dot-product reduction is the decoder convolution with 1090 input channels and a 3-tap kernel, for `gamma_3270 = 3270u/(1 - 3270u) ≈ 1.949452e-4`. Module-boundary float outputs therefore use `rtol=1e-3` and `atol=1e-5`, about 5.13 times that accumulation bound plus a near-zero floor. The final waveform uses normalized RMSE at most `1e-3` (about 60 dB signal-to-error ratio) and maximum absolute error at most `2e-3`. Integer IDs, masks, durations, alignment, and captured generator excitation compare exactly. A later native mismatch is diagnosed against these fixed limits; native output does not set or widen them.

# Phonon 40M development reference

This isolated `uv` project runs the pinned Hugging Face Transformers/PyTorch Phonon implementation against the hash-verified model and Mimi decoder assets in `PHONON_MODEL_DIR` (default `models/phonon-40m-en`). Assets are read-only and never copied into the repository. `uv.lock` pins Python dependencies and Transformers commit `50e35cc0ca18ec19b80653690de553e6decf74f5`; the checkpoint is `gradium/phonon-40m-en@cc738b7e4d182c2ba1bafb8062aa4846af000637`; decoder `gradium/phonon-mimi@33519be1a8e61c6fa00c6e154e2d38e76a367919`.

The locked HF converter currently sets Mimi's `sliding_window` to `transformer_context + 1` (251). Ticket19's reference corrects that config to 250 positions. This is the minimal documented reference patch route authorized for reconciliation: Python Pocket TTS trained semantics use `delta < context`, so context 250 allows exactly 250 keys, unlike the PR209 Rust behavior `delta <= context`. The patch is explicitly applied in `oracle.py`; `context.py` encodes the expected predicate and boundary test. Upstream PR209 documents the discovered discrepancy: https://github.com/gradium-ai/xn-ptts/pull/209. HF model source is Apache-2.0, xn-ptts source Apache-2.0/MIT; 40M weights are CC BY-NC 4.0 and the Mimi codec weights Apache-2.0. Access to the gated 40M model requires accepting its Hugging Face terms. To access the gated 40M snapshot, accept the model terms on its Hugging Face page, then authenticate through the Hugging Face CLI or a local token before downloading. The supplied reference voice is the model repository’s `default-voice.safetensors` from the same pinned revision; no external named voice catalogue is used. These assets are for development/reference use, not shipped runtime dependencies.

The pinned Rust `NormalRng::new` samples with standard deviation `sqrt(temperature)`; the pinned HF sampler multiplies Gaussian draws by the same `sqrt(temperature)`. Ticket19 explicitly sets temperature `0.8`, EOS threshold `4.0`, one flow step, and the pinned processor tail/fade settings, rather than inheriting the repository generation file’s temperature. The source revisions and file digests are in `source-hashes.json`.

Run from repository root:

```sh
UV_CACHE_DIR=/private/tmp/speech-mac-uv-cache uv run --project reference/phonon --locked python reference/phonon/oracle.py write
UV_CACHE_DIR=/private/tmp/speech-mac-uv-cache uv run --project reference/phonon --locked python reference/phonon/oracle.py verify
UV_CACHE_DIR=/private/tmp/speech-mac-uv-cache uv run --project reference/phonon --locked python reference/phonon/oracle.py write-q8
UV_CACHE_DIR=/private/tmp/speech-mac-uv-cache uv run --project reference/phonon --locked python reference/phonon/oracle.py verify-q8
UV_CACHE_DIR=/private/tmp/speech-mac-uv-cache uv run --project reference/phonon --locked python reference/phonon/oracle.py write-mimi
UV_CACHE_DIR=/private/tmp/speech-mac-uv-cache uv run --project reference/phonon --locked python reference/phonon/oracle.py verify-mimi
UV_CACHE_DIR=/private/tmp/speech-mac-uv-cache uv run --project reference/phonon --locked python reference/phonon/oracle.py write-temporal
UV_CACHE_DIR=/private/tmp/speech-mac-uv-cache uv run --project reference/phonon --locked python reference/phonon/oracle.py verify-temporal
make phonon-reference-check
UV_CACHE_DIR=/private/tmp/speech-mac-uv-cache uv run --project reference/phonon --locked python -m unittest discover -s reference/phonon/tests
```

## Native encoded-weight pack and projection check

`make phonon-native-pack` verifies the pinned source assets, then writes an ignored `.build/phonon-native-<source-hash>-v1.kpack` plus independent Q8 projection and BF16 decoding oracle files. The pack contains the 209 published tensor names and shapes, raw 34-byte-per-block Q8_0 tensors, raw BF16 tensors, and SHA-256 provenance for the GGUF, config, metadata, index, and payload. The C loader pins the canonical tensor-index digest as well as the source and config digests. No original F32 checkpoint tensors or expanded Q8 weights are copied into this runtime pack. The compact version-1 layout is validated by `src/phonon_pack.c`; tensor names/data are borrowed until `phonon_pack_close` and names are length-delimited.

The v1 file starts with a fixed header (`PHNQ8P1`, version, tensor count, index and payload lengths, then source/config/index/payload/metadata SHA-256 digests). A length-prefixed canonical provenance JSON block follows. Each index record stores a UTF-8 name length, encoding, rank, little-endian `u64` dimensions, payload offset and exact encoded length, then the name bytes. Payload starts on an 8-byte boundary. Q8_0 blocks are the published little-endian binary16 scale and 32 signed bytes; BF16 tensors retain their little-endian 16-bit words. The loader rejects unrecognized index schemas by their pinned digest.

Run `make phonon-native-check` on the machine with the pinned shared assets to inspect all 209 tensors, decode every BF16 tensor and compare representative values, and compare native C projections for all 36 Q8 matrices with the locked `gguf` dequantizer plus PyTorch CPU matrix-vector multiplication using the frozen `1e-3` relative / `1e-5` absolute tolerance. `make check` always exercises the same C parser against a synthetic weightless pack, including truncation, unsupported version/type, source/config mismatch, malformed metadata, dimension overflow, row-shape, overlap, bad length, invalid name, and payload corruption cases; it reports the real-weight projection census as a skip without the ignored real pack. `tests/test_phonon_q8.c` separately tests the scalar kernel with asymmetric rows and inputs.

The hello fixture stores the raw text and HF-tokenizer-normalized text, prepared IDs and masks, voice latents and projected voice/prefix states, exact captured Gaussian draws and temperature-scaled inputs, backbone and flow stages, raw EOS logits, Mimi transformer features, raw codec PCM, postprocessed PCM and stop lengths. The tokenizer output (for the structured quote/em-dash case) is tokenizer normalization only; it does not perform runtime number/currency/email/address rewriting. HF token IDs are produced by the pinned converter’s versioned tokenizer and model-config special IDs. Mimi temporal fixtures decode 263 latent frames (4,208 post-upsampling transformer positions) whole, one latent frame at a time (16 positions per frame), and in uneven [1,7,113,142]-frame chunks while retaining the transformer KV cache. The framewise/uneven feature sequences are passed through the causal SEANet decoder with a 250-position overlap; output is 1,920 samples per latent frame, 504,960 samples total, with zero deferred flush samples. A second Mimi fixture decodes deterministic 263/257-frame rows in an uneven batch and individually; both sequences cross the 250-position context. Their measured batch-vs-single PCM maximum absolute difference is 2.56113708e-9; the fixed allowance is 2e-7. The short fixture repeats bitwise on the pinned CPU toolchain. The checked-in Q8 snapshot reconstructs all 209 published GGUF tensors (36 Q8_0 and 173 BF16), then expands them to HF-loadable F32 tensors, including Mimi BF16 weights; the locked `gguf` reader verifies 36 Q8_0 and 173 BF16 tensors. Its fixed 40-frame replay and natural-stop replay have exact same-oracle budgets, while measured Q8-vs-BF16 deviations are reported separately as quantization effects and are not used to widen tolerances. The derived cache is keyed by the pinned GGUF digest and converter version, validates all 209 tensor names/shapes/types and the cached output digest, and never consumes the former partial hybrid cache. Pre-native C comparison targets are frozen in `budgets.json` and separately copied into `q8-budgets.json`: component stages use rtol=1e-3/atol=1e-5, discrete IDs/noise/stop counts are exact, and PCM uses normalized RMSE≤1e-3/max absolute error≤2e-3. The binary32 γ estimate is a component-level rationale only, not a proof of end-to-end nonlinear error; targets do not change to fit later native outputs. `source-hashes.json` records the pinned upstream flow sampler and HF converter/decoder sources. Verification commands are read-only; only explicit `write*` commands rewrite fixtures/manifests. The pinned HF converter builds a versioned tokenizer with its reserved audio-marker IDs. The adapter confirms those IDs against model config before inference. Outputs describe the corrected reference run and must not be compared against uncorrected Rust output as though equivalent.

## Projected voice and CLI checks

The development exporter verifies the pinned assets, expands the published BF16 speaker projection from `model.q8.gguf`, and applies Torch `F.linear` to the pinned raw default voice. It writes `.build/phonon-cli-projected.safetensors` with only `model_ext=ba7a5a45@300` metadata and an adjacent JSON provenance record containing source/output digests, model identity, compatible voice identity, weight name/type and output shape. It consumes no native output and changes no frozen fixture or budget.

```sh
UV_CACHE_DIR=/private/tmp/speech-mac-uv-cache uv run --project reference/phonon --offline --no-sync --locked python reference/phonon/export_projected_voice.py
UV_CACHE_DIR=/private/tmp/speech-mac-uv-cache uv run --project reference/phonon --offline --no-sync --locked python -m unittest discover -s reference/phonon/tests -p test_projected_voice.py
PHONON_CLI_PACK=.build/phonon-native-f6acc8f43cb296be-v1.kpack \
PHONON_DEFAULT_VOICE="$PHONON_MODEL_DIR/default-voice.safetensors" \
PHONON_PROJECTED_VOICE="$PWD/.build/phonon-cli-projected.safetensors" \
make test-one TEST=tests/test_speech_cli.c
```

The CLI test compares positional/file/stdin output exactly, compares raw and independently projected voices at the fixed PCM budget with exact sample count, and checks native controls, compatible local catalog selection, completed-WAV playback through a stub, 0600 temporary permissions, and cleanup on success/playback/removal failure. It saves audition WAVs without playing sound. Listening is unperformed. Model-free argument, input, catalog, voice-schema and WAV checks also run through `make check`.

## Asset setup

Install `uv`, accept the gated model terms on Hugging Face, and authenticate locally with the Hugging Face CLI. From the repository root, choose a persistent asset directory, install the locked project, and download only the pinned snapshots:

```sh
export PHONON_MODEL_DIR="$HOME/speech-mac-models/phonon-40m-en"
uv sync --project reference/phonon --locked
uv run --project reference/phonon --locked python reference/phonon/download_assets.py
uv run --project reference/phonon --locked python reference/phonon/verify_assets.py
make phonon-native-pack
mkdir -p models voices
cp .build/phonon-native-f6acc8f43cb296be-v1.kpack models/phonon-40m-en.kpack
cp "$PHONON_MODEL_DIR/default-voice.safetensors" voices/default.safetensors
make build
.build/speech-mac say 'Hello.' --output hello.wav
```

Downloads are a development setup action; the CLI never downloads assets. `pins.json` selects repository, revision, filename, size and SHA-256 for every required artifact; verification runs before conversion. Keep this directory outside disposable worktrees. Ordinary builds require neither the source assets nor the Python environment. See [licenses](licenses.md) before using or redistributing model assets.

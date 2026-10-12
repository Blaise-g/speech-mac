# Kokoro native for macOS

Native C inference for Kokoro-82M v1.0, targeting Apple Silicon macOS. The initial runtime will use Accelerate. Python is reserved for reference checks and weight conversion; it will not be part of the shipped inference path.

## Prerequisites

- An Apple Silicon Mac running macOS with Xcode Command Line Tools (`clang`, `make`, and `xcrun`).
- Git.

`uv` manages the locked Python reference and conversion tools. eSpeak NG will be used by the English text frontend. Neither Python nor eSpeak NG is part of the native inference runtime.

## Development and checks

Run `make` to list the available commands. Use `make setup` once in a clone to install the repository-local pre-commit hook, and `make dev` to build `.build/kokoro`. Run `make format` before committing, then `make check` for the full local check sequence. The hook runs `make check-fast` against the staged snapshot.

The C test suite uses `tests/test_*.c`; run a focused test with `make test-one TEST=tests/test_name.c`. `make check` covers format check, lint, typecheck, build, and the full test suite.

## Prepared phoneme CLI

The initial CLI accepts phonemes from the pinned Kokoro v1.0 vocabulary:

```sh
.build/kokoro say --ipa 'həˈloʊ'
.build/kokoro say --ipa 'həˈloʊ' --voice am_adam --speed 1.25 --output hello.wav
.build/kokoro --list-voices
```

The default voice is `af_heart`, and the default speed is `1.0`. Speed must be between `0.5` and `2.0`. `--output` writes mono PCM16 WAV at 24 kHz. Without it, the CLI writes a completed WAV to a mode-0600 temporary file, plays it with macOS `afplay`, and removes the file afterward. `--help` and `--version` show the command options and version.

The native model pack is `models/kokoro-v1_0-af_heart.kpack`; it contains the model and the initial `af_heart` style. `am_adam` uses the separately pinned `models/voices/am_adam.kvoice` style artifact, so it does not require a second copy of the model weights. Both are ignored by Git. Download and convert the pinned official artifacts with:

```sh
UV_CACHE_DIR=/private/tmp/kokoro-uv-cache uv run --project reference --locked python reference/assets.py download
UV_CACHE_DIR=/private/tmp/kokoro-uv-cache uv run --project reference --locked python reference/convert.py
```

`--list-voices` prints only voices whose model and style artifacts are installed and compatible.

## Project guidance

- [Agent instructions](AGENTS.md) point to the coding standards, domain decisions, and local tracker.
- [Contributing](CONTRIBUTING.md) describes branch, review, commit, and verification rules.
- [Local tracker](.scratch/kokoro-native/README.md) defines the authoritative spec and ticket locations.

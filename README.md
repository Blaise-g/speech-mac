# Kokoro native for macOS

Native C inference for Kokoro-82M v1.0, targeting Apple Silicon macOS. The initial runtime will use Accelerate. Python is reserved for reference checks and weight conversion; it will not be part of the shipped inference path.

## Prerequisites

- An Apple Silicon Mac running macOS with Xcode Command Line Tools (`clang`, `make`, and `xcrun`).
- Git.

`uv` manages the locked Python reference and conversion tools. Install eSpeak NG for English text input:

```sh
brew install espeak-ng
```

The C program invokes eSpeak NG as a separate process; eSpeak NG and Python are not part of C model inference.

## Development and checks

Run `make` to list the available commands. Use `make setup` once in a clone to install the repository-local pre-commit hook, and `make dev` to build `.build/kokoro`. Run `make format` before committing, then `make check` for the full local check sequence. The hook runs `make check-fast` against the staged snapshot.

The C test suite uses `tests/test_*.c`; run a focused test with `make test-one TEST=tests/test_name.c`. `make check` covers format check, lint, typecheck, build, and the full test suite.

## Performance baseline

Run `make benchmark` on Apple Silicon to build both binaries with release `-O2` flags and write `artifacts/12-performance-baseline.md` plus the raw `artifacts/12-performance-baseline.json`. The benchmark repeats a short English reply, a paragraph, and the reply as prepared IPA using `af_heart`, speed `1.0`, and a fixed seed. Each workload gets five fresh-process sessions with one warm-up and five measured resident requests per session.

The report separates fresh process startup, model-pack open and SHA-256 validation, engine creation, frontend, native synthesis, full waveform availability, and WAV close. In these completed-WAV baseline runs, first audio means the complete waveform buffer is ready after all chunks are combined; the metric does not measure `--stream` playback. A new process does not imply cleared OS file pages, so cache state is recorded as uncontrolled and the benchmark does not purge caches. Audio throughput is generated audio seconds divided by synthesis seconds; RTF is the inverse.

## English and prepared phoneme CLI

The `say` command reads one source: positional text, a UTF-8 file, or standard input. Ordinary text is passed to eSpeak NG and then mapped through the pinned Kokoro v1.0 symbol vocabulary. `--ipa` skips eSpeak NG and validates the selected source directly against that vocabulary.

```sh
.build/kokoro say 'Hello from the terminal.'
.build/kokoro say -f notes.txt --output notes.wav
printf 'Hello from standard input.' | .build/kokoro say --output stdin.wav
.build/kokoro say --stream -f notes.txt
printf 'Hello from standard input.' | .build/kokoro say --stream
.build/kokoro say --ipa 'həˈloʊ'
.build/kokoro say --ipa 'həˈloʊ' --voice am_adam --speed 1.25 --output hello.wav
.build/kokoro --list-voices
```

`--stream` plays prepared sentence chunks through the default audio device. It accepts positional text, `-f`/`--file`, or standard input; streamed playback cannot be combined with `-o`/`--output`.

The English frontend was checked with eSpeak NG 1.52.0 using `-q --ipa=3 -v en-us` and its default rate and pitch. Stress and length marks map through Kokoro's pinned symbol vocabulary. eSpeak also emits U+200D joiners; the frontend strips that formatting marker because Kokoro has no corresponding token. When eSpeak marks a syllabic consonant with U+0329, the frontend represents it as a schwa before that consonant because Kokoro has no syllabic-consonant token. Other output symbols outside the vocabulary fail clearly. eSpeak's line breaks are kept as sentence boundaries, and an overlong line is split at a whitespace boundary so each model request stays within 510 phoneme symbols. Completed-WAV output combines chunk audio in order; `--stream` submits each synthesized chunk to bounded playback buffers while later chunks are synthesized. These pronunciations can differ from Misaki.

The default voice is `af_heart`, and the default speed is `1.0`. Speed must be between `0.5` and `2.0`. `--output` writes mono PCM16 WAV at 24 kHz. Without `--output` or `--stream`, the CLI writes a completed WAV to a mode-0600 temporary file, plays it with macOS `afplay`, and removes the file afterward. `--help` and `--version` show the command options and version.

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

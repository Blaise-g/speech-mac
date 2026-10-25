# speech-mac CLI (Phonon migration)

`make build` produces `.build/speech-mac` for native Phonon 40M English and `.build/kokoro` for separately checking the legacy implementation during migration. The Phonon command needs only local assets and native macOS libraries at runtime. See [Phonon reference setup](reference/phonon/README.md) for the pinned model and native pack conversion.

```sh
.build/speech-mac --help
.build/speech-mac --version
.build/speech-mac voices --voices-dir /path/to/voices
.build/speech-mac say 'Hello.' --model /path/to/model.kpack --voices-dir /path/to/voices --output hello.wav
.build/speech-mac say --file text.txt --model /path/to/model.kpack --voice-file /path/to/voice.safetensors --output file.wav
printf 'Hello.' | .build/speech-mac say --stdin --model /path/to/model.kpack --voice-file /path/to/voice.safetensors --output stdin.wav
```

The default pack path is `models/phonon-40m-en.kpack`; the default voice is `voices/default.safetensors`. Install named voices as `<name>.safetensors` in the selected `--voices-dir`; names contain ASCII letters, digits, hyphens or underscores. `voices` lists only artifacts that pass compatibility validation. The catalog is local: filenames select artifacts, and artifact metadata plus the pinned pack establish compatibility. There is no recording encoder, cloud enrollment, or runtime download.

Without `--output` or `--stream`, the CLI writes the completed mono 24 kHz PCM16 WAV to a private temporary file, plays it using macOS `afplay`, then removes it even if playback fails. With `--output`, it writes the WAV without playing it. A single positional argument supplies text (`--` ends option parsing, so `say -- '--ipa'` supplies literal text); `--file PATH` reads UTF-8; `--stdin` or omitted text reads standard input. Inputs are limited to 1 MiB before and after normalization. Conflicting or duplicate inputs/options fail.

Supported controls are `--language en`, `--seed` (decimal 0 through 18446744073709551615, default 0), and finite `--duration-delta` (default 0), the model's native duration conditioning. `--speed`, `-s`, `--ipa`, and Kokoro voice names are unsupported; speed values are never converted.

`--stream` sends each 1,920-sample mono 24 kHz frame to AudioQueue while later frames are still being generated. Playback starts after the first submission. Three fixed playback buffers provide ordered copies and backpressure; no whole-utterance PCM queue is retained. `--stream` accepts the same text, file and stdin inputs, but conflicts with `--output`. SIGINT or SIGTERM cancels generation, wakes blocked playback, then joins and disposes the request before releasing the engine and voice. A running native numerical call may finish before cancellation is polled.

```sh
.build/speech-mac say 'Hello.' --stream --model /path/to/model.kpack --voice-file /path/to/voice.safetensors
```

First PCM readiness, buffer submission, successful running-state observation, buffer reclamation, synthesis completion and successful drain are separate monotonic events. Fake-consumer timings do not measure device or acoustic onset. Shared numerical operations still allocate bounded temporary buffers; streaming provides no allocation-free or real-time guarantee.

The opt-in functional device test mutes only its new AudioQueue to volume zero before submission and start. It never changes system volume. A test-only watchdog fails the process if the SDK probe does not finish within 30 seconds. Run it from the repository root with a converted pack and compatible voice:

```sh
PHONON_NATIVE_PACK=/path/to/model.kpack \
PHONON_DEFAULT_VOICE=/path/to/voice.safetensors \
PHONON_STREAM_DEVICE=1 \
make test-one TEST=tests/test_speech_stream.c \
  CPPFLAGS='-Isrc -DKOKORO_STREAM_SILENT_OUTPUT' BUILD_DIR=.build/stream-device
```

On the development host on 2026-10-25, real 40M batch/stream PCM equality and fake-device lifecycle checks passed, but native device acceptance remains blocked: `AudioQueueStart` returned `-66681` (`kAudioQueueErr_CannotStart`). The preserved silence-only adapter also failed. Standalone queue-muted probes at 24 kHz mono and 48 kHz stereo, with larger buffers, priming and main/worker starts, timed out inside the SDK start call. Successful device start, reclamation and drain still need verification in a working audio session. Acoustic onset and listening acceptance remain unmeasured.

Compatible voice schemas are finite F32 `speaker_wavs` shaped `[512,T]` or `[1,512,T]`, with raw metadata `sig=ba7a5a45`, `epoch=300`; or already projected `emb`/`audio_prompt` shaped `[T,512]` or `[1,T,512]`, with `model_ext=ba7a5a45@300`. Here `T` is 1 through 4096. Raw voices are projected exactly once; projected embeddings are used directly. An explicit conflicting signature, epoch or model extension fails. Unbound projected files are rejected; obtain an artifact with verifiable compatibility metadata. The compatible voice identity is the checkpoint's `embeddings_ext`, distinct from its full model identity `8aa279a6@100`. Arbitrary tensor names and multiple voice tensors are rejected.

Automated playback checks use a callback stub and inspect the completed WAV and temporary permissions/cleanup. Real-model CLI checks are opt-in through `PHONON_CLI_PACK`, `PHONON_DEFAULT_VOICE` and `PHONON_PROJECTED_VOICE`; they save audition WAVs but do not play sound. Listening acceptance remains separate from numerical validation.

The following sections describe the legacy implementation pending its removal.

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

Run `make stream-benchmark` on Apple Silicon to build the release `-O2` streaming helper and write a dated report with its raw measurements under `artifacts/`. Each benchmark AudioQueue is muted to volume `0.0` before PCM is submitted; the command does not change system output volume. The report measures streaming events from CLI call entry and keeps buffer reclamation separate from acoustic onset, which is unavailable without independent render evidence. See the [captured streaming report](artifacts/18-streaming-benchmark-20261012T161920Z/report.md); its profile-guided candidate failed the fixed parity budget, so the optimization criterion remains open.

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

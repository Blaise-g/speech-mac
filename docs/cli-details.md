# CLI and development details

`make build` produces `.build/speech-mac` for native Phonon 40M English on Apple Silicon macOS. The Phonon command needs only local assets and native macOS libraries at runtime. See [Phonon reference setup](../reference/phonon/README.md) for the pinned model and native pack conversion.

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
  CPPFLAGS='-Isrc -DSPEECH_STREAM_SILENT_OUTPUT' BUILD_DIR=.build/stream-device
```

On 2026-10-26 the queue-muted functional test passed from the user's Terminal: actual start, reclamation and drain of 30,720 samples. Earlier tool-context probes failed at `AudioQueueStart` or timed out; their cause remains unknown. Acoustic onset and listening acceptance remain unmeasured.

Compatible voice schemas are finite F32 `speaker_wavs` shaped `[512,T]` or `[1,512,T]`, with raw metadata `sig=ba7a5a45`, `epoch=300`; or already projected `emb`/`audio_prompt` shaped `[T,512]` or `[1,T,512]`, with `model_ext=ba7a5a45@300`. Here `T` is 1 through 4096. Raw voices are projected exactly once; projected embeddings are used directly. An explicit conflicting signature, epoch or model extension fails. Unbound projected files are rejected; obtain an artifact with verifiable compatibility metadata. The compatible voice identity is the checkpoint's `embeddings_ext`, distinct from its full model identity `8aa279a6@100`. Arbitrary tensor names and multiple voice tensors are rejected.

Automated playback checks use a callback stub and inspect the completed WAV and temporary permissions/cleanup. Real-model CLI checks are opt-in through `PHONON_CLI_PACK`, `PHONON_DEFAULT_VOICE` and `PHONON_PROJECTED_VOICE`; they save audition WAVs but do not play sound. Listening acceptance remains separate from numerical validation.


## Build and development

Use an Apple Silicon Mac, Xcode Command Line Tools (`clang`, `make`, `xcrun`), Git and `uv` for checks and development tooling. Runtime inference and English normalization are native C17 using Accelerate/NEON; they launch no Python, Rust or eSpeak process. Python and the pinned Rust normalizer are development references only.

```sh
make setup
make build
make format
make check
make test-one TEST=tests/test_speech_cli.c
```

`uv` manages the locked development project at `reference/phonon`. Follow its [asset setup](../reference/phonon/README.md#asset-setup) to download and verify the pinned gated model, create the native pack and install the default voice. Tests report explicit skips when optional real-model fixture paths are absent; a skip is not 40M validation. See [engine checks](phonon-engine.md) for actual-model fixture commands.

Historical numbered measurements in `artifacts/` describe their original Kokoro runs. The old runtime, oracle, converters, dependencies and active benchmark runners have been removed. New Phonon measurements belong to tickets 32 and 33. The installed external `/opt/homebrew/bin/kokoro` is outside this repository's changes. Local consumers were CLI, streaming, benchmarks and tests; no additional nearby or open-PR client was found. Undiscoverable private callers have no compatibility guarantee. The repository directory and historical tracker path remain unchanged. The GitHub repository is now `Blaise-g/speech-mac`; see the [remote setup record](remote-setup.md).

See [contributing](../CONTRIBUTING.md), [agent instructions](../AGENTS.md) and the [local tracker](../.scratch/kokoro-native/README.md).

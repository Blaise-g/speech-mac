# speech-mac

speech-mac is a native C17 implementation of Phonon 40M English text-to-speech for Apple Silicon macOS. The CLI accepts text, UTF-8 files or standard input, saves WAV files, and plays completed or streaming speech. Inference uses Accelerate/NEON and macOS system libraries; English text normalization also runs in C. Once set up, the CLI runs offline without Python, Rust or eSpeak.

## Quick start

You need an Apple Silicon Mac, Git, Xcode Command Line Tools and [uv](https://docs.astral.sh/uv/getting-started/installation/). Install the command line tools if needed:

```sh
xcode-select --install
```

Clone the repository and enter it:

```sh
git clone https://github.com/Blaise-g/speech-mac.git
cd speech-mac
```

The [Phonon 40M English model](https://huggingface.co/gradium/phonon-40m-en) is gated and licensed CC BY-NC 4.0. Read and accept its terms on Hugging Face, then authenticate locally with an account that has access:

```sh
uvx --from huggingface-hub hf auth login
make bootstrap
```

`make bootstrap` checks for Darwin/arm64, uv and the Xcode compiler before setup. It installs the locked Python setup environment, downloads and verifies all twelve pinned source assets, converts the native pack, installs the default voice, and builds `.build/speech-mac`. It does not play sound. Python is used for setup and development references, not inference.

See [setup details](docs/cli-details.md#setup-details) for offline setup, custom asset locations and destination conflicts.

## Usage

Save a mono 24 kHz PCM16 WAV without playback, then play it when ready:

```sh
.build/speech-mac say 'Hello there. I am ready to help you today.' --seed 0 --output hello.wav
afplay hello.wav
```

To hear speech as frames are generated, run the streaming command. This command plays audio:

```sh
.build/speech-mac say 'Hello there. I am ready to help you today.' --stream
```

Without `--stream` or `--output`, the CLI generates the complete WAV, plays it with `afplay`, then removes its private temporary file. `--stream` and `--output` cannot be combined.

```sh
.build/speech-mac say 'Hello.'
.build/speech-mac say --file text.txt --output file.wav
printf 'Hello.' | .build/speech-mac say --stdin --output stdin.wav
.build/speech-mac --help
.build/speech-mac --version
```

Omitting text also reads stdin. Text is limited to 1 MiB before and after normalization. Use one input source per command. `--` ends option parsing, for example `say -- '--literal-text'` reads that literal text.

## Voices and controls

The default voice comes from the pinned model revision. Compatible local voices can be selected by filename or installed in a local catalog:

```sh
.build/speech-mac voices
.build/speech-mac say 'Hello.' --voice-file '/path/to/voice.safetensors' --output local.wav
# After placing a compatible narrator.safetensors in voices/:
.build/speech-mac say 'Hello.' --voice narrator --output narrator.wav
.build/speech-mac voices --voices-dir '/path/to/voices'
```

`--voices-dir` selects the catalog directory; names use ASCII letters, digits, hyphens or underscores. The catalog lists only compatible artifacts. No cloud enrollment, recording encoder or runtime download is included. See [voice schemas and compatibility](docs/cli-details.md) before supplying a voice artifact.

Supported controls are `--language en`, `--seed` (decimal 0 through 18446744073709551615, default 0), and finite `--duration-delta` (default 0). Duration delta is the model's native duration conditioning. Only English and the controls listed here are supported.

```sh
.build/speech-mac say 'Hello.' --language en --seed 42 --duration-delta 0.1 --output controlled.wav
```

Use `--model /path/to/model.kpack` for a pack outside the default location. Run commands from the repository root when using the default relative model and voice paths.

## Build and design

```sh
make build
make format
make check
```

The implementation follows Salvatore Sanfilippo (antirez)'s [qwen-asr](https://github.com/antirez/qwen-asr) and [voxtral.c](https://github.com/antirez/voxtral.c) as architectural references for readable modular C, explicit weight and numerical-operation boundaries, and narrow platform interfaces. These references inform the C structure; Phonon supplies the TTS model.

The resident engine keeps model weights loaded across requests. The text frontend, Flow transformer, Mimi decoder, voice loader and playback code are separate C modules. The runtime uses Accelerate/NEON on the CPU and AudioQueue for progressive playback. Python and the pinned Rust normalizer are setup or development tools. Numerical code includes pinned SLEEF operations and adaptations from PyTorch; their provenance and licenses remain in [model and reference licenses](reference/phonon/licenses.md), [operator provenance](reference/phonon/operator-provenance.json), and [third-party notices](LICENSES/). See [coding standards](CODING_STANDARDS.md), the [runtime decision](docs/adr/0002-phonon-native-runtime.md) and [engine checks](docs/phonon-engine.md).

## Performance and limits

Streaming submits 1,920-sample frames through three fixed AudioQueue buffers with backpressure. It does not retain a whole-utterance PCM queue. Ctrl-C cancels generation and disposes playback; a running native numerical call may finish before cancellation is polled. Requests still allocate bounded temporary buffers. There is no allocation-free or real-time guarantee.

The [same-host M5 measurements](artifacts/33-linear-materialization-20261026/report.md) found resident synthesis speedups of 1.45×, 1.44× and 1.47× across the three measured workloads after preparing linear weights. This retains an extra 169.47 MiB per engine and makes fresh preparation-to-first-PCM 22 to 27 ms slower. One fresh structured pair was effectively tied. These are observations on a 16 GiB M5 running macOS 27.0.1, not predictions for every Mac. The report includes real-time factors (synthesis time divided by audio duration), cold preparation costs and process RSS high-water memory. Fresh processes did not purge OS caches.

First PCM readiness, submission, observed device running and acoustic onset are different events. Muted AudioQueue tests have passed; listening acceptance and acoustic onset remain unmeasured. See [CLI details](docs/cli-details.md) for playback ownership, voice schemas and the opt-in muted device test, and [engine checks](docs/phonon-engine.md) for numerical validation.

## Troubleshooting

- If bootstrap reports missing tools, install uv and Xcode Command Line Tools, then check `uv --version` and `xcrun --find clang`. Intel Macs and other operating systems are unsupported.
- If a download fails, check network access, accept the gated model terms and rerun `uvx --from huggingface-hub hf auth login`. Keep credentials local; do not put tokens in commands or repository files. Valid local assets can be reused without authentication.
- If verification fails, check `PHONON_MODEL_DIR`. Bootstrap attempts to recover missing or corrupt files from the pinned snapshots; it never trusts a private local verification manifest.
- If a destination conflicts, preserve or move that custom pack or voice, then retry. Paths containing spaces must be quoted. Select custom runtime paths with `--model`, `--voices-dir` or `--voice-file`.
- If there is no audible output, first save a WAV with `--output` and play it explicitly with `afplay`. `--output` intentionally does not play audio. Device failures do not invalidate a successfully saved WAV.

## Development and licenses

Run `make format` then `make check` before committing. CI runs model-free checks on `macos-15`; optional real-model tests report explicit skips when their ignored assets are absent. A skip is not 40M validation. See [contributing](CONTRIBUTING.md), [reference setup](reference/phonon/README.md), [CLI and development details](docs/cli-details.md), and the [local tracker](.scratch/kokoro-native/README.md).

There is currently no top-level license granting reuse of this repository's code. Public visibility does not grant a code license or remove model restrictions. Phonon 40M weights, tokenizer and supplied voice are CC BY-NC 4.0 and require the model's gated terms. Mimi weights and upstream source components have their own licenses. See [model and reference licenses](reference/phonon/licenses.md) and [retained third-party licenses](LICENSES/) for provenance and individual terms.

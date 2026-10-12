# 17: Add opt-in native streaming playback

## Context

Technical source: `.scratch/kokoro-native/spec.md` (Authorized streaming implementation), the [approved streaming design](../../../docs/streaming-plan.md), and [ticket 16](16-streaming-design.md).

Background: [Kokoro native inference spec](../spec.md). Existing engine and frontend behavior is defined in [ticket 10](10-resident-engine-phoneme-cli.md) and [ticket 11](11-english-espeak-frontend.md).

Worker Profile: implement-lite · xhigh · gpt-6-luna (override)

## What to build

An explicit `kokoro say --stream` request starts macOS playback after its first complete synthesized input chunk while later chunks continue through the resident engine. Completed-WAV playback remains the default.

## Acceptance criteria

- [ ] Preserve positional text, UTF-8 file, stdin, `--ipa`, both supported voices, and the existing finite speed range. Read and validate all input and prepare the existing sentence/context-bounded chunks before synthesis. Preserve completed-WAV playback and `-o/--output`; reject their combination with `--stream` clearly.
- [ ] Synthesize whole chunks through the existing engine in input order, preserving configured seed plus chunk index (default seed `104729`), style row `token_count - 3`, mono 24 kHz float samples, and all fixed numerical budgets. Playback of the first chunk begins while later chunks are still being synthesized; no within-request incremental decoding is added.
- [ ] Keep macOS playback in a narrow AudioQueue adapter. Bound queued audio to three 4096-frame buffers plus at most one complete current model-chunk waveform; make the producer wait for buffer reuse instead of accumulating the whole request. The output callback only makes buffers reusable and signals progress; it does not infer, allocate, or wait.
- [ ] Return normal completion only after graceful queue drain reports the queue stopped. Cancellation and errors stop playback promptly, retain the first terminal result, and join the producer before borrowed engine or chunk data is released. Append `KOKORO_CANCELLED` without changing existing status values. Handle Ctrl-C through signal-safe notification and ordinary control flow, without locks, allocation, or AudioQueue calls in the signal handler.
- [ ] Keep a fake playback consumer that consumes and reclaims first-chunk PCM while final-chunk synthesis is deliberately blocked; a submission or start request alone does not prove overlap. Cover batch-versus-stream float parity, sample completeness/order, full-buffer backpressure and resume, cancellation during synthesis/wait/drain, startup/enqueue/drain/synthesis errors and cleanup, repeated engine requests, and CLI option/error behavior.
- [ ] Separate deterministic fake-consumer tests from device and listening checks. Device lifecycle checks may use silence; do not run an unintended audible speaker benchmark.

## Tests

- Add defect-revealing tests at the engine/producer and playback-adapter seams for every acceptance behavior. Compare stream output to the existing batch result with identical chunks and options before WAV quantization.
- Run `make format`, then `make check`, and obtain independent review before committing.

## Blocked by

- `16-streaming-design` — verified design and interface/ownership decisions.

**Status:** ready-for-agent

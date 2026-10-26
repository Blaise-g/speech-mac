# 17: Add opt-in native streaming playback

## Context

Technical source: `.scratch/kokoro-native/spec.md` (Authorized streaming implementation), the [approved streaming design](https://github.com/Blaise-g/speech-mac/blob/3434f73aeada7e1818feb11cf4815b773a21188b/docs/streaming-plan.md), and [ticket 16](16-streaming-design.md).

Background: [Kokoro native inference spec](../spec.md). Existing engine and frontend behavior is defined in [ticket 10](10-resident-engine-phoneme-cli.md) and [ticket 11](11-english-espeak-frontend.md).

Worker Profile: implement-lite · xhigh · gpt-6-luna (override)

## What to build

An explicit `kokoro say --stream` request starts macOS playback after its first complete synthesized input chunk while later chunks continue through the resident engine. Completed-WAV playback remains the default.

## Acceptance criteria

- [x] Preserve positional text, UTF-8 file, stdin, `--ipa`, both supported voices, and the existing finite speed range. Read and validate all input and prepare the existing sentence/context-bounded chunks before synthesis. Preserve completed-WAV playback and `-o/--output`; reject their combination with `--stream` clearly.
- [x] Synthesize whole chunks through the existing engine in input order, preserving configured seed plus chunk index (default seed `104729`), style row `token_count - 3`, mono 24 kHz float samples, and all fixed numerical budgets. Playback of the first chunk begins while later chunks are still being synthesized; no within-request incremental decoding is added.
- [x] Keep macOS playback in a narrow AudioQueue adapter. Bound queued audio to three 4096-frame buffers plus at most one complete current model-chunk waveform; make the producer wait for buffer reuse instead of accumulating the whole request. The output callback only makes buffers reusable and signals progress; it does not infer, allocate, or wait.
- [x] Return normal completion only after graceful queue drain reports the queue stopped. Cancellation and errors stop playback promptly, retain the first terminal result, and join the producer before borrowed engine or chunk data is released. Append `KOKORO_CANCELLED` without changing existing status values. Handle Ctrl-C through signal-safe notification and ordinary control flow, without locks, allocation, or AudioQueue calls in the signal handler.
- [x] Keep a fake playback consumer that consumes and reclaims first-chunk PCM while final-chunk synthesis is deliberately blocked; a submission or start request alone does not prove overlap. Cover batch-versus-stream float parity, sample completeness/order, full-buffer backpressure and resume, cancellation during synthesis/wait/drain, startup/enqueue/drain/synthesis errors and cleanup, repeated engine requests, and CLI option/error behavior.
- [x] Separate deterministic fake-consumer tests from device and listening checks. Device lifecycle checks may use silence; do not run an unintended audible speaker benchmark.

## Tests

- Add defect-revealing tests at the engine/producer and playback-adapter seams for every acceptance behavior. Compare stream output to the existing batch result with identical chunks and options before WAV quantization.
- Run `make format`, then `make check`, and obtain independent review before committing.

## Blocked by

- `16-streaming-design` — verified design and interface/ownership decisions.

**Status:** done

## Verification — 2026-10-12

Coordinator verified commit `7111a9571b894dfc6ec7bfae6a877aeca97263d2`, its `Ticket: 17-native-streaming-playback` trailer, and the clean worktree. All acceptance criteria above are met. The producer preserves engine/chunk ownership, order, seed/style/speed and float samples; the narrow AudioQueue adapter bounds queued PCM to three 4096-frame buffers. Normal completion observes graceful stop; cancellation stops playback independently of active synthesis and joins before borrowed data is released.

Worker `17_native_streaming_playback` used `implement-lite · xhigh · gpt-6-luna (override)`. `make format`, full `make check` with real model assets, focused stream and CLI tests, and whitespace checks passed. Coordinator independently reran `make test-one TEST=tests/test_stream.c` on the committed tree: passed, including real-model parity. The normal suite explicitly skips opt-in device playback. The separate final `KOKORO_AUDIO_QUEUE_SMOKE=1 make test-one TEST=tests/test_stream.c` passed with CoreAudio access and 8192 zero samples; no audible speech benchmark or human listening acceptance is claimed. The commit hook passed with model-backed cases explicitly skipped in its asset-free staged snapshot; the separate full working-tree check exercised them.

The same independent reviewer examined all 15 changed paths with Correctness, Standards, Spec, Reuse, Quality, and Efficiency. No outstanding findings remained. Coordinator-assigned review record identifiers below group the reported fixes; verification refers to the committed source and focused/full checks above.

| Id | Lens | Finding | Disposition | Verified by |
| --- | --- | --- | --- | --- |
| 17-R1 | Correctness | SIGINT initialization could clear an interrupt after handler installation | Applied: reset before installing handler | `7111a95`; CLI checks and static review |
| 17-R2 | Correctness | Callback mutex could wait | Applied: lock-free atomic state and precreated semaphore notification | `7111a95`; stream tests and device silence smoke |
| 17-R3 | Correctness | Overlap assertion proved consumption without callback reclamation | Applied: require 9000 consumed samples and three reclaimed buffers while final synthesis is blocked | `7111a95`; stream test |
| 17-C1 | Correctness | Latched SIGINT caused repeated cancellation and a wait-loop spin | Applied: signal-triggered cancellation once, then ordinary wait | `7111a95`; blocked-synthesis latched-signal test |
| 17-R4 | Quality | Unused start state and redundant terminal expression | Applied: removed both | `7111a95`; static review and full check |
| 17-R5 | Quality | Usage and design status needed to reflect implemented playback | Applied: minimal README examples and timing-pending status | `7111a95`; independent doc review |
| 17-R6 | Efficiency | Callback notifications could accumulate stale wake tokens | Applied: coalesced notification and predicate recheck | `7111a95`; backpressure, drain and synchronous-callback tests |

Coordinator questions also settled prompt cancellation during active inference, serialization of cancellation against enqueue/start, and reference protection in the wait-error cleanup path. Regression checks cover these paths. Timing and any measured inference optimization belong to ticket 18.

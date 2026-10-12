# 16: Define a minimal native streaming plan

## Context

Technical source: `.scratch/kokoro-native/spec.md` (User follow-up — CoreML comparison and streaming plan), [ticket 15 comparison](15-coreml-comparison-baseline.md), [ticket 11 frontend](11-english-espeak-frontend.md), and the follow-up extension in the [approved Run Plan](../run-plan.md).

Background: [Kokoro native inference spec](../spec.md).

Worker Profile: implement-lite · xhigh · gpt-6-luna (override)

## What to build

A concise, benchmark-informed design document for adding chunk-level playback to the native CLI. This ticket delivers the design only; it does not add streaming code.

## Acceptance criteria

- [x] Use ticket 15's matched short and long workload results to explain the latency/resource tradeoff and choose the smallest first streaming slice. Define streaming as playback beginning after the first synthesized chunk while later chunks are still being produced; distinguish this from incremental synthesis inside one model request.
- [x] Specify a small C-facing interface for producing ordered mono 24 kHz audio chunks from the existing resident engine and English frontend. Define chunk ownership/lifetime, end-of-stream and error behavior, and how existing `af_heart`/`am_adam`, speed, IPA, and text-input options apply.
- [x] Define chunk boundaries using the existing sentence-aware frontend limits, the playback backend and startup behavior, and bounded buffering. State where producer backpressure waits, how cancellation reaches active synthesis and playback, and how cleanup handles queued audio, output errors, and WAV output if that mode is included.
- [x] Define first-audio timing precisely: process start through first synthesized chunk, first buffer submitted to the playback API, and first rendered buffer where observable. Separate API submission from acoustic onset, and give a benchmark-derived target or explain why the available measurements cannot yet set one.
- [x] List defect-revealing future tests for chunk order/no loss, batch-versus-stream output equivalence at the existing numerical budget, bounded-buffer backpressure, cancellation while producing or waiting, first playback before the final chunk completes, error cleanup, and repeat requests through one resident engine.
- [x] Separate the minimal first delivery from later work. Streaming does not imply a daemon, Metal backend, quantization, wider language support, or within-request model decoding unless a later measured proposal receives its own scope.
- [x] Keep the current completed-WAV default behavior and simple CLI requirements intact in the proposed migration; make any streaming opt-in behavior explicit.

## Blocked by

- `15-coreml-comparison-baseline` — matched workload measurements and documented resource limits.

**Status:** done

## Dependency verification — 2026-10-12

Ticket15 is verified in `fc8f34e5ea5651ea0727a0f27e4f1ab33b6e0ffd`; design-only work is unblocked.

## Verification — 2026-10-12

Commit `715a19e6bab1d693b70d3a07e15af49df4474f03`; coordinator verified clean tree, Ticket trailer and the complete plan/standards change. `make format`, Markdown/link/symbol checks, whitespace checks and staged-snapshot `make check-fast` passed. Model-backed hook cases were explicitly skipped because ignored assets are absent in the snapshot; no executable code changed. Coordinator's preceding full model-backed working-tree check passed for the unchanged runtime.

`docs/streaming-plan.md` defines opt-in sentence-level AudioQueue playback, existing engine/input/chunk reuse, seed/style/speed float-output parity, ownership, three fixed buffers, producer backpressure, graceful drain, distinct cancellation and race behavior. It distinguishes waveform availability, submission, start request and buffer reclamation; no callback proves acoustic onset. A deterministic fake consumer must receive first PCM while final-chunk synthesis is blocked. Single-chunk hello cannot demonstrate overlap. Initial scope rejects `--stream -o` and defers incremental input, daemon and new inference backends. Apple Silicon optimization follows measured streaming parity; no numeric TTFA promise is invented.

All six review lenses completed. Three design findings about drain, cancellation semantics and callback timing were applied by static contract review; no remaining findings. The persistent coding standard references antirez qwen-asr/voxtral.c and maps their modular C structure to Kokoro. Streaming implementation/runtime acceptance remain future work by requested design-only scope.

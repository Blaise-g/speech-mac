# 09: Verify complete direct C inference

## Context

Technical source: `.scratch/kokoro-native/spec.md`, official [Kokoro inference path](https://github.com/hexgrad/kokoro/blob/main/kokoro/model.py), [model modules](https://github.com/hexgrad/kokoro/blob/main/kokoro/modules.py), [ISTFTNet](https://github.com/hexgrad/kokoro/blob/main/kokoro/istftnet.py), and [approved Run Plan](../run-plan.md).

Background: [Kokoro native inference spec](../spec.md).

Worker Profile: implement-lite · xhigh · gpt-6-luna (override)

## What to build

Prepared phonemes, one voice style, and speed synthesize complete Kokoro audio through the assembled native pipeline.

## Acceptance criteria

- [x] The assembled inference path runs directly in C on Apple Silicon and matches the pinned official reference within ticket 01's pre-established numerical budgets.
- [x] Produce and retain a representative WAV audition artifact with recorded model, voice, and input provenance.
- [x] The native runtime inference path does not require Python.

## Blocked by

- `08-istft-waveform` — all inference stages through waveform reconstruction.

**Status:** done

## Verification — 2026-10-12

Commit `0e654ce9a2e2711ddcb726b3d9700921e2a8c47c`. Coordinator verified clean tree/trailer and reran assembled inference focused test: controlled-conditioning waveform NRMSE 1.67710831e-05/maxabs 8.2552433e-06 within fixed limits; ordinary native synthesis passes finite/shape/same-seed checks. Controlled source-STFT injection isolates reference FFT phase branch, not free-running sample parity. Worker full real-pack checks and independent review passed after error/lifetime/input/test fixes. Artifacts/provenance retained in artifacts/. User listened to native af_heart hello WAV and said "sounds clear"; final longer/optimized listening remains14. Per-call stage construction deferred10.

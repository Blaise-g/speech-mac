# 04: Predict aligned pitch and noise conditioning

## Context

Technical source: `.scratch/kokoro-native/spec.md`, official [Kokoro model modules](https://github.com/hexgrad/kokoro/blob/main/kokoro/modules.py), [inference path](https://github.com/hexgrad/kokoro/blob/main/kokoro/model.py), and [approved Run Plan](../run-plan.md).

Background: [Kokoro native inference spec](../spec.md).

Worker Profile: implement-lite · xhigh · gpt-6-luna (override)

## What to build

Duration-aligned prosody features and predictor style produce the F0 and noise sequences needed by the decoder.

## Acceptance criteria

- [x] Implement the v1.0 shared recurrent path, style-conditioned F0/noise blocks, and output projections.
- [x] Returned sequences align in length with duration-expanded frames and contain finite values.
- [x] F0/noise outputs match reference fixtures within ticket 01's pre-established budget; a miss prompts investigation rather than automatic tolerance widening.

## Blocked by

- `03-duration-alignment` — aligned prosody features.

**Status:** done

## Verification — 2026-10-12

Commit `0be4871b9bb951f66b66f28fcc58bc581cd8d364`. Coordinator verified clean tree/Ticket trailer and reran `make check` with real model weights: all prior native checks and F0/noise parity pass fixed budgets (max abs 6.10351562e-05 / 6.19888306e-06). Short/long inputs preserve official 2x temporal rate and finite output; malformed/non-finite inputs reject. Worker locked uv reference suite: 17 passed. Independent review fixed output reuse leak, lifetime docs, tensor binder duplication, convolution repacking, and test lifetime; no remaining findings.

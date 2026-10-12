# 06: Prepare style-conditioned decoder features

## Context

Technical source: `.scratch/kokoro-native/spec.md`, official [ISTFTNet decoder](https://github.com/hexgrad/kokoro/blob/main/kokoro/istftnet.py), [inference path](https://github.com/hexgrad/kokoro/blob/main/kokoro/model.py), and [approved Run Plan](../run-plan.md).

Background: [Kokoro native inference spec](../spec.md).

Worker Profile: implement-lite · xhigh · gpt-6-luna (override)

## What to build

Aligned text features, F0/noise sequences, and decoder style become the conditioned feature sequence consumed by Kokoro's waveform generator.

## Acceptance criteria

- [x] Implement the pre-generator ISTFTNet conditioning and residual blocks with the v1.0 channel and time shapes and style split.
- [x] Mismatched conditioning lengths return a clear error instead of partial output.
- [x] Pre-generator features match the reference fixture within ticket 01's pre-established budget; a miss prompts investigation rather than automatic tolerance widening.

## Blocked by

- `04-pitch-noise-prediction` — pitch and noise conditioning.
- `05-phoneme-acoustic-encoder` — aligned text features.

**Status:** done

## Verification — 2026-10-12

Commit `cb1a2847aa4d0303dd8fc693404ca258c33f565e`. Coordinator verified clean tree/trailer and native decoder focused test with real pack: mismatched inputs reject, one frame returns two finite frames, decoder fixture parity passed max abs 7.78436661e-05 fixed budgets. Worker full make check/locked uv suite (17) passed. Independent review removed redundant predicates; shared private conv/AdaIN/residual helpers deferred to final branch review. No open06 acceptance.

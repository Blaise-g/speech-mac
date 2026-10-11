# 03: Predict token durations and expand the alignment

## Context

Technical source: `.scratch/kokoro-native/spec.md`, official [Kokoro inference path](https://github.com/hexgrad/kokoro/blob/main/kokoro/model.py), [model modules](https://github.com/hexgrad/kokoro/blob/main/kokoro/modules.py), and [approved Run Plan](../run-plan.md).

Background: [Kokoro native inference spec](../spec.md).

Worker Profile: implement-lite · xhigh · gpt-6-luna (override)

## What to build

Contextual phoneme features and predictor voice style produce per-token durations and the frame-to-token alignment used by later acoustic stages.

## Acceptance criteria

- [x] Implement the v1.0 duration encoder, recurrent projection, speed scaling, rounding, and minimum-duration clamp.
- [x] Expand durations into a monotonic alignment where each output frame maps to exactly one input token.
- [x] Durations are positive and the alignment's output length equals the sum of durations.
- [x] Predictor features, durations, and alignment match reference fixtures within the budget established in ticket 01.

## Blocked by

- `02-phoneme-context-encoder` — contextual phoneme features.

**Status:** done

## Verification — 2026-10-11

Commit `51661e817ec17575099e800e25d2495b70dbbcda`. Coordinator verified clean tree/Ticket trailer and reran `make check` with actual model weights: native contextual/duration stage parity passed fixed budgets; durations/alignments match exactly, including 0.5 and 2.0 speed endpoints; one-token and invalid-input checks passed. Worker locked uv suite: 17 tests passed. Independent review applied helper simplification, reused captured logits for endpoint fixtures, and added token-length coverage. No open ticket blockers.

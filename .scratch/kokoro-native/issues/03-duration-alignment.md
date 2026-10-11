# 03: Predict token durations and expand the alignment

## Context

Technical source: `.scratch/kokoro-native/spec.md`, official [Kokoro inference path](https://github.com/hexgrad/kokoro/blob/main/kokoro/model.py), [model modules](https://github.com/hexgrad/kokoro/blob/main/kokoro/modules.py), and [approved Run Plan](../run-plan.md).

Background: [Kokoro native inference spec](../spec.md).

Worker Profile: implement-lite · xhigh · gpt-6-luna (override)

## What to build

Contextual phoneme features and predictor voice style produce per-token durations and the frame-to-token alignment used by later acoustic stages.

## Acceptance criteria

- [ ] Implement the v1.0 duration encoder, recurrent projection, speed scaling, rounding, and minimum-duration clamp.
- [ ] Expand durations into a monotonic alignment where each output frame maps to exactly one input token.
- [ ] Durations are positive and the alignment's output length equals the sum of durations.
- [ ] Predictor features, durations, and alignment match reference fixtures within the budget established in ticket 01.

## Blocked by

- `02-phoneme-context-encoder` — contextual phoneme features.

**Status:** ready-for-agent

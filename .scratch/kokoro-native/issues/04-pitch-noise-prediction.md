# 04: Predict aligned pitch and noise conditioning

## Context

Technical source: `.scratch/kokoro-native/spec.md`, official [Kokoro model modules](https://github.com/hexgrad/kokoro/blob/main/kokoro/modules.py), [inference path](https://github.com/hexgrad/kokoro/blob/main/kokoro/model.py), and [approved Run Plan](../run-plan.md).

Background: [Kokoro native inference spec](../spec.md).

Worker Profile: implement-lite · xhigh · gpt-6-luna (override)

## What to build

Duration-aligned prosody features and predictor style produce the F0 and noise sequences needed by the decoder.

## Acceptance criteria

- [ ] Implement the v1.0 shared recurrent path, style-conditioned F0/noise blocks, and output projections.
- [ ] Returned sequences align in length with duration-expanded frames and contain finite values.
- [ ] F0/noise outputs match reference fixtures within ticket 01's pre-established budget; a miss prompts investigation rather than automatic tolerance widening.

## Blocked by

- `03-duration-alignment` — aligned prosody features.

**Status:** ready-for-agent

# 06: Prepare style-conditioned decoder features

## Context

Technical source: `.scratch/kokoro-native/spec.md`, official [ISTFTNet decoder](https://github.com/hexgrad/kokoro/blob/main/kokoro/istftnet.py), [inference path](https://github.com/hexgrad/kokoro/blob/main/kokoro/model.py), and [approved Run Plan](../run-plan.md).

Background: [Kokoro native inference spec](../spec.md).

Worker Profile: implement-lite · xhigh · gpt-6-luna (override)

## What to build

Aligned text features, F0/noise sequences, and decoder style become the conditioned feature sequence consumed by Kokoro's waveform generator.

## Acceptance criteria

- [ ] Implement the pre-generator ISTFTNet conditioning and residual blocks with the v1.0 channel and time shapes and style split.
- [ ] Mismatched conditioning lengths return a clear error instead of partial output.
- [ ] Pre-generator features match the reference fixture within ticket 01's pre-established budget; a miss prompts investigation rather than automatic tolerance widening.

## Blocked by

- `04-pitch-noise-prediction` — pitch and noise conditioning.
- `05-phoneme-acoustic-encoder` — aligned text features.

**Status:** ready-for-agent

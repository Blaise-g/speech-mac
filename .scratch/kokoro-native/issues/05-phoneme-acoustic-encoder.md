# 05: Encode phonemes into aligned acoustic features

## Context

Technical source: `.scratch/kokoro-native/spec.md`, official [Kokoro model modules](https://github.com/hexgrad/kokoro/blob/main/kokoro/modules.py), [inference path](https://github.com/hexgrad/kokoro/blob/main/kokoro/model.py), and [approved Run Plan](../run-plan.md).

Background: [Kokoro native inference spec](../spec.md).

Worker Profile: implement-lite · xhigh · gpt-6-luna (override)

## What to build

Prepared phoneme IDs become text-encoder features expanded onto the duration-predicted acoustic frame sequence.

## Acceptance criteria

- [ ] Implement the v1.0 embedding, convolutional/residual stack, recurrent layer, padding mask, and alignment multiplication.
- [ ] Output time dimension matches duration expansion; padding does not contribute acoustic features.
- [ ] Text-encoder and aligned features match reference fixtures within ticket 01's pre-established budget; a miss prompts investigation rather than automatic tolerance widening.

## Blocked by

- `03-duration-alignment` — token-to-frame alignment.

**Status:** ready-for-agent

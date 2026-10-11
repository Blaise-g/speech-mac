# 09: Verify complete direct C inference

## Context

Technical source: `.scratch/kokoro-native/spec.md`, official [Kokoro inference path](https://github.com/hexgrad/kokoro/blob/main/kokoro/model.py), [model modules](https://github.com/hexgrad/kokoro/blob/main/kokoro/modules.py), [ISTFTNet](https://github.com/hexgrad/kokoro/blob/main/kokoro/istftnet.py), and [approved Run Plan](../run-plan.md).

Background: [Kokoro native inference spec](../spec.md).

Worker Profile: implement-lite · xhigh · gpt-6-luna (override)

## What to build

Prepared phonemes, one voice style, and speed synthesize complete Kokoro audio through the assembled native pipeline.

## Acceptance criteria

- [ ] The assembled inference path runs directly in C on Apple Silicon and matches the pinned official reference within ticket 01's pre-established numerical budgets.
- [ ] Produce and retain a representative WAV audition artifact with recorded model, voice, and input provenance.
- [ ] The native runtime inference path does not require Python.

## Blocked by

- `08-istft-waveform` — all inference stages through waveform reconstruction.

**Status:** ready-for-agent

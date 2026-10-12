# 08: Reconstruct the 24 kHz waveform

## Context

Technical source: `.scratch/kokoro-native/spec.md`, official [ISTFTNet implementation](https://github.com/hexgrad/kokoro/blob/main/kokoro/istftnet.py), [model config](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/config.json), and [approved Run Plan](../run-plan.md).

Background: [Kokoro native inference spec](../spec.md).

Worker Profile: implement-lite · xhigh · gpt-6-luna (override)

## What to build

Kokoro spectral frames become finite mono waveform samples at the model's output rate.

## Acceptance criteria

- [x] Implement the v1.0 inverse-STFT reconstruction with the official hop, window, and output sizing.
- [x] Return finite mono samples at 24 kHz with output length matching the model's frame semantics.
- [x] Match fixed spectral-frame fixtures within ticket 01's pre-established budget; keep platform acceleration behind a narrow numeric seam where needed for reference checks.

## Blocked by

- `07-neural-waveform-generator` — spectral frames.

**Status:** done

## Verification — 2026-10-12

Commit `cc57e7e547355ab28cd0b13d0188c3b9be718acd`. Coordinator verified clean tree/trailer and reran standalone ISTFT test without needing model weights: normalized RMSE 6.18711862e-07, max abs 3.27825546e-07 within fixed waveform limits. Worker full check passed with real weights; normal snapshot hook passed. Independent review fixed stale-error success behavior and documented zero-initialized output precondition; no remaining08 blocker. Combined generator→waveform propagation belongs09.

# 07: Generate ISTFTNet spectral frames

## Context

Technical source: `.scratch/kokoro-native/spec.md`, official [ISTFTNet decoder and generator](https://github.com/hexgrad/kokoro/blob/main/kokoro/istftnet.py), [model config](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/config.json), and [approved Run Plan](../run-plan.md).

Background: [Kokoro native inference spec](../spec.md).

Worker Profile: implement-lite · xhigh · gpt-6-luna (override)

## What to build

Decoder features, style, and pitch conditioning produce Kokoro's upsampled spectral representation, including its periodic/noise excitation.

## Acceptance criteria

- [ ] Implement the configured upsample stages, residual blocks, source excitation, and spectral output.
- [ ] Provide a deterministic test seam for excitation/noise inputs so comparisons can be reproduced.
- [ ] Spectral outputs match pinned fixtures within ticket 01's pre-established budget using captured excitation; output frame count and values are valid and finite.

## Blocked by

- `06-decoder-conditioning` — conditioned decoder features.

**Status:** ready-for-agent

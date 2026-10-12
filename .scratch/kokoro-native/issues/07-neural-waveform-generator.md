# 07: Generate ISTFTNet spectral frames

## Context

Technical source: `.scratch/kokoro-native/spec.md`, official [ISTFTNet decoder and generator](https://github.com/hexgrad/kokoro/blob/main/kokoro/istftnet.py), [model config](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/config.json), and [approved Run Plan](../run-plan.md).

Background: [Kokoro native inference spec](../spec.md).

Worker Profile: implement-lite · xhigh · gpt-6-luna (override)

## What to build

Decoder features, style, and pitch conditioning produce Kokoro's upsampled spectral representation, including its periodic/noise excitation.

## Acceptance criteria

- [x] Implement the configured upsample stages, residual blocks, source excitation, and spectral output.
- [x] Provide a deterministic test seam for excitation/noise inputs so comparisons can be reproduced.
- [x] Spectral outputs match pinned fixtures within ticket 01's pre-established budget using captured excitation; output frame count and values are valid and finite.

## Blocked by

- `06-decoder-conditioning` — conditioned decoder features.

**Status:** done

## Verification — 2026-10-12

Commit `3a04826f1f088aa41bd367614e8a35e81d11f5fb`. Coordinator verified clean tree/Ticket trailer and reran generator focused test with actual pack: source magnitude max abs 1.1920929e-07, reconstructed complex STFT max error 2.75269826e-07; learned spectral magnitude/phase max abs 9.15527344e-05/2.57398933e-05 pass unchanged per-element budgets. Native F0 excitation finite/seed-deterministic/seed-sensitive; malformed/overflow boundaries pass. Worker/reviewer full checks passed. Review fixed source STFT float overflow; overtone-phase concern retracted with locked Torch equality evidence.

Coordinator-approved deterministic conditioning seam: captured source STFT is optional test input so learned-generator parity uses identical actual conditioning; native source STFT is independently compared by magnitude/reconstructed complex coefficients because +/-pi represent equivalent complex phase but the neural network amplifies branch choices. Production remains fully native F0/excitation/STFT. This does not relax learned-spectrum or final-waveform budgets, and does not claim free-running sample parity with Torch. Integration09 must distinguish controlled-conditioning comparison from native production checks.

# 10: Reuse loaded weights and write phoneme-input WAVs

## Context

Technical source: `.scratch/kokoro-native/spec.md` (resident engine and WAV CLI acceptance) and [approved Run Plan](../run-plan.md).

Background: [Kokoro native inference spec](../spec.md).

Worker Profile: implement-lite · xhigh · gpt-6-luna (override)

## What to build

A reusable C engine keeps model weights resident across requests, and the CLI turns prepared phonemes into a valid WAV.

## Acceptance criteria

- [ ] Load model and voice data once into an engine and synthesize repeated requests through the same instance.
- [ ] The phoneme-input CLI writes mono 24 kHz WAV files with correct headers and sample counts.
- [ ] Document model and voice artifact inputs and return clear errors for missing or incompatible data.

## Blocked by

- `09-phoneme-to-audio-correctness` — complete native phoneme inference.

**Status:** ready-for-agent

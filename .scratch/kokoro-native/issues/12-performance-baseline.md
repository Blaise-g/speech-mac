# 12: Measure cold start, warm synthesis, and first audio

## Context

Technical source: `.scratch/kokoro-native/spec.md` (performance measurement acceptance), [ADR 0001](../../../docs/adr/0001-native-inference.md), and [approved Run Plan](../run-plan.md).

Background: [Kokoro native inference spec](../spec.md).

Worker Profile: implement-lite · xhigh · gpt-6-luna (override)

## What to build

Repeatable measurements show model-load cost, warm synthesis time, generated duration, throughput, and latency on an identified Apple Silicon Mac.

## Acceptance criteria

- [ ] Report hardware/OS, build and model revisions, voice, input, cold start, warm request time, generated duration, and throughput.
- [ ] Since inference returns a complete waveform rather than streaming, define first-audio latency as time until the full in-memory waveform is available; report WAV write completion separately.
- [ ] Run repeated measurements to establish observed variation for the optimization ticket. Do not assume a performance target in advance.

## Blocked by

- `11-english-espeak-frontend` — complete text-to-WAV path.

**Status:** ready-for-agent

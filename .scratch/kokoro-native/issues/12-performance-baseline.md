# 12: Measure cold start, warm synthesis, and first audio

## Context

Technical source: `.scratch/kokoro-native/spec.md` (performance measurement acceptance), [ADR 0001](../../../docs/adr/0001-native-inference.md), and [approved Run Plan](../run-plan.md).

Background: [Kokoro native inference spec](../spec.md).

Worker Profile: implement-lite · xhigh · gpt-6-luna (override)

## What to build

Repeatable measurements show model-load cost, warm synthesis time, generated duration, throughput, and latency on an identified Apple Silicon Mac.

## Acceptance criteria

- [x] Report hardware/OS, build and model revisions, voice, input, cold start, warm request time, generated duration, and throughput.
- [x] Since inference returns a complete waveform rather than streaming, define first-audio latency as time until the full in-memory waveform is available; report WAV write completion separately.
- [x] Run repeated measurements to establish observed variation for the optimization ticket. Do not assume a performance target in advance.

## Blocked by

- `11-english-espeak-frontend` — complete text-to-WAV path.

**Status:** done

## Verification — 2026-10-12

Commits `832032b97fb514848064b3498e9bba0b041ec1f6`, `029a9a98f0ff13b5b2fc05e8c5c1e2bb3a8374a1`. Coordinator verified clean tree/Ticket trailers, read committed report provenance and reran 6 locked uv report tests: pass. Apple M5 16GiB macOS27.0.1, release-O2.25warm samples/workload: short1.70saudio synthesis213.32ms median (4.9%range/median), paragraph9.825saudio1250.32ms (9.9%variation), throughput7.97x/7.86x. Generator87.8–88.9% dominant stage; coldCLI421.39ms/1466.85ms separate fromwarm. Full checks/repeated run and independent summary/hash recomputation pass. No cachepurge, stage timing benchmarkonly. Review fixed cache labels/effectivebuildprovenance/clockfailure/hostreturn.13 must profileinsidegenerator and exceed observedvariation for speedclaim.

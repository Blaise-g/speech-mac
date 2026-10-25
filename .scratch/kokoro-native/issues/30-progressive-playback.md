# 30: Bound playback, backpressure and cancellation

## Parent

None (Requirement Source).

## Context

- Technical source: `.scratch/kokoro-native/spec.md` and its pinned primary sources.
- Background: `.scratch/kokoro-native/spec.md`.
- User clarification dated 2026-10-25: subsequent implementation, fixes and ticket reviews use implement-lite / gpt-6.1-sol / low, overriding the repository model/effort defaults. Final whole-branch review uses gpt-6.1-sol / high. Actual earlier Luna/low execution records remain historical evidence.

## Footprint

stream driver, AudioQueue/platform adapter, CLI dispatch, request/cancel status contracts, deterministic concurrency tests/device probes/stream metrics.

## What to build

--stream consuming within-utterance frames through existing narrow AudioQueue adapter. Ordered bounded buffering/backpressure/drain/cancel/error lifecycle and resident reuse. Separate first PCM, submission/start, buffer reclaim and successful drain events. Prove first audio arrives before inference finishes, batch/stream equivalence including tail, cancellation during generation/wait/drain and single terminal outcome. Device checks mute queue only, never system volume; fake-consumer time is not acoustic/device onset.

## Blocked by

29-speech-cli-voices

Worker Profile: medium; implement-lite / gpt-6.1-sol / low (user clarification 2026-10-25)

## Acceptance criteria

- [ ] --stream consuming within-utterance frames through existing narrow AudioQueue adapter
- [ ] Ordered bounded buffering/backpressure/drain/cancel/error lifecycle and resident reuse
- [ ] Separate first PCM, submission/start, buffer reclaim and successful drain events
- [ ] Prove first audio arrives before inference finishes, batch/stream equivalence including tail, cancellation during generation/wait/drain and single terminal outcome
- [ ] Device checks mute queue only, never system volume
- [ ] fake-consumer time is not acoustic/device onset
- [ ] Complete the approved slice checks and keep unavailable model/device checks explicit.

Status: ready-for-agent

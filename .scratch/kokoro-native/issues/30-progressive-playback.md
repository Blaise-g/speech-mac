# 30: Bound playback, backpressure and cancellation

## Parent

None (Requirement Source).

## Context

- Technical source: `.scratch/kokoro-native/spec.md` and its pinned primary sources.
- Background: `.scratch/kokoro-native/spec.md`.
- User approval dated 2026-10-25: “use implement lite with effort low for all”; resolved workflow/model/effort are implement-lite / gpt-6-luna / low, overriding AGENTS xhigh. Reviews also use Luna/low.

## Footprint

stream driver, AudioQueue/platform adapter, CLI dispatch, request/cancel status contracts, deterministic concurrency tests/device probes/stream metrics.

## What to build

--stream consuming within-utterance frames through existing narrow AudioQueue adapter. Ordered bounded buffering/backpressure/drain/cancel/error lifecycle and resident reuse. Separate first PCM, submission/start, buffer reclaim and successful drain events. Prove first audio arrives before inference finishes, batch/stream equivalence including tail, cancellation during generation/wait/drain and single terminal outcome. Device checks mute queue only, never system volume; fake-consumer time is not acoustic/device onset.

## Blocked by

29-speech-cli-voices

Worker Profile: medium

## Acceptance criteria

- [ ] --stream consuming within-utterance frames through existing narrow AudioQueue adapter
- [ ] Ordered bounded buffering/backpressure/drain/cancel/error lifecycle and resident reuse
- [ ] Separate first PCM, submission/start, buffer reclaim and successful drain events
- [ ] Prove first audio arrives before inference finishes, batch/stream equivalence including tail, cancellation during generation/wait/drain and single terminal outcome
- [ ] Device checks mute queue only, never system volume
- [ ] fake-consumer time is not acoustic/device onset
- [ ] Complete the approved slice checks and keep unavailable model/device checks explicit.

Status: ready-for-agent

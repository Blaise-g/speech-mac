# 32: Record Apple Silicon correctness/performance baseline

## Parent

None (Requirement Source).

## Context

- Technical source: `.scratch/kokoro-native/spec.md` and its pinned primary sources.
- Background: `.scratch/kokoro-native/spec.md`.
- User clarification dated 2026-10-25: subsequent implementation, fixes and ticket reviews use implement-lite / gpt-6.1-sol / low, overriding the repository model/effort defaults. Final whole-branch review uses gpt-6.1-sol / high. Actual earlier Luna/low execution records remain historical evidence.

## Footprint

native/uv benchmark drivers, report schema/validators/tests, raw artifacts/docs, engine/stage/playback instrumentation; exclusive local device timing/AudioQueue during measurement.

## What to build

locked-uv benchmark/report with raw repetitions of short reply/structured text/long input, cold fresh-process and warm resident modes, source/build/model/quant/voice/settings/hardware provenance. Measure first PCM, total synthesis, RTF, CPU/RSS with scopes and stage profile. Separate muted AudioQueue from fake-consumer timing; acoustic onset unavailable without independent capture. Produce audition files for user listening; no autoplay/cache purge/system volume changes.

## Blocked by

31-remove-kokoro

Worker Profile: medium; implement-lite / gpt-6.1-sol / low (user clarification 2026-10-25)

## Acceptance criteria

- [ ] locked-uv benchmark/report with raw repetitions of short reply/structured text/long input, cold fresh-process and warm resident modes, source/build/model/quant/voice/settings/hardware provenance
- [ ] Measure first PCM, total synthesis, RTF, CPU/RSS with scopes and stage profile
- [ ] Separate muted AudioQueue from fake-consumer timing
- [ ] acoustic onset unavailable without independent capture
- [ ] Produce audition files for user listening
- [ ] no autoplay/cache purge/system volume changes
- [ ] Complete the approved slice checks and keep unavailable model/device checks explicit.

Status: ready-for-agent

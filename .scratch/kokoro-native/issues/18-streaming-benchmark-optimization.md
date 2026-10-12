# 18: Measure streaming timing and profile Apple Silicon optimization

## Context

Technical source: `.scratch/kokoro-native/spec.md` (Authorized streaming implementation), the [approved streaming design](../../../docs/streaming-plan.md), [ticket 15 comparison artifacts](15-coreml-comparison-baseline.md), and [ticket 13 optimization investigation](13-profile-guided-optimization.md).

Background: [Kokoro native inference spec](../spec.md).

Worker Profile: implement-lite · xhigh · gpt-6-luna (override)

## What to build

Repeatable streaming timing records show when chunks become available to the playback queue and how queue pressure affects production. Once streaming parity is established, a bounded profile-guided attempt tests whether a safe Apple Silicon change improves the native path.

## Acceptance criteria

- [ ] Preserve ticket 15's comparison artifacts and add immutable dated records with exact source/build, host, input, voice, configured seed, command, and raw measurements. Use ticket 15's short and paragraph inputs at `af_heart`, and repeat at `am_adam` for voice coverage.
- [ ] From a monotonic CLI-entry timestamp, measure first complete chunk ready, first buffer accepted, successful playback-start request, and first buffer reclaimed; report medians and observed ranges. Separate synthesis CPU time from playback wall time and backpressure, and report generated duration, queue bound, and any underruns. API events and buffer reclamation do not prove rendered or acoustic onset; report acoustic timing only with independent evidence.
- [ ] After stream/batch parity passes, profile the current native path and make one bounded attempt at a safe measured Apple Silicon optimization, favoring supported Accelerate/NEON work or layout/scratch reuse. Read ticket 13's rejected sine-approximation results first. Do not introduce Metal, quantization, or precision changes without separate scope.
- [ ] Recheck fixed numerical budgets and run paired before/after measurements on the same host and workload. Set the improvement threshold from contemporaneous paired-run variation; ticket 12 and 15 historical results provide context but are not a fixed threshold. Claim a gain only when it clears observed variation.
- [ ] If no safe change clears contemporaneous variation, retain the timing and optimization-attempt evidence and leave the optimization criterion open with the reason. Do not claim the ticket complete or invent a speedup. Streaming timing results remain reportable independently of that open criterion.

## Tests

- Add focused locked-`uv` fixture/stub tests for timing record and report validation, including provenance/sample retention, event and process timing scopes, and unavailable acoustic timing. Keep device measurements separate from automated tests.
- Do not run an audible speaker benchmark. If device timing cannot be collected without unintended audible output, mark it unavailable and report fake-consumer measurements separately.
- Run `make format`, then `make check`, and obtain independent review before committing.

## Blocked by

- `17-native-streaming-playback` — verified implementation and streaming/batch parity.

Ticket 15 is completed historical provenance; ticket 13's open investigation is context and does not block this bounded attempt.

**Status:** blocked

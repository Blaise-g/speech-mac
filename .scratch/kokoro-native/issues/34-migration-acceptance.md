# 34: Verify branch and record listening evidence

## Parent

None (Requirement Source).

## Context

- Technical source: `.scratch/kokoro-native/spec.md` and its pinned primary sources.
- Background: `.scratch/kokoro-native/spec.md`.
- User approval dated 2026-10-25: “use implement lite with effort low for all”; resolved workflow/model/effort are implement-lite / gpt-6-luna / low, overriding AGENTS xhigh. Reviews also use Luna/low.

## Footprint

acceptance/integration tests/recipes/report/artifacts/tracker evidence, whole native/reference/CLI contracts verified; queue-muted device probes. Repairs retain fixed budgets and get independent changed-path review.

## What to build

real40M acceptance report: format→check, locked-uv reference/Q8 parity, long window/flush, malformed input/assets, controls/voices, resident resets, cancellation/cleanup, runtime/dependency audit and final removal census. Each acceptance claim is verified here; the run-level whole-branch review follows ticket35. Fresh build/setup recipe and audition artifacts; user listening explicitly pending until user confirms, never inferred from numerical/ASR checks. Missing device/model checks remain open/blocking, not successful skips.

## Blocked by

33-measured-cpu-optimization

Worker Profile: medium

## Acceptance criteria

- [ ] real40M acceptance report: format→check, locked-uv reference/Q8 parity, long window/flush, malformed input/assets, controls/voices, resident resets, cancellation/cleanup, runtime/dependency audit and final removal census
- [ ] Complete independent changed-path review for this ticket before its commit; run-level whole-branch review follows ticket35
- [ ] Fresh build/setup recipe and audition artifacts
- [ ] user listening explicitly pending until user confirms, never inferred from numerical/ASR checks
- [ ] Missing device/model checks remain open/blocking, not successful skips
- [ ] Complete the approved slice checks and keep unavailable model/device checks explicit.

Status: ready-for-agent

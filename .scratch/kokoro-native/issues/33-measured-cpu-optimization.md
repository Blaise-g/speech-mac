# 33: Improve a measured CPU bottleneck

## Parent

None (Requirement Source).

## Context

- Technical source: `.scratch/kokoro-native/spec.md` and its pinned primary sources.
- Background: `.scratch/kokoro-native/spec.md`.
- User clarification dated 2026-10-25: subsequent implementation, fixes and ticket reviews use implement-lite / gpt-6.1-sol / low, overriding the repository model/effort defaults. Final whole-branch review uses gpt-6.1-sol / high. Actual earlier Luna/low execution records remain historical evidence.

## Footprint

hot-path numerical/model surface identified by32, relevant regression tests, before/after raw benchmark evidence/docs; exclusive device timing. Worker limits edits to one manageable measured surface.

## What to build

bounded Accelerate/NEON/packed-Q8/allocation improvement on dominant measured surface, paired on same host/workloads against fixed correctness budgets. Native packed Q8 is already correct in20. No speculative Mimi requantization/KleidiAI runtime/Metal. If nothing beats observed variation, preserve correct baseline and record failed experiments/bottlenecks candidly.

## Blocked by

32-phonon-baseline

Worker Profile: medium; implement-lite / gpt-6.1-sol / low (user clarification 2026-10-25)

## Acceptance criteria

- [ ] bounded Accelerate/NEON/packed-Q8/allocation improvement on dominant measured surface, paired on same host/workloads against fixed correctness budgets
- [ ] Native packed Q8 is already correct in20
- [ ] No speculative Mimi requantization/KleidiAI runtime/Metal
- [ ] If nothing beats observed variation, preserve correct baseline and record failed experiments/bottlenecks candidly
- [ ] Complete the approved slice checks and keep unavailable model/device checks explicit.

Status: ready-for-agent

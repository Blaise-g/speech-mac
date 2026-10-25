# 33: Improve a measured CPU bottleneck

## Parent

None (Requirement Source).

## Context

- Technical source: `.scratch/kokoro-native/spec.md` and its pinned primary sources.
- Background: `.scratch/kokoro-native/spec.md`.
- User approval dated 2026-10-25: “use implement lite with effort low for all”; resolved workflow/model/effort are implement-lite / gpt-6-luna / low, overriding AGENTS xhigh. Reviews also use Luna/low.

## Footprint

hot-path numerical/model surface identified by32, relevant regression tests, before/after raw benchmark evidence/docs; exclusive device timing. Worker limits edits to one manageable measured surface.

## What to build

bounded Accelerate/NEON/packed-Q8/allocation improvement on dominant measured surface, paired on same host/workloads against fixed correctness budgets. Native packed Q8 is already correct in20. No speculative Mimi requantization/KleidiAI runtime/Metal. If nothing beats observed variation, preserve correct baseline and record failed experiments/bottlenecks candidly.

## Blocked by

32-phonon-baseline

Worker Profile: medium

## Acceptance criteria

- [ ] bounded Accelerate/NEON/packed-Q8/allocation improvement on dominant measured surface, paired on same host/workloads against fixed correctness budgets
- [ ] Native packed Q8 is already correct in20
- [ ] No speculative Mimi requantization/KleidiAI runtime/Metal
- [ ] If nothing beats observed variation, preserve correct baseline and record failed experiments/bottlenecks candidly
- [ ] Complete the approved slice checks and keep unavailable model/device checks explicit.

Status: ready-for-agent

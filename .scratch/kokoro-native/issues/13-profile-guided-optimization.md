# 13: Improve a measured inference bottleneck

## Context

Technical source: `.scratch/kokoro-native/spec.md` (optimization acceptance), [ADR 0001](../../../docs/adr/0001-native-inference.md), and [approved Run Plan](../run-plan.md).

Background: [Kokoro native inference spec](../spec.md).

Worker Profile: implement-lite · xhigh · gpt-6-luna (override)

## What to build

One profile-backed change improves the measured native inference path while preserving reference correctness.

## Acceptance criteria

- [ ] Identify the dominant bottleneck from ticket 12's baseline and make one targeted optimization.
- [ ] Report paired before/after measurements on the same Mac and workload. Claim improvement only when it exceeds the run-to-run variation observed in ticket 12.
- [ ] Re-run stage and full-audio correctness checks against ticket 01's fixed budgets.
- [ ] Do not add Metal preemptively. If profiling shows a Metal-suitable bottleneck, request a separate bounded ticket with evidence.

## Blocked by

- `12-performance-baseline` — measured workload and run-to-run variation.

**Status:** ready-for-agent

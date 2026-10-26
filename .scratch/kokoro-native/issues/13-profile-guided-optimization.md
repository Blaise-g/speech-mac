# 13: Improve a measured inference bottleneck

## Context

Technical source: `.scratch/kokoro-native/spec.md` (optimization acceptance), [ADR 0001](https://github.com/Blaise-g/speech-mac/blob/3434f73aeada7e1818feb11cf4815b773a21188b/docs/adr/0001-native-inference.md), and [approved Run Plan](../run-plan.md).

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

**Status:** blocked

## Open record — 2026-10-12

Investigation commit `b35c7a151b279fcff7af15b48801653e61bf0126`; report artifacts/13-optimization-investigation.md. Coordinator verified clean tree/Ticket trailer and no native src diff from measured baseline e7c6465. Profile found scalar sine in generator residual hotpath. Tested sine substitutes miss fixed phase budget or have no measured gain/codegenerationbasis. Failed experiments discarded; no paired optimized speedclaim; no budgetrelaxation. Full model-backed checks passed on restored baseline, independent report review clear. Acceptance remains incomplete: no safe additional optimization above noise verified. Keep working measured~7.9x baseline; report13open.

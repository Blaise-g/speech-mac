# 19: Pin artifacts and a replayable 40M oracle

## Parent

None (Requirement Source).

## Context

- Technical source: `.scratch/kokoro-native/spec.md` and its pinned primary sources.
- Background: `.scratch/kokoro-native/spec.md`.
- User approval dated 2026-10-25: “use implement lite with effort low for all”; resolved workflow/model/effort are implement-lite / gpt-6-luna / low, overriding AGENTS xhigh. Reviews also use Luna/low.
- Oracle environment: Coordinator PATH lacked cargo/rustc. Pin a correct executable oracle or provision isolated development tools; a cached binary alone is not correctness evidence.

## Footprint

reference pins/tooling, uv project+lock, fixture manifest/budgets/exporters/tests/build integration, reference docs/licenses; shared assets read-only. Legacy oracle remains temporarily while replacement grows.

## What to build

a reproducible model/reference probe, fixed numerical budgets and intermediate fixtures before native work. Verify supplied assets against `local-verification.json` and authoritative revision/hash provenance; record licenses/access and catalog provenance. Pin xn-ptts `ac0e1524cfd9e71d57aadca0de542796f942d775` and model `cc738b7e4d182c2ba1bafb8062aa4846af000637`, oracle dependencies/locks. Upstream Rust may run only as a development oracle, never linked/shipped runtime. Resolve upstream Mimi context discrepancy (PR209) by identifying trained semantics and an independent pinned correct oracle, or a minimal documented reference patch with provenance. Decoder budget freezing is blocked until reconciled. Export exact normalized text/IDs/voice projection/prefix states/backbone output/raw EOS/noise draws/flow integration/latents/Mimi features/raw and final PCM/stop lengths. Fixtures cross >250 context positions with uneven batches. Do not silently treat buggy pinned output as truth or loosen tolerances to hide it.

## Blocked by

None (can start immediately)

Worker Profile: medium

## Acceptance criteria

- [ ] a reproducible model/reference probe, fixed numerical budgets and intermediate fixtures before native work
- [ ] Verify supplied assets against `local-verification.json` and authoritative revision/hash provenance
- [ ] record licenses/access and catalog provenance
- [ ] Pin xn-ptts `ac0e1524cfd9e71d57aadca0de542796f942d775` and model `cc738b7e4d182c2ba1bafb8062aa4846af000637`, oracle dependencies/locks
- [ ] Upstream Rust may run only as a development oracle, never linked/shipped runtime
- [ ] Resolve upstream Mimi context discrepancy (PR209) by identifying trained semantics and an independent pinned correct oracle, or a minimal documented reference patch with provenance
- [ ] Decoder budget freezing is blocked until reconciled
- [ ] Export exact normalized text/IDs/voice projection/prefix states/backbone output/raw EOS/noise draws/flow integration/latents/Mimi features/raw and final PCM/stop lengths
- [ ] Fixtures cross >250 context positions with uneven batches
- [ ] Do not silently treat buggy pinned output as truth or loosen tolerances to hide it
- [ ] Complete the approved slice checks and keep unavailable model/device checks explicit.

Status: ready-for-agent

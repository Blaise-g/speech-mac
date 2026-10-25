# 31: Finish maintained identity and remove legacy support

## Parent

None (Requirement Source).

## Context

- Technical source: `.scratch/kokoro-native/spec.md` and its pinned primary sources.
- Background: `.scratch/kokoro-native/spec.md`.
- User clarification dated 2026-10-25: subsequent implementation, fixes and ticket reviews use implement-lite / gpt-6.1-sol / low, overriding the repository model/effort defaults. Final whole-branch review uses gpt-6.1-sol / high. Actual earlier Luna/low execution records remain historical evidence.
- Consumer census: local API consumers are CLI/stream/benchmarks/tests. Nearby repositories and queried upstream PRs revealed no additional client. No UI exists. Installed `/opt/homebrew/bin/kokoro` is external and stays untouched. Census cannot rule out undiscoverable private callers. Recheck open PRs before removal/rename. Upstream PR209 context+1 discrepancy gates oracle reconciliation; PR186 Mimi Q8 was slower with quality tradeoff and is excluded.

## Footprint

repo-wide final contract/deletion across src/tests/reference/benchmarks/build/locks/config/docs/CI/public identifiers/generated fixtures. Historical tracker directory remains.

## What to build

speech-mac-only maintained runtime/dependencies/commands/libs/docs. Delete old model graph, phoneme/eSpeak frontend/style artifacts/options, old fixtures/oracle/conversion/reference dependencies and active old benchmark/comparison runners. Preserve historical numbered evidence, tracker00–18/archive and attributed artifacts. Rename surviving generic public APIs/headers/build outputs; update glossary/standards/ADR retaining history, setup/CI/maintained examples/locked uv dependencies and tracker guidance. Census below governs removals. Installed external kokoro tool stays untouched. No local root move or universal migration promise for undiscoverable external callers.

## Blocked by

30-progressive-playback

Worker Profile: medium; implement-lite / gpt-6.1-sol / low (user clarification 2026-10-25)

## Acceptance criteria

- [ ] speech-mac-only maintained runtime/dependencies/commands/libs/docs
- [ ] Delete old model graph, phoneme/eSpeak frontend/style artifacts/options, old fixtures/oracle/conversion/reference dependencies and active old benchmark/comparison runners
- [ ] Preserve historical numbered evidence, tracker00–18/archive and attributed artifacts
- [ ] Rename surviving generic public APIs/headers/build outputs
- [ ] update glossary/standards/ADR retaining history, setup/CI/maintained examples/locked uv dependencies and tracker guidance
- [ ] Census below governs removals
- [ ] Installed external kokoro tool stays untouched
- [ ] No local root move or universal migration promise for undiscoverable external callers
- [ ] Complete the approved slice checks and keep unavailable model/device checks explicit.

Status: ready-for-agent

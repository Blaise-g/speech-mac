# 28: Assemble native progressive text-to-PCM inference

## Parent

None (Requirement Source).

## Context

- Technical source: `.scratch/kokoro-native/spec.md` and its pinned primary sources.
- Background: `.scratch/kokoro-native/spec.md`.
- User clarification dated 2026-10-25: subsequent implementation, fixes and ticket reviews use implement-lite / gpt-6.1-sol / low, overriding the repository model/effort defaults. Final whole-branch review uses gpt-6.1-sol / high. Actual earlier Luna/low execution records remain historical evidence.

## Footprint

public speech engine/request API, synthesis assembly, callback/status/limit contracts, integration/full-model tests; consumes all native seams.

## What to build

resident weights and one request machine shared by incremental callbacks and batch PCM collector. First PCM is produced within the utterance at the model's 80ms latent cadence, not only after sentence completion. Full frontend→voice→Flow-LM→Mimi parity, requests/voices/text-chunk reset, bounded queues/context, cancellation/errors/finish/ownership and no leaked state. A synchronous producer/callback is sufficient; threads are not required for inference correctness.

## Blocked by

27-mimi-pcm

Worker Profile: medium; implement-lite / gpt-6.1-sol / low (user clarification 2026-10-25)

## Acceptance criteria

- [ ] resident weights and one request machine shared by incremental callbacks and batch PCM collector
- [ ] First PCM is produced within the utterance at the model's 80ms latent cadence, not only after sentence completion
- [ ] Full frontend→voice→Flow-LM→Mimi parity, requests/voices/text-chunk reset, bounded queues/context, cancellation/errors/finish/ownership and no leaked state
- [ ] A synchronous producer/callback is sufficient
- [ ] threads are not required for inference correctness
- [ ] Complete the approved slice checks and keep unavailable model/device checks explicit.

Status: ready-for-agent

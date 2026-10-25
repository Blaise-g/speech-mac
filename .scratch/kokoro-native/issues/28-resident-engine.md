# 28: Assemble native progressive text-to-PCM inference

## Parent

None (Requirement Source).

## Context

- Technical source: `.scratch/kokoro-native/spec.md` and its pinned primary sources.
- Background: `.scratch/kokoro-native/spec.md`.
- User approval dated 2026-10-25: “use implement lite with effort low for all”; resolved workflow/model/effort are implement-lite / gpt-6-luna / low, overriding AGENTS xhigh. Reviews also use Luna/low.

## Footprint

public speech engine/request API, synthesis assembly, callback/status/limit contracts, integration/full-model tests; consumes all native seams.

## What to build

resident weights and one request machine shared by incremental callbacks and batch PCM collector. First PCM is produced within the utterance at the model's 80ms latent cadence, not only after sentence completion. Full frontend→voice→Flow-LM→Mimi parity, requests/voices/text-chunk reset, bounded queues/context, cancellation/errors/finish/ownership and no leaked state. A synchronous producer/callback is sufficient; threads are not required for inference correctness.

## Blocked by

27-mimi-pcm

Worker Profile: medium

## Acceptance criteria

- [ ] resident weights and one request machine shared by incremental callbacks and batch PCM collector
- [ ] First PCM is produced within the utterance at the model's 80ms latent cadence, not only after sentence completion
- [ ] Full frontend→voice→Flow-LM→Mimi parity, requests/voices/text-chunk reset, bounded queues/context, cancellation/errors/finish/ownership and no leaked state
- [ ] A synchronous producer/callback is sufficient
- [ ] threads are not required for inference correctness
- [ ] Complete the approved slice checks and keep unavailable model/device checks explicit.

Status: ready-for-agent

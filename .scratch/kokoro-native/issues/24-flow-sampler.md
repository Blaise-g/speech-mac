# 24: Generate replayable latents and EOS

## Parent

None (Requirement Source).

## Context

- Technical source: `.scratch/kokoro-native/spec.md` and its pinned primary sources.
- Background: `.scratch/kokoro-native/spec.md`.
- User approval dated 2026-10-25: “use implement lite with effort low for all”; resolved workflow/model/effort are implement-lite / gpt-6-luna / low, overriding AGENTS xhigh. Reviews also use Luna/low.

## Footprint

flow head/sampler, PRNG+replay interface, generation controls/stop metadata, latent/EOS fixtures/numerical tests/build integration.

## What to build

single/multiple 32-d latent frames using six-block AdaLN flow head, RMS time embedding (not centred variance), SiLU, pinned one-step LSD integration and raw EOS threshold. Port BOS/previous-latent scaling, EOS frame/tail/frame-budget semantics; distinguish safety budget exhaustion from EOS. Native equal-seed repeatability is defined, but cross-runtime parity consumes exact saved noise draws.

## Blocked by

23-flow-transformer

Worker Profile: medium

## Acceptance criteria

- [ ] single/multiple 32-d latent frames using six-block AdaLN flow head, RMS time embedding (not centred variance), SiLU, pinned one-step LSD integration and raw EOS threshold
- [ ] Port BOS/previous-latent scaling, EOS frame/tail/frame-budget semantics
- [ ] distinguish safety budget exhaustion from EOS
- [ ] Native equal-seed repeatability is defined, but cross-runtime parity consumes exact saved noise draws
- [ ] Complete the approved slice checks and keep unavailable model/device checks explicit.

Status: ready-for-agent

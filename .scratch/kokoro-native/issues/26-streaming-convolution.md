# 26: Verify causal SEANet operators

## Parent

None (Requirement Source).

## Context

- Technical source: `.scratch/kokoro-native/spec.md` and its pinned primary sources.
- Background: `.scratch/kokoro-native/spec.md`.
- User approval dated 2026-10-25: “use implement lite with effort low for all”; resolved workflow/model/effort are implement-lite / gpt-6-luna / low, overriding AGENTS xhigh. Reviews also use Luna/low.

## Footprint

convolution/residual kernels/state, numerical interfaces, operator fixtures/C tests/build wiring.

## What to build

convolution/transposed-convolution and residual-block seam checks for pinned [6,5,4] SEANet decoder. Exact padding, trimming, overlap state, biases and activation ordering; frame-partition tests distinguish retained versus reset state, malformed shape/overflow handling. This bounded operator increment keeps full decoder assembly manageable.

## Blocked by

25-mimi-latent-transformer

Worker Profile: medium

## Acceptance criteria

- [ ] convolution/transposed-convolution and residual-block seam checks for pinned [6,5,4] SEANet decoder
- [ ] Exact padding, trimming, overlap state, biases and activation ordering
- [ ] frame-partition tests distinguish retained versus reset state, malformed shape/overflow handling
- [ ] This bounded operator increment keeps full decoder assembly manageable
- [ ] Complete the approved slice checks and keep unavailable model/device checks explicit.

Status: ready-for-agent

# 26: Verify causal SEANet operators

## Parent

None (Requirement Source).

## Context

- Technical source: `.scratch/kokoro-native/spec.md` and its pinned primary sources.
- Background: `.scratch/kokoro-native/spec.md`.
- User clarification dated 2026-10-25: subsequent implementation, fixes and ticket reviews use implement-lite / gpt-6.1-sol / low, overriding the repository model/effort defaults. Final whole-branch review uses gpt-6.1-sol / high. Actual earlier Luna/low execution records remain historical evidence.

## Footprint

convolution/residual kernels/state, numerical interfaces, operator fixtures/C tests/build wiring.

## What to build

convolution/transposed-convolution and residual-block seam checks for pinned [6,5,4] SEANet decoder. Exact padding, trimming, overlap state, biases and activation ordering; frame-partition tests distinguish retained versus reset state, malformed shape/overflow handling. This bounded operator increment keeps full decoder assembly manageable.

## Blocked by

25-mimi-latent-transformer

Worker Profile: medium; implement-lite / gpt-6.1-sol / low (user clarification 2026-10-25)

## Acceptance criteria

- [ ] convolution/transposed-convolution and residual-block seam checks for pinned [6,5,4] SEANet decoder
- [ ] Exact padding, trimming, overlap state, biases and activation ordering
- [ ] frame-partition tests distinguish retained versus reset state, malformed shape/overflow handling
- [ ] This bounded operator increment keeps full decoder assembly manageable
- [ ] Complete the approved slice checks and keep unavailable model/device checks explicit.

Status: ready-for-agent

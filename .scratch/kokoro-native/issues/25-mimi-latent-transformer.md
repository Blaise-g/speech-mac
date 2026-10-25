# 25: Verify Mimi projected latent features

## Parent

None (Requirement Source).

## Context

- Technical source: `.scratch/kokoro-native/spec.md` and its pinned primary sources.
- Background: `.scratch/kokoro-native/spec.md`.
- User clarification dated 2026-10-25: subsequent implementation, fixes and ticket reviews use implement-lite / gpt-6.1-sol / low, overriding the repository model/effort defaults. Final whole-branch review uses gpt-6.1-sol / high. Actual earlier Luna/low execution records remain historical evidence.

## Footprint

Mimi projection/upsampler/transformer, transposed-convolution primitive, shared transformer context interface, decoder feature fixtures/tests; consumes pack/numerical contracts.

## What to build

latent32 through quantizer output projection, config-derived 16x learned upsampling, and two-layer 512-wide/eight-head Mimi transformer to decoder-feature seam. Context policy must be the reconciled ticket19 decision, bounded state, layer scale .01 and erf-GELU. Framewise/uneven partitions cross >250 positions.

## Blocked by

24-flow-sampler

Worker Profile: medium; implement-lite / gpt-6.1-sol / low (user clarification 2026-10-25)

## Acceptance criteria

- [ ] latent32 through quantizer output projection, config-derived 16x learned upsampling, and two-layer 512-wide/eight-head Mimi transformer to decoder-feature seam
- [ ] Context policy must be the reconciled ticket19 decision, bounded state, layer scale .01 and erf-GELU
- [ ] Framewise/uneven partitions cross >250 positions
- [ ] Complete the approved slice checks and keep unavailable model/device checks explicit.

Status: ready-for-agent

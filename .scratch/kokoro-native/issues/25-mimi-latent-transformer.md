# 25: Verify Mimi projected latent features

## Parent

None (Requirement Source).

## Context

- Technical source: `.scratch/kokoro-native/spec.md` and its pinned primary sources.
- Background: `.scratch/kokoro-native/spec.md`.
- User approval dated 2026-10-25: “use implement lite with effort low for all”; resolved workflow/model/effort are implement-lite / gpt-6-luna / low, overriding AGENTS xhigh. Reviews also use Luna/low.

## Footprint

Mimi projection/upsampler/transformer, transposed-convolution primitive, shared transformer context interface, decoder feature fixtures/tests; consumes pack/numerical contracts.

## What to build

latent32 through quantizer output projection, config-derived 16x learned upsampling, and two-layer 512-wide/eight-head Mimi transformer to decoder-feature seam. Context policy must be the reconciled ticket19 decision, bounded state, layer scale .01 and erf-GELU. Framewise/uneven partitions cross >250 positions.

## Blocked by

24-flow-sampler

Worker Profile: medium

## Acceptance criteria

- [ ] latent32 through quantizer output projection, config-derived 16x learned upsampling, and two-layer 512-wide/eight-head Mimi transformer to decoder-feature seam
- [ ] Context policy must be the reconciled ticket19 decision, bounded state, layer scale .01 and erf-GELU
- [ ] Framewise/uneven partitions cross >250 positions
- [ ] Complete the approved slice checks and keep unavailable model/device checks explicit.

Status: ready-for-agent

# 23: Verify voice/text priming and cached Flow-LM

## Parent

None (Requirement Source).

## Context

- Technical source: `.scratch/kokoro-native/spec.md` and its pinned primary sources.
- Background: `.scratch/kokoro-native/spec.md`.
- User approval dated 2026-10-25: “use implement lite with effort low for all”; resolved workflow/model/effort are implement-lite / gpt-6-luna / low, overriding AGENTS xhigh. Reviews also use Luna/low.

## Footprint

transformer/numerical kernels, condition/voice projection loader contract, Flow-LM state, pack mappings, synthetic/real stage fixtures/tests.

## What to build

voice projection, text/condition priming and nine-layer 512-wide/eight-head/2048-FFN cached transformer outputs against prefix/step fixtures. Implement norm/erf-GELU/RoPE/causal attention/projections, bounded KV ownership, prefill/incremental parity and independent request states. Implement num_speakers=1 and native duration_delta summed conditioning per actual config. Compatible voice tensor parsing/projection needed for fixtures is included; catalog UX is later. No unsupported CFG/speculative architecture.

## Blocked by

22-native-unigram-tokenizer

Worker Profile: medium

## Acceptance criteria

- [ ] voice projection, text/condition priming and nine-layer 512-wide/eight-head/2048-FFN cached transformer outputs against prefix/step fixtures
- [ ] Implement norm/erf-GELU/RoPE/causal attention/projections, bounded KV ownership, prefill/incremental parity and independent request states
- [ ] Implement num_speakers=1 and native duration_delta summed conditioning per actual config
- [ ] Compatible voice tensor parsing/projection needed for fixtures is included
- [ ] catalog UX is later
- [ ] No unsupported CFG/speculative architecture
- [ ] Complete the approved slice checks and keep unavailable model/device checks explicit.

Status: ready-for-agent

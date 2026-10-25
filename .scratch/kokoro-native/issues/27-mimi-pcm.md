# 27: Produce complete 24k mono PCM from latents

## Parent

None (Requirement Source).

## Context

- Technical source: `.scratch/kokoro-native/spec.md` and its pinned primary sources.
- Background: `.scratch/kokoro-native/spec.md`.
- User approval dated 2026-10-25: “use implement lite with effort low for all”; resolved workflow/model/effort are implement-lite / gpt-6-luna / low, overriding AGENTS xhigh. Reviews also use Luna/low.

## Footprint

Mimi/SEANet assembly, DSP/latency state, PCM ownership/output APIs, raw/final waveform fixtures/tests. Consumes25/26 state contracts.

## What to build

full SEANet decoder assembly and chosen pinned-reference gain/limiter chain when applicable. Separate raw and postprocessed PCM seams. Explicit first-sample delay trimming, tail drain/flush, chunk reset and terminal behavior; no invented codec flush frames. Whole/framewise/uneven partitions must meet fixed budgets and sample counts, including tiny/empty/tail cases and >context streams.

## Blocked by

26-streaming-convolution

Worker Profile: medium

## Acceptance criteria

- [ ] full SEANet decoder assembly and chosen pinned-reference gain/limiter chain when applicable
- [ ] Separate raw and postprocessed PCM seams
- [ ] Explicit first-sample delay trimming, tail drain/flush, chunk reset and terminal behavior
- [ ] no invented codec flush frames
- [ ] Whole/framewise/uneven partitions must meet fixed budgets and sample counts, including tiny/empty/tail cases and >context streams
- [ ] Complete the approved slice checks and keep unavailable model/device checks explicit.

Status: ready-for-agent

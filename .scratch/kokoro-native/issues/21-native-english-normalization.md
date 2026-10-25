# 21: Normalize ordinary English in C

## Parent

None (Requirement Source).

## Context

- Technical source: `.scratch/kokoro-native/spec.md` and its pinned primary sources.
- Background: `.scratch/kokoro-native/spec.md`.
- User approval dated 2026-10-25: “use implement lite with effort low for all”; resolved workflow/model/effort are implement-lite / gpt-6-luna / low, overriding AGENTS xhigh. Reviews also use Luna/low.

## Footprint

UTF-8/normalizer module, public text interface, normalization corpus/reference fixtures/tests, frontend docs/build integration.

## What to build

actual native English default normalization at a probe seam: character/Unicode/punctuation policy plus numbers/currency/dashed digits/emails/URLs/abbreviations/elongations, with malformed UTF-8 and empty-after-normalization checks. Preserve default policy rather than adding optional date/phone/time rewrite controls. Validate en; no Python/Rust/eSpeak child process or fixture lookup implementation.

## Blocked by

20-native-model-q8

Worker Profile: medium

## Acceptance criteria

- [ ] actual native English default normalization at a probe seam: character/Unicode/punctuation policy plus numbers/currency/dashed digits/emails/URLs/abbreviations/elongations, with malformed UTF-8 and empty-after-normalization checks
- [ ] Preserve default policy rather than adding optional date/phone/time rewrite controls
- [ ] Validate en
- [ ] no Python/Rust/eSpeak child process or fixture lookup implementation
- [ ] Complete the approved slice checks and keep unavailable model/device checks explicit.

Status: ready-for-agent

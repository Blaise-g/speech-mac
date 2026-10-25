# 22: Tokenize and bound long requests in C

## Parent

None (Requirement Source).

## Context

- Technical source: `.scratch/kokoro-native/spec.md` and its pinned primary sources.
- Background: `.scratch/kokoro-native/spec.md`.
- User approval dated 2026-10-25: “use implement lite with effort low for all”; resolved workflow/model/effort are implement-lite / gpt-6-luna / low, overriding AGENTS xhigh. Reviews also use Luna/low.

## Footprint

tokenizer/table/converter representation, native request text planning APIs, token/prompt/chunk fixtures/tests, pack metadata/build wiring.

## What to build

exact tokenizer IDs and decoding: verified 4000-vocabulary Unigram, byte fallback, Prepend+Metaspace, special token behavior matching encode(text,false). Port prompt preparation, ordered sentence grouping and oversized-sentence fitting without silent loss. Explicit token/frame/context bounds; whitespace/Unicode/unknown bytes/ties/punctuation/long words/NUL/invalid text checks. No runtime tokenizer library or IPA input.

## Blocked by

21-native-english-normalization

Worker Profile: medium

## Acceptance criteria

- [ ] exact tokenizer IDs and decoding: verified 4000-vocabulary Unigram, byte fallback, Prepend+Metaspace, special token behavior matching encode(text,false)
- [ ] Port prompt preparation, ordered sentence grouping and oversized-sentence fitting without silent loss
- [ ] Explicit token/frame/context bounds
- [ ] whitespace/Unicode/unknown bytes/ties/punctuation/long words/NUL/invalid text checks
- [ ] No runtime tokenizer library or IPA input
- [ ] Complete the approved slice checks and keep unavailable model/device checks explicit.

Status: ready-for-agent

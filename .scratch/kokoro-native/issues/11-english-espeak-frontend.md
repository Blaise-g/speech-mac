# 11: Synthesize English text, files, or standard input through eSpeak NG

## Context

Technical source: `.scratch/kokoro-native/spec.md` (English frontend and text-source CLI acceptance) and [approved Run Plan](../run-plan.md).

Background: [Kokoro native inference spec](../spec.md).

Worker Profile: implement-lite · xhigh · gpt-6-luna (override)

## What to build

Ordinary English text is converted by eSpeak NG and synthesized by the resident native engine. The CLI accepts one source: a positional text argument, `-f/--file PATH` for UTF-8 text, or standard input. `--ipa` bypasses eSpeak and treats that source as IPA/phoneme input.

## Acceptance criteria

- [x] Support the initial English path with a documented mapping from eSpeak output to Kokoro-supported symbols; record the eSpeak NG version and settings used.
- [x] `kokoro say TEXT`, `kokoro say -f/--file PATH`, and `... | kokoro say` consume the selected source. Require exactly one source; reject conflicting sources, invalid UTF-8, empty input, and unmappable output clearly.
- [x] With `--ipa`, bypass the English frontend and validate the supplied IPA/phoneme symbols against the supported inventory.
- [x] Preserve the existing prepared-phoneme reference path. Select `af_heart` by default and allow `-v/--voice am_adam`; `--list-voices` reports only installed supported English voices.
- [x] Consume long text completely using bounded, sentence-aware phoneme chunks (splitting an overlong sentence at a safe boundary) and combine all synthesized audio in order before WAV output or default playback. Never silently truncate input or output.
- [x] Document that pronunciation can differ from Misaki. Keep eSpeak NG and Python out of C model inference and runtime model execution.

## Tests

- Check representative English inputs against expected frontend tokens/IDs and run a text-to-WAV smoke test.
- Test positional/file/stdin source selection, conflicts, invalid UTF-8, empty input, `--ipa` bypass, voice selection, and a long multi-sentence input that crosses chunk boundaries; verify all expected chunks are synthesized in order, the final sample count equals the sum of chunk samples, and WAV metadata remains valid.
- Verify output through the WAV and audio checks from ticket 10. Do not require incremental playback or a daemon.

## Blocked by

- `10-resident-engine-phoneme-cli` — resident engine and shared CLI options.

**Status:** done

## Verification — 2026-10-12

Commit `d3bdaa39b297214864e63349bfff37b62284e9f0`. Coordinator verified clean tree/trailer and reran tokenizer and CLI tests: positional/file/stdin/IPA/source-errors/long ordered chunk PCM/playback checks pass. Independently exercised ordinary English and am_adam stdin synthesis and afinfo validated24kmonoPCM16. Worker full real-pack make check passed. Review fixed exact510-symbol split, UTF8 byte-vs-codepoint separator handling, shared decoding, and expectedfrontendtoken assertion. Retained57.525s text/WAV provenance in artifacts/. eSpeakNG1.52 installed. Human shortEnglish sample listening pending separately14; no11 blocker.

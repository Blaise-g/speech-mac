# 11: Synthesize ordinary English through eSpeak NG

## Context

Technical source: `.scratch/kokoro-native/spec.md` (English frontend decision and acceptance) and [approved Run Plan](../run-plan.md).

Background: [Kokoro native inference spec](../spec.md).

Worker Profile: implement-lite · xhigh · gpt-6-luna (override)

## What to build

Ordinary English text is converted by eSpeak NG and synthesized by the resident native engine through the CLI.

## Acceptance criteria

- [ ] Support the initial English path with a documented mapping from eSpeak output to Kokoro-supported symbols.
- [ ] Record the eSpeak NG version and settings used for reproducible frontend checks.
- [ ] Keep prepared phoneme input available as the reference path; reject empty or unmappable output clearly.
- [ ] Document that pronunciation can differ from Misaki and verify representative English text produces a valid WAV.
- [ ] Keep eSpeak NG and Python out of the C model inference path.

## Blocked by

- `10-resident-engine-phoneme-cli` — resident engine and phoneme WAV path.

**Status:** ready-for-agent

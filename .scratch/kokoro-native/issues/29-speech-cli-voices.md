# 29: Ship speech-mac tasks and compatible voices

## Parent

None (Requirement Source).

## Context

- Technical source: `.scratch/kokoro-native/spec.md` and its pinned primary sources.
- Background: `.scratch/kokoro-native/spec.md`.
- User approval dated 2026-10-25: “use implement lite with effort low for all”; resolved workflow/model/effort are implement-lite / gpt-6-luna / low, overriding AGENTS xhigh. Reviews also use Luna/low.

## Footprint

CLI/main, WAV/playback adapters, voice catalog/schema/compatibility resolution, engine consumers, speech-mac build product, CLI tests/docs.

## What to build

speech-mac say positional/file/stdin, valid mono24k PCM16 WAV, completed-WAV default playback, help/version, installed named/default and explicit compatible local voice artifacts. Validate en/seed/--duration-delta with native meaning and only tested supported controls. Actionable failures for old voice names, --ipa and --speed/-s; no silent reinterpretation. No cloud voice creation, recording encoder or runtime network requirement. Preserve temp permissions/cleanup; tests stub playback or write audition WAVs, no unsolicited speech. Legacy remains separately checkable until31, but no final compatibility alias.

## Blocked by

28-resident-engine

Worker Profile: medium

## Acceptance criteria

- [ ] speech-mac say positional/file/stdin, valid mono24k PCM16 WAV, completed-WAV default playback, help/version, installed named/default and explicit compatible local voice artifacts
- [ ] Validate en/seed/--duration-delta with native meaning and only tested supported controls
- [ ] Actionable failures for old voice names, --ipa and --speed/-s
- [ ] no silent reinterpretation
- [ ] No cloud voice creation, recording encoder or runtime network requirement
- [ ] Preserve temp permissions/cleanup
- [ ] tests stub playback or write audition WAVs, no unsolicited speech
- [ ] Legacy remains separately checkable until31, but no final compatibility alias
- [ ] Complete the approved slice checks and keep unavailable model/device checks explicit.

Status: ready-for-agent

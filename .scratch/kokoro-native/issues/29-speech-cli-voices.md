# 29: Ship speech-mac tasks and compatible voices

## Parent

None (Requirement Source).

## Context

- Technical source: `.scratch/kokoro-native/spec.md` and its pinned primary sources.
- Background: `.scratch/kokoro-native/spec.md`.
- User clarification dated 2026-10-25: subsequent implementation, fixes and ticket reviews use implement-lite / gpt-6.1-sol / low, overriding the repository model/effort defaults. Final whole-branch review uses gpt-6.1-sol / high. Actual earlier Luna/low execution records remain historical evidence.

## Footprint

CLI/main, WAV/playback adapters, voice catalog/schema/compatibility resolution, engine consumers, speech-mac build product, CLI tests/docs.

## What to build

speech-mac say positional/file/stdin, valid mono24k PCM16 WAV, completed-WAV default playback, help/version, installed named/default and explicit compatible local voice artifacts. Validate en/seed/--duration-delta with native meaning and only tested supported controls. Actionable failures for old voice names, --ipa and --speed/-s; no silent reinterpretation. No cloud voice creation, recording encoder or runtime network requirement. Preserve temp permissions/cleanup; tests stub playback or write audition WAVs, no unsolicited speech. Legacy remains separately checkable until31, but no final compatibility alias.

## Blocked by

28-resident-engine

Worker Profile: medium; implement-lite / gpt-6.1-sol / low (user clarification 2026-10-25)

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

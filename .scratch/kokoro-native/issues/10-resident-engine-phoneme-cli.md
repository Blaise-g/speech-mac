# 10: Reuse loaded weights and provide the native `say` CLI

## Context

Technical source: `.scratch/kokoro-native/spec.md` (resident engine and CLI acceptance) and [approved Run Plan](../run-plan.md).

Background: [Kokoro native inference spec](../spec.md).

Worker Profile: implement-lite · xhigh · gpt-6-luna (override)

## What to build

A reusable C engine keeps model weights resident across requests. The `kokoro say` command accepts prepared phonemes and exposes voice, speed, WAV output, and basic macOS playback options.

## Acceptance criteria

- [x] Load model and voice data once into an engine and synthesize repeated requests through the same instance.
- [x] Support the initial `af_heart` voice and the requested English `am_adam` voice; pin the immutable source revision/hash and validate compatibility of each voice artifact. `--list-voices` lists installed supported voices only.
- [x] `kokoro say --ipa TEXT` accepts prepared IPA/phoneme input, with `-v/--voice` and `-s/--speed`; reject unknown voices and speeds that are non-finite or outside the documented 0.5–2 range.
- [x] `-o/--output PATH` writes a mono 24 kHz WAV with correct headers and sample counts. Without `-o`, play the completed WAV through macOS `afplay` and remove the securely created temporary file after playback.
- [x] Provide `--help`, `--version`, and `--list-voices`; document defaults and model/voice artifact inputs.
- [x] Invoke `afplay` without a shell and pass the generated path as an argument; report missing player or playback errors clearly.

## Tests

- Exercise repeated engine requests and compare results to the direct inference seam.
- Run CLI checks for prepared phonemes, voice/speed selection, WAV metadata, help/version/list output, invalid options, and missing/incompatible model or voice artifacts.
- Use an injectable or stub player to verify playback selection, argument handling, and temporary-file cleanup without requiring audible output in the automated suite.

## Blocked by

- `09-phoneme-to-audio-correctness` — complete native phoneme inference.

**Status:** done

## Verification — 2026-10-12

Commit `65873d88f9f24f97f826966e2204f35e4293a9da`. Coordinator verified clean tree/trailer and reran engine/CLI focused tests: repeated resident requests, voice selection, WAV/player cleanup pass. Independently invoked am_adam IPA speed1.3 output and inspected valid mono24kPCM16 WAV. Worker full make check with model/voice artifacts and locked uv suite (20) passed. Single review fixed cross-voice comparison bounds and added CLI artifact/voice coverage. am_adam compact style artifact shares model pack; no daemon. English/file/stdin remains11.

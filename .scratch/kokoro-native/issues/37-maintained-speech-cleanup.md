#37: Remove obsolete model leftovers and clarify the C implementation

## Context and approved Run Plan

User2026-10-26: “make sure to also clean up and remove all deprecated kokoro references, or other old things no longer needed while making the readme clear thats its a clean c implementaion based on antirez work $unslop”; “and dapt the readme as well in its style!”

The Coordinator approves this bounded cleanup from merged3434f73aeada7e1818feb11cf4815b773a21188b on chore/maintained-speech-cleanup. The explicit cleanup request supersedes earlier retention of obsolete model artifacts in the maintained checkout; Git history preserves them. Historical tracker records remain committed at the configured path. Current Phonon numerical/measurement/provenance evidence remains byte-exact. No inference algorithm, model configuration, compatibility engine, performance claim, code-license grant or external cache change is authorized.

One sequential implementation worker uses implement-lite / GPT-6.1 Sol / low and one independent low reviewer. Root alone maintains tracker and archival links. Final whole-branch review uses GPT-6.1 Sol / high. Capacity is four threads, with implementation peak root + worker + reviewer = three; no concurrent worker mutates the tree. Use the existing checkout, no new worktree. Publish one reviewed cleanup PR, wait for successful check, merge under the user's standing merge instruction and clean up the merged branch.

## Footprint

Worker: README.md, docs/cli-details.md, docs/remote-setup.md, docs/adr/0002-phonon-native-runtime.md, reference/README.md, benchmarks/README.md, CODING_STANDARDS.md; src/speech_cli.c, tests/test_speech_cli.c and reference/phonon/tests/test_maintained_surface.py. Read domain-modeling for ADR edits and unslop for prose. No other source/test/dependency/fixture/vendor/license/lock edits without a Worker Question.

Delete exactly these tracked files; directory entries below mean all tracked files under that directory, not ignored local files:

- docs/streaming-plan.md
- docs/adr/0001-native-inference.md
- artifacts/09-af-heart-hello-controlled.wav
- artifacts/09-af-heart-hello-native.wav
- artifacts/09-af-heart-hello-provenance.md
- artifacts/11-af-heart-english-provenance.md
- artifacts/11-af-heart-english.txt
- artifacts/11-af-heart-english.wav
- artifacts/12-performance-baseline.json
- artifacts/12-performance-baseline.md
- artifacts/13-optimization-investigation.md
- artifacts/15-coreml-comparison-20261012T113233Z/
- artifacts/15-coreml-comparison-20261012T113654Z/
- artifacts/18-streaming-benchmark-20261012T161920Z/
- artifacts/18-streaming-benchmark-postreview-build-20261012T171907Z.json

Root: this ticket, active spec, tracker archival note and Markdown links into removed files, using immutable3434f73 Git targets. No historical facts or raw execution records are relabelled. The configured tracker and local absolute checkout retain their historical names. Negative regression guards may name forbidden old files/symbols. Third-party notices and current Phonon artifacts32–35 remain intact. Ignored builds/models/voices, original shared assets, user backup and external installed executables are outside cleanup.

## Acceptance criteria

- [ ] Maintained C CLI and usage docs contain no obsolete product-specific support or migration diagnostics. Unsupported options still fail with exit2 before asset access; invalid voices still fail with exit1. The -- literal-text boundary and successful synthesis remain unchanged.
- [ ] Obsolete streaming/initial-model design and tracked09–18 result families are removed; their historical tracker links resolve to immutable Git history. Current Phonon evidence, pins, fixtures, budgets, locks and notices remain byte-exact.
- [ ] README uses the direct antirez presentation pattern: native C/model/dependency introduction, early setup and runnable examples, concise features/usage/build/design/limits. Explicitly credit Salvatore Sanfilippo (antirez)'s qwen-asr and voxtral.c as architectural references, without claiming he authored Phonon or supplied copied code. Preserve actual setup/auth/voice/control/resource/license/testing facts, moving detail to linked docs where useful.
- [ ] Maintained developer/reference/benchmark/remote/architecture docs describe the current implementation and commands, without stale migration instructions or old model performance claims. Historical evidence remains identifiable, never rewritten to impersonate a new result.
- [ ] Defect-revealing focused CLI and maintained-reference checks pass, with red proof for changed diagnostic contracts. Independent changed-path review precedes commit; make format then make check pass. Final Sol/high review and PR/main CI pass; repository stays public and merged branches/worktrees are cleaned up.

Worker Profile: medium; implement-lite / gpt-6.1-sol / low.

Status: ready-for-agent

## Audit

Read-only GPT-6.1 Sol/low audit: /private/tmp/speech-mac-37-cleanup-audit.md. Root inspected official antirez qwen-asr and voxtral.c READMEs; both introduce the C model pipeline and dependencies directly, put quickstart commands early, then explain features/usage/build/limits. Their ASR algorithms and platform/backend claims do not transfer to this TTS runtime. Current CODING_STANDARDS already establishes the structural influence. No evidence establishes copied antirez source. Existing PyTorch/SLEEF-derived numerical work and notices must remain attributed.

Independent plan review found that the exact deletion list depended on the temporary audit. Applied: the authoritative Footprint now lists the exact files/directories above. No correctness/spec finding; implementation evidence remains pending.

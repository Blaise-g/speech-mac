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

Root: this ticket, active spec, tracker archival note and Markdown links into removed files, using immutable3434f73 Git targets. No historical facts or raw execution records are relabelled. The configured tracker and local absolute checkout retain their historical names. Negative regression guards may name forbidden old files/symbols. Third-party notices and current Phonon artifacts from tickets32 through35 remain intact. Ignored builds/models/voices, original shared assets, user backup and external installed executables are outside cleanup.

## Acceptance criteria

- [x] Maintained C CLI and usage docs contain no obsolete product-specific support or migration diagnostics. Unsupported options still fail with exit2 before asset access; invalid voices still fail with exit1. The -- literal-text boundary and successful synthesis remain unchanged.
- [x] Obsolete streaming/initial-model design and tracked result families from tickets09 through18 are removed; their historical tracker links resolve to immutable Git history. Current Phonon evidence, pins, fixtures, budgets, locks and notices remain byte-exact.
- [x] README uses the direct antirez presentation pattern: native C/model/dependency introduction, early setup and runnable examples, concise features/usage/build/design/limits. Explicitly credit Salvatore Sanfilippo (antirez)'s qwen-asr and voxtral.c as architectural references, without claiming he authored Phonon or supplied copied code. Preserve actual setup/auth/voice/control/resource/license/testing facts, moving detail to linked docs where useful.
- [x] Maintained developer/reference/benchmark/remote/architecture docs describe the current implementation and commands, without stale migration instructions or old model performance claims. Historical evidence remains identifiable, never rewritten to impersonate a new result.
- [x] Defect-revealing focused CLI and maintained-reference checks pass, with red proof for changed diagnostic contracts. Independent changed-path review precedes commit; make format then make check pass. Final Sol/high review and PR/main CI pass; repository stays public and merged branches/worktrees are cleaned up.

Worker Profile: medium; implement-lite / gpt-6.1-sol / low.

Status: done

## Audit

Read-only GPT-6.1 Sol/low audit: /private/tmp/speech-mac-37-cleanup-audit.md. Root inspected official antirez qwen-asr and voxtral.c READMEs; both introduce the C model pipeline and dependencies directly, put quickstart commands early, then explain features/usage/build/limits. Their ASR algorithms and platform/backend claims do not transfer to this TTS runtime. Current CODING_STANDARDS already establishes the structural influence. No evidence establishes copied antirez source. Existing PyTorch/SLEEF-derived numerical work and notices must remain attributed.

Independent plan review found that the exact deletion list depended on the temporary audit. Applied: the authoritative Footprint now lists the exact files/directories above. No correctness/spec finding; implementation evidence remains pending.

## Source delivered, 2026-10-26

Worker commit b4c0609f8ffda10b24c60fce9094c9b8c10f6fb9 starts from reviewed plan c7aea897be74cf75c7fd49f8c5bc89512c4568b1. Root collected the complete atomic report from the run Briefs directory. GPT-6.1 Sol/low implement-lite and one independent Sol/low six-lens reviewer found no issues. Exact authorized footprint is ten updated source/test/doc files and21 obsolete tracked deletions. No numerical source, current evidence, pin, fixture, budget, lock or legal notice changed.

Worker obtained red proof at the CLI boundary for unsupported-option rejection before asset access and a product-neutral missing-voice error. Focused C and maintained-reference tests passed, followed by make format then make check and the staged hook. The literal-text boundary and rejection exit codes are preserved. Seven existing analyzer warnings remain unchanged; optional actual projected-voice/device fixtures were not selected. No audio was played. Root historical-link verification, actual default WAV smoke, final Sol/high review and publication checks follow before completion.

Root independently verified the exact21-file deletion set, Ticket trailer and unchanged protected index/content. Current runtime C/header files have no old product-name match. Four unsupported options fail with exit2 before asset errors, one missing voice fails with exit1 and a neutral diagnostic, and the default catalog lists default. Actual default-path synthesis produces mono PCM16/24k/76800 samples, byte-identical to the accepted WAV (SHA256 e1653a11cf7d39baf4a7b394ab727e995d802f9a74585bec50e1c28e0084faf2), without playback. Root make format then full make check passed, including the maintained CLI and bootstrap tests, with optional fixture skips explicit. Results/logs: /private/tmp/speech-mac-37-root-result.json, speech-mac-37-root-verify.log and speech-mac-37-root-check.log.

Root redirected12 historical Markdown links to exact pre-cleanup3434f73 files and added the [historical-record boundary](../history.md). Earlier facts remain unchanged. The tracker name, raw execution paths and negative regression guards intentionally retain their provenance; obsolete runtime code, user-facing product diagnostics and obsolete checked-in model results do not. Final Sol/high review and PR/main CI remain pending.

## Final High review triage

The independent GPT-6.1 Sol/high reviewer examined the complete branch and pending archival metadata since3434f73, across all six lenses. No executable/numerical finding or open-PR overlap; two low-severity documentation findings are accepted. Report: /private/tmp/speech-mac-37-final-high-review.report.md. Derived Footprint: reference/phonon/README.md, only its inbound Mac quickstart anchor, is added to repair the README heading change within the same requirement.

| Id | Finding | Disposition | Evidence |
| --- | --- | --- | --- |
| R1 | The reference guide still targets the renamed README setup heading. | Applied | Commit0845f0c targets README.md#quick-start; static checks and independent High refresh pass. |
| R2 | The reference overview says the project contains source assets, which are downloaded outside the clone. | Applied | Commit0845f0c says asset pins, conversion tools and checks; independent High refresh passes. |
| T1 | Cleanup PR/main CI, merge and branch cleanup. | Resolved | PR6 check40353413331 and main check40353813228 passed; source merged as2885c31 and branch deleted. |

The source fix is limited to these two documentation lines, using the standing Sol/low implementation profile and independent review. No behavior change warrants a new executable test. Final High refresh and CI follow the fixes.

## Repaired source merged

The Sol/low worker repaired R1/R2 in0845f0ca7fa734d381f39563e43d3bcb8851da00 from cleanf0be12fda91177716044bbaaed4e328a7ce1a64a. Only the two authorized documentation lines changed. Static checks, make format then make check, independent Low review and staged hook passed. The final Sol/high focused refresh independently confirmed both committed blobs and no remaining finding; the original whole-branch review remains applicable to unchanged paths. Reports: /private/tmp/speech-mac-37-high-doc-fix.report.md and speech-mac-37-final-high-refresh.report.md.

[PR6](https://github.com/Blaise-g/speech-mac/pull/6) passed check run40353413331 at0845f0ca7fa734d381f39563e43d3bcb8851da00 and merged as2885c31ce97669b7b81232c14ab7860871c761d0. Its source branch is deleted, no new worktree was created, and the checkout now holds only the main worktree. Public visibility/description remain correct; anonymous Git resolves HEAD/main to2885c31 and the public raw README returns HTTP200 with bytes identical to the checkout. The updated116-commit redacted reachable-history scan finds no leaks beyond the previously reviewed exact tokenizer-hash exclusions. Main CI and final tracker completion follow this dated record.

## Completion verified, 2026-10-26

Merged-main [CI40353813228](https://github.com/Blaise-g/speech-mac/actions/runs/40353813228) completed successfully at2885c31ce97669b7b81232c14ab7860871c761d0. All acceptance criteria are verified; R1/R2 are applied and independently cleared by the High refresh, and T1 is resolved. The final tracker-only record starts from that merged commit on docs/cleanup-verified. Root make format then make check passed again for this closeout (/private/tmp/speech-mac-37-closeout-check.log). No further executable or model verification is claimed by this metadata update.

The optional code-license choice remains unanswered, so no grant was added. Listening/acoustic onset remain unverified, unchanged from source acceptance. Seven existing analyzer warnings remain in unchanged files; they are not suppressed or presented as cleanup regressions. No user models, caches, backups or external installed commands were removed.

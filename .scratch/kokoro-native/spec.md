# speech-mac: native Phonon migration

Status: in-progress

This is the active Requirement Source for the Phonon migration. It supersedes the Kokoro product and architecture requirements. The previous source is preserved in `kokoro-spec-archive.md`; tickets 00–18 and their evidence remain historical records, including their open criteria. They are not blockers for this migration.

## Run preferences

- Use the orchestrate workflow and present a concrete Run Plan for approval before implementation.
- Rename the project, CLI, and GitHub repository to `speech-mac`.
- Start with the largest published Phonon size, the linked English 40M model. Additional language variants are deferred.
- Remove Kokoro support. Retain the current user-facing tasks with Phonon-specific options replacing Kokoro-specific options.
- Use readable, fast native C in the spirit of antirez / Salvatore Sanfilippo's inference projects, optimized for Apple Silicon macOS.
- Accept named and local voice selection, language controls where supported, a seed, and the model's native speaking-rate control.
- The user can supply model downloads or a locally saved Hugging Face access token as needed.
- On 2026-10-25 the user approved the complete Run Plan for tickets 19–35, including the final GitHub rename, branch push, and one pull request to main.
- Use `implement-lite` with low effort for implementation and fixes. The user clarified that the intended model is **GPT-6.1 Sol**, not GPT-6 Luna; subsequent implementation, fix, and ticket review workers use `gpt-6.1-sol` / `low`.
- The final whole-branch review uses **GPT-6.1 Sol / high**, including review of code already implemented before this clarification.
- On 2026-10-26 the user asked the assistant to run the prepared timing check and inspect the necessary results, rather than hand execution back to the user.

## Problem Statement

The repository is tied to Kokoro, but the user wants the same local speech-generation workflows using Phonon 40M English. The name and maintained interfaces should describe a macOS speech tool. The replacement must produce correct speech and use the Mac efficiently, with measurements supporting optimization claims.

## Decisions

- Use `speech-mac` for the maintained project and command. A model-specific name was rejected because the user wants a general project identity. General naming does not require a plugin system or speculative support for other models.
- Use `gradium/phonon-40m-en` initially. The 31M model was rejected because the user requested the largest version. Multilingual and other single-language checkpoints are outside this first migration.
- Replace the Kokoro backend completely by the end of the run. Do not ship a compatibility backend. Preserve useful platform and I/O behavior when it passes the new requirements; do not preserve obsolete model code merely for reuse.
- Keep shipped inference native C17 on Apple Silicon, with explicit ownership and narrow numerical/platform interfaces. Python is permitted for development reference checks and conversion. Wrapping Gradium's Rust runtime or using an embedded inference runtime would not meet the native-C objective. Existing standards favor Accelerate/NEON first and allow Metal only when profiling justifies it.
- Preserve positional text, UTF-8 files, standard input, completed WAV output, local playback, streaming playback, and a reusable engine with resident model weights. Generation must deliver progressive audio according to the new model rather than depend on the former sentence-completion boundary.
- Replace Kokoro's phoneme frontend, IPA vocabulary, style artifacts, voice names, and speed semantics with the target model's text, voice, and generation semantics. Reject obsolete options clearly and document migration. Do not advertise unsupported controls or silently reinterpret the old speed scale.
- Support the default voice, compatible installed named voices, and compatible local voice artifacts. Voice cloning/design through a cloud service and direct cloning from recordings are outside scope.
- Measure correctness before optimizing. Preserve documented numerical budgets, replayable generation inputs, and streaming/batch equivalence where applicable. Report first PCM readiness, playback events, and acoustic onset separately.
- Keep model assets ignored and independently licensed. Access is needed for real-model checks. A browser download supplied by the user is acceptable; do not substitute 31M or mark skipped checks as passing 40M validation.

## Scope

- Replace model asset/reference tooling, inference, frontend, and model-specific tests for Phonon 40M English.
- Deliver the current CLI and resident-engine tasks under the new identity, including voices, supported generation controls, error handling, streaming, cancellation, and bounded buffering.
- Update build/check tooling, documentation, glossary, architecture records, tracker guidance, and maintained examples for the migration.
- Establish reproducible correctness and Apple Silicon performance results, then apply and measure useful optimizations.
- Rename the GitHub repository as a separately sequenced administrative step, preserving history and verifying remote/CI behavior. The Run Plan must name the exact action and timing.
- Preserve historical tracker/evidence records with their historical identity. Remove Kokoro from supported runtime paths, dependencies, commands, and maintained setup instructions.

## Out of Scope

Other model families or platforms; additional Phonon languages in this run; model training; hosted speech services; voice cloning/design APIs; a GUI; speculative abstraction frameworks; merging and deployment.

## Acceptance

- A documented build produces `speech-mac` on Apple Silicon. Inference and the ordinary text frontend require no Python, Rust, eSpeak, Core ML, or embedded model runtime at execution time.
- The pinned 40M English model, tokenizer, configuration, and default voice are verified and reproducible. Manually supplied artifacts are checked against authoritative provenance before use. Required license notices and access instructions are documented.
- Ordinary English text works from positional input, a UTF-8 file, and standard input. WAV output is valid mono 24 kHz audio. Default playback and progressive streaming work; cancellation, long input, malformed input, and resource cleanup have meaningful checks.
- Voice discovery/selection and local compatible voice files work. Supported generation options are documented and validated. Removed Kokoro-specific options fail with actionable guidance.
- Repeated requests reuse loaded weights, reset request state correctly, and do not leak generation state between voices or requests. Streaming buffers and model context have explicit bounds or limits.
- Reference comparisons cover deterministic inputs, meaningful intermediate outputs, and final audio. Random inputs can be replayed rather than assuming different runtimes share a PRNG. Tolerances and stop/flush semantics are explicit. Listening evidence is distinguished from numerical checks and user listening acceptance.
- Benchmarks identify the Mac, model/config/voice revisions, source/build, generation settings, sample counts, raw timings, cold/warm state, first PCM readiness, throughput/RTF, and memory. Playback timing is kept distinct from synthesized PCM availability and acoustic onset.
- Optimization is measured against a correct baseline on the same host/workload, with unchanged correctness budgets. Only gains beyond observed variation are claimed; unsuccessful attempts and remaining bottlenecks are reported candidly.
- Run `make format` then `make check`, relevant locked-uv reference checks, independent changed-path review before commits, and whole-branch review. Missing model/device checks remain explicit blockers or open acceptance criteria.
- Maintained code/build/reference dependencies and setup no longer support Kokoro. Project/CLI identity is consistent; the planned GitHub rename and remote/CI checks are verified if authorized in the Run Plan. Historical records remain attributable to their original runs.

## Glossary and ADRs

- Read `GLOSSARY.md`, `CODING_STANDARDS.md`, and `docs/adr/0001-native-inference.md`. Replace their Kokoro-specific active definitions as part of the migration, preserving historical decision context.
- The new frontend normalizes and tokenizes ordinary text. A voice conditions generation using a compatible local artifact. A resident engine holds model weights across requests; request-specific generation and decoder state must be reset or explicitly retained by the API contract.
- Primary technical sources: https://huggingface.co/gradium/phonon-40m-en and https://github.com/gradium-ai/xn-ptts. Architectural references: https://github.com/antirez/qwen-asr and https://github.com/antirez/voxtral.c.
- Planning-time model revision: `cc738b7e4d182c2ba1bafb8062aa4846af000637`. Verify and pin the artifacts actually used; do not assume current-main browser downloads match without verification.
- Shared user download location: `/Volumes/Code/Personal/speech-mac-models/phonon-40m-en/`. Keep it outside worktrees so cleanup cannot remove it.

## Planning evidence

- Base: `67e66a3486c68aa2ad2c0ce33b0e01eda8d12669`, confirmed against GitHub's default branch through the API on 2026-10-25. No open pull requests were returned at entry.
- Worktree-parent and Git-common-directory write probes passed. Launcher preflight passed the base hook, with ignored model-backed checks explicitly skipped in its staged snapshot.
- Pinned 40M access initially returned HTTP 403, then returned HTTP 200 after the user saved a replacement local token on 2026-10-25. Eight required artifacts were downloaded and verified against revision-specific authoritative sizes and LFS SHA-256 or Git blob IDs. The manifest is `/Volumes/Code/Personal/speech-mac-models/phonon-40m-en/local-verification.json`. Real-model inference and parity have not yet been executed.
- The initial cleanup fetch timed out. Cleanup succeeded using the exact API-confirmed base SHA, removing two clean merged historical worktrees and their branches. Main-checkout ignored cache symlinks into an old worktree were found dangling after cleanup. The locked reference environment and hash-verified original assets/native packs were rebuilt under main's ignored `.local-cache/`; both links now point there. Main's tracked tree remains unchanged.
- A per-command `gh auth git-credential` helper made `git ls-remote` succeed at the exact base without changing global configuration. GitHub returned HTTP 404 for the proposed `Blaise-g/speech-mac` target at preflight; recheck before renaming.

## Implementation Decisions

1. Pin source/runtime/model/artifact/voice revisions and hashes above. Actual config: nine-layer Flow-LM512/heads8/FFN2048, flow-depth6/width512/latent32, RMS-time norm, erf-GELU, temp.8/one LSD step/EOS4; Mimi float two-layer512/heads8/context250, ratios6,5,4,12.5Hz→24k. Speaker encoder unnecessary for precomputed local voices; required speaker projection and compatibility metadata `ba7a5a45@300` retained. Development oracle may execute pinned Rust; shipped runtime may not.
2. Native C17 boundaries: checked packed assets; English normalization+Unigram frontend; numerical kernels; voice/condition priming; cached Flow-LM; replayable flow sampler; bounded Mimi; one resident request machine; CLI/WAV/platform playback. Reject speculative plugin framework, embedded Rust/Python/eSpeak/CoreML and sentence-complete streaming.
3. Preserve published Q8 transformer encodings plus required floats; version/hash/layout explicit. Correct packed CPU path first, Accelerate/NEON behind narrow seams. No unsupported Q4, experimental Mimi Q8 or Metal. Read actual downloaded tensor types before implementation promises.
4. Match pinned default English normalization/tokenizer semantics natively. Ordered bounded fitting preserves input or errors. Validate en, seed and duration_delta; do not advertise extra speculative controls. Old Kokoro speed scale is removed, not reinterpreted.
5. Installed named/default/local compatible voices are local inputs with manifest/catalog/metadata validation. No cloud service, audio-recording encoder or runtime network requirement. All mutable generation/decoder/DSP state resets between requests; weights stay resident. Public buffer ownership, callbacks, cancellation, drain and limits are explicit.
6. Reference gate19 identifies trained Mimi context semantics and independent pinned correct oracle (or minimal provenance-bearing patch). Freeze decoder budgets only after reconciliation. Include >250-position/uneven-batch fixtures. Never use buggy pinned output silently as truth or loosen tolerances to hide context/tail defects. Record intentional upstream differences.
7. Expand Phonon alongside legacy only for independently checkable intermediate commits, then migrate tasks and delete maintained legacy support. Preserve historical identity/evidence. speech-mac API/CLI/build identity is maintained; final separate remote rename uses exact action above, with no local root move.
8. Implementation Workers: implement-lite/gpt-6.1-sol/low local override; record starting SHA. Defect-revealing tests, format then check and relevant locked-uv reference checks, independent changed-path review before Ticket-trailed commits, whole-branch independent review at acceptance. No behavior means explicitly empty test suite.


9. User clarification on 2026-10-25: subsequent implementation, fix and ticket review workers use `implement-lite` / `gpt-6.1-sol` / `low`; final whole-branch review uses `gpt-6.1-sol` / `high`. The Coordinator initially applied low effort to the repository's Luna model. Tickets19–23 and the ticket24 checkpoint were implemented and reviewed with Luna/low. Preserve those actual execution records; the final Sol/high review covers that code too. These local overrides do not change global configuration.

## Testing Decisions

1. Fixture manifest records config/model/voice/reference hashes/revisions, normalized text/IDs/prefix conditions, exact random draws, backbone/flow/raw EOS/latents, raw decoder/postprocessed PCM and sample counts. Separate fixed stage float/Q8 budgets based on repeated oracle behavior. Shape/mask/length/finite/error invariants supplement highest useful numerical seam. Equal seeds are not a cross-runtime PRNG contract.
2. Exact normalizer/tokenizer corpus: English structures, Unicode/byte fallback/whitespace/unknowns, prompt/chunk fitting, source parity, empty/malformed inputs; prove runtime needs no child interpreter/tokenizer. Test actual algorithms, never hardcoded fixture dispatch.
3. Transformer prefill/incremental, Mimi framewise/whole/uneven partitions cross >250 context positions and cache wrap. Repeat requests, voice alternation, text boundaries, budgets and resets. Same replayed inputs for stream and batch; fixed waveform tolerance plus exact count/stop/tail semantics. Distinguish DSP drain from EOS tail; no added silence to conceal dropped output.
4. Q8 loaded blocks and matvec/matmul compared to published tensor reference and complete graph. Malformed/truncated/unsupported formats rejected. Quantization error against float source separated from port error. Synthetic fixtures run CI without gated weights; full40M checks must execute for acceptance, never count skips as passing.
5. Engine/CLI/fake playback cover every task, compatibility/controls, bounded ownership, cancellation during numerical work/buffer waits, ordering/backpressure/drain/errors. Real AudioQueue checks queue-muted only; fake consumer labelled; acoustic onset unknown without measurement. Generate audition files, keep user listening status separate.
6. Benchmarks record exact host/build/model/quant/voice/settings, repeats/counts/cold/warm, firstPCM/synthesis/playback scopes and raw timings. Paired same-host optimizations report spread and rejected attempts. No cache purge, system-volume changes or unsolicited speech; historical Kokoro measurements are not relabelled.

## Resolved numerical issue

- **E24-1 resolved without an acceptance change.** Source commit `94e1fefa63a896a31276eaff7f715bd3c60d0449`, implement-lite / GPT-6.1 Sol / low with independent Sol/low review, matches all frozen autonomous stages and independently decoded PCM exactly. The Coordinator verified real normal/sanitizer/optimized gates, exact noise/EOS/counts, earlier pack/transformer parity and pinned source/license provenance. The repair uses source-shaped batched prefix/voice projections, source reduction/order and a private compiled NEON math boundary; ordinary builds/inference need no external SLEEF runtime, generator, CMake or network. Frozen fixtures/budgets/settings and ticket20 pack/kernel source remain unchanged. Formal repair1 counts against23 and24, each1 of2; the withdrawn protocol-change proposal was never applied. [Ticket24](issues/24-flow-sampler.md) records the full evidence, derived Footprint approvals, remaining warnings and handoff. No open Escalation remains. Ticket25 may start after this verification record is committed. Final whole-branch review still uses GPT-6.1 Sol / high and has not occurred.

## Resident engine and further numerical verification — 2026-10-25

Ticket 28 is verified at `b3b5b6097f33351e77991101a10a1cfd4dd65f1c`: resident native weights, one progressive request machine and bounded batch collection, with actual frontend→voice→Flow→Mimi comparisons. Long requests exposed two shared rounding gaps corrected under the existing source-equivalence contract: scalar LayerNorm M2 FMA order and generic one-key accumulating SDPA blocks. Frozen budgets and fixtures were unchanged. The Coordinator independently passed regenerated references, 24 exact attention shapes, actual five-case ASAN, all preserved native gates and canonical format→check with real fixtures.

This preapproved shared correction consumes repair 2 of 2 for each 23 and 24; 28 remains attempt 1/formal 0. Details and all Worker Question/Footprint answers are in [ticket 28](issues/28-resident-engine.md). CLI/voices 29 through administration 35 remain pending. Final whole-branch GPT-6.1 Sol / high review has not occurred.

## CLI and compatible voices verified — 2026-10-25

Ticket 29 is verified at `2f994e469a86e64f78f988850742ea76faa6253e`: native speech-mac input/output tasks, completed-WAV playback adapter, local compatible voices and native controls. Raw voices are projected once; bound emb/audio_prompt artifacts are used directly. The reproducible independent projected-voice exporter and actual CLI/ASAN/engine/native/full-suite checks passed, including byte-identical built-executable WAV output. Source assets, numerical kernels and frozen budgets were unchanged; no sound or listening acceptance was claimed. [Ticket 29](issues/29-speech-cli-voices.md) records all checks, approved test-tool Footprint and resolved review findings. Tickets 30–35 and the final whole-branch Sol/high review remain pending.

## Streaming implementation saved; acceptance blocked — 2026-10-25

Ticket30 implementation is committed at `8d09e66cbbff86eacdde5617e7c75d63751a784f`. Actual model/fake-device stream equality, cancellation/races, sanitizer/TSAN, preserved numerical and canonical checks pass. Successful native AudioQueue start/reclamation/drain remains unverified: new, preserved and standalone queue-muted probes fail or hang at SDK start on this host. No device failure or skip is credited as passing. A comments-only ownership clarification is preapproved as ticket30 repair1of2. [Ticket30](issues/30-progressive-playback.md) records exact evidence and the pending bounded Terminal check. Tickets31–35 and final whole-branch Sol/high review remain held; no acceptance criterion was waived.

The ownership clarification is now applied at `8390afcffcb26848b23f82b4ae8d7b0c63abe589`; independent review, format/check, staged hook and the Coordinator's exact noncomment-byte comparison pass. Formal ticket30 repair1of2 is applied without changing executable behavior. E30-1 remains open: the Coordinator's ordinary and PTY silent SDK checks both fail at AudioQueueStart -66681. A reviewed private helper, `sh /private/tmp/speech-mac-30-terminal-check.sh`, is ready for the user's Terminal and records its bounded queue-muted result in `.build/phonon-stream-terminal-device.log`. A successful actual start/reclamation/drain or an explicit user-approved plan change is required before dependent tickets launch. No user waiver or listening/device success has been recorded.

## Streaming acceptance verified; migration resumed — 2026-10-26

E30-1 is resolved without an acceptance change. The user ran the prepared silent integration from Terminal; the Coordinator read its saved log and verified `TERMINAL_DEVICE_EXIT=0`, `muted AudioQueue result=0` and actual queue-volume-zero start/reclamation/drain success for30720 samples. Ticket30 is verified at `15139c5c03fb1d534602fc6a0e7959ee1ef9817d` with unchanged implementation/header-repair SHAs and existing independent software/numerical evidence. [Ticket30](issues/30-progressive-playback.md) records the log hash and limits: fake timings remain fake, acoustic onset/listening are unmeasured, prior tool-context failures are preserved and no system settings changed. No Escalation remains open. Ticket31 may start after this record is committed; tickets31–35 and final GPT-6.1 Sol/high whole-branch review remain pending.

## Maintained Kokoro support removed — 2026-10-26

Ticket 31 is verified at `43eb085d6cffbaa8287dc63d30ae323b66b05917`: the maintained runtime/build/APIs/docs are speech-mac-only, legacy graph/frontend/artifacts/reference dependencies and active benchmark runners are removed, and historical evidence/licenses are retained. The Coordinator applied the root-owned tracker-guidance update after independent source verification. Fresh build/removal/dependency audits, actual engine/stream sanitizers, full real-fixture checks, locked reference/native replay and independent six-lens review passed without changing frozen budgets or numerical behavior. [Ticket31](issues/31-remove-kokoro.md) records source/report recovery, naming decisions, all evidence and explicit device/listening limits. No Escalation is open. Ticket 32 may begin after this record is committed; measurements, optimization, integration acceptance, GitHub rename and final GPT-6.1 Sol/high whole-branch review remain pending.

## Baseline committed; current queue timing pending — 2026-10-26

Ticket32 source `0f7ea8eab3e52f3b388ed810de0a3f71218935e9` and evidence `054355f35e2645f6bc53115be14114c71d5cbc5a` deliver repeated real 40M fresh/resident baselines, source/provenance/CPU/RSS/stage profiles and unplayed audition WAVs. Worker and Coordinator software/reference/frozen replay/full-fixture checks pass; repeated normal/profile F32 output is identical. Shared linear expansion is a measured candidate for ticket33; no optimization is claimed. E32-1 remains open: the current tool-context queue-muted timing probe fails at AudioQueueStart -66681. The reviewed helper `sh /private/tmp/speech-mac-32-terminal-timing.sh` collects the required current raw events in the user's Terminal during an idle window. [Ticket32](issues/32-phonon-baseline.md) records exact evidence, validation and limits. Earlier ticket30 functional success is not reused as ticket32 timing. Ticket32 and dependent tickets33–35 remain held until that evidence or an explicit user-approved plan change; final GPT-6.1 Sol/high whole-branch review remains pending. Acoustic onset/listening are not established.

## Current playback timing verified; evidence integration resumed — 2026-10-26

E32-1 is resolved by an assistant-run PTY execution of the reviewed silent helper:76800 samples/40 frames/40 reclaims, all success exits and validators zero. First PCM/submission/running observation were0.057648/0.057729/0.462776s; drain3.761503s, with queue volume zero and acoustic onset unknown. [Ticket32](issues/32-phonon-baseline.md) records raw event boundaries, hashes, actual execution context and preserved earlier failures. No cause is claimed for the earlier failures; no system audio settings or numerical contracts changed. The remaining32 work is reviewed artifact integration and completion writeback;33–35 and final GPT-6.1 Sol/high review remain pending. No user listening or acoustic acceptance is inferred.

## Baseline complete — 2026-10-26

Ticket32 is verified at `8c1458bc1517f0c081e58c84237fa5416f710aa1`: repeated real40M release baselines, distinct stage/operator profiles, audited current real queue timing and unplayed audition artifacts are committed and independently verified. Original compute data and executable source are unchanged by the final timing integration. [Ticket32](issues/32-phonon-baseline.md) records all gates, hashes, actual assistant PTY context and preserved failures. No Escalation remains open. Ticket33 may launch after this record is committed to improve one measured CPU surface against the fixed baseline/budgets. Listening/acoustic limits remain explicit;33–35 and final GPT-6.1 Sol/high whole-branch review are pending.

## Measured CPU improvement verified — 2026-10-26

Ticket33 is verified at `9eed9617c949ba2ed97dc758255d39c6d0c1c7a7`. Immutable pack-owned exact linear materializations give observed resident paired median speedups1.448×/1.437×/1.468× on this M5, with baseline/candidate F32 identity and unchanged budgets. Adoption explicitly includes169.47MiB extra materializations,22–27ms worse fresh preparation-to-first-PCM and a tied/slower fresh structured pair. [Ticket33](issues/33-measured-cpu-optimization.md) records preserved raw data, memory/preparation costs, all tests, independent recomputation/reproduction and scope limits. No Escalation is open. Ticket34 may start after completion writeback is committed; final current acceptance, GitHub rename and GPT-6.1 Sol/high whole-branch review remain pending.

## Native migration acceptance verified — 2026-10-26

Ticket34 is verified at `13387fd0f41d79465a2cc5cc60655728643f4067`: full real40M numerical/voice/control/bounds/long/reset/cancellation/cleanup gates, fresh isolated locked setup/conversion/build, native dependency/removal audit and current assistant-run muted SDK check pass. Current audition files match the tracked baseline files byte-for-byte. [Ticket34](issues/34-migration-acceptance.md) records all commands, immutable hashes, preserved offline-cache miss, successful isolated online recovery and independent verification. No model/device gap or Escalation remains. User listening remains pending and acoustic onset unknown. After this completion record is committed,35 may execute the pre-approved GitHub rename/origin sequence; final whole-branch GPT-6.1 Sol/high review still precedes push/PR. No merge/deploy is authorized.

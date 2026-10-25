# speech-mac: native Phonon migration

Status: ready-for-agent

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

## Numerical issue under repair

- **E24-1 — 24-flow-sampler.** Reviewed checkpoint `604d244c505d7d03db4c313438e98ab32e726776` passes identical-input component checks but fails frozen autonomous condition/integration/latent/PCM targets. The Coordinator reproduced raw/final PCM normalized RMSE `0.00359485159` and max error `0.00610744208` against `0.001`/`0.002`; exact noise, EOS/tail and 61,440 samples pass. After the user's model clarification, proceed with preapproved repair follow-up 1 using implement-lite / GPT-6.1 Sol / low under the unchanged acceptance contract. The previously drafted protocol-change proposal is withdrawn pending that investigation; no user decision or tolerance change is being applied. Details and evidence are in [ticket24](issues/24-flow-sampler.md). Dependent tickets25–35 remain held. Final whole-branch review uses GPT-6.1 Sol / high. Escalate if the repair cannot resolve the failure or requires an acceptance change.

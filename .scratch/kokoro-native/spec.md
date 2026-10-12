# Native Kokoro inference for macOS

Status: ready-for-agent

## Problem Statement

Build a fast, working Kokoro text-to-speech implementation for macOS on Apple Silicon, with readable C inference in the spirit of antirez's inference projects. Prioritize getting correct speech working and measuring performance; polish can follow.

## Decisions

- Implement Kokoro-82M v1.0 inference directly in C. Embedding an existing inference runtime does not meet the agreed objective.
- Target macOS on Apple Silicon first. Additional platforms are deferred to keep the initial delivery focused.
- Permit Accelerate, Apple frameworks, and a small Objective-C bridge. Start with Accelerate; add Metal only where profiling justifies it.
- Permit Python for reference validation and weight conversion, but not as a shipped inference dependency.
- Accept ordinary English text using eSpeak NG initially. Differences from the official Misaki pronunciation frontend are acceptable; reproducing Misaki is deferred.
- Also accept prepared phonemes so model correctness can be assessed independently of pronunciation differences.
- Use one voice for initial validation and generate WAV audio through a CLI. Keep model weights resident in the reusable engine for repeated requests.
- Prioritize short replies and first-audio latency, then sustained generation speed. Establish numerical targets from measurements, not assumptions.
- Require reference comparisons and intelligible audio without obvious artifacts. Packaging and interface polish may be deferred; correctness may not.
- User specifies GPT-6 Luna at xhigh effort with implement-lite for implementation workers. Setup is one worker applying project-setup. This explicit profile overrides the orchestration skill's normal implementation-profile selection.

## Scope

Local development foundation and tracker; reproducible reference and conversion tools; complete native C model inference; English frontend and phoneme input; reusable resident-model engine; WAV CLI; correctness and listening checks; cold and warm performance measurement; targeted optimization supported by measurements.

## Out of Scope

Additional languages or platforms, broad voice coverage, polished packaging, elaborate APIs, live playback and incremental audio delivery in the first milestone, model training, remote publication, merging, and deployment.

## Acceptance

- A documented local build runs the model on Apple Silicon without Python in the runtime inference path.
- The CLI turns ordinary English text into a valid WAV using the initial supported voice.
- Direct phoneme inputs can be compared against the official reference with identical model inputs and recorded numerical tolerances.
- Reference checks cover meaningful intermediate outputs as needed to isolate errors; final audio is checked for intelligibility and artifacts.
- Repeated requests reuse loaded weights through the engine.
- Cold start, warm synthesis time, generated audio duration, and throughput are measured on an identified Mac; first-audio latency is reported with its exact output semantics.
- Measured bottlenecks guide optimization, with correctness checked after changes.
- Verification reports distinguish executed checks, empty suites, blocked checks, and unverified listening judgments.

## Glossary and ADRs

Use the foundation's glossary and decision records where present. Here, frontend means conversion from English text into phonemes; inference means conversion from phonemes, voice style, and speed into audio; resident engine means weights remain loaded across synthesis requests. No other domain decisions are implied.

## Implementation Decisions

- Implement the inference graph in readable C17, using Accelerate behind narrow numerical-operation seams when it serves the accepted Apple Silicon target. Do not embed a model runtime; Python remains development-only for reference, fixture, and weight conversion work.
- Use the official Kokoro-82M v1.0 config and checkpoint as the model contract. Pin immutable revisions and hashes in the reference/conversion work before producing fixtures. Use `af_heart` for the initial reference and add the requested `am_adam` as an English voice artifact in ticket 10; keep eSpeak NG in the English frontend boundary.
- Treat the module boundaries in tickets 02–08 as the stable backend seams. Use a versioned C-readable weight pack produced from official weights, stored outside Git, and load it into a resident engine that accepts prepared phonemes, voice style, and speed and returns mono 24 kHz audio.
- Add no Metal path or abstraction in advance. A separate ticket can be proposed only if the measured profile demonstrates a suitable bottleneck and likely benefit.

## Testing Decisions

- Before native C inference work, set per-stage and final-audio numerical budgets from repeated-reference behavior and documented float-precision rationale. Test each C module at its stable input/output seam against oracle fixtures using those fixed budgets; a miss triggers diagnosis and a fix, with budget changes requiring independent evidence and written rationale. Test shape, masking, alignment, length, error, and finite-value invariants separately where exact numeric equality is not the right assertion.
- Test the assembled engine using the highest available seam: direct phoneme-to-waveform reference comparison, controlled excitation for reproducibility, repeat requests against one resident engine, and CLI WAV inspection. Ticket 09 records audition artifacts; ticket 14 holds final human listening acceptance so pending listening does not block technical follow-on tickets.
- Run `make format` followed by `make check` and an independent review before each commit. The English frontend gets token/ID tests and CLI checks for positional text, UTF-8 file, standard input, IPA bypass, voice selection, and long input across sentence-aware chunks; benchmark and optimization tickets record the exact Mac/workload and use repeated paired measurements. Ticket 12 continues to define first-audio latency as availability of the complete in-memory waveform, with WAV write completion reported separately. Set any performance acceptance bound from observed baseline variation, never from an assumed number.

## User scope amendment — 2026-10-11

The user requests a minimal useful CLI: `kokoro say` with positional text, `-f/--file` UTF-8 text file, or stdin; `-v/--voice`, `-s/--speed`, `-o/--output`, `--ipa`, `--list-voices`, help/version. Use completed-WAV macOS playback when no output is requested. Process long input without silent truncation. Support the initial English voices af_heart and am_adam. This supersedes the original one-voice and playback-deferral scope for tickets 10–11. Streaming, daemon mode, quantization and broad voice/language coverage remain deferred at the user’s request to keep the first release simple. Reference performance/size claims are not native implementation promises.

## Whole-branch review — 2026-10-12

Reviewed base `963ce48eb1da6f6b067588b067e6d325313bcf93` through `b69274b9cfc5abc32567f39353131536822cdfa6`. Report-only review covered Correctness, Spec, Standards, Reuse, Quality and Efficiency. Generated binaries and full lock records were not visually inspected; their producers, consumers, manifests and provenance were reviewed. No configured Git remote exists, so forge overlap is not applicable.

| Id | Finding | Applied / rejected / deferred | Reason | Verified by |
| --- | --- | --- | --- | --- |
| R-1 | Public voice-style row size could overflow when multiplying by sizeof(float). | applied | Check full F32 row size before multiplication; reject crafted [2^54+1,1,256] view. | `a36e879e63d2fafff8d210346ced578e02b0589b`; coordinator parser regression passed; worker full real-model check passed. |

Review summary: one accepted correctness finding fixed, no remaining accepted findings. Earlier ticket06 helper-sharing question requires no refactor: stage weights/shapes/layouts differ and no concrete benefit justifies a larger common layer for this release. Tickets13/14 remain open as recorded.

## User follow-up — CoreML comparison and streaming plan, 2026-10-12

### Problem statement

Compare the working native C implementation with the already-installed Jud/kokoro-coreml CLI using the user's example `kokoro say "hello from the terminal"` and representative longer input. Measure speed and computational resources, preserve a baseline for future improvements, and then plan streaming. Do not assume C is faster or more efficient merely because of the implementation language.

### Decisions and scope

- Use the installed CoreML executable `/opt/homebrew/bin/kokoro` (observed version0.11.2) and the current native release binary; do not upgrade or replace the existing installation. Pin executable/source/model provenance where observable.
- Compare matched voice, speed and text, plus identical prepared IPA when both interfaces support it. Report generated durations and frontend/precision/output-format differences.
- Repeat runs and separate cold fresh-process CLI, existing resident/native-engine, and CoreML resident/daemon modes. Do not label a daemon client-only CPU/RSS sample as the entire inference service's resource usage. No OS/model cache purge.
- Record end-to-end latency, audio throughput, CPU seconds, peak process RSS and model asset sizes where measurable. State GPU/ANE attribution and memory-accounting limits instead of inventing total hardware efficiency.
- Benchmark without speaker playback; write outputs into project-owned or temporary paths. Any benchmark-started daemon must be cleaned up and original state restored; do not stop a pre-existing user daemon or change persistent launch settings.
- Retain raw results and a comparison report with repeatable commands and a run/provenance schema so future improvements can be compared without losing historical baselines.
- After the comparison, produce a minimal native streaming plan with requirements, proposed interface, chunks/playback/backpressure/cancellation, first-audio metrics and acceptance tests. Stream means playback begins after the first synthesized chunk while later chunks are still being produced. Implementation is not authorized by this request; this follow-up delivers the plan only.
- Preserve current simple CLI and accuracy constraints. A daemon, Metal backend, quantization or wider language support remains separate future work unless measurements justify a later proposal.

### Acceptance

A benchmark report distinguishes measured results from source claims, includes the exact installed version and native revision, documents cold/warm/resource scopes, and says which implementation wins for each workload without presuming a winner. Data and rerun commands support future comparison. The streaming plan follows the measured latency/resource bottlenecks and names a small first delivery plus deferred work.

The user explicitly requested this follow-up, so the existing run is extended by tickets15 comparison then16 streaming plan, with the same Luna xhigh worker profile and local branch/worktree/no remote publication. Original13 optimization and14 listening remain open rather than blocking the new comparison.

## Implementation Decisions

- Ticket 15 compares the installed `/opt/homebrew/bin/kokoro` v0.11.2 executable with the native release build from the current run. Do not upgrade or replace the installed tool. Preserve every raw comparison run under new `artifacts/15-coreml-comparison*` paths, with executable/source/model/build provenance and rerun commands; leave ticket 12 artifacts intact. Record CLI, resident-engine, and daemon modes only where their interfaces support them, and identify whether CPU/RSS samples cover the client, service, or both.
- Reuse ticket 12's native benchmark driver and reporting conventions, adding only comparison-specific runner/report code or minor native-driver instrumentation when required. Distinguish OS-reported peak RSS from interval-sampled RSS, and daemon-lifetime resource use from per-request CPU deltas. Treat unavailable GPU/ANE and associated memory attribution as an explicit measurement limit rather than a blocker.
- Match the exact short user example and a representative longer input at `af_heart`, speed `1.0`; compare identical prepared IPA only where both CLIs support that input. Keep pronunciation frontend, numeric precision, output sample format, and audio format differences visible. Report fresh-process state without purging operating-system or model caches. Do not assume a winner, or claim GPU/ANE or total hardware efficiency when the metric cannot observe it.
- Ticket 16 produces a design document only. The first proposed slice is opt-in, sentence-aware chunk playback using the existing frontend's input bounds and resident engine; one model request still produces each complete audio chunk. The design names the smallest playback API and bounded buffer supported by ticket 15 measurements, documents ownership, ordering, backpressure, cancellation and error cleanup, and leaves the existing completed-WAV default intact. A daemon, Metal, quantization, wider language support, and within-request incremental decoding require separate evidence and scope.

## Testing Decisions

- Ticket 15 adds focused Python report/parser tests using the locked `uv` environment and fixtures or stubs, not a live CoreML model or daemon. Tests cover sample/provenance preservation, mismatched workloads or measurement modes, unknown metric and process scopes, and per-workload outcomes without an assumed global winner. The device benchmark is separately reported with exact commands and retains all raw repeats.
- Ticket 16's design specifies future defect-revealing tests at the highest useful engine/playback seams: ordered chunk completeness and batch equivalence under the existing fixed numerical budget; bounded-buffer producer backpressure; cancellation during synthesis and while waiting; playback of the first chunk before later chunks finish; error cleanup; and repeated engine requests. It adds no streaming implementation or runtime test in this planning extension.

## Architectural standard reaffirmed — 2026-10-12

The user reaffirmed antirez's [qwen-asr](https://github.com/antirez/qwen-asr) and [voxtral.c](https://github.com/antirez/voxtral.c) as the continuing structural standard. Keep native C inference, explicit weight/numerical/model/CLI boundaries, a small reusable API, and narrow platform adapters. Their ASR algorithms are references for structure; Kokoro retains its own TTS graph. The standard is recorded in `CODING_STANDARDS.md` and mapped to the current modules in `docs/streaming-plan.md`. CoreML is a comparison target, not the native runtime architecture.

The user also confirmed further Apple Silicon optimization after streaming. Keep the saved comparison baseline, profile the resulting C path, and prioritize measured Accelerate/NEON opportunities. An isolated Metal path needs a suitable measured bottleneck and separate scope. Preserve the fixed accuracy budgets and report gains against observed variation; no backend or speedup is assumed.

## Follow-up per-ticket review triage — 2026-10-12

| Id | Finding | Applied / rejected / deferred | Reason | Verified by |
| --- | --- | --- | --- | --- |
| T15-R1 | Daemon startup/resource failure could escape cleanup. | applied | Cleanup checks ownership before stopping; preserve foreign processes. | `fc8f34e`; coordinator 20 comparison tests, including lifecycle failures, passed. |
| T15-R2 | External output paths failed after measurements. | applied | Reject unsupported paths before starting the benchmark. | `fc8f34e`; output-path regression passed. |
| T15-R3 | Provenance mismatch/preservation lacked regression coverage. | applied | Exercise mismatch rejection and raw-record preservation. | `fc8f34e`; provenance regressions passed. |
| T15-R4 | Mach ticks were named/converted as nanoseconds. | applied | Use host125/3 timebase, retain original capture and tick values, document correction. | `fc8f34e`; conversion tests and coordinator original-hash/report recomputation passed. |
| T15-R5 | Daemon records were duplicated under each workload. | applied | Retain one top-level daemon record. | `fc8f34e`; artifact review and report recomputation passed. |
| T15-R6 | Resident throughput used mismatched frontend timing scopes. | applied | Compare waveform-ready timing for both tools. | `fc8f34e`; scope regression and raw-report recomputation passed. |
| T16-C01 | Buffer reuse did not establish successful playback drain. | applied | Graceful stop plus running-state notification defines completion. | `715a19e`; static design review. |
| T16-C02 | Cancellation result and completion race were undefined. | applied | Proposed appended cancellation status; first terminal result wins under lock. | `715a19e`; static design review. |
| T16-C03 | Callback timing overstated audio consumption/rendering. | applied | Name reclamation separately; render/acoustic timing unavailable without independent evidence. | `715a19e`; static design review. |

No remaining accepted per-ticket findings. Ticket15 benchmark is verified; ticket16 design is verified. Original13 optimization and14 listening remain open. Streaming runtime code and device acceptance are not implemented by the design ticket.

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

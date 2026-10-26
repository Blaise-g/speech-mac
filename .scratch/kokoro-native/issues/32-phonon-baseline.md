# 32: Record Apple Silicon correctness/performance baseline

## Parent

None (Requirement Source).

## Context

- Technical source: `.scratch/kokoro-native/spec.md` and its pinned primary sources.
- Background: `.scratch/kokoro-native/spec.md`.
- User clarification dated 2026-10-25: subsequent implementation, fixes and ticket reviews use implement-lite / gpt-6.1-sol / low, overriding the repository model/effort defaults. Final whole-branch review uses gpt-6.1-sol / high. Actual earlier Luna/low execution records remain historical evidence.

## Footprint

native/uv benchmark drivers, report schema/validators/tests, raw artifacts/docs, engine/stage/playback instrumentation; exclusive local device timing/AudioQueue during measurement.

## What to build

locked-uv benchmark/report with raw repetitions of short reply/structured text/long input, cold fresh-process and warm resident modes, source/build/model/quant/voice/settings/hardware provenance. Measure first PCM, total synthesis, RTF, CPU/RSS with scopes and stage profile. Separate muted AudioQueue from fake-consumer timing; acoustic onset unavailable without independent capture. Produce audition files for user listening; no autoplay/cache purge/system volume changes.

## Blocked by

31-remove-kokoro

Worker Profile: medium; implement-lite / gpt-6.1-sol / low (user clarification 2026-10-25)

## Acceptance criteria

- [x] locked-uv benchmark/report with raw repetitions of short reply/structured text/long input, cold fresh-process and warm resident modes, source/build/model/quant/voice/settings/hardware provenance
- [x] Measure first PCM, total synthesis, RTF, CPU/RSS with scopes and stage profile
- [x] Separate muted AudioQueue from fake-consumer timing
- [x] acoustic onset unavailable without independent capture
- [x] Produce audition files for user listening
- [x] no autoplay/cache purge/system volume changes
- [x] Complete the approved slice checks and keep unavailable model/device checks explicit.

Status: done

## Implementation and partial verification — 2026-10-26

- Wave base: `649b9b61e776c0d6f1540d08ba90593f4f14a7e2`; branch `feat/phonon-migration`, existing run worktree. Worker `/root/32_phonon_baseline`, attempt 1, implement-lite / GPT-6.1 Sol / low; formal repair count 0.
- Reviewed source commit: `0f7ea8eab3e52f3b388ed810de0a3f71218935e9`. Reviewed evidence commit: `054355f35e2645f6bc53115be14114c71d5cbc5a`. Both carry Ticket32 trailers; clean tree confirmed before this record.
- Canonical atomic report: `/private/var/folders/j1/4vzqsp717kbb3gth_48lxdkc0000gn/T/orchestrate-briefs/kokoro-mac-feat-phonon-migration-f11dbf4f/32-phonon-baseline.report.md`, status blocked. Source, compute measurements, audition artifacts and software checks are complete; current successful queue timing remains required.
- Durable evidence: `artifacts/32-phonon-baseline-20261026/{raw.json,report.md,verification.md}`, correctness/fake/failed-device logs, texts and six tracked WAVs. No optimization or full ticket completion is claimed.

### Delivered measurements and profile

The locked-uv orchestrator measures five fresh-process and five resident requests for each short, structured and long workload: 30 raw repetitions, retained excluded warmups and separate profiled requests. It records committed source/binary/model/quant/voice/settings/hardware provenance, preparation, request-relative first PCM and synthesis, CPU delta, Darwin process-lifetime RSS bytes, exact counts and output duration/RTF. Fresh process is explicitly not a cold OS cache. Model/voice preparation and process launch are excluded from the request clock; raw voice projection remains once per request inside synthesis.

On this Apple M5/16 GiB/macOS 27.0.1 host, fresh/resident synthesis medians are 0.7933/0.7767s short, 3.9706/3.9615s structured and 7.8100/7.8278s long; RTF is about 0.242. The committed report preserves ranges and boundaries. Output durations are 3.2, 16.4 and 32.24s. Every native F32 hash matches across repeated fresh/resident requests and separate profile builds; paired WAVs are byte-identical, mono 24kHz PCM16. Files are ready for listening but have not been played.

Top-level profile scopes are disjoint and compiled separately from release timing. Mimi/PCM occupies about 63–65% of those profiled requests. Nested shared-linear BF16/Q8 expansion is about 37% of synthesis and SGEMM about 6%; these overlap the top-level stages and cannot be added to them or attributed entirely to Mimi. Expansion is a measured candidate for ticket33, not a demonstrated speedup or the sole bottleneck. Ordinary shipping builds compile the profiling hooks away.

### Coordinator scope answer and review findings

The Coordinator approved updating `reference/phonon/tests/test_maintained_surface.py` as the planned ticket32 integration of new benchmark tools. The exact permitted new non-Markdown benchmark files are `phonon_baseline.c` and `phonon_queue.c`. Forbidden old runners, identifiers, binaries and dependencies remain checked; a temporary obsolete runner makes the check fail before restoration. This changes no runtime or numerical contract and consumes no formal repair.

Independent GPT-6.1 Sol/low SingleReviewer applied all six lenses to source and then evidence/helper. C1 corrected repeat 20 plus resident warmup exceeding the native 20 bound (orchestration supports 5–19); C2 rejects negative/fractional counts and bytes; Q1 distinguishes nested operator scopes from disjoint stages. All accepted findings are fixed and tested; final findings:none. Driver process-exit cleanup warnings were corrected before commit. The inherited seven normalizer/SLEEF analyzer warnings remain visible. An initial reference invocation omitted the now-required shared-model environment; the correctly configured rerun passed.

### Independent verification

Worker format/check, both staged hooks, all 23 locked reference tests, actual five-case engine and fake-stream gates, exact frozen generation replay and report validation passed. `verification.md` records commands, mutation red proofs and omissions. No source model asset, vendor code, frozen fixture/budget or numerical algorithm changed.

The Coordinator independently ran `/private/tmp/speech-mac-32-coordinator-verify.py` on committed source/evidence, exit 0. It verified unchanged vendor/fixtures/budgets/lock/tracker, source identity, every local source-asset hash, raw report arithmetic/provenance and all six WAV headers/counts/hashes. Independently rebuilt normal/profile drivers generated matching committed F32 hashes in two requests per build on all three workloads (76800,393600,773760 samples). Driver ASAN/UBSAN and native count 0/21 rejection passed. All 23 reference tests, frozen noise/head/autonomous/EOS/latent/61440-sample PCM replay and `make format`→full actual-fixture `make check TEST_CFLAGS=-O2` passed. Shipping profile functions are absent; the pinned silent queue binary is unchanged, helper syntax passes and source is clean. Root logs: `/private/tmp/speech-mac-32-coordinator-{verify,asan-build,driver-asan,reference,frozen,format,check}.log`; reproduction details remain in ignored `.build/ticket32-root-probes.json`. These reproduction runs establish correctness, not replacement performance statistics.

### E32-1: actual queue timing — open

The sole bounded queue-muted tool-context probe failed at `AudioQueueStart -66681`, child exit 4 after 15.54s. `queue-agent-attempt.json` preserves source `0f7ea8e`, binary hash, build command and failure; it contains no successful timing. Actual-model/fake-consumer timings are separately labelled. The earlier ticket30 user-Terminal functional success remains historical evidence, not a new32 timing. Cause of the tool-context failure remains unknown; no service, system-volume or default-device setting changed and no diagnostic matrix was repeated.

All concrete code, evidence, independent review and software checks are complete. The recommended next step is the separately reviewed silent helper in the user's Terminal, with all other measurements/tests idle:

```sh
sh /private/tmp/speech-mac-32-terminal-timing.sh
```

It verifies binary/pack/voice hashes, uses only a new queue at volume zero and has a 60-second native alarm. It writes raw JSON, source/hash metadata and exits to `.build/phonon-queue-terminal-timing.log`; it changes no source/evidence/tracker file. Success requires `TERMINAL_DEVICE_EXIT=0`, `QUEUE_TIMING_EXIT=0`, `TERMINAL_QUEUE_VALIDATED=1`, `TERMINAL_VALIDATION_EXIT=0`, finite correctly ordered events, 76800 samples/40 frames/40 reclaims and null acoustic onset. Root independently confirmed the helper's pinned binary remains SHA256 `35e3997d6feef019b19f38d9f114987ccb8216bb458ce68e5df762459c9830fe`.

On a successful result, the same ticket32 scope resumes to integrate the verified timing evidence and independent artifact review before completion. Alternatively the user may explicitly approve a plan change leaving physical timing open while later work proceeds; none has been given. Ticket32 remains blocked and 33–35 remain held. Acoustic onset is unavailable without capture; user listening stays pending until confirmed. Final whole-branch GPT-6.1 Sol/high review, GitHub rename, push and PR remain future steps.

### Current queue timing verified; E32-1 resolved — 2026-10-26

The user instructed: “run it for me and check the things as needeed”. The Coordinator ran the reviewed `sh /private/tmp/speech-mac-32-terminal-timing.sh` through `exec_command` with `tty=true`. It returned exit 0 with all four success markers. Independent parsing confirmed the original pinned source/binary/pack/voice metadata, finite ordered queue events, 76800 samples/40 frames/40 reclaimed buffers and null acoustic onset. This is an assistant-run real SDK measurement, not a user-entered Terminal run. The helper's literal `CONTEXT=user Terminal` header is superseded by the appended `ACTUAL_RUNNER` annotation.

| Current real queue event | Seconds from synthesis begin |
| --- | ---: |
| First PCM | 0.057648 |
| First submission | 0.057729 |
| Running state observed | 0.462776 |
| First buffer reclaimed | 0.537393 |
| Synthesis completed under queue backpressure | 3.417466 |
| Successful drain | 3.761503 |

Log `.build/phonon-queue-terminal-timing.log`, preserved at `/private/tmp/speech-mac-32-assistant-pty-success.log`, SHA256 `0c0e2a81ae0d5b224ab310d570756b97f32cac1bc303ee3bb796471b589c0e7b`. Markers: `QUEUE_TIMING_EXIT=0`, `TERMINAL_DEVICE_EXIT=0`, `TERMINAL_QUEUE_VALIDATED=1`, `TERMINAL_VALIDATION_EXIT=0`. Queue volume remained zero; no system audio settings changed. These are observed SDK/producer events, including backpressure, not acoustic onset or replacement unpaced compute measurements. Listening remains pending.

Before this success, the Coordinator also tried the same reviewed helper through a private `.command` wrapper opened by Terminal.app. That automated launch failed with AudioQueueStart -66681/child4/validation1; it was not a manual user run. Its exact output is preserved at `/private/tmp/speech-mac-32-terminal-app-failed.log`, SHA256 `1be234c04bdfb66a7a4566dc8b7eafa765aa3f272bf037739bfcabfd21044fc2`. The earlier direct-tool failure remains separately preserved in the committed artifact. Why those attempts failed and the subsequent PTY run succeeded is not established; no service restart, device/volume change, numerical repair or diagnostic matrix was applied.

The current measurement satisfies E32-1 without a plan waiver; no Escalation remains open. Ticket32 now resumes only its original evidence-integration scope: preserve failed attempts, add the successful raw timing with truthful execution context, update report/verification, independently review artifacts and commit. Source 0f7ea8e, prior compute baselines and numerical/software gates are unchanged; formal ticket32 repair count remains 0. The initial blocked atomic report was archived at `/private/tmp/speech-mac-32-initial-blocked.report.md` before resuming. Ticket32 is not marked complete until the integrated evidence is verified; tickets33–35 still wait for that completion record. Final whole-branch Sol/high review remains pending.

### Verified completion — 2026-10-26

The same Sol/low worker completed original-scope evidence integration in attempt2 at `8c1458bc1517f0c081e58c84237fa5416f710aa1`, with Ticket32 trailer and clean tree. Source `0f7ea8e` and original baseline evidence `054355f` are unchanged. Formal repair count remains 0; this was environment/evidence continuation. The final canonical atomic report is attempt2/status done and preserves the archived initial blocked report's history.

Five artifact paths changed: `queue-assistant-pty.json`, exact successful raw log, exact prior automated Terminal.app failure log, report and verification. The successful assistant PTY context, pins, four success markers, current raw events/counts and null acoustic onset are integrated. The original tool failure remains separate. All original compute/profile/warmup/text/WAV bytes remain unchanged. Independent Sol/low SingleReviewer applied all six lenses to every changed artifact and found none. Worker format→check and staged hook passed; no executable edit or remeasurement occurred.

The Coordinator independently checked the exact artifact-only footprint, compared both integrated logs byte-for-byte with immutable captures and their recorded SHA256s, parsed the successful JSON and verified all pins/markers/finite ordering/76800 samples/40 frames/40 reclaims. The existing benchmark validator passes; every prior compute/profile/warmup/text/WAV/agent-failure byte is identical to evidence 054355f. Executable, test, reference, tracker, vendor, fixture and budget source is unchanged; format-check, whitespace and clean-tree checks pass. Prior full actual-fixture, reference, frozen and sanitizer gates above remain applicable; no broad repeat was needed for artifact prose/data.

Ticket32 is verified on `feat/phonon-migration` at `8c1458bc1517f0c081e58c84237fa5416f710aa1`. Result commits: `0f7ea8eab3e52f3b388ed810de0a3f71218935e9`, `054355f35e2645f6bc53115be14114c71d5cbc5a`, `8c1458bc1517f0c081e58c84237fa5416f710aa1`. No Escalation remains open. Ticket33 may start after this completion record is reviewed and committed. Listening remains pending; acoustic onset is unavailable; prior SDK failure cause remains unknown. Optimization, final integration acceptance, GitHub rename and final whole-branch GPT-6.1 Sol/high review remain future work.

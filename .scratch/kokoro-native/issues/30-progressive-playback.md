# 30: Bound playback, backpressure and cancellation

## Parent

None (Requirement Source).

## Context

- Technical source: `.scratch/kokoro-native/spec.md` and its pinned primary sources.
- Background: `.scratch/kokoro-native/spec.md`.
- User clarification dated 2026-10-25: subsequent implementation, fixes and ticket reviews use implement-lite / gpt-6.1-sol / low, overriding the repository model/effort defaults. Final whole-branch review uses gpt-6.1-sol / high. Actual earlier Luna/low execution records remain historical evidence.

## Footprint

stream driver, AudioQueue/platform adapter, CLI dispatch, request/cancel status contracts, deterministic concurrency tests/device probes/stream metrics.

## What to build

--stream consuming within-utterance frames through existing narrow AudioQueue adapter. Ordered bounded buffering/backpressure/drain/cancel/error lifecycle and resident reuse. Separate first PCM, submission/start, buffer reclaim and successful drain events. Prove first audio arrives before inference finishes, batch/stream equivalence including tail, cancellation during generation/wait/drain and single terminal outcome. Device checks mute queue only, never system volume; fake-consumer time is not acoustic/device onset.

## Blocked by

29-speech-cli-voices

Worker Profile: medium; implement-lite / gpt-6.1-sol / low (user clarification 2026-10-25)

## Acceptance criteria

- [ ] --stream consuming within-utterance frames through existing narrow AudioQueue adapter
- [ ] Ordered bounded buffering/backpressure/drain/cancel/error lifecycle and resident reuse
- [ ] Separate first PCM, submission/start, buffer reclaim and successful drain events
- [ ] Prove first audio arrives before inference finishes, batch/stream equivalence including tail, cancellation during generation/wait/drain and single terminal outcome
- [ ] Device checks mute queue only, never system volume
- [ ] fake-consumer time is not acoustic/device onset
- [ ] Complete the approved slice checks and keep unavailable model/device checks explicit.

Status: blocked

## Implementation and partial verification — 2026-10-25

- Wave base: `d1476b9de1d6fa5bcd1e70c14928d647f24c3a42`; branch `feat/phonon-migration`, existing run worktree.
- Worker `/root/30_progressive_playback`, attempt 1, implement-lite / GPT-6.1 Sol / low.
- Implementation SHA: `8d09e66cbbff86eacdde5617e7c75d63751a784f`, with `Ticket: 30-progressive-playback`; clean tree and all 13 approved paths confirmed. No numerical kernels, frozen fixtures/budgets, source model assets or tracker records changed in that source commit.
- Atomic attempt1 report: `/private/var/folders/j1/4vzqsp717kbb3gth_48lxdkc0000gn/T/orchestrate-briefs/kokoro-mac-feat-phonon-migration-f11dbf4f/30-progressive-playback.report.md`; archived before follow-up at `/private/tmp/30-progressive-playback-initial-blocked.report.md`.

This ticket is implemented but not verified done. Successful native queue-muted start/reclamation/drain remains blocking. A narrow public ownership-documentation correction is also required below. Tickets 31–35 remain held; no device skip or failed attempt is counted as a pass.

### Delivered behavior and independent evidence

The producer calls actual `speech_synthesize` and submits each borrowed 1920-sample frame to the narrow AudioQueue adapter, which owns a copy. The first submission starts playback; three fixed buffers provide backpressure without retaining an utterance. Cancellation chains the original request callback, wakes submission/drain waits, aborts and joins; terminal status is stable. Monotonic events distinguish first PCM, submission, observed start, reclamation, synthesis end and successful drain. CLI --stream uses the native driver, rejects --output and returns 130 for cancellation. Nonstream completed-WAV behavior is retained; its normal signal behavior was regression-tested.

The worker passed final canonical default `make format`→`make check`, full model-selected O2 checks, actual stream ASAN/UBSAN and O3, synthetic controller TSAN with no suppressions, preserved legacy fake ASAN, exact frozen replay and staged hook. All five engine cases and actual raw/projected CLI checks passed. Stream/batch equality is exact at 30720 samples/16 frames for the same text/voice/settings; submission and start precede synthesis completion. Fake timings are labelled and establish no device or acoustic onset. Source EOS tail and codec finish behavior are unchanged.

The Coordinator independently passed synthetic TSAN, actual stream ASAN/UBSAN and O3, legacy fake ASAN and complete native checks (old prefix/39-step consumers, pack209/36Q8 and frozen 32-frame head/conditions/EOS/latent/raw/final PCM exact0/61440 samples). Canonical format/check then passed with real CLI assets selected. Other unselected model/device skips are not actual validation; focused real gates supply the evidence above. One initial Coordinator full-suite invocation supplied a native pack without its convolution fixture and failed test configuration; this is retained in `-check-config-fail.log` and was resolved by selecting fixtures only for their focused commands. No code repair arose from it. Root logs: `/private/tmp/speech-mac-30-coordinator-{tsan,stream-sanitizer,legacy-sanitizer,stream-optimized,preserved,format,check}.log`.

### Coordinator scope answers

- A compile-time-only controlled PCM producer harness is approved for deterministic, weightless controller lifecycle tests. It does not ship, cannot replace neural operations in ordinary binaries and is labelled separately from actual model/device evidence. Playback dependency fakes remain the external seam.
- ThreadSanitizer on the fake/controller suite is approved as additional concurrency evidence, separate from SDK/model timing; it passed without suppression.
- Queue-volume-zero SDK probes, read-only HAL/system/bootstrap diagnostics and an unchanged silence-only legacy smoke are within the explicit device-check criterion. No service restart, system-volume/default-device change, production-format fallback or hardware-setting write is authorized or was performed.
- A test-only 30-second watchdog is approved around opt-in SDK checks; it reports failure and exits124 for a hung SDK. An isolated held-call proof verifies the timeout. Production handlers/cancellation behavior are unchanged by that watchdog.
- Extracted fake tests may snapshot state at disposal while the mock is valid, preserving live assertions, to fix the observed sanitizer UAF. Retaining leaked mocks to hide the UAF was not used.

### Findings addressed before the implementation commit

Independent Sol/low SingleReviewer examined all 13 changed paths with Correctness, Standards, Spec, Reuse, Quality and Efficiency. The final source review had no remaining code finding. The subsequent Coordinator contract review below adds F30-C1.

| Id | Finding | Disposition/evidence |
| --- | --- | --- |
| R1 | Global CLI handlers changed nonstream termination | Applied: stream-only, value/-- aware signal setup; FIFO/SIGTERM red/green |
| W1 | Legacy tests read a disposed/freed fake queue | Applied: owned dispose snapshots, live pre-release assertion retained; ASAN red/green |
| W2 | Cancellation during SDK property read could become successful drain | Applied: post-read cancel check; deterministic barrier red/green, TSAN/ASAN |
| W3 | Cancellation could win before delayed drain-success publication | Applied: mutex orders cancellation against drained timestamp/benchmark event; deterministic red/green |
| D1 | Successful native device functional check is absent | Open/blocking; actual executed failures and diagnostic modes below |

Deferred-start, dropped-tail, unchained-cancel and CLI-return-success mutations fail their assertions; source copies were restored. The worker's earlier model-selected check used a relative projected fixture path, producing a broken generated symlink; it removed only that link and reran successfully with an absolute path. Earlier failed configurations remain evidence, not passing results. Seven inherited analyzer warnings remain visible; no numerical or race suppression was added.

### Native device blocker

The Coordinator's own bounded queue-muted integration also failed (command exit2): queue creation/mute/enqueue succeeded, then `AudioQueueStart` returned -66681, kAudioQueueErr_CannotStart. Log: `/private/tmp/speech-mac-30-coordinator-device.log`. This confirms a failed executed check on the committed implementation, not a skip. No sound or system settings changed.

Worker isolation also found the preserved silence-only legacy adapter failing at the same start call. Four standalone fresh queue-muted processes—24k mono one1920-frame buffer;24k mono three4096 buffers with successful prime12288;native48k stereo three4096 unprimed;worker-thread48k stereo with successful prime—hung inside SDK start until15s SIGALRM. Logs: `.build/phonon-stream-device-{diagnose,bounded,probe-modes,info}.log`, `.build/phonon-stream-legacy-device.log`; SDK stack `.build/phonon-stream-device-probe-sample.txt`. Read-only HAL reports default output78/alive1/48k/buffer512/one output stream/runningSomewhere0; system_profiler identifies MacBook Pro Speakers. Root bootstrap is Aqua, user501; no context switching was attempted. The cause remains unknown.

A bounded silent Terminal integration command is prepared in README/report to distinguish a working user audio session from this execution context. The user has not yet provided that result or answered the optional other-app audio question. Successful start, all reclamations and drain remain required; acoustic onset and listening are unmeasured. No later ticket starts while this criterion remains unresolved without an explicit user-approved plan change.

### Accepted follow-up F30-C1: public ownership wording

A focused independent Sol/low review confirmed two unsupported implications in `src/speech_stream.h`: allowing cancel concurrently with an owner who may free, and describing cancel_context as immutable. `playback_users` protects the playback handle, not the lifetime of the SpeechStream allocation. The supported contract allows cancel concurrent with wait; callers must prevent new cancel calls and join/synchronize existing cancel callers before free. Cancellation and playback context storage stays borrowed/alive through producer completion; callback state may be mutable with caller synchronization. The playback ops table, engine, voice, text and language storage remain valid and unmodified while borrowed.

The Coordinator accepts a comments-only correction to that header, without ABI, runtime or numerical changes. This is preapproved formal repair1of2 for ticket30 only; tickets20/23/24 repair counts remain unchanged. Preserve the initial report and implementation evidence. Require format→check, independent Sol/low review of the contract and a noncomment-byte comparison. Existing behavioral matrices remain applicable because executable code is unchanged. The canonical follow-up report must remain blocked solely on native device acceptance after the wording correction.

### F30-C1 applied; device escalation E30-1 — 2026-10-25

Attempt2 started at `e4d1c355e80d94f3fbb27effbb7848a566856831` and committed the header correction at `8390afcffcb26848b23f82b4ae8d7b0c63abe589`, with the required Ticket trailer. Only comments in `src/speech_stream.h` changed. The worker passed `make format`→`make check`, independent GPT-6.1 Sol/low six-lens review and the staged hook. Its atomic attempt2 report preserves the full attempt1 report and remains blocked only on device acceptance. Formal repair1of2 for ticket30 is now applied; other ticket repair counts are unchanged.

The Coordinator independently confirmed the single-header scope, clean tree, format-check and diff whitespace. Removing C block comments leaves all 1145 bytes identical to both the repair base and original implementation; SHA256 `cacb9f7eb227fcd85f4acdde32baf5e0e628c92c3bcf595ead40ed3397e5d739`. The existing software evidence therefore remains applicable; no behavioral/model/device matrix was repeated for this documentation-only change. An independent `nm .build/speech-mac` check found none of `speech_stream_test_start`, `speech_test_synthesize`, `phonon_attention_test` or `phonon_prefill_attention_test` in the shipping binary.

The Coordinator also ran the same compiled bounded silent integration through a PTY. It still failed at `AudioQueueStart -66681`; `/private/tmp/speech-mac-30-coordinator-device-pty.log` preserves the output. Both ordinary tool execution and PTY execution fail; the cause remains unknown. No service, device or system-volume setting changed.

**E30-1, open, ticket30:** obtain a successful native queue-muted start/reclamation/drain result from the user's Terminal, or an explicit decision to change the approved dependency/acceptance plan. The recommended next step is the prepared helper:

```sh
sh /private/tmp/speech-mac-30-terminal-check.sh
```

The private helper invokes the committed focused test with the actual pack/default voice, `PHONON_STREAM_DEVICE=1`, `KOKORO_STREAM_SILENT_OUTPUT`, a separate build directory and O2. It was inspected and passed `sh -n`; it has not been run from the user's Terminal. It changes no system audio settings and mutes its own queue before submission/start. The existing SDK watchdog fails after 30 seconds rather than hanging indefinitely. Results go to `.build/phonon-stream-terminal-device.log`, including `TERMINAL_DEVICE_EXIT`. Acceptance requires exit0 and the test's `ACTUAL queue-volume-zero AudioQueue functional start/reclaim/drain: ok` marker; fake parity output alone is insufficient. This may also fail in Terminal and promises no environment fix.

Alternatively the user may explicitly approve leaving physical AudioQueue verification open while the remaining migration proceeds. That would change the approved plan and must be recorded before any dependent launch; no such approval has been received. The optional other-app audio question remains unanswered and would not itself establish device acceptance. Tickets31–35 remain held, final whole-branch GPT-6.1 Sol/high review has not run, and no criterion is marked complete by this partial record.

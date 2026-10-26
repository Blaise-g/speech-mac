# Ticket 32 verification — 2026-10-26

Source implementation: `0f7ea8eab3e52f3b388ed810de0a3f71218935e9`, starting at `649b9b61e776c0d6f1540d08ba90593f4f14a7e2`. Raw measurements identify this committed implementation, not the later evidence commit. No optimization, precision change, fixture change or budget change was made.

## Checks

| Command or check | Result |
|---|---|
| `make format` then `make check` | Passed before source commit and again before evidence commit; staged source hook passed. Ordinary suite explicitly skips ignored real-model/device checks unless configured. Existing English-normalizer and SLEEF analyzer warnings remain; new driver cleanup warnings were fixed. |
| `uv run --project reference/phonon --offline --no-sync --locked python -m unittest discover -s reference/phonon/tests -p test_benchmark.py` | Three tests passed, including unsupported repeat boundary and meaningful corrupted-measurement cases. |
| In-memory mutation replacing `validate` with a no-op | Expected red: all twelve then-current corruption cases stopped raising. No source mutation persisted. Additional negative/fractional count cases pass their rejection tests. |
| Removal test with temporary `benchmarks/run_legacy.py`, then probe removal | Expected red, then full maintained-surface suite green. Exact approved allowance is only `phonon_baseline.c` and `phonon_queue.c`; all old runtime/reference/dependency checks remain. |
| `PHONON_MODEL_DIR=/Volumes/Code/Personal/speech-mac-models/phonon-40m-en uv run --project reference/phonon --offline --no-sync --locked python -m unittest discover -s reference/phonon/tests` | All 23 tests passed. Initial invocation without the explicit model environment failed on missing portable-default assets, then the correctly configured invocation passed. |
| `PHONON_NATIVE_PACK=.build/phonon-native-f6acc8f43cb296be-v1.kpack PHONON_DEFAULT_VOICE=/Volumes/Code/Personal/speech-mac-models/phonon-40m-en/default-voice.safetensors PHONON_ENGINE_FIXTURE=.build/phonon-engine-v1.bin make test-one TEST=tests/test_speech.c TEST_CFLAGS=-O2` | Actual five-case engine gate passed exact counts/chunks and frozen waveform budgets; see `engine-correctness.log`. |
| `PHONON_NATIVE_PACK=.build/phonon-native-f6acc8f43cb296be-v1.kpack PHONON_DEFAULT_VOICE=/Volumes/Code/Personal/speech-mac-models/phonon-40m-en/default-voice.safetensors make test-one TEST=tests/test_speech_stream.c TEST_CFLAGS=-O2` | Actual engine/fake consumer exact batch/stream samples, cancellation, errors, resident reuse and CLI checks passed; device intentionally unselected. See `fake-stream.log`. |
| `PHONON_MODEL_DIR=/Volumes/Code/Personal/speech-mac-models/phonon-40m-en uv run --project reference/phonon --offline --no-sync --locked python reference/phonon/verify_native_generation_trace.py` | Frozen replay passed exact noise, head, autonomous trajectory, EOS, latent and 61440-sample PCM; zero numerical error. See `frozen-replay.log`. |
| Locked-uv `reference/phonon/benchmark.py` command in `benchmarks/README.md`, output `artifacts/32-phonon-baseline-20261026` | Completed all thirty baseline repetitions, retained excluded warmups and separate profiles. Revalidated final enriched `raw.json`; actual F32 hashes agree throughout. All six WAV headers/counts/hashes verified. |
| One `-O2 -DSPEECH_STREAM_SILENT_OUTPUT` actual SDK probe | Failed in tool context at AudioQueueStart -66681, exit 4; no successful hardware timing claimed. See `queue-agent-attempt.json`. No repeated diagnostic matrix. |
| `sh -n /private/tmp/speech-mac-32-terminal-timing.sh` and extracted validator with synthetic logs | Shell syntax passed; one valid synthetic log accepted and seven malformed/failure logs rejected. No device invoked by these validator tests. |
| `git diff --check` | Passed. |

The initial canonical removal check failed because ticket 31's then-empty benchmark guard forbade every non-Markdown file. The coordinator approved the exact two-driver allowance as ticket 32 integration, and its red/green probe verifies that obsolete runners still fail.

All Python execution used the checked-in locked uv project. Compute measurements ran exclusively, with no concurrent test, trace or benchmark writer; queue probing followed compute completion. No cache purge, service restart, autoplay, system-volume or default-device change occurred.

## Independent review

One GPT-6.1 Sol / low SingleReviewer applied Correctness, Spec, Standards, Reuse, Quality and Efficiency to all changed source, test, documentation and evidence paths. Source review finished before the implementation commit; evidence and the private helper received a separate follow-up. Binary WAV content was checked through headers, sample counts and hashes without playback. No lens was omitted. The later assistant PTY device result received an evidence-only follow-up review. Acoustic and listening results remain unavailable.

| ID | Lens | Finding | Disposition | Reason | Verified by |
|---|---|---|---|---|---|
| C1 | Correctness | Repeat count 20 exceeded native resident limit after adding warmup | Applied | Reject unsupported 5–19 boundary before any output/build/measurement | `check: uv run --project reference/phonon --offline --no-sync --locked python -m unittest discover -s reference/phonon/tests -p test_benchmark.py` |
| C2 | Correctness | Negative audio counts/durations could satisfy ratios | Applied | Require positive integral counts/bytes and bounded frames; negative/fractional corruption cases reject | Same focused check |
| Q1 | Quality | Header called nested operator scopes disjoint | Applied | Document top-level disjoint stages and nested operators consistently | Static |
| — | All six | Final implementation/evidence/helper review | No remaining finding | Raw arithmetic, provenance, WAV/F32 equality and honest unavailable-device labels verified | Static and read-only JSON/WAV/hash inspection |

## Current queue timing integration — 2026-10-26

The initial implementation/evidence checkpoint was blocked on current queue timing; its report is archived at `/private/tmp/speech-mac-32-initial-blocked.report.md`. The coordinator then ran the exact reviewed helper `sh /private/tmp/speech-mac-32-terminal-timing.sh` using assistant `exec_command` with `tty=true`, after the user asked the assistant to run it. This succeeded without a manual user Terminal run. The raw log's historical `CONTEXT=user Terminal` line is corrected by its appended `ACTUAL_RUNNER` annotation.

`queue-assistant-pty-success.log` is the immutable successful log, SHA-256 `0c0e2a81ae0d5b224ab310d570756b97f32cac1bc303ee3bb796471b589c0e7b`. `queue-assistant-pty.json` retains parsed metrics/provenance and references the separate prior automated Terminal.app failure, `queue-terminal-app-failed.log`, SHA-256 `1be234c04bdfb66a7a4566dc8b7eafa765aa3f272bf037739bfcabfd21044fc2`. That earlier coordinator-automated Terminal.app attempt failed SDK -66681, child exit 4, validation exit 1; it was not manual user execution. The original `queue-agent-attempt.json` remains untouched.

The successful log pins source `0f7ea8eab3e52f3b388ed810de0a3f71218935e9`, binary `35e3997d6feef019b19f38d9f114987ccb8216bb458ce68e5df762459c9830fe`, and the original pack/voice hashes. All four success markers pass: `QUEUE_TIMING_EXIT=0`, `TERMINAL_DEVICE_EXIT=0`, `TERMINAL_QUEUE_VALIDATED=1`, `TERMINAL_VALIDATION_EXIT=0`. Counts are 76800 samples, 40 frames and 40 reclaimed buffers. Finite ordered events are first PCM 0.057648 s, submission 0.057729 s, running observed 0.462776 s, first reclaim 0.537393 s, synthesis end 3.417466 s and drain 3.761503 s, relative to synthesis begin. Acoustic onset is null. This single muted queue-backpressured observation is separate from release compute timings and fake-consumer timings.

Locked-uv evidence validation checked both immutable log hashes, pinned metadata, markers, event finiteness/order, counts, acoustic null and failure exits. The existing `benchmark.validate` accepted unchanged `raw.json`. Byte comparisons against starting HEAD `f1a9a0faec437340eae5ed5ee504103694874d32` verified every original compute/profile/warmup/WAV/text/failure artifact unchanged; only this report and `report.md` were subsequently updated. The follow-up changes no executable behavior, so no new behavioral test or remeasurement is needed. Prior actual engine, frozen replay, stream and reference gates remain applicable to unchanged source; they were not rerun. Required format/check and staged hook run normally for this evidence commit.

One independent GPT-6.1 Sol / low SingleReviewer applies all six lenses to the changed evidence/docs; no lens or changed evidence path is omitted. No new review finding remains. This environment continuation consumes no numerical/implementation repair (formal repair 0).

Current actual queue timing is now established for this observed run. The cause of prior failures remains unknown; no code, service, cache, system-volume or default-device fix is claimed. No audition file was played. User listening remains pending and acoustic onset requires independent capture. The seven pre-existing analyzer warnings (three English-normalizer and four SLEEF) remain attributed; no new tool or lint rule is missing.

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

One GPT-6.1 Sol / low SingleReviewer applied Correctness, Spec, Standards, Reuse, Quality and Efficiency to all changed source, test, documentation and evidence paths. Source review finished before the implementation commit; evidence and the private helper received a separate follow-up. Binary WAV content was checked through headers, sample counts and hashes without playback. No lens was omitted. Future Terminal/device, acoustic and listening results could not be reviewed because they do not exist yet.

| ID | Lens | Finding | Disposition | Reason | Verified by |
|---|---|---|---|---|---|
| C1 | Correctness | Repeat count 20 exceeded native resident limit after adding warmup | Applied | Reject unsupported 5–19 boundary before any output/build/measurement | `check: uv run --project reference/phonon --offline --no-sync --locked python -m unittest discover -s reference/phonon/tests -p test_benchmark.py` |
| C2 | Correctness | Negative audio counts/durations could satisfy ratios | Applied | Require positive integral counts/bytes and bounded frames; negative/fractional corruption cases reject | Same focused check |
| Q1 | Quality | Header called nested operator scopes disjoint | Applied | Document top-level disjoint stages and nested operators consistently | Static |
| — | All six | Final implementation/evidence/helper review | No remaining finding | Raw arithmetic, provenance, WAV/F32 equality and honest unavailable-device labels verified | Static and read-only JSON/WAV/hash inspection |

## Pending hardware timing and listening

The reviewed private helper is `sh /private/tmp/speech-mac-32-terminal-timing.sh`. It pins the silent queue binary, pack and voice hashes; the native binary has a 60-second alarm. It writes `.build/phonon-queue-terminal-timing.log` and requires zero child/validator exits, finite ordered queue events, 76800 samples, 40 frames/reclaimed buffers and null acoustic onset. The coordinator must reserve an idle measurement window before the user runs it. A zero exit with `TERMINAL_QUEUE_VALIDATED=1` establishes the current queue event measurement only.

Current actual queue timing remains unavailable. Historical ticket 30 functional success is not reused as a new timing. The three audition files are ready for user listening; listening remains pending, and acoustic onset requires independent capture.

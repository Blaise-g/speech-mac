# Ticket33 verification — 2026-10-26

Starting clean commit: `f9ac91830db6ce6cde4ea9fe7ff597c6201a23d1`. Reviewed source commit: `cef25447a45840db848f6e3aadf0f0d9afc8272c`. The coordinator owns tracker writeback. This slice changes one measured allocation/materialization surface; no earlier-ticket defect repair, budget change or asset rewrite occurred.

## Checks and scope

| Command/check | Result and evidence |
|---|---|
| `sh .build/33-check-command.sh` | Final `make format` → `make check TEST_CFLAGS=-O2` passed with every real fixture and absolute projected voice. Exact environment/command retained in `check-command.txt`; complete log `33-check-final-corrected.log`. Full engine stage/PCM, long/reset/voice, concurrent requests, CLI and fake-stream gates passed. Real device explicitly unselected. |
| `PHONON_NATIVE_PACK=.build/phonon-native-f6acc8f43cb296be-v1.kpack PHONON_Q8_ORACLE=.build/phonon-native-q8-projections-v1.bin PHONON_BF16_ORACLE=.build/phonon-native-bf16-oracle-v1.bin make test-one TEST=tests/test_phonon_pack.c` | Final focused prepared-pack test passed (`33-pack.log`). Explicitly prepares the pack, checks 75 matrices/177,702,912 bytes, retains original packed Q8 oracle checks, compares cached Q8 SGEMM against 36 independent projection fixtures and cached BF16 samples against source values. |
| Same prepared-pack command with `TEST_CFLAGS='-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer'` | Passed on the final 75-matrix preparation/filter path (`33-pack-asan.log`), including partial-limit rollback, cleanup, independent pack destruction, concurrent reads, exact synthetic BF16/Q8 and nonfinite rejection. |
| `PHONON_NATIVE_PACK=… PHONON_DEFAULT_VOICE=… PHONON_ENGINE_FIXTURE=.build/phonon-engine-v1.bin make test-one TEST=tests/test_speech.c TEST_CFLAGS=-O2` | All five prepared-engine cases passed exact counts/chunks and unchanged stage/PCM budgets (`33-engine.log`). Repeated in final canonical check after excluding six unused/nonlinear matrices and adding simultaneous requests. |
| Same full five-case engine command with `TEST_CFLAGS='-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer'` | Passed (`33-engine-asan.log`). This full ASAN run preceded only the six nonconsumer exclusions and concurrency-test addition. Final production preparation/cleanup is additionally covered by final prepared-pack ASAN, constructor/concurrency TSAN and full release suite. |
| `PHONON_NATIVE_PACK=… PHONON_DEFAULT_VOICE=… make test-one TEST=tests/test_speech_stream.c TEST_CFLAGS='-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer'` | Actual batch/stream equality, cancellation/error/resident reuse and fake AudioQueue ownership passed (`33-stream-asan.log`). Before the skip-only six-tensor narrowing; final release stream gate repeated in canonical suite. No hardware/listening claim. |
| `PHONON_NATIVE_PACK=… PHONON_DEFAULT_VOICE=… PHONON_ENGINE_FIXTURE=.build/phonon-engine-v1.bin PHONON_ENGINE_RECORD=0 make test-one TEST=tests/test_speech.c TEST_CFLAGS='-O1 -g -fsanitize=thread'` | Final narrowed constructor plus two simultaneous public `speech_collect` requests sharing one engine/voice/options passed exact serial PCM equality with no TSAN report (`33-engine-tsan.log`). |
| `PHONON_MODEL_DIR=/Volumes/Code/Personal/speech-mac-models/phonon-40m-en make phonon-native-check` | Passed (`33-native.log`): unchanged standalone raw fallback, fixed/native projections, prefill/incremental, exact closed-loop noise/trajectory/EOS/latent and 61,440-sample PCM. This is not relabelled as a prepared-cache stage test. |
| `PHONON_MODEL_DIR=… make phonon-reference-check` | Pinned assets/reference checks and all 23 locked-uv tests passed (`33-reference.log`). |
| Cache-discard mutation, then focused pack test | Red at required published cache pointer; exact source hash restored; focused real-pack green (`33-red-cache.log`, `33-pack.log`). |
| Rollback pointer-clear mutation, then focused pack test | Red at the required empty cache after partial preparation failure; exact source hash restored (`33-red-rollback.log`). Final prepared-pack ASAN/green verifies actual cleanup. |
| Flow and Mimi cached-QKV offset mutations, each through actual record0 engine test | Flow red at first condition component; Mimi red at PCM NRMSE 1.62818/maxabs .86649. Both exact source hashes restored (`33-red-slices.log`); final record0 green including concurrent requests (`33-engine-final-focused.log`). |
| Independent SingleReviewer, all six lenses | Source/header/test review cleared before source commit. Measurement method reviewed before execution. Final evidence review also cleared on the same reviewer; no extra child reviewer. |
| Source staged hook | Passed (`33-source-commit.log`). Its ordinary unconfigured tests explicitly skip ignored assets; the separately configured complete canonical suite above supplies actual-model coverage. |
| `PHONON_MODEL_DIR=… uv run --project reference/phonon --offline --no-sync --locked python .build/33-measure.py` | All 72 child executions completed, then deterministic metadata postprocessing failed; original complete native data, method and failure log retained. No remeasurement/exclusion. Recovery, validator and independent native-field/source-hash/summary checks passed; `validation.json`, `raw.json`, `native-process-data.json`, `measurement-method.txt`, `postprocessing-recovery.txt`. |
| Evidence-only `make format` → `make check` | Passed after final artifact preparation (`33-evidence-check.log`); ordinary asset/device skips are explicit and do not replace the earlier complete real-fixture suite. |
| `git diff --check` | Passed throughout. |

All Python reference checks, mutations, measurement and evidence postprocessing use `uv` with the checked-in locked `reference/phonon` project. The native source and numerical assets remain pinned. Three existing English-normalizer analyzer warnings and four existing vendor SLEEF warnings remain; no new warning or suppression was introduced.

The initial unoptimized full real-fixture check was stopped while in its slow PCM gate when the source/test scope was finalized (`33-check.log`); no pass is claimed for that superseded run. A subsequent release canonical run failed only because the invocation supplied a relative projected-voice path: the test-created symlink resolved relative to its nested voices directory and dangled (`33-check-final.log`). Correcting the ignored symlink to the absolute existing artifact made the focused CLI test pass (`33-cli.log`); the final absolute-environment full suite then passed. No production/test source fix or numerical repair was made for this invocation error.

## Change-to-evidence map

| Behavior/obligation | External anchor | Evidence |
|---|---|---|
| Exact materialization and unchanged projection arithmetic | Published BF16/Q8 encodings; frozen source projection fixtures and unchanged stage/PCM budgets | Synthetic unit-scale Q8 and hand-derived BF16 dot product; prepared real 36-Q8/BF16 oracles; cached engine trace/PCM; all measured baseline/candidate F32 hashes identical |
| Reuse, bounded ownership and cleanup | Approved per-model immutable ownership, 256 MiB bound, complete failure/destruction cleanup | Cache creation/rollback mutations; idempotent pointer; independent pack lifetime; final ASAN; pinned inventory accounting |
| Concurrent use and request isolation | Public speech API permits separate requests sharing an immutable engine | Concurrent shared projections; two actual simultaneous requests equal serial PCM exactly; TSAN; final full suite |
| Correct sliced materialized views | Existing full batched source-shaped QKV prefill and immutable raw layout | Each QKV offset mutation fails actual engine stage/PCM gate; exact restoration/focused green; full five-case suite; paired F32 identity |
| Unchanged fallback | Compatibility with existing unprepared numerical callers | Existing standalone native/frozen/pack/Mimi/PCM gates and exact replay |
| Measured gain and tradeoffs | Same-host paired release measurements; all raw results; unchanged assets/settings | Six AB/BA pairs per workload/mode, retained warmups, current baseline distributions, cold preparation/first/completion totals and RSS; fresh structured tied pair explicitly retained |
| Header/documentation changes | Actual ownership/allocation and evidence boundaries | Static review; no separate behavioral test required |

## Review findings

Paths carrying behavior/standards: `src/phonon_mimi.c`, `src/phonon_pack.c`, `src/phonon_pack.h`, `src/phonon_pcm.h`, `src/phonon_transformer.c`, `src/phonon_transformer.h`, `src/speech.c`, `tests/phonon_pack_test.h`, `tests/test_phonon_pack.c`, `tests/test_speech.c`. All Correctness, Spec, Standards, Reuse, Quality and Efficiency lenses were used; none omitted. Every final evidence path under this artifact directory received the same six-lens follow-up, including independent recomputation of all summaries, paired ratios, fresh preparation sums and immutable native/source hashes. Raw data are inspected through values/hashes rather than treated as code.

| ID | Lens | Finding | Disposition | Reason | Verified by |
|---|---|---|---|---|---|
| Q1 | Quality | Old linear comment described every expansion as call-owned/transient | Applied | Distinguish immutable borrowed prepared weights from call-owned fallback | Static |
| M1 | Correctness / Spec | Warm preparation-to-event sums omitted the intervening excluded request | Applied | Report those elapsed-scope sums only for fresh processes; retain warm request-relative timings | Static method/summary review and `validation.json` summary recomputation |
| M2 | Correctness / Spec | Original method omitted derived duration/RTF/consumer/profiled metadata and stopped after native measurements | Applied | Preserve original native data; add deterministic fields without rerun or exclusion | Independent native-field comparison, hashes and summary recomputation |
| — | All six | Final source/test/evidence review | Cleared | No actionable source finding remains; numerical/ownership checks reviewed | Static and retained check logs |

Listening remains pending. Acoustic onset is unavailable without independent capture. No new device result is claimed; actual current ticket32 SDK timing remains historical evidence of that run, and ticket34 final acceptance is separate.

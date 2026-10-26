# Exact resident linear materialization — 2026-10-26

Prepared immutable linear weights improve resident synthesis on this host. Median paired speedups are 1.45× short, 1.44× structured and 1.47× long; all measured resident candidate ranges are below their baseline ranges. The tradeoff is about 169.47 MiB of retained F32 weights per engine and slower fresh preparation-to-first-PCM. Fresh structured timing includes one effectively tied pair, so a uniform fresh-request gain is not claimed.

Baseline: `f9ac91830db6ce6cde4ea9fe7ff597c6201a23d1` (private archive of the clean ticket32 base). Candidate: `cef25447a45840db848f6e3aadf0f0d9afc8272c`. Both use identical C17 warning flags, `-O2`, Accelerate and AudioToolbox on Apple M5 / 16 GiB / macOS 27.0.1. `raw.json` records exact compiler, OS build, source-file and binary hashes, assets/settings and unchanged thread environment.

## Repeated release measurements

| Workload | Mode | Baseline synthesis median (range), s | Candidate synthesis median (range), s | Paired speedup median (range) |
|---|---|---|---|---|
| short | fresh_process | 0.8355 (0.7894–0.9835) | 0.5852 (0.5719–0.6110) | 1.415× (1.295–1.720) |
| short | warm_resident | 0.8130 (0.7807–0.9220) | 0.5686 (0.5400–0.5923) | 1.448× (1.410–1.557) |
| structured | fresh_process | 4.2582 (4.0732–4.6690) | 2.9811 (2.8766–4.3602) | 1.440× (1.000–1.533) |
| structured | warm_resident | 4.2931 (3.9718–6.0015) | 3.1008 (2.8096–3.9075) | 1.437× (1.355–1.643) |
| long | fresh_process | 8.5920 (7.9408–10.1552) | 6.0641 (5.5136–7.6337) | 1.408× (1.186–1.643) |
| long | warm_resident | 8.4743 (8.1310–10.9790) | 5.9501 (5.4129–6.8453) | 1.468× (1.363–1.783) |

Six pairs per workload/mode alternate baseline→candidate and candidate→baseline, with three pairs in each order. All 72 child executions, 72 measured requests and 36 excluded resident warmups are retained. Every resident child runs one excluded request followed by one measured request on the same engine and voice. Each request resets its seed/state. The native measurement window lasted 437.52 s, exclusively: no concurrent tests, traces, profiling, playback or other benchmark writer. No OS cache purge or system setting changed. Fresh means a new process with ordinary OS caches.

No timing is excluded as an outlier. Baseline drift is visible, including the 10.979 s long resident result. Fresh structured pair 4 is 4.358527 s baseline versus 4.360225 s candidate (0.99961×); its observed ranges overlap. Resident pairs all improve, with minimum paired speedups 1.410× / 1.355× / 1.363×. These are observed same-host results, not a promise for every future host or request.

The request clock starts after engine and voice preparation. First PCM is callback readiness; synthesis ends after synchronous memory-copy consumption. CPU is synthesis user+system delta. No device pacing or acoustic timing enters these results. Fresh totals below sum the separately measured engine preparation, voice open and request event; they exclude child launch and input-file overhead.

## Preparation, first PCM and memory

| Workload | Fresh engine preparation median, baseline→candidate ms | Fresh preparation+voice+first PCM median, baseline→candidate ms | Fresh preparation+voice+synthesis median, baseline→candidate s | Resident request first PCM median, baseline→candidate ms |
|---|---|---|---|---|
| short | 36.812→53.633 | 83.805→105.517 | 0.887→0.650 | 42.998→37.453 |
| structured | 33.710→56.702 | 86.315→112.961 | 4.292→3.047 | 54.159→42.902 |
| long | 34.962→56.465 | 91.969→113.746 | 8.626→6.117 | 51.877→42.914 |

Engine preparation increases by roughly 17–23 ms at the fresh medians. Fresh preparation-to-first-PCM increases by roughly 22–27 ms even though completed synthesis is faster. The resident request first-PCM medians improve. Resident preparation-to-event totals are deliberately absent: the intervening excluded warmup would make those sums misleading.

| Workload | Baseline process RSS high-water range, MiB | Candidate process RSS high-water range, MiB |
|---|---|---|
| short | 114.27–116.23 | 279.14–281.16 |
| structured | 114.97–122.66 | 279.84–287.81 |
| long | 119.42–147.80 | 279.77–312.61 |

RSS ranges combine measured fresh/resident processes for each workload. Darwin RSS is process lifetime high water, not current retained memory or a request allocation delta. Raw bytes and all per-mode distributions remain in `raw.json`. Each pack owns exactly 75 prepared matrices / 177,702,912 F32 bytes (169.47 MiB), below its 256 MiB cap. The original packed bytes remain resident for validation and other consumers. Requests and fallback projections still allocate; complete inference is not allocation-free.

## Bounded implementation

One measured surface changed: repeated BF16/Q8 linear allocation and exact expansion moves into pack preparation during engine creation, before model views escape. Cached views are immutable and borrowed until pack close. Failure frees partial materializations; destruction frees every prepared matrix. Separate engines own separate packs. The exact existing BF16/Q8 conversion is shared by preparation and transient fallback. SGEMM shapes, flags, bias accumulation, decoding arithmetic and reduction order are unchanged. Flow and Mimi QKV slices advance both packed and materialized row offsets.

Two lookup tensors are excluded: `flow_lm.condition_provider.conditioners.num_speakers.embed.weight` and `flow_lm.condition_provider.conditioners.transcript_in_segment.embed.weight`. Four checkpoint matrices have no native linear consumer and are excluded: `flow_lm.flow.w_s_t.0.weight`, `.3.weight`, `.6.weight`, `.9.weight` (each suffix expands under the same prefix). Their packed representations and fallback behavior remain available. `materializations.json` records every exact included/excluded name, shape, size and reason. Excluding these six avoids 8,204,672 unnecessary F32 bytes.

No Metal, Mimi requantization, KleidiAI, global mutable cache, backend framework or product flag was added. Voice projection counts, planned chunk resets, RNG cursor, ownership, full batched prefill, NEWLAPACK/legacy-K1 semantics, fixtures and correctness budgets remain unchanged. No old-ticket defect repair was made.

## Correctness and evidence boundaries

All measured F32 hashes are identical across baseline/candidate, all repetitions and retained warmups, and match ticket32. Benchmark counts remain 76,800 samples / 40 frames / 1 chunk short; 393,600 / 205 / 2 structured; 773,760 / 403 / 4 long. The prepared engine also passes all five independent stage/PCM cases with exact sample/chunk counts 30,720/1, 49,920/1, 72,960/1, 503,040/2 and 1,242,240/2, unchanged stage rtol=.001/atol=1e-5 and PCM NRMSE≤.001/maxabs≤.002.

Prepared-path evidence is the explicit prepared-pack Q8/BF16 oracle test, prepared actual-engine trace/PCM suite, concurrent public requests, and paired C-engine F32 identity. Standalone native frozen transformer/Mimi/PCM probes open unprepared packs; their exact replay is fallback-preservation calibration, not proof of cached stages. Both cached QKV offset mutations fail at their intended stage/PCM gates. See `verification.md` and retained logs.

The original measurement method completed all native requests, then its validator stopped because derived duration/RTF/consumer/profiled fields had not been attached. `native-process-data.json` preserves the original complete data; `postprocessing-recovery.txt` adds only those deterministic fields and resumes validation/summarization. Every original timing/count/hash/provenance field and sequence was independently compared unchanged. `33-measure.log` preserves the failure. No native measurement was rerun. `measurement-method.txt`, `validation.json`, source/binary hashes and archive/build commands make the recovery reviewable.

No new candidate profile is used for speedup statistics. Ticket32 identified expansion as ~37% of synthesis and Mimi/PCM as 63–65% of its separate profile; those historical fractions are not relabelled as candidate measurements. Remaining candidate bottlenecks need their own profile before another optimization.

Actual engine/fake-consumer stream equality, cancellation and ownership gates pass. No new real-device run or listening occurred in ticket33. Ticket32 current muted SDK evidence remains attributable to ticket32; ticket34 current final-device/listening acceptance is not inferred here. Acoustic onset remains unavailable.

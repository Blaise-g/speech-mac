# 33: Improve a measured CPU bottleneck

## Parent

None (Requirement Source).

## Context

- Technical source: `.scratch/kokoro-native/spec.md` and its pinned primary sources.
- Background: `.scratch/kokoro-native/spec.md`.
- User clarification dated 2026-10-25: subsequent implementation, fixes and ticket reviews use implement-lite / gpt-6.1-sol / low, overriding the repository model/effort defaults. Final whole-branch review uses gpt-6.1-sol / high. Actual earlier Luna/low execution records remain historical evidence.

## Footprint

hot-path numerical/model surface identified by32, relevant regression tests, before/after raw benchmark evidence/docs; exclusive device timing. Worker limits edits to one manageable measured surface.

## What to build

bounded Accelerate/NEON/packed-Q8/allocation improvement on dominant measured surface, paired on same host/workloads against fixed correctness budgets. Native packed Q8 is already correct in20. No speculative Mimi requantization/KleidiAI runtime/Metal. If nothing beats observed variation, preserve correct baseline and record failed experiments/bottlenecks candidly.

## Blocked by

32-phonon-baseline

Worker Profile: medium; implement-lite / gpt-6.1-sol / low (user clarification 2026-10-25)

## Acceptance criteria

- [x] bounded Accelerate/NEON/packed-Q8/allocation improvement on dominant measured surface, paired on same host/workloads against fixed correctness budgets
- [x] Native packed Q8 is already correct in20
- [x] No speculative Mimi requantization/KleidiAI runtime/Metal
- [x] If nothing beats observed variation, preserve correct baseline and record failed experiments/bottlenecks candidly
- [x] Complete the approved slice checks and keep unavailable model/device checks explicit.

Status: done

## Implementation and verified completion — 2026-10-26

- Wave base: `f9ac91830db6ce6cde4ea9fe7ff597c6201a23d1`; branch `feat/phonon-migration`, existing run worktree. Worker `/root/33_measured_cpu_optimization`, attempt 1, implement-lite / GPT-6.1 Sol / low; formal repair count 0.
- Source: `cef25447a45840db848f6e3aadf0f0d9afc8272c`; evidence: `9eed9617c949ba2ed97dc758255d39c6d0c1c7a7`. Both Ticket33 trailers and clean tree confirmed. Canonical atomic report `33-measured-cpu-optimization.report.md` is done/attempt 1 in the run Briefs directory.
- Durable source/provenance/raw distributions/validation/inventory/method/failure/correctness evidence: `artifacts/33-linear-materialization-20261026/`. No tracker files were edited by the worker.

### Bounded change and approved decisions

One measured surface changed: repeated exact BF16/Q8 linear expansion/allocation is retained as immutable pack-owned F32 materializations during engine preparation, before views escape. SGEMM shapes/flags/bias, conversion arithmetic and reduction order remain unchanged. Packed bytes remain authoritative for validation/nonlinear consumers. Flow/Mimi QKV slices advance both packed and materialized offsets. A 256MiB cap, checked sizes, idempotence, partial rollback, independent-pack cleanup and immutable shared requests are tested.

The Coordinator approved this implementation within ticket33's measured footprint, not as an old ticket20 defect repair. Two lookup embeddings and four unused checkpoint w_s_t matrices were excluded after inspection; raw fallback remains valid. `materializations.json` records exact names/reasons and 75 matrices/177,702,912 F32 bytes (169.47MiB), avoiding 8,204,672 unnecessary bytes. No global mutable cache, backend framework, product flag, Metal, Mimi requantization or KleidiAI was added. Remaining request/fallback allocations prevent an allocation-free claim.

### Paired measurements and adoption limits

Six AB/BA pairs for each short/structured/long workload in fresh and resident modes retain 72 measured requests and 36 excluded warmups across 72 serial native processes. Baseline f9ac918 and candidate cef2544 use identical -O2 flags, assets/settings and this Apple M5/16GiB/macOS27.0.1 host. The 437.52s native window was exclusive. Every measured/warmup F32 hash equals its baseline/candidate counterpart and ticket32; no sample was excluded or remeasured.

Resident paired median speedups are 1.448× short, 1.437× structured and 1.468× long; all observed candidate resident ranges are below baseline ranges. Fresh medians improve about 1.41–1.44×, but structured pair 4 is 0.99961× (4.358527s baseline, 4.360225s candidate); there is no universal fresh win. The retained long baseline outlier 10.979s and all variation remain visible.

Tradeoffs are part of adoption: candidate RSS high-water is 279–313MiB versus 114–148MiB baseline. Fresh engine preparation adds 17–23ms and preparation+voice+first PCM adds 22–27ms. Fresh preparation+voice+completion medians improve 0.887→0.650s short, 4.292→3.047s structured, 8.626→6.117s long. Resident request first-PCM medians improve; warm preparation-to-event sums are not reported because an excluded warmup intervenes. The Coordinator approved this measured resident/completion gain with its documented cold-onset and memory costs; no threshold, acoustic or cross-host promise is invented.

The original method completed every native run, then its validator stopped on omitted deterministic duration/RTF/consumer/profiled fields. Original native values, sequence, method and error log are preserved; recovery added only those four fields and recomputed summaries. No timing was rerun or selected. Independent review recomputed all native/source/method hashes, distributions, paired ratios and fresh sums.

### Checks and independent verification

Worker final format→full actual-fixture make check-O2, prepared 75-matrix/36-Q8/BF16 oracles, final pack ASAN, five prepared-engine cases, actual concurrent shared-engine/voice serial-equivalent requests under TSAN, stream ASAN and native/reference gates passed. Initial engine/stream ASAN preceded only six nonconsumer exclusions; final narrowed pack ASAN, concurrent TSAN and full release suite cover the final constructor. Cache-discard, rollback and both cached-QKV-offset mutations fail intended assertions; exact source hashes were restored. A superseded O0 run and an incorrect relative projected-voice fixture invocation remain recorded; final corrected absolute-env canonical checks pass without a code repair. Standalone frozen probes open unprepared packs and are labelled fallback calibration; prepared pack/engine and paired F32 results prove the cached path.

The independent GPT-6.1 Sol/low SingleReviewer covered all six lenses over all 10 source/test paths and all evidence. Q1 corrected a stale transient-only ownership comment; M1 restricted preparation-to-event totals to fresh runs; M2 preserved/recovered deterministic postprocessing. Final findings:none. Both staged hooks passed; seven inherited analyzer warnings remain visible without new suppression. Vendor, pinned assets, frozen fixtures/budgets, voice projection/reset/RNG/count/stop rules and source math are unchanged.

The Coordinator independently ran `/private/tmp/speech-mac-33-coordinator-verify.py`, exit 0. It verified original versus recovered native values, 72/36/72 counts, AB/BA serial ordering, every source/method/data hash, all F32 identities against ticket32, every summary statistic and fresh sum, and the 75-matrix bounded inventory. Prepared-pack, full five-case/concurrent engine and stream ASAN/UBSAN passed on final source. Raw-fallback native calibration passed; make format→full actual-fixture make check-O3 passed; shipping test/profiling symbols are absent. Two reversed-order representative warm pairs on each workload match F32 exactly and observe 1.26–1.49× gains; these separate sanity runs do not replace the published 72-request statistics. Root results are in ignored `.build/ticket33-root-paired.json`; logs `/private/tmp/speech-mac-33-coordinator-{verify,prepared-pack-asan,engine-asan,stream-asan,raw-calibration,format,check}.log`. Final source remains clean.

Ticket33 is verified at `9eed9617c949ba2ed97dc758255d39c6d0c1c7a7` on the run branch. No Escalation or missing ticket33 software/model check remains. Ticket34 may launch after this reviewed completion record is committed. No new physical device timing/listening was performed here; final current acceptance belongs to ticket34. Acoustic onset is unavailable, listening pending, prior SDK failure cause unknown. GitHub rename and final whole-branch GPT-6.1 Sol/high review still follow tickets34/35.

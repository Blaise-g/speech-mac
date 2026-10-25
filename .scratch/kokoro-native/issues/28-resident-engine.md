# 28: Assemble native progressive text-to-PCM inference

## Parent

None (Requirement Source).

## Context

- Technical source: `.scratch/kokoro-native/spec.md` and its pinned primary sources.
- Background: `.scratch/kokoro-native/spec.md`.
- User clarification dated 2026-10-25: subsequent implementation, fixes and ticket reviews use implement-lite / gpt-6.1-sol / low, overriding the repository model/effort defaults. Final whole-branch review uses gpt-6.1-sol / high. Actual earlier Luna/low execution records remain historical evidence.

## Footprint

public speech engine/request API, synthesis assembly, callback/status/limit contracts, integration/full-model tests; consumes all native seams.

## What to build

resident weights and one request machine shared by incremental callbacks and batch PCM collector. First PCM is produced within the utterance at the model's 80ms latent cadence, not only after sentence completion. Full frontend→voice→Flow-LM→Mimi parity, requests/voices/text-chunk reset, bounded queues/context, cancellation/errors/finish/ownership and no leaked state. A synchronous producer/callback is sufficient; threads are not required for inference correctness.

## Blocked by

27-mimi-pcm

Worker Profile: medium; implement-lite / gpt-6.1-sol / low (user clarification 2026-10-25)

## Acceptance criteria

- [x] resident weights and one request machine shared by incremental callbacks and batch PCM collector
- [x] First PCM is produced within the utterance at the model's 80ms latent cadence, not only after sentence completion
- [x] Full frontend→voice→Flow-LM→Mimi parity, requests/voices/text-chunk reset, bounded queues/context, cancellation/errors/finish/ownership and no leaked state
- [x] A synchronous producer/callback is sufficient
- [x] threads are not required for inference correctness
- [x] Complete the approved slice checks and keep unavailable model/device checks explicit.

Status: done

## Execution and verification — 2026-10-25

- Wave base: `caf0e88a6e940d89f7a1c79451efb89c995d0c31`; branch `feat/phonon-migration`, existing run worktree.
- Worker `/root/28_resident_engine`, attempt 1, implement-lite / GPT-6.1 Sol / low. No formal ticket 28 repair was needed.
- Result and verified source SHA: `b3b5b6097f33351e77991101a10a1cfd4dd65f1c`. The Coordinator verified its three Ticket trailers (23,24,28), clean tree and all 14 changed paths within the ticket and approved derived Footprint. Tracker files were untouched by the worker.
- Atomic report: `/private/var/folders/j1/4vzqsp717kbb3gth_48lxdkc0000gn/T/orchestrate-briefs/kokoro-mac-feat-phonon-migration-f11dbf4f/28-resident-engine.report.md`.

The public `speech.h` API owns immutable mapped weights/decoder models in a resident engine. Requests own RNG, Flow caches, PCM state and scratch. The native path runs default English normalization, source prompt preparation/grouping/fitting and raw Unigram tokenization, projects the voice in a batch and prefills each complete prefix in a batch. It then generates/decodes one latent and calls back with1920 mono samples at 24 kHz samples before evaluating the next latent. The bounded collector calls this same producer. An actual head observer and a delayed-callback mutation prove that output starts within the utterance.

Flow and Mimi reset per source text chunk; the request RNG cursor continues across chunks. Actual voice length (1..4096 frames) participates in the 8192-position prefix+generation bound. Inputs before/after normalization are capped at 1 MiB; per-chunk generation at 4096 frames; total request generation at 8192. Callbacks borrow PCM only during the call; errors/cancellation free request state. The collector returns owned PCM only on success and NULL/zero on error, with generated partial progress reported separately. Shared linears still allocate bounded expanded-weight temporaries; no allocation-free or real-time guarantee is made. [Engine contracts](../../../docs/phonon-engine.md) retain the exact status/ownership details.

### Worker Questions and derived Footprint

All answers below come from the approved spec, ticket and pinned source, without an acceptance change.

- Q28-1: approved cancellation-aware variants in `phonon_transformer.[ch]` and `phonon_sampler.[ch]`, with explicit request poll/context and unchanged NULL wrappers. Poll between layers, attention blocks and head blocks; no promise to interrupt a BLAS call. Cancelled work does not publish output or commit a cache. Direct numerical cancellation, edge-triggered polling, callback rejection and clean retry are tested.
- Preparation question: retain before/after synchronous normalization/planning cancellation boundaries. Additional 21/22/tokenizer variants were not required or added; document those boundaries and retain bounds/latched cancellation before numerical work.
- Trace question: approved compile-time-only normalized-latent/condition/EOS observations and new independent bridge fields to diagnose long requests. They cannot substitute model math and their exported entry points are absent from production binaries.
- Runtime settings: confirmed temperature 0.8 as the approved runtime override, distinct from downloaded generation config 0.3. EOS threshold 4, one LSD step, minimum EOS 0 and source 1/3 tail remain unchanged. Pinned `run_inline` establishes per-chunk Flow/Mimi reset with request RNG retention. Source grouping target 50 differs from hard fitting limit 200.
- Oracle correction: the new exporter initially used a converted HF tokenizer that rewrote sentence-marker punctuation. It now uses pinned raw Unigram IDs and an independent marker assertion. Expected normalization/preparation comes from Rust, planning from the source-checked independent reference, and audio from all 209 actual GGUF tensors (36 Q8 + 173 BF16 including Mimi). No expected frontend/audio is taken from the native engine; frozen 19 fixtures/budgets are unchanged.
- Q28-2: approved two narrow source-order corrections in shared transformer code, provenance and defect-revealing tests. Details follow.

### Shared numerical repair 2 for tickets 23 and 24

The original request with 27 repetitions of Hello! had correct counts but failed waveform parity. Teacher-forcing and captured stages proved the full prefix/KV, every layer intermediate and final unnormalized 512-vector bit exact. Final LayerNorm variance differed by 1 ULP because the scalar incoming-M2 merge lacked the pinned source FMA. Only that scalar product/add contraction changed; the outer addition stays unfused. A checked-in 512-vector, exact-output-bit regression fails the old code and passes the correction without model assets.

The request containing 220 words crossed 512 attention keys. Identical-operand probes isolated a second 1 ULP difference: pinned Torch imports legacy Accelerate SGEMM, whose K=1/beta=1 update rounds the product before adding the destination, while NEWLAPACK contracts it. Only that generic later one-key block uses an unfused product/add in single-query and prefill paths. K>1 GEMM, max/rescale/total ordering and all other math remain unchanged. A blanket beta0-GEMM-plus-add candidate was rejected because it broke other block shapes. An analytic model-free regression catches the old behavior; an independent gate proves 14 single-query and 10 causal-prefill lengths through 2049 and more than two blocks.

The Coordinator approved this ongoing correction as formal repair 2 of 2 for each 23 and 24, both previously 1 of 2. Ticket 28 remains attempt 1/formal 0. Both tickets' repair allowances are now exhausted; a further post-verification repair traced to them requires user escalation. Pack 20 source/contracts, vector moments, epsilon, vendor header, frozen fixtures and budgets were unchanged. Source commit carries all three Ticket trailers.

The Coordinator independently fetched pinned PyTorch `e2d141dbde55c2a4370fac5165b0561b6af4798b` sources and verified new provenance hashes: LayerNorm `254240e1c9ee8b04a9e4bdb656d2c4f08ba36effb3f75e8a86720f4e87b5e367`; CPUBlas `03a6cb1a2e8778fd29bb331571d1d6a426dff78f3b26e39314c7905bff167a8b`. No legacy ABI switch, dependency or warning suppression was introduced.

### Checks and review

The Coordinator independently ran `/private/tmp/speech-mac-28-coordinator-verify.sh` on the committed source, exit 0: regenerated the independent five-case bridge; ran the 24-shape identical-operand attention gate; ran all five actual requests under ASAN/UBSAN; passed complete `make phonon-native-check`; then `make format` followed by canonical `make check` with all Phonon model/fixture variables selected. No sanitizer diagnostic occurred. The full suite executed the real pack, Flow, Mimi, convolution, PCM and engine checks; unrelated legacy/model/device skips remain explicit. Production speech/attention test entry points are absent; the source tree remained clean.

Root logs: `/private/tmp/speech-mac-28-coordinator-{export,attention,sanitizer,preserved,format,check}.log`. The regenerated bridge SHA-256 is `a5dd6df4d962a6270360514c6b70255f043149d157306807c9e9736bb0ebeec1`; its expected data comes solely from pinned independent sources. The Coordinator also independently fetched and checked the two added pinned numerical-source hashes below.

The worker's final single sequential normal→ASAN/UBSAN→O3 matrix passed all five engine cases, transformer/sampler, Mimi, convolutions, all 11 PCM records across 3 partitions, pack validation with 36 Q8 projection comparisons, and the independent generation verifier in every mode. The original frozen 32-frame Flow/head/EOS/latent and decoded PCM remain exact 0 with 61440 raw/final samples.

All engine sample/chunk counts are exact: 30720/1,49920/1,72960/1,503040/2,1242240/2. Batch/callback PCM is bitwise equal. Worst engine PCM NRMSE is 2.74799634e-5, max absolute error 8.28951597e-5, within unchanged 0.001/0.002 gates. Stage gates remain rtol 1e-3/atol 1e-5; independent raw Gaussian draws are exact. Both long stress cases remain in the suite.

Independent GPT-6.1 Sol/low Single Reviewer examined all 14 paths with Correctness, Standards, Spec, Reuse, Quality and Efficiency lenses; final refresh found none. Before that refresh, the Coordinator's test-voice portability finding was applied through `PHONON_DEFAULT_VOICE`; the collector comment was corrected to generated partial progress, including a rejected callback frame. The worker proved temporary-copy mutations for delayed output, wrong temperature, ignored cancellation, ignored request cap and bypassed currency normalization fail; all edits were reversed and tracked source hashes retained. Its first normalization negative control used an unchanged case and stayed green; the currency case supplies the valid red proof.

The inherited seven analyzer warnings remain visible (three normalizer, four unused unmodified generated-SLEEF dead stores), with no extra suppression. Historical Kokoro asset/device skips are not Phonon validation. No audible playback or listening claim is made here. CLI/voice catalog 29, playback 30, removal 31, measurements 32–33, acceptance 34 and administration 35 remain later work. Final whole-branch GPT-6.1 Sol/high review has not occurred.

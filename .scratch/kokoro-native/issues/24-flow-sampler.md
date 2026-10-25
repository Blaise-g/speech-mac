# 24: Generate replayable latents and EOS

## Parent

None (Requirement Source).

## Context

- Technical source: `.scratch/kokoro-native/spec.md` and its pinned primary sources.
- Background: `.scratch/kokoro-native/spec.md`.
- User clarification dated 2026-10-25: subsequent implementation, fixes and ticket reviews use implement-lite / gpt-6.1-sol / low, overriding the repository model/effort defaults. Final whole-branch review uses gpt-6.1-sol / high. Actual earlier Luna/low execution records remain historical evidence.

## Footprint

flow head/sampler, PRNG+replay interface, generation controls/stop metadata, latent/EOS fixtures/numerical tests/build integration.

## What to build

single/multiple 32-d latent frames using six-block AdaLN flow head, RMS time embedding (not centred variance), SiLU, pinned one-step LSD integration and raw EOS threshold. Port BOS/previous-latent scaling, EOS frame/tail/frame-budget semantics; distinguish safety budget exhaustion from EOS. Native equal-seed repeatability is defined, but cross-runtime parity consumes exact saved noise draws.

## Blocked by

23-flow-transformer

Worker Profile: medium; implement-lite / gpt-6.1-sol / low (user clarification 2026-10-25)

## Acceptance criteria

- [x] single/multiple 32-d latent frames using six-block AdaLN flow head, RMS time embedding (not centred variance), SiLU, pinned one-step LSD integration and raw EOS threshold
- [x] Port BOS/previous-latent scaling, EOS frame/tail/frame-budget semantics
- [x] distinguish safety budget exhaustion from EOS
- [x] Native equal-seed repeatability is defined, but cross-runtime parity consumes exact saved noise draws
- [x] Complete the approved slice checks and keep unavailable model/device checks explicit.

Status: done

## Execution record — 2026-10-25

- Wave base: `5495b64bfb2b905b03c45af565c03bb340495344`.
- Branch: `feat/phonon-migration`; worktree: `/Volumes/Code/Personal/kokoro-mac/.worktrees/phonon-migration-planning`.
- Worker: `/root/24_flow_sampler`; attempt 1; implement-lite / gpt-6-luna / low. No formal repair follow-up used.
- Reviewed checkpoint: `604d244c505d7d03db4c313438e98ab32e726776`, with `Ticket: 24-flow-sampler`. Seven changed paths fit the Footprint. The Coordinator confirmed the commit, trailer, clean tree, and unchanged frozen fixtures/budgets. This is not a verified ticket completion.
- Atomic blocked report: `/private/var/folders/j1/4vzqsp717kbb3gth_48lxdkc0000gn/T/orchestrate-briefs/kokoro-mac-feat-phonon-migration-f11dbf4f/24-flow-sampler.report.md`. An initial missing report was delivered by a report-only correction; no source edit or test rerun accompanied that correction.

The checkpoint implements all 94 flow-head tensors, six AdaLN blocks, stored-frequency RMS time embeddings, canonical division-form SiLU, one-step LSD, raw standard-Gaussian replay and seeded generation, normalized feedback, final Mimi-space denormalization, EOS frame/tail handling, and distinct safety-budget exhaustion. State is request-owned; weights are borrowed. The bulk generator is an internal convenience: progressive engine integration remains ticket28 work.

### Coordinator verification

The Coordinator ran `make format` then `make check` successfully on the checkpoint. Focused sampler and real-model transformer/generation tests passed both with `TEST_CFLAGS='-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer'` and `TEST_CFLAGS='-O3 -DNDEBUG'`. Real checks used the native Q8 pack, sampler/prefix/condition bridges and default voice. They exercised 40 fixed and 32 natural component frames, 140 prefix/39 incremental positions, seeded repeatability, exact replay consumption, EOS plus three tail frames, and replay exhaustion rejection. These C checks expose trajectory diagnostics; their zero exit does not mean the autonomous numerical budgets pass.

The Coordinator independently ran the locked-uv `reference/phonon/verify_native_generation_trace.py` against the optimized trace. It exited 1 as required because the frozen autonomous stage/PCM targets fail. Exact raw noise replay, finite values, 32 generated frames, EOS/tail outcome, and 61,440 raw/final PCM samples pass. The worker's complete `make phonon-native-check` exited 2 at the same numerical gate.

| Comparison | Maximum absolute error | Elements over frozen stage target |
| --- | --- | --- |
| Flow integration, identical dynamic inputs | 0.00000634789467 | 0 / 1024 |
| EOS projection, identical dynamic inputs | 0.000000953674316 | 0 / 32 |
| Autonomous flow condition | 0.00108897686 | 495 / 16384 |
| Autonomous normalized integration | 0.00275814533 | 27 / 1024 |
| Autonomous denormalized latent | 0.00263963267 | 27 / 1024 |
| Autonomous EOS | 0.0014629364 | 0 / 32 |

Raw and final PCM both have normalized RMSE `0.00359485159` and maximum absolute error `0.00610744208`, exceeding the unchanged targets `0.001` and `0.002`. Decoding for this diagnostic uses the independent pinned Q8/BF16 Mimi oracle; native Mimi is not implemented by this ticket. The same result was reproduced separately with `/private/tmp/speech-mac-24-pcm-check.py` before the checked-in verifier was completed.

Logs: `/private/tmp/speech-mac-24-coordinator-{format,check,sampler-sanitizer,real-sanitizer,sampler-optimized,real-optimized,native-gate}.log`. Verification script: `/private/tmp/speech-mac-24-coordinator-verify.sh`; `/private/tmp/speech-mac-24-coordinator-native-gate.status` contains `1`. Ordinary `make check` explicitly skips unrelated legacy model-backed and opt-in AudioQueue tests. The three previously reviewed normalizer analyzer warnings remain. No listening or native decoded-audio acceptance is claimed.

### Review disposition

Independent `/root/24_flow_sampler/sampler_review`, GPT-6 Luna / low, reviewed all seven changed paths under Correctness, Spec, Standards, Reuse, Quality, and Efficiency against pinned HF/Rust sources. Final gate refresh found no further concerns.

Independent Luna/low tracker review confirmed the blocked outcome and metrics. Its profile-label concern was addressed by writing the resolved user override beside the unchanged abstract medium tier; the gate status filename was made explicit. The reviewer refreshed both clarifications and found no further issues.

| Id | Finding | Disposition and verification |
| --- | --- | --- |
| 24-R1 | Autonomous stage failures were only printed | Applied: budget-driven verifier aggregates failing condition, normalized integration, EOS, latent and PCM stages; root reproduced its failing exit |
| 24-R2 | Synthetic head expectations lacked derivation provenance | Applied: test documents locked Torch 2.7.1 operators and deterministic BF16 construction; optimized always-active test passed |
| 24-R3 | Exact noise, finite PCM, stage-specific budgets, normalized integration and EOS needed explicit gate coverage | Applied: verifier reads authoritative budgets, requires exact noise/finite values/counts and all relevant stages; C test requires 1024 replay draws consumed; reviewer and Coordinator checked final gate |

Coordinator pre-report findings were also fixed: double temperature scaling, fixture record layout, normalized versus Mimi-space comparison, tolerance scale, NDEBUG-disabled assertions, optional fixture dependency selection, and fixed-run EOS threshold `1e9`. No frozen reference output was regenerated from native output. A reciprocal-multiply SiLU variant improved this one autonomous waveform but matched the pinned PyTorch operator less closely on a broad input grid; it was rejected. Canonical `x / (1 + exp(-x))` remains. Shared Q8/transformer experiments were reverted; their source files are unchanged by this ticket.

### Worker Question 24-1 — 2026-10-25

The first inventory treated absent sampler code as a blocker and ended without a report. Coordinator answer: implementing that missing code is this ticket's approved assignment; continue the same attempt through implementation, tests and review. Source: this ticket's What to build and Footprint. Subsequent progress continuations and report delivery correction changed neither scope nor profile and were not post-verification repairs.

### Numerical issue E24-1 — historical repair decision; now resolved

At checkpoint `604d244c505d7d03db4c313438e98ab32e726776`, independently reviewed component operations met the frozen targets on identical inputs, while tiny differences grew when native and PyTorch generation fed their own outputs into later frames. Autonomous condition/integration/latent/PCM targets were unmet. No tolerance or validation protocol was changed, and tickets25–35 were held during the repair. The verified repair below resolves that failure.

Protocol options considered before the model clarification, retained as historical context and not currently requested:

1. Recommended: retain unchanged strict stage budgets for identical-input component comparisons, unchanged waveform budgets for fixed-input Mimi decoding and native batch/stream equivalence, plus exact replay/count/stop and state-reset checks. Report autonomous cross-backend trajectory/waveform error separately as a diagnostic. Preserve broad generation and listening acceptance in ticket34. This explicitly relaxes the current autonomous whole-generation gate; it does not establish audio quality by itself.
2. Keep the existing autonomous acceptance gate and continue source-justified floating-point matching before later tickets. No fixture-specific formula selection or threshold widening.

The recommendation has not taken effect. A change requires user approval under orchestrate's Approved plan rule.

After the user clarified the intended model as GPT-6.1 Sol / low, the Coordinator withdrew the pending protocol-change proposal and scheduled preapproved repair follow-up 1 with that model. The new worker must investigate source-justified numerical fixes under all existing frozen gates. The checkpoint remains unverified and tickets25–35 remain held. There is no acceptance decision awaiting the user during this repair. The final whole-branch review will use GPT-6.1 Sol / high and cover the earlier Luna-written code too.

Repair failure record: checkpoint `604d244c505d7d03db4c313438e98ab32e726776` is red on the autonomous native gate; last verified predecessor tip was `5495b64bfb2b905b03c45af565c03bb340495344`. Repair 1 is within the two preapproved follow-ups; no extra repair or scope change is authorized. The new launch receipt records its current run-tip/base after this tracker update is committed.

## Verified repair — 2026-10-25

- Worker: `/root/24_flow_sampler_sol_repair`; attempt 2, implement-lite / GPT-6.1 Sol / low, following the user's model clarification. Repair base: `6e77e1257f3be433f426b9d2398db6d2595852b5`.
- Result and verified run-branch SHA: `94e1fefa63a896a31276eaff7f715bd3c60d0449`. Both `Ticket: 23-flow-transformer` and `Ticket: 24-flow-sampler` trailers are present; all 16 changed paths fit the original or Coordinator-approved derived Footprint. Clean tree verified. Nothing pushed.
- Formal repair follow-up 1 counts against both tickets23 and24, each now 1 of 2 preapproved repairs. Ticket20 source, contracts and pack format remain unchanged; no third ticket20 repair was undertaken.
- Atomic attempt2 report: `/private/var/folders/j1/4vzqsp717kbb3gth_48lxdkc0000gn/T/orchestrate-briefs/kokoro-mac-feat-phonon-migration-f11dbf4f/24-flow-sampler.report.md`. Attempt1 report retained at `/private/tmp/24-flow-sampler-attempt1-luna.report.md`.

The repair preserves the frozen acceptance contract. Native B1 flow-head/EOS outputs, autonomous conditions, normalized integration, denormalized latents, EOS, and independent decoded raw/final PCM now match the frozen natural replay exactly: maximum errors and PCM normalized RMSE are zero. Exact saved noise, finite values, 32 frames, EOS plus three tail frames, and 61,440 samples pass. Native Mimi implementation remains later work; this comparison decodes native latents with the independent pinned Q8/BF16 oracle.

### Source diagnosis and approved Footprint answers

The same pinned Torch model exactly replays frozen outputs with full-prefix processing, but fails the original waveform budget when only its prefix is processed incrementally: normalized RMSE `0.00619996898`, max error `0.0105209351`. This controlled diagnostic justified matching the source's batch shapes rather than weakening the gate. Native matching-prefix state and all nine cache key/value stages are now exact.

Coordinator-approved additions were limited to existing native numerical/prefill behavior and its tests/tooling:

- BF16 and Q8 expansion plus source-shaped Accelerate SGEMM in `phonon_transformer.c`, preserving public signatures, raw pack representation, finite/overflow/error checks and explicit transient ownership. `phonon_q8.[ch]` and pack source stayed unchanged.
- Float32 RoPE operation boundaries, width512 source moments/affine order with generic-width fallback, source NEON GELU polynomial, CPU SDPA reductions and bounded key blocks. Raw cache/QKV remains interleaved; only attention scratch uses the HF Q/K component reduction order.
- Internal batched-prefix API, true batch-linear projections, source query/key blocking and batched voice projection. Scratch is request-owned; score storage is at most `256 × 512` floats, without a quadratic context allocation. Cache length commits only on success; started failed prefills reset. Incremental generation follows the completed prefix.
- Private `phonon_math.[ch]`, the unmodified 579,669-byte generated SLEEF ADVSIMD header, provenance, exact upstream notices and a development generator. Only exp/sin/cos vector entrypoints are used; SiLU retains canonical division and scalar tails use system libm. Ordinary builds/inference require no external SLEEF library, CMake, generator or network.
- Focused regression tests, source/operator and execution-strategy diagnostics, B1-per-frame head verification, and header ownership documentation. No public speech API, settings, schema, fixture or budget change.

Source pins are the actual Torch 2.7.1 revision `e2d141dbde55c2a4370fac5165b0561b6af4798b` and its SLEEF submodule `56e1f79cb140fb9326d612d0be06b5250565cade`. The Coordinator independently checked the locked Torch revision, upstream source hashes/full BSD license, GitHub submodule mapping, pinned SLEEF archive hash/full BSL license and generated header size/hash. A fresh-directory worker regeneration reproduced header SHA256 `9c67f846daa49394628b4390396cb7223197e383de940e02cc3a58d9f33b477f`.

### Independent Coordinator checks on the committed repair

Passed `/private/tmp/speech-mac-24-sol-coordinator-verify.sh` with the real pack, Q8/BF16 oracles, flow/sampler/condition bridges and default voice explicitly selected:

- `make format` then `make check`, including the actual pack/projection and old/new transformer gates.
- Native math unit and real transformer/generation tests under ASan/UBSan and `-O3 -DNDEBUG`; always-active assertions execute in optimized tests.
- Locked-uv `verify_native_generation_trace.py` after each build: all original stage and raw/final PCM errors zero.
- Locked-uv `trace_native_operators.py`: 64 BF16 plus64 Q8 projections, 51 normalization patterns, 11 RoPE positions through8191 and65,537 GELU inputs exact; single-query attention through8192 max error `9.31322575e-10`; 14 causal-prefill boundary lengths max error `7.4505806e-09`, within the unchanged backbone target.
- Locked-uv `trace_prefill_strategies.py`: exact full-prefix reconstruction, exact native matched-prefix/cache stages; incremental Torch drift remains a separately labelled diagnostic.
- Complete `make phonon-native-check`: passed 209-tensor pack/36-projection oracle, fixed40/natural32 sampler, existing140-prefix/39-step replay, new batched prefill/39 consumers and autonomous PCM gate. Existing sequential replay max error `5.84125519e-06` remains within its original budget.
- `git diff --check`, clean tree and protected-path diff: tracker, frozen fixtures/budgets, packed kernels and pack format unchanged by the source commit.

Logs: `/private/tmp/speech-mac-24-sol-coordinator-{format,check,math-sanitizer,real-sanitizer,gate-sanitizer,math-optimized,real-optimized,gate-optimized,operators,prefill-strategies,native-check}.log`. Earlier failing evidence remains in the separate attempt1 logs. The worker also passed all14 locked reference tests, the full pack/sampler sanitizer and optimized tests, and fresh vendor regeneration; command evidence is in its atomic report.

### Repair review disposition

Independent `/root/24_flow_sampler_sol_repair/repair_review`, GPT-6.1 Sol / low, applied code-review-diff and code-simplify across all16 paths and all six lenses. The generated header received pinned hash/license/regeneration review, not a line-by-line algorithm review; wrapper/callers and native batching were reviewed directly. Final voice-test refresh found no further corrective findings.

| Id | Finding | Disposition and verification |
| --- | --- | --- |
| 24-S1 | Header's allocation-free promise conflicted with transient expanded weights | Applied: explicit linear/prefill ownership documentation; full checks |
| 24-S2 | Private half decoder duplicates the packed-kernel helper | Deferred: keep ticket20 API/source unchanged under the approved boundary; no correctness issue |
| 24-S3 | Broad attention/prefill differential only printed errors | Applied: shape/finite and existing backbone `rtol=1e-3`, `atol=1e-5` gate; independent Coordinator run passed |

Defect-revealing evidence retained under `/private/tmp/phonon24-trace.dsQEuW`: old double-accumulation cancellation, old GELU and old libm math fail the new independent golden tests; removing the causal mask in a temporary source copy fails the enforced prefill comparison with max error `3.0720346` and15,336 over-budget elements. Production source passes; no mutation remains.

Remaining limitations are explicit: optional half-helper consolidation is deferred; three historical normalizer analyzer warnings and four dead stores in unused, unmodified generated SLEEF functions remain visible. The Coordinator confirmed all seven warnings and absence of the unused remainder/erfc/cospi symbols from the test binary. Legacy Kokoro model/device checks still report skips; relevant Phonon checks executed. Transient expansion and prefix costs need the planned baseline/optimization work; no speed or listening claim is made.

### Handoff

- Use batched `phonon_voice_project` and `phonon_flow_stack_prefill` when establishing a generation request. A loop of single-token prefix steps is mathematically equivalent but does not satisfy the frozen autonomous source replay. The returned prefill state is unnormalized; apply `phonon_flow_output_norm` before the head, as the generator does.
- `phonon_math` and source-equivalent shared primitives are available to Mimi work. Keep raw Q/K representation separate from HF reduction order, inspect the actual oracle attention backend, and reuse narrow helpers when their contracts match.
- Engine28 must still provide progressive output/cancellation; the bulk generator is not the streaming engine. Allocate/check actual voice+text+generation positions, not a fixed512 voice reserve. Preserve request-owned RNG across the source-defined chunk lifecycle.
- Preserve vendor notices/provenance through legacy removal and documentation updates. Later measured optimization may improve transient expansion/ownership without changing the correctness gates.

## Verified shared repair 2 from resident-engine integration — 2026-10-25

Verified source `b3b5b6097f33351e77991101a10a1cfd4dd65f1c` carries Ticket trailers for 23,24 and 28. The ticket 28 worker and independent reviewer used GPT-6.1 Sol / low. The Coordinator approved the narrow derived shared-numerical/test/provenance Footprint, independently verified the committed result, and left frozen fixtures, budgets, pack 20 and vendor code unchanged.

Long autonomous requests exposed two rounding gaps: the scalar incoming-M2 LayerNorm merge needs an explicit source FMA followed by an unfused outer add; later SDPA blocks with K=1/beta=1 need an unfused product/add matching pinned Torch legacy SGEMM. Other GEMM shapes remain unchanged; a broader accumulation candidate was rejected. Both have regressions that run without model assets, and 24 independent single-query/causal-prefill shapes match exactly. Full five-case engine parity, exact counts/batch-callback equality, all preserved component gates, normal/ASAN/O3 worker modes, Coordinator ASAN/full-suite checks and the original exact 32-frame autonomous replay pass.

This is formal repair 2 of 2 for each ticket 23 and 24, both previously 1 of 2; their preapproved repair allowances are exhausted. Ticket 28 remains attempt 1/formal 0. A further post-verification repair traced to these tickets requires user escalation. The shared cancellation variants are new ticket 28 integration behavior, separately approved as Q28-1. Full causes, decisions, source hashes, review and verification evidence are recorded in [ticket 28](28-resident-engine.md). Final whole-branch GPT-6.1 Sol / high review remains future work.

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
- [ ] Complete the approved slice checks and keep unavailable model/device checks explicit.

Status: in-progress

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

### Numerical issue E24-1 — repair under the unchanged contract

The independently reviewed component operations meet the frozen targets on identical inputs, while tiny differences grow when native and PyTorch generation feed their own outputs into later frames. The current autonomous condition/integration/latent/PCM acceptance targets are unmet. No tolerance or validation protocol has been changed, and ticket25 through ticket35 remain held by this unverified predecessor.

Protocol options considered before the model clarification, retained as historical context and not currently requested:

1. Recommended: retain unchanged strict stage budgets for identical-input component comparisons, unchanged waveform budgets for fixed-input Mimi decoding and native batch/stream equivalence, plus exact replay/count/stop and state-reset checks. Report autonomous cross-backend trajectory/waveform error separately as a diagnostic. Preserve broad generation and listening acceptance in ticket34. This explicitly relaxes the current autonomous whole-generation gate; it does not establish audio quality by itself.
2. Keep the existing autonomous acceptance gate and continue source-justified floating-point matching before later tickets. No fixture-specific formula selection or threshold widening.

The recommendation has not taken effect. A change requires user approval under orchestrate's Approved plan rule.

After the user clarified the intended model as GPT-6.1 Sol / low, the Coordinator withdrew the pending protocol-change proposal and scheduled preapproved repair follow-up 1 with that model. The new worker must investigate source-justified numerical fixes under all existing frozen gates. The checkpoint remains unverified and tickets25–35 remain held. There is no acceptance decision awaiting the user during this repair. The final whole-branch review will use GPT-6.1 Sol / high and cover the earlier Luna-written code too.

Repair failure record: checkpoint `604d244c505d7d03db4c313438e98ab32e726776` is red on the autonomous native gate; last verified predecessor tip was `5495b64bfb2b905b03c45af565c03bb340495344`. Repair 1 is within the two preapproved follow-ups; no extra repair or scope change is authorized. The new launch receipt records its current run-tip/base after this tracker update is committed.

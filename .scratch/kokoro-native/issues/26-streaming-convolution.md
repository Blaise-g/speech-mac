# 26: Verify causal SEANet operators

## Parent

None (Requirement Source).

## Context

- Technical source: `.scratch/kokoro-native/spec.md` and its pinned primary sources.
- Background: `.scratch/kokoro-native/spec.md`.
- User clarification dated 2026-10-25: subsequent implementation, fixes and ticket reviews use implement-lite / gpt-6.1-sol / low, overriding the repository model/effort defaults. Final whole-branch review uses gpt-6.1-sol / high. Actual earlier Luna/low execution records remain historical evidence.

## Footprint

convolution/residual kernels/state, numerical interfaces, operator fixtures/C tests/build wiring.

## What to build

convolution/transposed-convolution and residual-block seam checks for pinned [6,5,4] SEANet decoder. Exact padding, trimming, overlap state, biases and activation ordering; frame-partition tests distinguish retained versus reset state, malformed shape/overflow handling. This bounded operator increment keeps full decoder assembly manageable.

## Blocked by

25-mimi-latent-transformer

Worker Profile: medium; implement-lite / gpt-6.1-sol / low (user clarification 2026-10-25)

## Acceptance criteria

- [x] convolution/transposed-convolution and residual-block seam checks for pinned [6,5,4] SEANet decoder
- [x] Exact padding, trimming, overlap state, biases and activation ordering
- [x] frame-partition tests distinguish retained versus reset state, malformed shape/overflow handling
- [x] This bounded operator increment keeps full decoder assembly manageable
- [x] Complete the approved slice checks and keep unavailable model/device checks explicit.

Status: done

## Execution record — 2026-10-25

- Base: `50e38318cf0ddb5bb7385626d1620a4981e4b174`; branch `feat/phonon-migration`; existing `phonon-migration-planning` worktree.
- Worker: `/root/26_streaming_convolution`; attempt 1; implement-lite / GPT-6.1 Sol / low. No repair follow-up or scope expansion used; no open Escalation.
- Result and verified run-branch SHA: `5f6d280662ec494d56185f7bc432ebc4f91aa275`, with `Ticket: 26-streaming-convolution`. All four new files fit the published Footprint. Shared code, vendor files, tracker, frozen fixtures and budgets were unchanged by the source commit; clean tree verified.
- Atomic report: `/private/var/folders/j1/4vzqsp717kbb3gth_48lxdkc0000gn/T/orchestrate-briefs/kokoro-mac-feat-phonon-migration-f11dbf4f/26-streaming-convolution.report.md`.

The native operator module implements dense causal Conv1d with stride/dilation 1 and kernels 1/3/7, dense ConvTranspose1d with strides 4/5/6 and kernel `2*stride`, and identity residual blocks with `ELU → Conv3 → ELU → Conv1 → add`. Models own expanded copies of BF16 weights/biases and may outlive the source pack. Request states borrow immutable models and own bounded history, overlap and work buffers. Processing allocates nothing.

Padding, transpose right trim, retained overlap and bias placement follow the pinned source. Each input position emits `stride` output positions immediately, with no deferred operator flush. Buffers are row-major positions/channels; capacities count output positions. Invalid arguments preserve state; arithmetic failures poison it until reset. Empty input is a no-op. Cumulative input is bounded at 15,728,640 positions. Residual partial alias and overflow/reset cases are covered.

### Independent reference and Coordinator checks

The new `export_conv_fixture.py` loads the independent pinned Q8-mode reference, profiles actual Torch `_slow_conv2d_forward`/`slow_conv_transpose2d` dispatch, and exports all eight actual BF16 decoder module seams. It uses asymmetric seeded random/impulse inputs with a zero tail. Tests verify exported weights against the native pack, then compare outputs. ELU follows inspected Torch 2.7.1 `exp(x)-1`; the existing native exp4 interface suffices. No new math entrypoint or vendor change was needed.

The Coordinator independently ran `/private/tmp/speech-mac-26-coordinator-verify.sh` successfully on the committed source:

- Regenerated the ignored Q8/BF16 bridge and provenance. Bridge SHA256 independently confirmed as `c397c35fb111e64e8333d0c80b59bead90bfa2e6fcc5ab4ffd30a40f0ec8097d`.
- Actual operator tests under ASan/UBSan and `-O3 -DNDEBUG` with active checks. All eight seams pass unchanged `rtol=1e-3`, `atol=1e-5`; max absolute error `2.24113464e-05` in the initial 512-channel kernel7 convolution. Native whole/uneven outputs compare bitwise equal; reset-at-each-position outputs differ.
- Tiny asymmetric cross-correlation/scatter oracles and an independent double ELU residual oracle, max residual error `5.96046448e-08`, plus padding/phase/bias/overlap/capacity/config/size/nonfinite/alias/poison-reset cases.
- Complete preserved `make phonon-native-check`: 209-tensor/36-Q8/BF16 checks and earlier Flow/head/prefix/autonomous stages remain passing, with exact generated latents and independently decoded PCM.
- `make format` then `make check` with actual convolution, Mimi, pack and Flow fixture flags. All nine ticket25 feature gates still pass and native feature partitions remain exact. `git diff --check` and clean tree verified.

Coordinator logs: `/private/tmp/speech-mac-26-coordinator-{export,sanitizer,optimized,preserved,format,check}.log`. The worker additionally tested selected-missing fixtures and preserved model gates separately. No model skip was counted as actual parity evidence.

### Review disposition

Independent `/root/26_streaming_convolution/review26`, GPT-6.1 Sol / low, applied code-review-diff and code-simplify across all four paths and all six lenses. No runtime defect or other lens finding remained after the focused refresh.

| Id | Finding | Disposition and verification |
| --- | --- | --- |
| 26-C1 | Comparator could accept nonfinite reference values | Applied: native and expected values must both be finite; a deliberately NaN-corrupted bridge passed the old comparator and fails the fixed one; all actual seams remain green |

Temporary mutations dropping transpose overlap, changing causal insertion phase, removing residual identity and removing the full-width residual alias guard each fail focused tests. Source hashes confirmed immediate restoration; final sanitizer, optimized, standard and staged-hook checks passed. Worker evidence is in `.build/conv-red-{overlap,phase,residual,alias,oracle-nan}.log` and `.build/conv-green-oracle-nan.log`.

Only the seven inherited analyzer warnings and explicitly unavailable legacy/device checks remain. No performance, complete native waveform, engine or listening claim is made. Full assembly/postprocessing belongs to ticket27.

### Handoff

- Create immutable operator models from the actual pack views, then request states; destroy states before their models. Expanded weights are model-owned, so the operator models themselves need not retain the source pack.
- Compose the pinned decoder order and its outer ELUs in ticket27. Residual ELUs are already inside the residual seam. Preserve source bias/overlap placement and zero deferred flush; do not append synthetic codec tail frames.
- Input to the decoder is ticket25's 200Hz feature positions. Decoder ratios 6,5,4 lead to 24kHz mono samples and 1920 samples per 12.5Hz latent frame. Capacity units must remain explicit when assembling these APIs.

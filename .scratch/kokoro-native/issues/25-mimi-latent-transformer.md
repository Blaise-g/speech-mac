# 25: Verify Mimi projected latent features

## Parent

None (Requirement Source).

## Context

- Technical source: `.scratch/kokoro-native/spec.md` and its pinned primary sources.
- Background: `.scratch/kokoro-native/spec.md`.
- User clarification dated 2026-10-25: subsequent implementation, fixes and ticket reviews use implement-lite / gpt-6.1-sol / low, overriding the repository model/effort defaults. Final whole-branch review uses gpt-6.1-sol / high. Actual earlier Luna/low execution records remain historical evidence.

## Footprint

Mimi projection/upsampler/transformer, transposed-convolution primitive, shared transformer context interface, decoder feature fixtures/tests; consumes pack/numerical contracts.

## What to build

latent32 through quantizer output projection, config-derived 16x learned upsampling, and two-layer 512-wide/eight-head Mimi transformer to decoder-feature seam. Context policy must be the reconciled ticket19 decision, bounded state, layer scale .01 and erf-GELU. Framewise/uneven partitions cross >250 positions.

## Blocked by

24-flow-sampler

Worker Profile: medium; implement-lite / gpt-6.1-sol / low (user clarification 2026-10-25)

## Acceptance criteria

- [x] latent32 through quantizer output projection, config-derived 16x learned upsampling, and two-layer 512-wide/eight-head Mimi transformer to decoder-feature seam
- [x] Context policy must be the reconciled ticket19 decision, bounded state, layer scale .01 and erf-GELU
- [x] Framewise/uneven partitions cross >250 positions
- [x] Complete the approved slice checks and keep unavailable model/device checks explicit.

Status: done

## Execution record — 2026-10-25

- Base: `3f424ea886cd660c65b817db7e169716cb23bb6c`; branch `feat/phonon-migration`; existing `phonon-migration-planning` worktree.
- Worker: `/root/25_mimi_latent_transformer`; attempt 1; implement-lite / GPT-6.1 Sol / low. No repair follow-up used; no open Escalation.
- Result and verified run-branch SHA: `ecf18438875fd3ee06166c1bc1fc0c91bb969ce9`, with `Ticket: 25-mimi-latent-transformer`. All six paths fit the published Footprint; the shared transformer change only exposes/renames the existing batched linear helper without changing its arithmetic. Clean tree and protected tracker/fixture/budget diff verified.
- Atomic report: `/private/var/folders/j1/4vzqsp717kbb3gth_48lxdkc0000gn/T/orchestrate-briefs/kokoro-mac-feat-phonon-migration-f11dbf4f/25-mimi-latent-transformer.report.md`.

`phonon_mimi` now executes continuous 32D Mimi-space latents through learned BF16 projection, depthwise transpose convolution `[512,1,32]` with stride 16/right trim 16, and the actual two-layer 512-wide/eight-head transformer with learned layer-scale vectors and erf-GELU. Each latent produces sixteen feature positions immediately; there is no startup padding delay or deferred feature flush. Request state owns fixed ring caches/workspace, previous projected latent, absolute position and failure state. The model borrows pack views; state borrows the immutable model. Both must outlive the request.

The window is exactly `0 <= query - key < 250`. Keys are inserted query by query so a sixteen-position batch cannot evict history needed by an earlier query. Cache slots wrap while RoPE positions remain absolute. The API limits each request to 8192 latent frames. Invalid arguments leave progress untouched; processing failure poisons the state until reset and output may contain a partial prefix. Reset, interleaved requests and repeated runs are tested.

### Independent reference and Coordinator verification

Existing `mimi-temporal-263.npz` and `mimi-uneven-263-257.npz` contain original-F32 Mimi outputs. Their frozen latent inputs and trained-context semantics remain useful; their feature values are not actual GGUF-BF16 parity targets. The new independent `export_mimi_fixture.py` loads pinned `oracle.load('q8')` with all 209 published tensors, including 44 BF16 Mimi tensors, verifies frozen input/file hashes, and generates a separate ignored Q8-mode bridge. It first proves both Q8 hello feature arrays bitwise equal to their frozen expectations. No frozen fixture or budget was rewritten.

The Coordinator independently regenerated `.build/phonon-mimi-q8-v1.bin` plus provenance, then ran `/private/tmp/speech-mac-25-coordinator-verify.sh` successfully on the committed source:

- Real-model `tests/test_phonon_mimi.c` under ASan/UBSan and `-O3 -DNDEBUG`, with always-active checks.
- All nine independent feature expectations: hello40/32, long263 whole/framewise/uneven `[1,7,113,142]`, and masked263/257 rows plus their independent single-row references. Maximum absolute error `1.4603138e-05`; every element passes unchanged `rtol=1e-3`, `atol=1e-5` targets. Exact valid feature counts are640,512,4208,4208,4112.
- Native whole/framewise/uneven long outputs are bitwise identical, max error 0, within the frozen temporal allowance `1e-5`.
- Synthetic scalar graph covers causal overlap/startup/right trim, exact250-key boundary, independently chosen `.5`/`.25` learned scale values, resets, independent requests, overlap/size/config/nonfinite errors, failure/reset and cumulative limit enforcement.
- Complete `make phonon-native-check`: original209-tensor/36-Q8 and BF16 checks, existing Flow/head/prefix gates and exact autonomous generation/raw/final PCM remain passing.
- `make format` then `make check`, `git diff --check`, clean tree and unchanged frozen/tracker/pack paths.

Coordinator logs: `/private/tmp/speech-mac-25-coordinator-{export,sanitizer,optimized,preserved,format,check}.log`. The bridge's JSON records Q8 weights mode, model/Mimi/HF/Torch/source/input/budget/exporter hashes and corrected context; bridge SHA256 is `679a465e6336de3dc281d68d03022f97fb520865e7dea2771ec85e3dbac6e061` in the worker report. The worker also verified gate selection: a shared pack alone skips Mimi; an explicitly selected missing fixture/dependency fails.

### Review disposition

Independent `/root/25_mimi_latent_transformer/single_review`, GPT-6.1 Sol / low, applied code-review-diff and code-simplify across all six paths and Correctness, Spec, Standards, Reuse, Quality and Efficiency. No path/lens omitted.

| Id | Finding | Disposition and verification |
| --- | --- | --- |
| 25-R1 | Cumulative 8192-frame budget was not tested across calls | Applied: after 19 accepted frames/304 positions, an otherwise per-call-valid 8174-frame request exceeds remaining 8173 and is rejected without progress; independent review refreshed |

Temporary mutations to249-key context, dropped upsampler overlap and substituted constant scales fail the focused tests; removing cumulative budget enforcement fails ASan. Worker logs `.build/mimi-red-{window249,drop-overlap,constant-scale,cumulative}.log` retain the red evidence; every mutation was immediately restored with equal source hashes. Final sanitizer/optimized/full checks passed after the additions. No source failure or repair remained.

Seven inherited analyzer warnings remain visible, as already recorded in ticket24. Legacy model-backed/device checks explicitly skip when unavailable; all relevant Phonon gates executed. This ticket claims native decoder features only; PCM, streaming convolutions, engine/CLI integration, device/listening and measured performance remain later work.

### Handoff

- Call `phonon_mimi_model_load` on the resident pack, then create/reset/destroy request states while keeping model/pack alive. Inputs are row-major `[latent_frames,32]` in denormalized Mimi units; output is `[latent_frames*16,512]`. Capacity is feature rows, not float elements.
- Subsequent convolution work consumes these feature positions at200Hz and reaches24kHz samples through ratios6,5,4. The later PCM contract is1920 samples per latent with source-defined crop/flush behavior; no PCM was produced by this ticket.
- Preserve source-equivalent shared math, true ring semantics, learned weights and actual BF16 Q8-mode reference provenance. Use a new independent bridge for additional convolution/PCM seams, retaining all frozen source inputs/budgets.

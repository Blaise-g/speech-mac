# 20: Load verified assets and execute packed-Q8 CPU projections

## Parent

None (Requirement Source).

## Context

- Technical source: `.scratch/kokoro-native/spec.md` and its pinned primary sources.
- Background: `.scratch/kokoro-native/spec.md`.
- User approval dated 2026-10-25: “use implement lite with effort low for all”; resolved workflow/model/effort are implement-lite / gpt-6-luna / low, overriding AGENTS xhigh. Reviews also use Luna/low.

## Footprint

C asset loader, converter, pack schema/mappings, tensor/numerical APIs, scalar/Accelerate operations, C/Python tests, Makefile/notices, model inputs read-only.

## What to build

C asset inspection and genuine packed-Q8 transformer projection checks against the oracle. Use a checked versioned pack preserving published GGUF blocks, dtypes/layouts/source hashes, with development conversion allowed. Implement only actual supported formats verified in the downloaded artifact (expected Q8_0) plus required float tensors. Shape/config/overflow/truncation/ownership checks reject incompatible assets. Native CPU packed-weight dot/matmul must execute; expanding everything to F32 and calling it quantized is insufficient. Mimi and flow head stay float where reference does. No Q4, experimental Mimi quantization or Metal.

## Blocked by

19-phonon-reference

Worker Profile: medium

## Acceptance criteria

- [ ] C asset inspection and genuine packed-Q8 transformer projection checks against the oracle
- [ ] Use a checked versioned pack preserving published GGUF blocks, dtypes/layouts/source hashes, with development conversion allowed
- [ ] Implement only actual supported formats verified in the downloaded artifact (expected Q8_0) plus required float tensors
- [ ] Shape/config/overflow/truncation/ownership checks reject incompatible assets
- [ ] Native CPU packed-weight dot/matmul must execute
- [ ] expanding everything to F32 and calling it quantized is insufficient
- [ ] Mimi and flow head stay float where reference does
- [ ] No Q4, experimental Mimi quantization or Metal
- [ ] Complete the approved slice checks and keep unavailable model/device checks explicit.

Status: ready-for-agent

## Attempt 1 — partial, not verified, 2026-10-25

- Worker: `/root/20_native_model_q8`, `implement-lite` / `gpt-6-luna` / low, attempt 1. Wave base: `136f82d48e22e6f2bbb2a121aab9f4ee0c23f95a`; run branch: `feat/phonon-migration`; worktree: `/Volumes/Code/Personal/kokoro-mac/.worktrees/phonon-migration-planning`.
- Partial commit: `593daadb24c2928a1ba33c3bb38fbb60a653e298`, with `Ticket: 20-native-model-q8`. It adds only `src/phonon_q8.c`, `src/phonon_q8.h`, and `tests/test_phonon_q8.c`. Coordinator verified the commit/trailer, in-scope paths and clean tree. The partial result is preserved on the run branch but does not satisfy the ticket.
- Coordinator passed `make format`, `make test-one TEST=tests/test_phonon_q8.c`, and `make check`. Logs are `/private/tmp/speech-mac-20-partial-format.log`, `/private/tmp/speech-mac-20-partial-focused.log`, and `/private/tmp/speech-mac-20-partial-check.log`. Legacy Kokoro model/voice and device cases were explicitly skipped. The worker additionally reported passing `make phonon-reference-check`; that validates the Python reference, not the missing native model projections.
- The independent review fixed a signed-fixture initialization issue and identified the remaining specification gaps. The versioned asset converter/pack, C loader/inspection, required floating-point handling, malformed-pack/overflow checks, and actual pinned-model native projections remain unimplemented. No acceptance checkbox is marked complete from the synthetic kernel alone. Ticket 21 remains held.
- Integration outcome: failed acceptance coverage; not verified. The current ready-for-agent status is retained until the complete ticket can be verified under the local-tracker rules.
- Repair follow-up 1 of the two approved follow-ups is authorized, retaining the original scope, footprint, frozen budgets and low profile. Its instructions are `/private/tmp/speech-mac-20-repair1.md`; attempt 2 starts from the new run tip after this record is committed. It must complete the remaining original criteria and re-review all changed paths since the original wave base before reporting completion.
- Attempt-1 report preserved at `/private/tmp/speech-mac-20-attempt1.report.md`. No external environment blocker or product decision was reported; this continuation changes no approved requirement.

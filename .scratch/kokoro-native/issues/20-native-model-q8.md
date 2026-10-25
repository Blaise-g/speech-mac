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

## Attempt 2 — implemented, verification repair required, 2026-10-25

- Repair follow-up 1 delivered `0cf89eeb59822269a3f69ed787ff20a98d01ad95`, with `Ticket: 20-native-model-q8`, from follow-up base `44de7b938b6d114bac5feb3baa3decbb8ff63dd9`. Coordinator checked the commit, clean tree and in-scope loader/converter/tests/Makefile changes. The original kernel commit remains on the branch.
- The completed implementation preserves all 209 published tensors and implements checked ownership, canonical schema/provenance, metadata/index/payload integrity, bounds/alignment/type/shape rejection, BF16 decoding and actual packed-Q8 projections. The independent Luna/low reviewer reported all six findings applied and no remaining issue. The attempt-2 report is preserved at `/private/tmp/speech-mac-20-attempt2.report.md`.
- Coordinator independently passed `make phonon-native-check`: the actual pinned pack loaded, its 209 tensors were checked, all 36 Q8 projections matched the oracle, and BF16 comparisons ran. Log: `/private/tmp/speech-mac-20-coordinator-native.log`. Production-symbol inspection confirmed the test-only parser API is absent from `.build/kokoro`.
- Coordinator verification then stopped at `make test-one TEST=tests/test_phonon_pack.c TEST_CFLAGS='-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer'`. Compilation fails at three `sprintf` calls in `src/phonon_pack.c` because the macOS SDK marks them deprecated under this build and warnings are errors. The sanitizer did not execute. Log: `/private/tmp/speech-mac-20-coordinator-sanitizer.log`.
- The conventional `make format` and `make check` rerun passed; logs are `/private/tmp/speech-mac-20-pre-repair2-format.log` and `/private/tmp/speech-mac-20-pre-repair2-check.log`. The default suite runs weightless parser cases and explicitly skips the separately executed real-model pack path, unavailable legacy model cases, and opt-in device smoke.
- Integration outcome: still unverified pending the compiler correction and sanitizer run; ticket 21 remains held. Repair follow-up 2 of 2 is authorized at the same low profile and original footprint, using `/private/tmp/speech-mac-20-repair2.md`. Attempt 3 starts after this tracker-only record is committed. It replaces the deprecated digest formatting, retains schema/weights/frozen budgets, runs the exact sanitizer command and native checks, and obtains independent review before committing. No third follow-up is pre-approved.

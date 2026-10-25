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

# 02: Produce Kokoro contextual phoneme features in C

## Context

Technical source: `.scratch/kokoro-native/spec.md`, the official [model configuration](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/config.json), [inference path](https://github.com/hexgrad/kokoro/blob/main/kokoro/model.py), and [approved Run Plan](../run-plan.md).

Background: [Kokoro native inference spec](../spec.md).

Worker Profile: implement-lite · xhigh · gpt-6-luna (override)

## What to build

Prepared phoneme IDs and Kokoro v1.0 model data produce the contextual token features consumed by the duration and acoustic paths.

## Acceptance criteria

- [x] C loads the required model tensors from the versioned pack and runs the v1.0 PL-BERT/ALBERT embedding, attention, masking, and projection path.
- [x] The C pack loader validates tensor names and shapes and rejects truncated, malformed, or incompatible model packs.
- [x] Valid token sequences return the expected feature shape; invalid IDs and sequences beyond the configured context fail clearly.
- [x] Stage outputs match the pinned fixtures within ticket 01's pre-established budget. A miss prompts investigation and a fix; any budget change needs independent numerical evidence and written rationale.
- [x] No Python inference runtime is required by the native path.

## Blocked by

- `01-reference-oracle-and-weight-pack` — fixture and weight-pack contract.

**Status:** done

## Verification — 2026-10-11

Commit: `421542d2f43bea8fcc8984b9e35967b51095cbde`. Coordinator reran `make check`: native context parity passed (max abs 2.0980835e-05, unchanged atol=1e-5/rtol=1e-3), parser checks passed. Locked uv reference suite: 17 tests passed. Verified clean tree and Ticket trailer. Independent review fixed payload/tensor alignment; no remaining blocker. Runtime CLI is a later ticket. Relative converter output status-print issue deferred as tooling follow-up.

# 01: Pin the reference and produce C-readable model data

## Context

Technical source: `.scratch/kokoro-native/spec.md`, the official [Kokoro-82M model card](https://huggingface.co/hexgrad/Kokoro-82M), [model config](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/config.json), and [approved Run Plan](../run-plan.md).

Background: [Kokoro native inference spec](../spec.md).

Worker Profile: implement-lite · xhigh · gpt-6-luna (override)

## What to build

Create a repeatable reference path that prepares official v1.0 weights and one voice for later native inference work, and produces authoritative intermediate and full-audio fixtures without adding Python to shipped inference.

## Acceptance criteria

- [x] Pin immutable official model, config, inference source, voice (`af_heart` proposed), and reference tool revisions by revision or content hash.
- [x] Convert the checkpoint and voice style to a documented C-readable tensor pack; keep full model and voice weights outside Git.
- [x] Generate compact fixtures from fixed prepared phoneme IDs for every planned module boundary and final waveform. Record provenance, shapes, and dtypes.
- [x] Verify fixture repeatability across reference runs. Before native C inference is implemented, establish per-stage and final-audio error metrics and budgets from reference repeatability plus a documented numerical-precision rationale; do not use native C output to choose budgets.
- [x] Provide reference-side validation for tensor names, shapes, metadata, and malformed/truncated/incompatible packs. Keep C inference implementation out of this ticket so budgets are fixed first.

## Blocked by

- None (can start immediately).

**Status:** done

## Verification — 2026-10-11

Commits: `54e6e276f5f640433c15305af110f2a47d2c6a59`, `12d3ad58fea8b2edc5dbd5aa1f1434cfba540e41`. Coordinator verified clean tree, both Ticket trailers, and reran documented locked uv suite: 15 tests passed. `make check` succeeds with explicit no-C/empty-suite skips. Worker verified deterministic 549-tensor conversion and 23-array oracle repeatability. Independent review fixed digest/group validation and documentation issues. Low-priority memory copies and duplicate pins deferred; fixtures currently cover hello only.

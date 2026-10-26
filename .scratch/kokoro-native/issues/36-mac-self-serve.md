#36: Public Mac setup and README

## Context

User2026-10-26: “how can i test it via command? also make the repo public and serf serveable on mac with a proper readme using $unslop”. Scope/Run Plan approved by Coordinator in active spec. Prior migration is merged in897b3da. Use implement-lite / GPT-6.1 Sol / low, final review Sol/high. Root is sole tracker/admin writer.

## Footprint

Makefile; scripts/ bootstrap helper(s); reference/phonon canonical setup/conversion/download helpers and tests only as needed; README.md, reference/phonon/README.md, CONTRIBUTING.md, docs/remote-setup.md; code LICENSE only after explicit user selection; setup-focused tests and CI wiring for those model-free tests. No inference C/math/fixtures/budgets/vendor/lock changes without a Worker Question. No historical artifact edits. Root-only GitHub visibility/description/readback and tracker.

## Acceptance criteria

- [ ] `make bootstrap` prepares an Apple Silicon Mac checkout using installed Xcode CLI tools and uv: locked environment, verified pinned assets, canonical native pack, ignored default model/voice and release CLI. Existing `make setup` remains hook setup.
- [ ] Prerequisites/unsupported platform/missing gated access fail clearly; HF credentials stay local; verified existing source assets can be reused offline; repeated setup is safe and idempotent; existing unrelated/custom files are preserved. No automatic audio/system/global environment changes.
- [ ] Model-free tests expose setup failure/reuse/destination behavior; actual fresh-checkout setup and deterministic default-output WAV smoke pass using verified existing40M assets, without user speech playback.
- [ ] README unslop explains prerequisites, fresh clone/auth/setup, explicit audible playback versus saved WAV, files/stdin/stream, voices/native controls, resource costs, troubleshooting and separate model/third-party/code licensing. Portable paths; no claim of unsupported languages/Intel/server/global install.
- [ ] Independent changed-path review before Ticket36 commits; make format then make check; relevant locked setup tests; final Sol/high review and CI pass.
- [ ] Root verifies public GitHub identity/history/settings/CI/anonymous read and updates stale repository description/current private-only documentation, after the redacted reachable-history credential audit.

Worker Profile: medium; implement-lite / gpt-6.1-sol / low.

Status: in-progress

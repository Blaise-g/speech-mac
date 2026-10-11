---
status: done
---

# Local development foundation

Establish the local Git repository, reproducible Make-based C checks, repository-local staged-snapshot hook, macOS CI workflow, agent and contributor guidance, and committed local tracker conventions.

## Verification

Verified 2026-10-11 on Apple Silicon macOS with Xcode Command Line Tools:

- `make setup` installed `.githooks` for this clone.
- `make format` completed; the repository has no C sources or headers yet.
- `make check` ran format check, lint, typecheck, build, and tests in order. The first four steps explicitly skipped because there are no C sources; the test suite explicitly reported empty.
- `xcrun --find clang-format` and `xcrun clang-format --version` resolved Apple clang-format 21.0.0.
- In disposable repositories, `make test-one TEST=tests/test_smoke.c` built and ran a focused C test; an out-of-repository test path was rejected. Removing `src/` made `make check` fail in `source-discovery`.
- The staged-snapshot hook accepted a representative C implementation and focused test, rejected a staged missing-header change with a nonzero result, then accepted a staged valid edit while an unstaged invalid edit remained untouched.

The CI workflow is configured for the GitHub-hosted `macos-15` runner; no remote repository was created or pushed, so no remote CI run was observed. The host is Apple Silicon, but model inference, reference comparison, listening checks, and performance measurements cannot run until implementation and model weights are present. eSpeak NG and model weights are not installed; neither is required by this foundation. Python reference/conversion dependencies will be introduced with their first tool.

The product spec at `.scratch/kokoro-native/spec.md` is intentionally deferred to the coordinator after setup; this commit contains the tracker rules and setup ticket only.

## Review record

The independent reviewer used fixed point `3d0f7cf296d29ba16e84f363ba22ee5701fe0e1b` and examined all setup paths under Correctness, Standards, Spec, Reuse, Quality, and Efficiency. The follow-up checked the source-discovery and focused-test fixes under Correctness. No paths or lenses were omitted; the absent product spec is deferred to the coordinator as described above.

| ID | Lens | Finding | Disposition | Reason | Verified by |
| --- | --- | --- | --- | --- | --- |
| C1 | Correctness | Failed source discovery could be hidden and checks skipped. | Applied | `source-discovery` now propagates `find` failures. | `check: make check`; `check: disposable missing-src case` |
| C2 | Correctness | `test-one` could run a matching test file outside the repository. | Applied | It resolves the physical test directory and requires a regular, non-symlink file directly under `tests/`. | `check: disposable focused test and outside-path rejection` |
| S1 | Spec | The product spec file is not present yet. | Deferred | The request assigns feature spec and tickets to the coordinator after setup. | `not verified` |

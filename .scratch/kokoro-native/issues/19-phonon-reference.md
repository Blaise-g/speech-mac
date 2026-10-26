# 19: Pin artifacts and a replayable 40M oracle

## Parent

None (Requirement Source).

## Context

- Technical source: `.scratch/kokoro-native/spec.md` and its pinned primary sources.
- Background: `.scratch/kokoro-native/spec.md`.
- User approval dated 2026-10-25: “use implement lite with effort low for all”; resolved workflow/model/effort are implement-lite / gpt-6-luna / low, overriding AGENTS xhigh. Reviews also use Luna/low.
- Oracle environment: Coordinator PATH lacked cargo/rustc. Pin a correct executable oracle or provision isolated development tools; a cached binary alone is not correctness evidence.

## Footprint

reference pins/tooling, uv project+lock, fixture manifest/budgets/exporters/tests/build integration, reference docs/licenses; shared assets read-only. Legacy oracle remains temporarily while replacement grows.

## What to build

a reproducible model/reference probe, fixed numerical budgets and intermediate fixtures before native work. Verify supplied assets against `local-verification.json` and authoritative revision/hash provenance; record licenses/access and catalog provenance. Pin xn-ptts `ac0e1524cfd9e71d57aadca0de542796f942d775` and model `cc738b7e4d182c2ba1bafb8062aa4846af000637`, oracle dependencies/locks. Upstream Rust may run only as a development oracle, never linked/shipped runtime. Resolve upstream Mimi context discrepancy (PR209) by identifying trained semantics and an independent pinned correct oracle, or a minimal documented reference patch with provenance. Decoder budget freezing is blocked until reconciled. Export exact normalized text/IDs/voice projection/prefix states/backbone output/raw EOS/noise draws/flow integration/latents/Mimi features/raw and final PCM/stop lengths. Fixtures cross >250 context positions with uneven batches. Do not silently treat buggy pinned output as truth or loosen tolerances to hide it.

## Blocked by

None (can start immediately)

Worker Profile: medium

## Acceptance criteria

- [x] a reproducible model/reference probe, fixed numerical budgets and intermediate fixtures before native work
- [x] Verify supplied assets against `local-verification.json` and authoritative revision/hash provenance
- [x] record licenses/access and catalog provenance
- [x] Pin xn-ptts `ac0e1524cfd9e71d57aadca0de542796f942d775` and model `cc738b7e4d182c2ba1bafb8062aa4846af000637`, oracle dependencies/locks
- [x] Upstream Rust may run only as a development oracle, never linked/shipped runtime
- [x] Resolve upstream Mimi context discrepancy (PR209) by identifying trained semantics and an independent pinned correct oracle, or a minimal documented reference patch with provenance
- [x] Decoder budget freezing is blocked until reconciled
- [x] Export exact normalized text/IDs/voice projection/prefix states/backbone output/raw EOS/noise draws/flow integration/latents/Mimi features/raw and final PCM/stop lengths
- [x] Fixtures cross >250 context positions with uneven batches
- [x] Do not silently treat buggy pinned output as truth or loosen tolerances to hide it
- [x] Complete the approved slice checks and keep unavailable model/device checks explicit.

Status: done

## Coordinator verification — 2026-10-25

- Worker: `/root/19_phonon_reference`, attempt 1, `implement-lite` / `gpt-6-luna` / low, matching the approved override.
- Wave base: `9d2359507652e1999a6432c534dd96195c700c23`; branch: `feat/phonon-migration`; worktree: `/Volumes/Code/Personal/kokoro-mac/.worktrees/phonon-migration-planning`.
- Result commit and verified run-branch tip: `7347886ef53cd5520d79aa7d1af92c97f2154ffd`, with `Ticket: 19-phonon-reference`. Coordinator checked the clean tree, commit trailer and changed-path footprint.
- Coordinator independently passed `make phonon-reference-check` (14 tests, 12 verified assets, float and published-Q8 replay, corrected actual attention mask, uneven-batch and temporal decoder checks), then `make format` and `make check`. The temporal probe verified 504,960 PCM samples and zero deferred flush samples. Logs: `/private/tmp/speech-mac-19-coordinator-reference.log`, `/private/tmp/speech-mac-19-coordinator-format.log`, `/private/tmp/speech-mac-19-coordinator-check.log`.
- `make check` explicitly skipped the unavailable legacy Kokoro model/voice cases and the opt-in AudioQueue device smoke. Real Phonon reference checks executed; native C parity and playback are later tickets, not evidence claimed here.
- Independent review accepted and fixed two findings: derived-Q8 config/dtype cache validation, and tokenizer temporary-directory cleanup. Regression checks and reviewer recheck passed. No remaining review finding was reported. Worker report: `/private/var/folders/j1/4vzqsp717kbb3gth_48lxdkc0000gn/T/orchestrate-briefs/kokoro-mac-feat-phonon-migration-f11dbf4f/19-phonon-reference.report.md`.
- Frozen float/Q8 native targets are in `reference/phonon/budgets.json`. Same-oracle replay, native targets, and float-versus-Q8 observations remain distinct. The published GGUF contains 36 Q8_0 and 173 BF16 tensors; all 209, including BF16 Mimi weights, are reconstructed for its reference. The float reference retains the original decoder weights. Native tickets must meet the fixed targets without tuning them to their own output.
- HF tokenizer character normalization is labelled separately from the English runtime rewrite policy to be implemented in ticket 21. Generation explicitly uses temperature 0.8 as variance and the pinned processor's EOS tail settings.
- Integration outcome: verified on the run branch; ticket 20 may start after this tracker record is committed.

## Coordinator questions and answers — 2026-10-25

- **Q19-1, local tooling:** The worker asked whether `Environment access: none` prohibited missing development dependencies/network setup. Answer: no; the approved footprint/spec already permit isolated reference tooling and pinned local dependencies. No deployment or administrative environment was granted. Use locked `uv` tooling under the worktree and normal temporary caches; an isolated development-only toolchain is permitted if needed, with no global install/profile changes. This clarified existing scope and did not split or reduce the ticket.
- **Q19-2, reference discrepancy:** HF conversion also used 251 keys. Answer: take the already-approved minimal documented reference-patch route, pin the executable HF source, set Mimi to the trained 250-position rule, and test the actual mask at the boundary plus temporal partitions. Evidence and the intentional difference from uncorrected upstream are recorded in the reference project. No tolerance widening or fixture-only substitute was accepted.
- **Q19-3, old cache cleanup:** A guardrail rejected recursive removal of `.build/reference-phonon-q8` because `.build` was outside its recursive-removal allowlist. Answer: do not retry or bypass it; leaving the ignored hybrid cache unused is sufficient. The corrected cache has a separate content-derived identity and validated source/output/config digests. Deletion was not required by acceptance.

## Deferred environment item

The unused ignored hybrid reference cache remains for optional future cleanup. It is neither consumed nor shipped and does not block ticket 19. Native parity and device validation remain assigned to their later tickets.

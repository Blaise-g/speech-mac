# 21: Normalize ordinary English in C

## Parent

None (Requirement Source).

## Context

- Technical source: `.scratch/kokoro-native/spec.md` and its pinned primary sources.
- Background: `.scratch/kokoro-native/spec.md`.
- User approval dated 2026-10-25: “use implement lite with effort low for all”; resolved workflow/model/effort are implement-lite / gpt-6-luna / low, overriding AGENTS xhigh. Reviews also use Luna/low.

## Footprint

UTF-8/normalizer module, public text interface, normalization corpus/reference fixtures/tests, frontend docs/build integration.

## What to build

actual native English default normalization at a probe seam: character/Unicode/punctuation policy plus numbers/currency/dashed digits/emails/URLs/abbreviations/elongations, with malformed UTF-8 and empty-after-normalization checks. Preserve default policy rather than adding optional date/phone/time rewrite controls. Validate en; no Python/Rust/eSpeak child process or fixture lookup implementation.

## Blocked by

20-native-model-q8

Worker Profile: medium

## Acceptance criteria

- [x] actual native English default normalization at a probe seam: character/Unicode/punctuation policy plus numbers/currency/dashed digits/emails/URLs/abbreviations/elongations, with malformed UTF-8 and empty-after-normalization checks
- [x] Preserve default policy rather than adding optional date/phone/time rewrite controls
- [x] Validate en
- [x] no Python/Rust/eSpeak child process or fixture lookup implementation
- [x] Complete the approved slice checks and keep unavailable model/device checks explicit.

Status: done

## Execution record — 2026-10-25

- Wave base: `a27d96bbf2043f4f4cc40fa24075fe8dacc7cc81`.
- Branch: `feat/phonon-migration`; worktree: `/Volumes/Code/Personal/kokoro-mac/.worktrees/phonon-migration-planning`.
- Worker: `/root/21_native_english_normalization`; attempt 1; implement-lite / gpt-6-luna / low, as approved. No repair follow-up used.
- Result and verified run-branch SHA: `d0c63e751477d1a4750859ab19bdb653ff1929b6`, with `Ticket: 21-native-english-normalization`. The Coordinator confirmed the trailer, clean tree, and all 33 changed paths within the Footprint.
- Report: `/private/var/folders/j1/4vzqsp717kbb3gth_48lxdkc0000gn/T/orchestrate-briefs/kokoro-mac-feat-phonon-migration-f11dbf4f/21-native-english-normalization.report.md`.

The native API implements the pinned default English character pass, seven ordered word rules, and multiword spans. It validates UTF-8 and `en`, rejects embedded NUL, returns caller-owned output including empty output, and reports allocation failure without partial output. Generated Unicode predicates follow the isolated Rust reference toolchain rather than process locale. Generic Kokoro status/error helpers remain temporary shared infrastructure until ticket31 removes the old identity.

The development-only dependency-free Cargo oracle copies pinned xn-ptts preprocessing modules, with licenses and provenance. The Coordinator independently matched the two main module hashes against checkout `ac0e1524cfd9e71d57aadca0de542796f942d775`; the independent reviewer checked the copied rewrite modules. Optional upstream language/rule source in that oracle is reference material, not a shipped runtime control. Official Rust 1.99.1 was installed under ignored `.build/reference-toolchain`, without global configuration or PATH changes. The product normalizer does not execute it.

### Coordinator verification

- Passed `make test-one TEST=tests/test_english_normalize.c TEST_CFLAGS='-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -DPHONON_ENGLISH_NORMALIZE_TEST_ALLOCATOR'`. Allocation injection distinguishes a triggered failure from an allocation index never reached, requires out-of-memory/null/zero-length output for each triggered failure, and checks successful output.
- Passed `UV_CACHE_DIR=/private/tmp/kokoro-uv-cache uv run --project reference/phonon --offline --no-sync --locked python reference/phonon/normalization/compare.py`: all 173 cases matched both pinned Rust output and native C. The runner rebuilds its native probe on every run.
- Passed `make format`, then `make check`. Logs: `/private/tmp/speech-mac-21-coordinator-{sanitizer,differential,format,check}.log`.
- `nm .build/kokoro` showed the public normalizer/free functions and no normalizer test-allocation API. The final working tree was clean.
- Worker additionally passed ordinary focused tests and optimized `-O3 -DNDEBUG` focused tests. Its final independent Luna/low review found no remaining correctness findings after fixes.
- Existing Kokoro model-backed checks skipped because that ignored legacy pack is absent in this worktree; AudioQueue device checks remain opt-in. The new real Phonon pack check is separate and was already executed for ticket20. This weightless frontend ticket makes no new synthesis/device claim.

### Review disposition

| Id | Lens | Finding | Applied / rejected / deferred | Reason | Verified by |
| --- | --- | --- | --- | --- | --- |
| 21-R1 | Correctness | Fixed-size range, decimal, currency and exponent buffers could truncate or overread long inputs | Applied | Preserve supported input and owned bounds | Sanitizer focused tests and differential cases at `d0c63e7` |
| 21-R2 | Spec | Unicode predicates, currency cents/signs, malformed email overlap and rule order differed from source | Applied | Pinned default English policy is authoritative | 173-case executable differential at `d0c63e7` |
| 21-R3 | Correctness / Spec | Missing rank span; its first fix overread EOF and accepted partial second words | Applied | Match whole-token upstream suffix handling | EOF, punctuation and repeated-space regressions; final independent review |
| 21-R4 | Correctness / Spec | Ending-in compared only the last output word instead of the whole prior emitted item | Applied | Rewrites must preserve upstream item boundaries | Slash/email context regressions and differential check |
| 21-R5 | Correctness | Failed rewrite allocations could be treated as an unclaimed rule | Applied | Fail without silently changing text behavior | Trigger-aware allocation injection under ASan/UBSan |
| 21-R6 | Correctness | Differential runner could use a stale native probe | Applied | Compare the current implementation | Runner always invokes its make target; Coordinator differential passed |
| 21-R7 | Standards | Three unsuppressed Clang analyzer warnings remain in bounded dashed-digit/URL reads | Rejected as defects | Independent review traced bounds to NUL-terminated token lengths and parsed domain ends | Static review plus sanitizer checks; analyzer still reports these warnings |
| 21-R8 | Reuse / Quality / Efficiency | No additional changes requested by simplification review | Rejected | No actionable finding | Independent Luna/low review |

### Worker Question 21-1 — 2026-10-25

Question: the initial worker stopped before edits because the port was substantial and behavior had not yet been established, without a failed command or an external blocker.

Answer: the pinned source and executable oracle define the behavior; establishing that behavior and porting it are this ticket's approved work. Continue within attempt1, staging the oracle, character pass, ordered rules, and differential checks internally. Task size alone does not change scope or require a ticket split. Source: this ticket and the approved Requirement Source. The worker resumed at the same base and completed the full slice; this clarification was not a post-report verification repair.

Deferred environment finding: the three analyzer warnings remain visible and unsuppressed as reviewed false positives. No user escalation or unresolved ticket criterion remains.

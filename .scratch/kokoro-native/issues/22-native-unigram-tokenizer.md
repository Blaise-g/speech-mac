# 22: Tokenize and bound long requests in C

## Parent

None (Requirement Source).

## Context

- Technical source: `.scratch/kokoro-native/spec.md` and its pinned primary sources.
- Background: `.scratch/kokoro-native/spec.md`.
- User approval dated 2026-10-25: “use implement lite with effort low for all”; resolved workflow/model/effort are implement-lite / gpt-6-luna / low, overriding AGENTS xhigh. Reviews also use Luna/low.

## Footprint

tokenizer/table/converter representation, native request text planning APIs, token/prompt/chunk fixtures/tests, pack metadata/build wiring.

## What to build

exact tokenizer IDs and decoding: verified 4000-vocabulary Unigram, byte fallback, Prepend+Metaspace, special token behavior matching encode(text,false). Port prompt preparation, ordered sentence grouping and oversized-sentence fitting without silent loss. Explicit token/frame/context bounds; whitespace/Unicode/unknown bytes/ties/punctuation/long words/NUL/invalid text checks. No runtime tokenizer library or IPA input.

## Blocked by

21-native-english-normalization

Worker Profile: medium

## Acceptance criteria

- [x] exact tokenizer IDs and decoding: verified 4000-vocabulary Unigram, byte fallback, Prepend+Metaspace, special token behavior matching encode(text,false)
- [x] Port prompt preparation, ordered sentence grouping and oversized-sentence fitting without silent loss
- [x] Explicit token/frame/context bounds
- [x] whitespace/Unicode/unknown bytes/ties/punctuation/long words/NUL/invalid text checks
- [x] No runtime tokenizer library or IPA input
- [x] Complete the approved slice checks and keep unavailable model/device checks explicit.

Status: done

## Execution record — 2026-10-25

- Wave base: `ec2cbde7b1740b6a7c082090295fa16d04b1d4ee`.
- Branch: `feat/phonon-migration`; worktree: `/Volumes/Code/Personal/kokoro-mac/.worktrees/phonon-migration-planning`.
- Worker: `/root/22_native_unigram_tokenizer`; attempt 1; implement-lite / gpt-6-luna / low, as approved. No repair follow-up used.
- Result and verified run-branch SHA: `110b1177ee5fb2758e9e477635bcb38a22587677`, with `Ticket: 22-native-unigram-tokenizer`. The Coordinator confirmed the trailer, clean tree, and all 16 changed paths within the Footprint.
- Atomic report: `/private/var/folders/j1/4vzqsp717kbb3gth_48lxdkc0000gn/T/orchestrate-briefs/kokoro-mac-feat-phonon-migration-f11dbf4f/22-native-unigram-tokenizer.report.md`.

The native table and generator preserve the pinned 4,000-piece tokenizer's scores, byte IDs, special-token policy, preprocessing, and decoder order. Ordinary input is NUL-free valid UTF-8; literal spellings such as `<s>` do not become NUL-prefixed reserved tokens. Viterbi search uses f64 scores and UTF-8 scalar edges, with source-matched unknown scoring and byte expansion. Candidate buckets bound prefix work for long words. Decode filters special IDs before the Metaspace, ByteFallback, and Fuse sequence.

The planner consumes normalized ordinary text from the preceding frontend seam. It prepares prompts with Rust Unicode 17.0 semantics, groups sentences in source order, and fits oversized pieces using sentence/clause/nearest-token boundaries or Unicode-scalar word splits. The grouping target is distinct from the default hard fit cap of 200 tokens; `phonon_prompt_fit` can apply a smaller hard limit and preserves the original plan on failure. Bounds include 1 MiB input, `3 * input_limit + 4` token IDs, 4,096 frames, and 8,192 context positions, with 512 prompt-headroom positions. Impossible fits return an error; fitting does not silently truncate text. The runtime uses native C without an interpreter or tokenizer library.

### Coordinator verification

- Passed focused `make test-one TEST=tests/test_phonon_tokenizer.c TEST_CFLAGS='-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer'` and `TEST_CFLAGS='-O3 -DNDEBUG'`. Checks remain active under NDEBUG. Fixtures cover exact chunk text/IDs/order, EOS tails, frame and context budgets, punctuation, non-divisible fit limits, ties, malformed text, bounds, long input, and impossible fits.
- Passed locked-uv `python tools/verify_phonon_tokenizer.py`: all 6,308 encoding and 9,004 decoding cases matched actual tokenizers 0.23.3. Coverage includes every ordinary vocabulary piece, all 4,000 single IDs, seeded combinations, valid Unicode, invalid byte runs, and mixed byte/special sequences.
- Passed locked-uv `python tools/verify_prompt_prepare.py`: all 9,133 cases matched the exact extracted pinned Rust preparation function, compiled with the isolated Rust toolchain. Source SHA-256: `a9a57097d5786570a48948c02a91840409e2725bfb7887ce00b3d65e2ceffbe5`; function SHA-256: `7ac23d079feed46473aede020e582a622f2dbf3ce5fa2108a6a359a540a00e93`.
- Python commands used `UV_CACHE_DIR=/private/tmp/kokoro-uv-cache uv run --project reference/phonon --offline --no-sync --locked`. The planner fixtures use a separate source-port oracle for their corpus; full Unicode preparation is checked against Rust directly.
- Passed `make format`, then `make check`; final tree clean. Logs: `/private/tmp/speech-mac-22-coordinator-{sanitizer,optimized,tokenizer,prepare,format,check}.log`.
- Independent Luna/low review covered the changed behavior, generators, verifiers, fixtures, and pinned sources under all six implement-lite lenses. Final refresh found no actionable findings.
- Existing legacy Kokoro model/voice checks skipped because their ignored artifacts are absent; AudioQueue smoke remains opt-in. `make check` does not invoke the separate `make phonon-native-check`; ticket20 already verified the available Phonon pack and Q8 projections. Phonon inference and playback are subsequent tickets, not checks completed by this frontend slice. The worker corrected its report's initial misleading missing-Phonon-pack explanation without code or git changes.

### Review disposition

| Id | Lens | Finding | Applied / rejected / deferred | Reason | Verified by |
| --- | --- | --- | --- | --- | --- |
| 22-R1 | Correctness | Initial encoder scanned an unterminated ID array and approximated fallback/scoring | Applied | Match the actual scalar Viterbi algorithm with checked ownership and bounds | Focused sanitizer tests and 6,308 encoding comparisons at `110b1177` |
| 22-R2 | Correctness | Byte decoding removed a Metaspace marker created only by byte fallback | Applied | Metaspace precedes ByteFallback; special IDs are filtered first | Red case `[630,0,554,533]`, C regression, 9,004 decoding comparisons |
| 22-R3 | Spec | Python Unicode 15 tables differed from the Rust Unicode 17 reference | Applied | Generate properties and full uppercase expansions from the pinned toolchain | 9,133 direct Rust comparisons |
| 22-R4 | Correctness / Spec | Fitting used floor instead of ceil, applied punctuation minimums to fallback cuts, and included an extra closer | Applied | Preserve pinned cut order, token imbalance, first ties, and exact closer set | Source-port fixture comparisons, including limits 8 and 9 |
| 22-R5 | Correctness | Whitespace-only preparation leaked; joined sentence text could outgrow the original byte buffer | Applied | Free empty preparation and grow grouped text with checked bounds | Sanitizer tests and `a.a.a` expansion regression |
| 22-R6 | Efficiency | Draft prefix search performed excessive work on long unbroken input | Applied | Generate bounded first-byte candidate buckets | Long-word tests, optimized focused checks, broad tokenizer comparisons |
| 22-R7 | Correctness | Reviewer proposed changing the sentence-marker extraction algorithm | Rejected | Pinned `tts_model.rs` uses the same encoding of `.!...?` and skips its first ID | Source review and exact punctuation/grouping fixtures |
| 22-R8 | Correctness / Quality | Output initialization, assertion-only tests, and tooling provenance/paths needed hardening | Applied | Clear outputs before validation, retain optimized checks, use explicit hash/version gates and portable tool paths | Sanitized/optimized tests, locked oracle commands, independent review |

### Worker Question 22-1 — 2026-10-25

Question: whether a provenance-bearing generated vocabulary include was outside the Footprint; the worker initially stopped before implementation without a concrete external blocker or report file.

Answer: the include, generator, and manifest are explicitly within “tokenizer/table/converter representation” and the fixture/build Footprint. Source: this ticket and the approved native frontend/provenance requirements. Continue the same attempt at the unchanged base, implementing the table, actual algorithm, planner, tests, and review. Task size does not require a scope change. A later progress-only final was also resumed in the same assignment with existing work preserved; no implementation verification repair or user escalation was needed.

No unresolved ticket criterion or user escalation remains. The three previously reviewed normalizer analyzer warnings from ticket21 remain visible in the full check; this ticket introduced no additional analyzer warning requiring a disposition.

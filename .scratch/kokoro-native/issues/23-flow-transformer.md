# 23: Verify voice/text priming and cached Flow-LM

## Parent

None (Requirement Source).

## Context

- Technical source: `.scratch/kokoro-native/spec.md` and its pinned primary sources.
- Background: `.scratch/kokoro-native/spec.md`.
- User approval dated 2026-10-25: “use implement lite with effort low for all”; resolved workflow/model/effort are implement-lite / gpt-6-luna / low, overriding AGENTS xhigh. Reviews also use Luna/low.

## Footprint

transformer/numerical kernels, condition/voice projection loader contract, Flow-LM state, pack mappings, synthetic/real stage fixtures/tests.

## What to build

voice projection, text/condition priming and nine-layer 512-wide/eight-head/2048-FFN cached transformer outputs against prefix/step fixtures. Implement norm/erf-GELU/RoPE/causal attention/projections, bounded KV ownership, prefill/incremental parity and independent request states. Implement num_speakers=1 and native duration_delta summed conditioning per actual config. Compatible voice tensor parsing/projection needed for fixtures is included; catalog UX is later. No unsupported CFG/speculative architecture.

## Blocked by

22-native-unigram-tokenizer

Worker Profile: medium

## Acceptance criteria

- [x] voice projection, text/condition priming and nine-layer 512-wide/eight-head/2048-FFN cached transformer outputs against prefix/step fixtures
- [x] Implement norm/erf-GELU/RoPE/causal attention/projections, bounded KV ownership, prefill/incremental parity and independent request states
- [x] Implement num_speakers=1 and native duration_delta summed conditioning per actual config
- [x] Compatible voice tensor parsing/projection needed for fixtures is included
- [x] catalog UX is later
- [x] No unsupported CFG/speculative architecture
- [x] Complete the approved slice checks and keep unavailable model/device checks explicit.

Status: done

## Execution record — 2026-10-25

- Wave base: `e9a34867481451ed98a75eef62af1b0270f4c2a1`.
- Branch: `feat/phonon-migration`; worktree: `/Volumes/Code/Personal/kokoro-mac/.worktrees/phonon-migration-planning`.
- Worker: `/root/23_flow_transformer`; attempt 1; implement-lite / gpt-6-luna / low, as approved. No repair follow-up used.
- Result and verified run-branch SHA: `d16f948718523c20f93067acabb91062f981b0ac`, with `Ticket: 23-flow-transformer`. The Coordinator confirmed the trailer, clean tree, and all seven changed paths within the Footprint.
- Atomic report: `/private/var/folders/j1/4vzqsp717kbb3gth_48lxdkc0000gn/T/orchestrate-briefs/kokoro-mac-feat-phonon-migration-f11dbf4f/23-flow-transformer.report.md`.

Native C now executes the nine-layer 512-wide/eight-head/2048-FFN Flow-LM using actual packed Q8 projections and BF16 parameters. The path includes affine LayerNorm, erf GELU, interleaved RoPE, causal attention, residuals, final norm, token and voice projections, BOS, summed speaker-count/duration conditions, and bounded request-owned caches. One captured position is used by every layer, with one cache-length commit after the full token. Separate caches, interleaving, reset, and capacity errors have executable checks.

The raw voice reader accepts F32 `speaker_wavs` shaped `[512,T]` or `[1,512,T]`, with `1 <= T <= 4096`. It checks structural JSON, decoded escaped keys/values, UTF-8 and surrogate pairs, duplicate fields, string metadata, signature/epoch, optional `model_ext`, exact dtype/shape/offsets, file bounds, and finite data. It returns owned row-major speaker latents `[T,512]`, transposed from file layout. Named/catalog UX remains ticket29 work.

### Coordinator verification

- Regenerated the independent bridge with `UV_CACHE_DIR=/private/tmp/kokoro-uv-cache uv run --project reference/phonon --offline --no-sync --locked python reference/phonon/export_flow_fixture.py`. It reads the frozen Q8 trace and pinned HF model, writes ignored binary bridges plus provenance, and changes no source fixture or budget.
- Passed the real-model transformer test under `TEST_CFLAGS='-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer'` and `TEST_CFLAGS='-O3 -DNDEBUG'`. Both ran 140 prefix and 39 cached incremental positions, with maximum absolute error `6.73532486e-06`, within unchanged `rtol=1e-3`, `atol=1e-5` gates. Separate comparisons covered real voice projection, BOS projection, and all 512 conditioning values at positive/negative duration deltas.
- Real test command: `PHONON_NATIVE_PACK=.build/phonon-native-f6acc8f43cb296be-v1.kpack PHONON_FLOW_FIXTURE=.build/phonon-flow-prefix-v1.bin PHONON_DEFAULT_VOICE=/Volumes/Code/Personal/speech-mac-models/phonon-40m-en/default-voice.safetensors PHONON_CONDITION_FIXTURE=.build/phonon-flow-condition-v1.bin make test-one TEST=tests/test_phonon_transformer.c`, with the flags above.
- Passed `make test-one TEST=tests/test_phonon_voice.c` with the same ASan/UBSan flags.
- Independently passed 144 voice-reader cases against the committed reader, including exact transposed values, both ranks, variable/max frames, all field-order permutations, valid Unicode escapes accepted by the safetensors library, malformed JSON/UTF-8/surrogates, duplicates, incompatible metadata, numeric/offset corruption, truncation, and NaN. Verifier: `/private/tmp/speech-mac-23-voice-check.py`; final log: `/private/tmp/speech-mac-23-coordinator-voice-matrix.log`.
- Passed `make format`, then `make check`; final tree clean. Logs: `/private/tmp/speech-mac-23-coordinator-{export,real-sanitizer,voice-sanitizer,real-optimized,voice-matrix,format,check}.log`.
- Worker also passed `make phonon-native-check`, `make phonon-reference-check`, focused checks, and the commit hook. Independent Luna/low review covered all seven changed paths under Correctness, Spec, Standards, Reuse, Quality, and Efficiency; final report found no actionable findings.
- The default full suite still explicitly skips unrelated legacy Kokoro model/voice checks and opt-in AudioQueue smoke. The new real Phonon gates above executed. The three reviewed normalizer analyzer warnings from ticket21 remain; no warning remains in this ticket's changed paths. No synthesized-audio or device claim is made by this component ticket.

### Review disposition

| Id | Lens | Finding | Applied / rejected / deferred | Reason | Verified by |
| --- | --- | --- | --- | --- | --- |
| 23-R1 | Correctness | Backbone norms require affine LayerNorm, and every layer must use the same captured token position | Applied | Preserve source normalization and cache sequencing | Nonzero-bias, independent-state/reset tests and real Q8 stage parity |
| 23-R2 | Correctness | Draft voice substring parser could dereference a missing colon, use unrelated fields, and assume 125 frames | Applied | Parse structure and bounded variable shapes instead of matching a fixture layout | Voice ASan/UBSan tests and 144-case independent matrix |
| 23-R3 | Correctness / Spec | Escaped strings, UTF-8/surrogates, metadata types/model_ext, duplicates, and unsupported top-level entries needed validation | Applied | Accept equivalent valid JSON and reject malformed/incompatible raw voice files | Voice regressions and independent matrix |
| 23-R4 | Spec | Zero-duration tests did not prove scale, sign, cosine/sine order, period, or exponent denominator | Applied | Check nonzero frequencies and full real conditioning vectors | Synthetic indices 0/32/63 and all-channel HF comparisons at positive/negative delta |
| 23-R5 | Correctness | Generated replay latents require mean/std conversion before the backbone; BOS has different units | Applied | Avoid double normalization and preserve actual generation behavior | Frozen 39-step replay and separate BOS comparison |
| 23-R6 | Correctness | Reviewer questioned adjacent-pair RoPE against HF rotate-half | Rejected / withdrawn | HF mapping permutes raw Flow-LM Q/K rows; native consumes raw packed rows and applies interleaved RoPE | Pinned converter/mapping review and all-layer real parity |
| 23-R7 | Efficiency | Redundant inner position guard repeated the entry check | Applied | Remove unreachable repeated work | Refreshed focused and full checks |
| 23-R8 | Standards | Two test failure branches leaked file handles | Applied | Clean up every test failure path | Refreshed analyzer/full checks and sanitizer tests |

### Worker Question 23-1 — 2026-10-25

Question: the worker initially stopped because native Flow-LM/conditioning code did not yet exist, without a failed command or concrete external blocker; the atomic report file was absent and the base remained clean.

Answer: creating that code is this new-feature ticket's approved work. Source: the ticket, Requirement Source, and Run Plan. Resume the same attempt/base, implement numerical stages, the full graph, voice/condition loading, real comparisons, tests, review, and commit. A later primitive-only progress final was also resumed with existing work preserved. These were logical continuations, not post-verification repairs or scope/profile changes. Progress thereafter used messages while implementation continued.

### Handoff to later tickets

- `phonon_flow_embed_latent` consumes generated 32-dimensional decoder/Mimi-space audio latents and standardizes with `emb_mean`/`emb_std`; the learned BOS is already in its own normalized space and is not standardized again. The voice reader's 512-dimensional speaker latents are a different representation. Ticket24 must preserve these units when integrating the sampler.
- Precomputed `emb` and `audio_prompt` voice forms remain for ticket29's voice compatibility/schema work. This ticket supplies the raw parsing/projection required by its fixtures and does not claim all upstream voice forms.
- A raw voice can have more than 512 frames. Ticket28 must size/check the actual voice prefix plus text and generation positions against the 8,192-position bound; the planner's default 512-position headroom alone cannot cover every accepted voice.
- The reviewer could not find `generation_pocket_tts.py` at the requested path; it inspected the actual pinned modeling/conversion sources and the worker separately traced generation semantics. This was not a missing implementation dependency.

No unresolved criterion or user escalation remains for ticket23.

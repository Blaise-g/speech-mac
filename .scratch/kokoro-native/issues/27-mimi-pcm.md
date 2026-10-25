# 27: Produce complete 24k mono PCM from latents

## Parent

None (Requirement Source).

## Context

- Technical source: `.scratch/kokoro-native/spec.md` and its pinned primary sources.
- Background: `.scratch/kokoro-native/spec.md`.
- User clarification dated 2026-10-25: subsequent implementation, fixes and ticket reviews use implement-lite / gpt-6.1-sol / low, overriding the repository model/effort defaults. Final whole-branch review uses gpt-6.1-sol / high. Actual earlier Luna/low execution records remain historical evidence.

## Footprint

Mimi/SEANet assembly, DSP/latency state, PCM ownership/output APIs, raw/final waveform fixtures/tests. Consumes25/26 state contracts.

## What to build

full SEANet decoder assembly and chosen pinned-reference gain/limiter chain when applicable. Separate raw and postprocessed PCM seams. Explicit first-sample delay trimming, tail drain/flush, chunk reset and terminal behavior; no invented codec flush frames. Whole/framewise/uneven partitions must meet fixed budgets and sample counts, including tiny/empty/tail cases and >context streams.

## Blocked by

26-streaming-convolution

Worker Profile: medium; implement-lite / gpt-6.1-sol / low (user clarification 2026-10-25)

## Acceptance criteria

- [x] full SEANet decoder assembly and chosen pinned-reference gain/limiter chain when applicable
- [x] Separate raw and postprocessed PCM seams
- [x] Explicit first-sample delay trimming, tail drain/flush, chunk reset and terminal behavior
- [x] no invented codec flush frames
- [x] Whole/framewise/uneven partitions must meet fixed budgets and sample counts, including tiny/empty/tail cases and >context streams
- [x] Complete the approved slice checks and keep unavailable model/device checks explicit.

Status: done

## Execution record — 2026-10-25

- Base: `c8a722b5214f977832c01ae0bb31438ce457b0e4`; branch `feat/phonon-migration`; existing `phonon-migration-planning` worktree.
- Worker: `/root/27_mimi_pcm`; attempt 1; implement-lite / GPT-6.1 Sol / low, including independent ticket review.
- Implementation source: `f742e31d197260adba85d06fedef3d62ebb3e823`, with `Ticket: 27-mimi-pcm`. Seven paths fit the published or approved derived Footprint; clean tree and unchanged frozen fixtures/budgets verified. The subsequent documentation correction below is also verified.
- Derived Footprint answer Q27-1: expose the identical existing ELU helper in `phonon_conv.[ch]` for outer SEANet activations. Source layers1/4/7/10 require it; reuse changes no arithmetic, vendor or public speech API. This is new ticket27 assembly work, not a ticket26 defect/repair.
- Original atomic report collected and archived at `/private/tmp/27-mimi-pcm-original.report.md` before follow-up rendering.

### Coordinator numerical verification

The Coordinator independently passed `/private/tmp/speech-mac-27-coordinator-verify.sh` on the committed candidate: regenerated independent Q8-mode raw/final expectations from frozen inputs; actual full PCM under ASan/UBSan, `-O3 -DNDEBUG` and the standard full suite; preserved complete native pack/Flow/autonomous checks and Mimi/convolution gates; `make format` then `make check`; clean diff/status. The synthetic graph injection symbol is absent from the ordinary production binary.

All11 records ×3 partitions pass raw/final waveform budgets, exact counts and native byte-identical partition outputs. Maximum normalized RMSE is `6.9840846e-06` and maximum absolute error `1.37686729e-05`, against unchanged `0.001`/`0.002` targets. The runtime emits1920 mono24k samples per retained latent frame, no first-sample trim or deferred flush; selected pinned processor fade is0 with no gain/limiter. Whole-utterance buffering is not used. These algorithm checks remain valid for a comment-only correction.

Logs: `/private/tmp/speech-mac-27-coordinator-{export,sanitizer,optimized,preserved,format,check}.log`. Seven inherited analyzer warnings and explicit legacy/device skips remain unchanged. Listening and performance claims remain later work.

### Root review finding 27-C1 — allocation contract

`phonon_pcm.h` and the original completion report incorrectly promise that complete processing allocates nothing. The actual call chain is `phonon_pcm_process → phonon_mimi_process → phonon_linear/_batch`, where the shared linear helper allocates/frees bounded expanded weights. Local PCM buffers and SEANet processing are preallocated; complete Mimi decoding still includes the explicitly accepted transient expansion baseline from ticket24.

Preapproved repair follow-up1 of2 was applied against ticket27 only: correct the header comment and report wording, with no executable code, ABI, algorithm, settings, fixture or budget change. Ticket33 optimization was not performed. Existing ticket25/26 and shared linear contracts were accurate; no repair was counted against them. The Coordinator recorded the finding at `46265e83413fae8a83aacdcbe907debdd93fd48d`, then rendered the precise follow-up from that tip. All numerical evidence above is retained.

### Verified correction and completion — 2026-10-25

The same worker and independent GPT-6.1 Sol / low reviewer completed formal repair1 in `57b0c3f87a0c60ef5a66289ff1b3cfe05afa9179`, with `Ticket: 27-mimi-pcm`. Only the `phonon_pcm.h` comment changed. The Coordinator independently compared the old/new headers after removing C comments: every non-comment byte is identical, including signatures and macros. It also verified the transitive allocation call chain, final report wording, commit/trailer and clean tree.

The header now states that PCM/Mimi caches and scratch are bounded, SEANet processing uses preallocated state, and Mimi shared linear calls allocate/free temporary expanded weights. There is no whole-utterance runtime buffering, but complete decoding is not allocation free. No new test or expensive model matrix rerun was needed for this comment-only change. Required `make format` then `make check`, staged hook and independent six-lens refresh passed; the Coordinator read the final canonical logs. Original independent normal/sanitizer/optimized numerical evidence remains valid for unchanged executable behavior.

Final result commits: `f742e31d197260adba85d06fedef3d62ebb3e823` and `57b0c3f87a0c60ef5a66289ff1b3cfe05afa9179`. Verified run-branch tip: the latter. The canonical report at `/private/var/folders/j1/4vzqsp717kbb3gth_48lxdkc0000gn/T/orchestrate-briefs/kokoro-mac-feat-phonon-migration-f11dbf4f/27-mimi-pcm.report.md` retains all prior checks, metrics, mutation restoration history and limitations, with inaccurate allocation claims replaced. Repair logs: `.build/phonon-pcm-repair1-{check,static,commit}.log`.

| Id | Finding | Disposition and verification |
| --- | --- | --- |
| 27-C1 | Complete-decoding allocation promise omitted transitive shared linear allocations | Applied: header/report corrected, exact comment-only comparison, static call-chain verification, canonical checks and independent Sol/low review |

All six ticket criteria are complete. No user Escalation or selected Phonon verification gap remains. Ticket28 may start after this record is committed. Final whole-branch Sol/high review, listening, CLI/playback integration and measured optimization remain later work.

### Handoff

- Resident `PhononPCMModel` borrows Mimi pack views and owns eleven expanded SEANet convolution models; keep the pack alive until model destruction. Requests borrow the immutable model and own bounded caches/scratch. Destroy request states before the model.
- Each retained 32D Mimi-space latent immediately yields1920 mono24k samples. Optional final output is separate or aliases raw exactly; with the pinned fade0/no-limiter processor, final equals the retained raw prefix. Only pass retained mask-prefix frames. EOS tail is generated upstream, not by decoder finish.
- Finish emits zero samples, marks terminal, and repeated finish/process after finish fail until reset. Empty input is a no-op only before finish. Invalid arguments preserve state; arithmetic failure poisons until reset, with possible partial output. Capacities count samples.
- The test-only sparse graph constructor uses the production state path and is absent from the ordinary binary. Runtime inference requires no Python/Rust/model framework or external SLEEF library. Do not interpret bounded buffering as an allocation-free or hard real-time guarantee.

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

- [ ] full SEANet decoder assembly and chosen pinned-reference gain/limiter chain when applicable
- [ ] Separate raw and postprocessed PCM seams
- [ ] Explicit first-sample delay trimming, tail drain/flush, chunk reset and terminal behavior
- [ ] no invented codec flush frames
- [ ] Whole/framewise/uneven partitions must meet fixed budgets and sample counts, including tiny/empty/tail cases and >context streams
- [ ] Complete the approved slice checks and keep unavailable model/device checks explicit.

Status: in-progress

## Execution record — 2026-10-25

- Base: `c8a722b5214f977832c01ae0bb31438ce457b0e4`; branch `feat/phonon-migration`; existing `phonon-migration-planning` worktree.
- Worker: `/root/27_mimi_pcm`; attempt 1; implement-lite / GPT-6.1 Sol / low, including independent ticket review.
- Candidate source: `f742e31d197260adba85d06fedef3d62ebb3e823`, with `Ticket: 27-mimi-pcm`. Seven paths fit the published or approved derived Footprint; clean tree and unchanged frozen fixtures/budgets verified. Not yet marked done because of the contract correction below.
- Derived Footprint answer Q27-1: expose the identical existing ELU helper in `phonon_conv.[ch]` for outer SEANet activations. Source layers1/4/7/10 require it; reuse changes no arithmetic, vendor or public speech API. This is new ticket27 assembly work, not a ticket26 defect/repair.
- Original atomic report collected and archived at `/private/tmp/27-mimi-pcm-original.report.md` before follow-up rendering.

### Coordinator numerical verification

The Coordinator independently passed `/private/tmp/speech-mac-27-coordinator-verify.sh` on the committed candidate: regenerated independent Q8-mode raw/final expectations from frozen inputs; actual full PCM under ASan/UBSan, `-O3 -DNDEBUG` and the standard full suite; preserved complete native pack/Flow/autonomous checks and Mimi/convolution gates; `make format` then `make check`; clean diff/status. The synthetic graph injection symbol is absent from the ordinary production binary.

All11 records ×3 partitions pass raw/final waveform budgets, exact counts and native byte-identical partition outputs. Maximum normalized RMSE is `6.9840846e-06` and maximum absolute error `1.37686729e-05`, against unchanged `0.001`/`0.002` targets. The runtime emits1920 mono24k samples per retained latent frame, no first-sample trim or deferred flush; selected pinned processor fade is0 with no gain/limiter. Whole-utterance buffering is not used. These algorithm checks remain valid for a comment-only correction.

Logs: `/private/tmp/speech-mac-27-coordinator-{export,sanitizer,optimized,preserved,format,check}.log`. Seven inherited analyzer warnings and explicit legacy/device skips remain unchanged. Listening and performance claims remain later work.

### Root review finding 27-C1 — allocation contract

`phonon_pcm.h` and the original completion report incorrectly promise that complete processing allocates nothing. The actual call chain is `phonon_pcm_process → phonon_mimi_process → phonon_linear/_batch`, where the shared linear helper allocates/frees bounded expanded weights. Local PCM buffers and SEANet processing are preallocated; complete Mimi decoding still includes the explicitly accepted transient expansion baseline from ticket24.

Preapproved repair follow-up1 of2 is required against ticket27 only: correct the header comment and report wording, with no executable code, ABI, algorithm, settings, fixture or budget change. Do not perform ticket33 optimization here. Existing ticket25/26 and shared linear contracts are accurate; no repair is counted against them. The Coordinator will render the precise documentation follow-up from the new tracker tip, keeping all numerical evidence above. Source model/PCM behavior passed; this allocation promise must be accurate before engine integration.

No user Escalation is open. Ticket28 and dependents remain held until this correction is reviewed, committed and recorded verified.

# 31: Finish maintained identity and remove legacy support

## Parent

None (Requirement Source).

## Context

- Technical source: `.scratch/kokoro-native/spec.md` and its pinned primary sources.
- Background: `.scratch/kokoro-native/spec.md`.
- User clarification dated 2026-10-25: subsequent implementation, fixes and ticket reviews use implement-lite / gpt-6.1-sol / low, overriding the repository model/effort defaults. Final whole-branch review uses gpt-6.1-sol / high. Actual earlier Luna/low execution records remain historical evidence.
- Consumer census: local API consumers are CLI/stream/benchmarks/tests. Nearby repositories and queried upstream PRs revealed no additional client. No UI exists. Installed `/opt/homebrew/bin/kokoro` is external and stays untouched. Census cannot rule out undiscoverable private callers. Recheck open PRs before removal/rename. Upstream PR209 context+1 discrepancy gates oracle reconciliation; PR186 Mimi Q8 was slower with quality tradeoff and is excluded.

## Footprint

repo-wide final contract/deletion across src/tests/reference/benchmarks/build/locks/config/docs/CI/public identifiers/generated fixtures. Historical tracker directory remains.

## What to build

speech-mac-only maintained runtime/dependencies/commands/libs/docs. Delete old model graph, phoneme/eSpeak frontend/style artifacts/options, old fixtures/oracle/conversion/reference dependencies and active old benchmark/comparison runners. Preserve historical numbered evidence, tracker 00–18/archive and attributed artifacts. Rename surviving generic public APIs/headers/build outputs; update glossary/standards/ADR retaining history, setup/CI/maintained examples/locked uv dependencies and tracker guidance. Census below governs removals. Installed external kokoro tool stays untouched. No local root move or universal migration promise for undiscoverable external callers.

## Blocked by

30-progressive-playback

Worker Profile: medium; implement-lite / gpt-6.1-sol / low (user clarification 2026-10-25)

## Acceptance criteria

- [x] speech-mac-only maintained runtime/dependencies/commands/libs/docs
- [x] Delete old model graph, phoneme/eSpeak frontend/style artifacts/options, old fixtures/oracle/conversion/reference dependencies and active old benchmark/comparison runners
- [x] Preserve historical numbered evidence, tracker 00–18/archive and attributed artifacts
- [x] Rename surviving generic public APIs/headers/build outputs
- [x] update glossary/standards/ADR retaining history, setup/CI/maintained examples/locked uv dependencies and tracker guidance
- [x] Census below governs removals
- [x] Installed external kokoro tool stays untouched
- [x] No local root move or universal migration promise for undiscoverable external callers
- [x] Complete the approved slice checks and keep unavailable model/device checks explicit.

Status: done

## Implementation and verified completion — 2026-10-26

- Wave/source base: `350144c45fbf878b4e5b5f770b71d179f6266d21`; branch `feat/phonon-migration`, existing run worktree.
- Worker `/root/31_remove_kokoro`, implement-lite / GPT-6.1 Sol / low. Source commit: `43eb085d6cffbaa8287dc63d30ae323b66b05917`, with `Ticket: 31-remove-kokoro`; clean tree confirmed.
- Source scope: 159 files, 701 insertions/22281 deletions. No tracker, historical artifacts, vendor, Phonon frozen fixtures/budgets/pins or Phonon lock changed in the source commit.
- Atomic report: `/private/var/folders/j1/4vzqsp717kbb3gth_48lxdkc0000gn/T/orchestrate-briefs/kokoro-mac-feat-phonon-migration-f11dbf4f/31-remove-kokoro.report.md`, attempt2. Attempt1 completed implementation, review, checks and commit, then hit a model-capacity error before reporting. The same Sol/low worker recovered the report without source edits or additional tests. This is runtime recovery, not a repair; formal ticket31 repair count remains 0.

### Delivered scope and coordinator decisions

The maintained build now produces speech-mac. The Kokoro graph, eSpeak/phoneme frontend, IPA tokenizer, style/artifact reader, old CLI/engine interfaces, old fixtures/oracle/conversion/reference dependency project/lock and active legacy benchmark runners are removed. Historical numbered reports, tracker 00–18/archive, attributed artifacts and license notices are retained. Existing generic UTF-8, SHA, WAV, AudioQueue and playback helpers use Speech/speech names. No compatibility binary or alias is built.

The worker asked how to avoid colliding with the existing generation `SpeechStatus`. The Coordinator approved `SpeechIOStatus`, `SpeechError` and `SPEECH_IO_*` for shared helper/platform errors while retaining the request-machine status; explicit mappings and ownership were preserved. This is within the generic-interface footprint, without an architecture or scope change.

README, contribution instructions, standards, glossary, source-owned agent guidance, setup, CI and examples now describe Phonon. ADR0001 and the older streaming plan retain attributed history; ADR0002 records the native Phonon architecture. The new downloader fetches exact pinned upstream filenames/revisions, verifies size/hash before installing and preserves the local codec subdirectory. The portable default is `models/phonon-40m-en`, overridden by `PHONON_MODEL_DIR`. Native checks select the canonical pack instead of an arbitrary glob. Ordinary checks use a stdlib-only removal census via uv; actual reference checks retain the unchanged Phonon lock. CI uses the verified official setup-uv action and existing macos-15 runner. No ordinary runtime dependency on Python/Rust/eSpeak/CoreML was introduced.

The Coordinator applied the worker's private tracker-guidance proposal here after source verification: the tracker README now resolves profiles through the approved Run Plan/user override and links to AGENTS for completion checks, preserving historical profiles. It replaces the stale Luna/xhigh default without duplicating current profile values. This completes the coordinated root-owned guidance step.

The Coordinator's pre-removal read-only GitHub census returned no open PRs on 2026-10-26. Removals follow the published local consumer census; no universal promise is made for undiscoverable private clients. `/opt/homebrew/bin/kokoro` and main's unattributed untracked test file were left untouched. No local root/worktree move, GitHub rename, remote mutation, branch push or PR occurred.

### Checks and independent review

The worker passed default `make format`→full actual-fixture `make check`, all 20 locked reference tests, complete reference/native gates, all 173 Rust/fixture/native normalization comparisons, actual CLI/stream ASAN/UBSAN, a fresh build and all 3 removal checks. New removal tests fail against a private prior-state tree; a private corruption-guard mutation makes the new download test fail. The staged-snapshot hook passed. An early accidental deletion of test-only headers, mechanically renamed rejection-message assertion and overly broad census were corrected before final validation; none remains open.

The Coordinator independently ran `/private/tmp/speech-mac-31-coordinator-verify.sh` against the committed source, exit 0:

| Gate | Result |
| --- | --- |
| Clean committed source and unchanged vendor/Phonon fixtures/budgets/lock/artifacts/tracker | PASS |
| Fresh build and removal census | PASS; only speech-mac produced, 3 tests |
| Native pack/generation replay | PASS; 209 tensors/36 Q8 projections; all autonomous stages and 61440 raw/final PCM samples exact 0 |
| Locked reference checks | PASS; 20 tests, trained Mimi window/uneven replay retained |
| Actual five-case engine ASAN/UBSAN | PASS; exact sample/chunk counts and fixed PCM budgets |
| Actual stream ASAN/UBSAN | PASS; 30720 samples/16 frames bitwise batch equality and lifecycle/cancellation/error checks |
| `make format`→full actual-fixture `make check TEST_CFLAGS=-O2` | PASS |
| Fresh binary symbol/dependency audit | PASS; no kokoro exports, test-only hooks or removed inference/runtime libraries |
| Generated SLEEF header | PASS; 579669 bytes, unchanged SHA256 `9c67f846daa49394628b4390396cb7223197e383de940e02cc3a58d9f33b477f` |

Root logs: `/private/tmp/speech-mac-31-coordinator-{verify,build,removal,native,reference,engine-asan,stream-asan,format,check}.log`. Worker logs: `.build/ticket31-*`, fully indexed in its atomic report.

All five engine cases retain sample/chunk counts 30720/1, 49920/1, 72960/1, 503040/2 and 1242240/2. Worst engine PCM NRMSE 2.74799634e-5 and maximum absolute error 8.28951597e-5 remain inside unchanged 0.001/0.002 budgets. Counts/noise/stops, source-shaped math, voice projection semantics and reset behavior are unchanged.

The independent GPT-6.1 Sol/low SingleReviewer examined every modified/added behavior, standards, build, documentation and test path with all six lenses; final findings:none. Complete deletion bodies and legacy fixture/lock bytes were omitted from line-by-line review because removal is required; selection and absence of surviving deleted-header includes were checked. External execution and census were verified separately above.

Ticket31 is verified on the run branch at `43eb085d6cffbaa8287dc63d30ae323b66b05917`. Seven inherited analyzer warnings remain visible, without added suppression. No new physical device replay was required for the identifier/deletion refactor; the verified user-Terminal ticket30 queue-muted start/reclamation/drain result remains attributed to its original run. Tool-context SDK failures retain unknown cause, fake timings remain fake, and acoustic onset/listening are unmeasured. New measurements, fresh setup/acceptance, final whole-branch Sol/high review and GitHub administration remain tickets 32–35 and later run steps. No Escalation is open. Ticket32 may launch after this guidance/completion record is reviewed and committed.

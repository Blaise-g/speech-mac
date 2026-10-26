# 34: Verify branch and record listening evidence

## Parent

None (Requirement Source).

## Context

- Technical source: `.scratch/kokoro-native/spec.md` and its pinned primary sources.
- Background: `.scratch/kokoro-native/spec.md`.
- User clarification dated 2026-10-25: subsequent implementation, fixes and ticket reviews use implement-lite / gpt-6.1-sol / low, overriding the repository model/effort defaults. Final whole-branch review uses gpt-6.1-sol / high. Actual earlier Luna/low execution records remain historical evidence.

## Footprint

acceptance/integration tests/recipes/report/artifacts/tracker evidence, whole native/reference/CLI contracts verified; queue-muted device probes. Repairs retain fixed budgets and get independent changed-path review.

## What to build

real40M acceptance report: format→check, locked-uv reference/Q8 parity, long window/flush, malformed input/assets, controls/voices, resident resets, cancellation/cleanup, runtime/dependency audit and final removal census. Each acceptance claim is verified here; the run-level whole-branch review follows ticket35. Fresh build/setup recipe and audition artifacts; user listening explicitly pending until user confirms, never inferred from numerical/ASR checks. Missing device/model checks remain open/blocking, not successful skips.

## Blocked by

33-measured-cpu-optimization

Worker Profile: medium; implement-lite / gpt-6.1-sol / low (user clarification 2026-10-25)

## Acceptance criteria

- [x] real40M acceptance report: format→check, locked-uv reference/Q8 parity, long window/flush, malformed input/assets, controls/voices, resident resets, cancellation/cleanup, runtime/dependency audit and final removal census
- [x] Complete independent changed-path review for this ticket before its commit; run-level whole-branch review follows ticket35
- [x] Fresh build/setup recipe and audition artifacts
- [x] user listening explicitly pending until user confirms, never inferred from numerical/ASR checks
- [x] Missing device/model checks remain open/blocking, not successful skips
- [x] Complete the approved slice checks and keep unavailable model/device checks explicit.

Status: done

## Implementation and verified completion — 2026-10-26

- Wave base: `4753207c6f5e3c188f665686a5c9c9f1b22d88ac`; branch `feat/phonon-migration`, existing run worktree. Worker `/root/34_migration_acceptance`, attempt 1, implement-lite / GPT-6.1 Sol / low; formal repair count 0.
- Result `13387fd0f41d79465a2cc5cc60655728643f4067`, Ticket34 trailer, clean tree. Canonical atomic `34-migration-acceptance.report.md` is done/attempt 1 in the run Briefs directory. Eighteen acceptance artifact paths under `artifacts/34-acceptance-20261026/`; runtime source remains cef2544. No product/test/reference/fixture/budget/lock/vendor/tracker change.

The complete actual acceptance runner passed format→check-O2→locked reference→native gates, with absolute fixture/model paths and no engine-record filter. It verifies all five prepared-engine cases (30720/49920/72960/503040/1242240 samples,1/1/1/2/2 chunks), all 33 raw PCM partition records, trained 250-key Mimi/uneven boundaries, malformed/truncated/nonfinite assets/input/caps, CLI controls/default/named/local/raw/projected voices, cancellation/cleanup/resident/concurrent resets, 23 reference tests, 12 pinned assets and 36 Q8/173 BF16 tensor gates. Frozen noise/head/conditions/integration/EOS/latents/61440 raw/final PCM replay exactly. Standalone unprepared probes remain fallback calibration; prepared pack/engine/paired results supply cached-path proof. The sole suite device skip is covered by a separate actual current-code result.

The Coordinator granted an idle window for a newly compiled current source helper with SPEECH_STREAM_SILENT_OUTPUT/O2, not the old32 binary. Its first assistant exec_command tty=true run passed: 76800 samples/40 frames/40 reclaims, status/exit 0, finite ordered SDK events and null acoustic onset. First PCM/submission/running observation were 0.051353/0.051453/0.221702s; drain 3.517704s. Source4753207/runtimecef2544 and binary SHA256 `c9996750f31522ee1d7ae56f42d31043e54511777a0e2d049fc3a45a709236d1` are recorded. Queue-local volume 0 and 60-second watchdog only; no system-volume/device/service/cache changes or autoplay. Earlier failures remain unexplained historical attempts; no cause/fix is inferred.

Fresh setup uses a private clean tracked-source archive, new locked uv environment/cache, independent native conversion/projection oracles, copied default pack/voice, new build and native CLI output. The initial offline cache lacked SymPy1.14; its failure is retained. The Coordinator approved online exact-lock dependency downloads in a new isolated task cache/venv, without global installs/config changes. Recovery passed, all source/lock/shared asset bytes remain unchanged, and no new network model download is claimed. Copies protect shared originals.

Current and fresh binary dependency/symbol audits pass; source process-creation audit finds only the literal afplay seam for explicit completed-WAV playback. This is a source/dependency audit, not an OS-level child trace. Restricted-PATH CLI output succeeds. Current short/structured/long audition WAVs are byte-identical to tracked32 files with text/settings/count/hash provenance; no playback occurred.

Independent GPT-6.1 Sol/low SingleReviewer applied all six lenses to underlying acceptance artifacts and independently matched 66 source/binary/audition/text hashes; no findings. Verbatim review is retained with evidence. Missing-fixture preflight exits 2; copied-log wrong-count/model-skip mutations fail, original source/logs unchanged. Worker final actual suite and staged hook pass; seven existing analyzer warnings remain attributed, no new warning or suppression.

The Coordinator independently ran `/private/tmp/speech-mac-34-coordinator-verify.py`, exit 0. Exact artifact-only footprint and unchanged runtime/test/reference/vendor/tracker were confirmed; current source/asset/lock/SDK binary hashes and raw event ordering/counts match. Full-suite validator passes original log and rejects both copied corruptions; missing-fixture preflight stops before tests. Both native binaries, library/symbol audit, all three current audition identities and fresh lock/default pack/voice match. A new restricted-PATH default CLI seed123 run in the private fresh checkout reproduces both its setup WAV and the actual CLI fixture byte-for-byte. Format/whitespace checks pass; branch clean. Prior source33 full actual/sanitizer gates remain applicable to unchanged C code; no broad mathematical remeasurement was needed. Root log `/private/tmp/speech-mac-34-coordinator-verify.log`.

Ticket34 is verified at `13387fd0f41d79465a2cc5cc60655728643f4067`. No required model/device check, tool or log is missing; no Escalation remains open. Listening stays PENDING user confirmation, including intelligibility/speaker/artifacts; acoustic onset is unknown without independent capture. These limits are not inferred from numeric/ASR/muted-device success. After reviewed completion writeback is committed,35 may perform only its approved repository administration, followed by the mandatory final whole-branch GPT-6.1 Sol/high review before branch push/PR. No rename/push/PR/merge/deployment occurred here.

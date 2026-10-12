# 18: Measure streaming timing and profile Apple Silicon optimization

## Context

Technical source: `.scratch/kokoro-native/spec.md` (Authorized streaming implementation), the [approved streaming design](../../../docs/streaming-plan.md), [ticket 15 comparison artifacts](15-coreml-comparison-baseline.md), and [ticket 13 optimization investigation](13-profile-guided-optimization.md).

Background: [Kokoro native inference spec](../spec.md).

Worker Profile: implement-lite · xhigh · gpt-6-luna (override)

## What to build

Repeatable streaming timing records show when chunks become available to the playback queue and how queue pressure affects production. Once streaming parity is established, a bounded profile-guided attempt tests whether a safe Apple Silicon change improves the native path.

## Acceptance criteria

- [x] Preserve ticket 15's comparison artifacts and add immutable dated records with exact source/build, host, input, voice, configured seed, command, and raw measurements. Use ticket 15's short and paragraph inputs at `af_heart`, and repeat at `am_adam` for voice coverage.
- [x] From a monotonic CLI-entry timestamp, measure first complete chunk ready, first buffer accepted, successful playback-start request, and first buffer reclaimed; report medians and observed ranges. Separate synthesis CPU time from playback wall time and backpressure, and report generated duration, queue bound, and any underruns. API events and buffer reclamation do not prove rendered or acoustic onset; report acoustic timing only with independent evidence.
- [ ] After stream/batch parity passes, profile the current native path and make one bounded attempt at a safe measured Apple Silicon optimization, favoring supported Accelerate/NEON work or layout/scratch reuse. Read ticket 13's rejected sine-approximation results first. Do not introduce Metal, quantization, or precision changes without separate scope.
- [x] Recheck fixed numerical budgets and run paired before/after measurements on the same host and workload. Set the improvement threshold from contemporaneous paired-run variation; ticket 12 and 15 historical results provide context but are not a fixed threshold. Claim a gain only when it clears observed variation.
- [x] If no safe change clears contemporaneous variation, retain the timing and optimization-attempt evidence and leave the optimization criterion open with the reason. Do not claim the ticket complete or invent a speedup. Streaming timing results remain reportable independently of that open criterion.

## Tests

- Add focused locked-`uv` fixture/stub tests for timing record and report validation, including provenance/sample retention, event and process timing scopes, and unavailable acoustic timing. Keep device measurements separate from automated tests.
- Do not run an audible speaker benchmark. If device timing cannot be collected without unintended audible output, mark it unavailable and report fake-consumer measurements separately.
- Run `make format`, then `make check`, and obtain independent review before committing.

## Blocked by

- `17-native-streaming-playback` — verified implementation and streaming/batch parity.

Ticket 15 is completed historical provenance; ticket 13's open investigation is context and does not block this bounded attempt.

**Status:** blocked

## Dependency verification — 2026-10-12

Ticket 17 is verified in `7111a9571b894dfc6ec7bfae6a877aeca97263d2`: full real-model checks, independent review, coordinator focused rerun, and the separate silence-only real AudioQueue lifecycle test passed. Timing and the bounded optimization attempt are unblocked.

## Verification and open criterion — 2026-10-12

Coordinator verified commit `1a7a427f2b0526448537e6a0fcc2517b0987a7e1`, its Ticket trailer, clean worktree and all five tracked artifact files. Ticket 15 comparison artifacts, ticket 09 audition WAVs and the neural-generator implementation are unchanged. Worker profile: `implement-lite · xhigh · gpt-6-luna (override)`.

Timing is verified in [report.md](../../../artifacts/18-streaming-benchmark-20261012T161920Z/report.md) and [raw.json](../../../artifacts/18-streaming-benchmark-20261012T161920Z/raw.json): 20 baseline samples (five per text/voice pair) and ten alternating optimization pairs, retaining 40 real-model invocations. Each queue was muted without changing system volume. Times start at the benchmark call into the production CLI and exclude earlier wrapper startup. Acoustic onset and actual device underruns remain unavailable. Queue-empty observations while the producer submission phase is open are possible-gap proxies only. The fixed queue is 49,152 bytes, without a claim about process-wide RSS.

The `af_heart` paragraph first-chunk median is 0.524 s [0.501, 0.539]; successful playback-start return is 0.584 s [0.545, 0.772]. Complete medians/ranges, synthesis and process CPU, playback/backpressure, source/build/model/voice/host provenance, and raw driver lines are retained. Single-chunk hello cannot prove intra-request decoding overlap.

The bounded float32 AdaIN fusion candidate failed the unchanged phase budget: one of 72,611 elements (`3.05166468e-05` error versus `1.87394089e-05` allowed). Diagnostic median gains of 0.65% and 1.68% did not clear contemporaneous range thresholds of 10.77% and 5.71%. Candidate runtime code is removed; its patch, original/normalized hashes, check log and qualitative profile remain. **A safe measured optimization is open; this ticket is not done.** The approved run permits independent timing delivery after this bounded attempt.

`make format`, full real-asset `make check`, muted benchmark helper build, focused release-`-O2` generator/inference checks, eight locked-`uv` report tests, whitespace checks and hook passed. The staged snapshot skipped ignored model-backed cases; the separate full check exercised them. Coordinator independently passed the report tests, validated the raw record and byte-for-byte report regeneration, and confirmed final source fingerprint `82886dd958e024f8fd74d86a547f410592ad2cd9487168ac9466e3aeeb2ffcf5`. Measurement-time identity is recorded separately from the final reporting/safety build.

Independent review covered all six lenses without outstanding findings. Two findings were applied: preflight rejection of binaries lacking the silent/timing build marker, and distinct saved-measurement versus final-source provenance. Schema-1 samples were retained without retroactively claiming the marker; schema-2 future runs require it. Patch-only normalization passed `git apply --check` and reviewer confirmation.

Coordinator answers settled partial event ordering, custom build-flag propagation, compile-out of benchmark counters, proxy wording, historical artifact preservation, and release-build parity. Q18-1: the sandbox startup failed with `-66680` after queue mute succeeded; coordinator authorized the same inaudible local-device operation with CoreAudio access. Escalated collection succeeded. No unresolved approval rejection or device blocker remains.

## Whole-branch review fixes — 2026-10-12

R1 and R2 are applied in `ceb0854e42ec8e3b05fecaf952faf4539b4061ed`, with both IDs and this ticket's trailer in the commit. Coordinator verified the clean tree, helper regression, nine locked-`uv` report tests, unchanged original timing artifacts, and the separate [post-review build record](../../../artifacts/18-streaming-benchmark-postreview-build-20261012T171907Z.json). Current helper/tool source fingerprint is `8f72867104eacd1f40bcf8be238dd4003d5dbbf2e26dca9205e56ae61e9f6b81`; that record is build verification with no new performance measurements. Historical raw/report identities remain unchanged. Full real-model checks and independent fix review passed. The optimization criterion remains open for the numerical and timing reasons above.

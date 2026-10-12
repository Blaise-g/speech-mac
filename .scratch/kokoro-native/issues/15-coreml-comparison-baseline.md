# 15: Compare the installed CoreML CLI with native Kokoro

## Context

Technical source: `.scratch/kokoro-native/spec.md` (User follow-up — CoreML comparison and streaming plan), [ticket 12 baseline](12-performance-baseline.md), and the follow-up extension in the [approved Run Plan](../run-plan.md).

Background: [Kokoro native inference spec](../spec.md).

Worker Profile: implement-lite · xhigh · gpt-6-luna (override)

## What to build

A repeatable comparison records matched native C and installed Jud/kokoro-coreml CLI behavior, latency, and measurable process resources, retaining each raw run and rerun instructions for future comparisons.

## Acceptance criteria

- [x] Use the already-installed `/opt/homebrew/bin/kokoro` version `0.11.2` unchanged and the native release build from this run. Record exact executable path/hash, observable CoreML source/version and model asset provenance, native commit/build provenance, host and OS, and all relevant voice/frontend/output-format details. Do not upgrade or replace the installed CLI.
- [x] Compare the exact text `hello from the terminal` and a checked-in or fully recorded representative longer text at voice `af_heart`, speed `1.0`, with speaker playback disabled. Run the same prepared IPA string through both implementations only if both expose a compatible input; otherwise mark that seam unsupported and do not present an unmatched comparison as model-only evidence.
- [x] Measure repeated fresh-process CLI completion and resident inference for both tools where supported. Include CoreML resident/daemon measurements only when its existing interface supports them; start a temporary daemon only from the known stopped state, record whether samples cover the client, service process, or both, and clean it up to restore the original state. Do not stop a pre-existing daemon or change persistent launch settings.
- [x] Record end-to-end completion, generated audio duration, throughput, process CPU time and RSS with explicit measurement scope, and installed model asset sizes where observable. Label OS-reported peak RSS separately from interval-sampled RSS. For a resident daemon, separate daemon-lifetime CPU/RSS from per-request CPU deltas; never present client-only usage as total service usage. State that exact GPU/ANE attribution and associated memory accounting are unavailable if the tools cannot measure them; this is a documented limit, not a blocker. Do not purge OS/model caches; label fresh-process runs accurately without claiming a cold hardware cache.
- [x] Preserve the complete raw samples, per-run provenance, exact rerun commands, and a human-readable comparison report in new `artifacts/15-coreml-comparison*` outputs without overwriting ticket 12's baseline or prior comparison runs. State which implementation wins for each matched workload, or report a tie/inconclusive result when evidence does not support a winner.
- [x] Keep measurements separate from vendor/source claims and from listening judgments. Document frontend, precision, sample-rate, codec, and output-writing differences that limit direct comparison.

## Tests

- [x] Reuse ticket 12's native benchmark driver and reporting conventions, adding only a small comparison-specific runner/report path or minor native-driver instrumentation where required. Do not create a general benchmark SDK or harness.
- [x] Add focused, locked-`uv` tests for the report/parser or normalizer: preserve raw sample counts and provenance, reject incomplete or mismatched workload/mode records, label unsupported metrics and their process scope instead of fabricating values, and avoid a global winner when results differ by workload or fall within observed variation.
- [x] Test command construction and report generation with fixtures or stubs; do not require an installed daemon, CoreML model, or speaker playback for the automated report tests.
- [x] Record the actual on-device benchmark command and result separately from automated test results.

## Blocked by

- `12-performance-baseline` — native workload, build provenance, and repeated-run measurement conventions.

Tickets `13-profile-guided-optimization` and `14-listening-acceptance` are not prerequisites; this comparison uses the current native release build and does not claim final listening acceptance.

**Status:** done

## Verification — 2026-10-12

Commit `fc8f34e5ea5651ea0727a0f27e4f1ab33b6e0ffd`; clean tree and Ticket trailer verified by coordinator. Coordinator reran 20 locked-uv comparison tests, recomputed report and outcomes from raw records, verified original-capture SHA and all sample counts, and ran full `make check` with real model assets: pass. The staged-snapshot commit hook passed non-model checks and explicitly skipped ignored model assets; coordinator's working-tree run executed those checks.

On Apple M5 16GiB/macOS27.0.1, matched af_heart/speed1 English: fresh hello native0.492s/CoreML4.229s; fresh paragraph1.729s/5.419s. Resident waveform-ready hello0.263s/0.296s; paragraph1.500s/1.406s. Native outputs are longer; resident paragraph throughput ranges overlap, so no global efficiency or speed winner. Fresh hello CPU0.470s/4.010s and peakRSS0.569GiB/1.569GiB are direct-process metrics. GPU/ANE work remains unattributed. Pack327624400B versus CoreML installed-assets110755755B; asset coverage and precision differ.

Five fresh samples per tool/case, 25 native warm requests and five CoreML resident requests per English workload. Immutable raw, preserved pre-correction capture, failure manifest and rerun commands are in `artifacts/15-coreml-comparison-20261012T113654Z/` and the documented earlier failed normalization attempt. Mach-time CPU correction is recorded without altering latency/audio samples. Daemon verified stopped after the run. Review fixes cover owned-PID cleanup, early output-path validation, provenance, CPU units, deduplication and matched throughput scope; no remaining findings.

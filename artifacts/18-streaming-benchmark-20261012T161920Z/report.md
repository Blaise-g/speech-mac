# Native streaming timing and optimization attempt

Captured: `2026-10-12T16:19:21Z` · source commit `9d77f30384ef9a6c9b2f1e8d0240e20acaa048de`.

Each event is measured from the benchmark process immediately before `kokoro_cli_run`. The benchmark invokes the production CLI parser and stream path. Per-queue volume is set to 0.0 before any PCM submission; it does not change system output volume.

The successful-start event is timestamped after `AudioQueueStart` returns successfully. An AudioQueue callback can run while that SDK call is still in progress, so the first buffer reclamation may precede the successful-start timestamp; validation checks only the ordering constraints implied by the adapter.

The callback reclamation timestamp is a buffer reuse event, not evidence that audio was rendered. Acoustic onset is unavailable because no output loopback or device-clock measurement was used. The queue-starvation count is an adapter observation that the pending AudioQueue buffer count reached zero while the producer submission phase was still open; actual device underruns remain unavailable without independent device evidence. On long inputs it flags a potential playback-gap condition, not a confirmed audible gap.

The sandbox probe reached `AudioQueueStart` and failed with OSStatus `-66680` after per-queue mute succeeded before PCM enqueue. The repeated collection used the authorized escalated local device access; all benchmark queues remained muted and system output volume was unchanged.

## Streaming samples

All values are medians with observed sample ranges. Event times include full frontend and model initialization from the benchmark CLI-call entry.

| Workload | Voice | Samples | First chunk ready | First buffer accepted | Start success | First buffer reclaimed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| hello_from_terminal | af_heart | 5 | 0.525 s [0.508, 0.576] | 0.525 s [0.508, 0.576] | 0.675 s [0.671, 0.893] | 0.856 s [0.852, 1.074] |
| hello_from_terminal | am_adam | 5 | 0.546 s [0.536, 0.564] | 0.546 s [0.537, 0.564] | 0.723 s [0.671, 0.726] | 0.904 s [0.852, 0.907] |
| representative_paragraph | af_heart | 5 | 0.524 s [0.501, 0.539] | 0.524 s [0.501, 0.539] | 0.584 s [0.545, 0.772] | 0.765 s [0.726, 0.953] |
| representative_paragraph | am_adam | 5 | 0.509 s [0.485, 0.520] | 0.509 s [0.485, 0.520] | 0.545 s [0.539, 0.799] | 0.725 s [0.720, 0.980] |

## Synthesis and AudioQueue timing

Synthesis wall and thread CPU cover synthesis calls only. AudioQueue open completion is measured from CLI-call entry; open span and start API are separate SDK intervals.

| Workload | Voice | Samples | Synthesis wall / thread CPU | AQ open span / completion from entry | Start API | Submit backpressure | Playback wall | Process CPU / wall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hello_from_terminal | af_heart | 5 | 0.259 s [0.253, 0.314] / 0.257 s [0.252, 0.286] | 0.044 s [0.041, 0.064] / 0.259 s [0.255, 0.297] | 0.154 s [0.099, 0.337] | 1.365 s [1.365, 1.365] | 1.877 s [1.877, 1.877] | 0.487 s [0.477, 0.514] / 2.554 s [2.549, 2.771] |
| hello_from_terminal | am_adam | 5 | 0.286 s [0.279, 0.297] / 0.283 s [0.276, 0.293] | 0.044 s [0.041, 0.048] / 0.266 s [0.256, 0.270] | 0.165 s [0.135, 0.179] | 1.536 s [1.536, 1.536] | 2.080 s [2.080, 2.080] | 0.513 s [0.505, 0.528] / 2.805 s [2.752, 2.808] |
| representative_paragraph | af_heart | 5 | 1.480 s [1.395, 1.581] / 1.477 s [1.391, 1.542] | 0.049 s [0.048, 0.052] / 0.298 s [0.278, 0.326] | 0.054 s [0.029, 0.233] | 8.187 s [8.080, 8.258] | 9.930 s [9.930, 9.931] | 1.729 s [1.644, 1.805] / 10.516 s [10.478, 10.704] |
| representative_paragraph | am_adam | 5 | 1.460 s [1.429, 1.486] / 1.458 s [1.426, 1.476] | 0.047 s [0.044, 0.050] / 0.292 s [0.276, 0.299] | 0.047 s [0.030, 0.279] | 8.601 s [8.579, 8.635] | 10.453 s [10.453, 10.453] | 1.706 s [1.667, 1.726] / 10.999 s [10.994, 11.254] |

## Generated audio and bounded queue

The buffer-reclamation callback and queue-starvation observation do not prove device render completion or an acoustic gap.

| Workload | Voice | Samples | Generated audio | Queue bound | Queue-starvation observations |
| --- | --- | ---: | ---: | ---: | ---: |
| hello_from_terminal | af_heart | 5 | 1.775 s [1.775, 1.775] | 49,152 B / 0.512 s | 0 [0, 0] |
| hello_from_terminal | am_adam | 5 | 1.975 s [1.975, 1.975] | 49,152 B / 0.512 s | 0 [0, 0] |
| representative_paragraph | af_heart | 5 | 9.825 s [9.825, 9.825] | 49,152 B / 0.512 s | 1 [1, 2] |
| representative_paragraph | am_adam | 5 | 10.350 s [10.350, 10.350] | 49,152 B / 0.512 s | 1 [1, 1] |

## Profile-guided attempt

The qualitative native profile is retained in [`profile.txt`](profile.txt), with its capture command and SHA-256 in `raw.json`. It identifies the sampled hot path but does not provide cycle-accurate attribution.

Profile-guided neural-generator AdaIN affine fusion candidate (float32). The unchanged fixed parity budget rejected it: 1 of 72,611 spectral phase elements exceeded tolerance.

The failed numerical candidate is preserved as [`candidate.patch`](candidate.patch) (SHA-256 `08f6a24ec41e5a328c2ea666e45369faa84abed38ffce7d8504f49b2695f9741`); its fixed-budget check output is [`candidate-check.log`](candidate-check.log) (SHA-256 `7f2ce4bedd104b08ff82c9c00c1d438fe1bbd4adcd1368247324ac71116b8942`).

Paired same-host measurements use `af_heart`, speed 1.0, seed 104729, and the ticket 15 short and paragraph text. The threshold is each workload's larger contemporaneous baseline/candidate full-range variation.

| Workload | Pairs | Baseline synthesis median [range] | Candidate synthesis median [range] | Observed variation threshold | Median gain | Clears variation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| hello_from_terminal | 5 | 0.254 s [0.250, 0.277] | 0.253 s [0.243, 0.263] | 10.77% | 0.65% | no |
| representative_paragraph | 5 | 1.337 s [1.307, 1.377] | 1.314 s [1.301, 1.376] | 5.71% | 1.68% | no |

Candidate rejected before eligibility; no tolerance changed. Paired timings are diagnostic only and do not qualify as a safe gain. Candidate source code is removed from the final tree, and the optimization criterion remains open.

## Provenance and rerun

Raw measurements and full source/build/host/model provenance: [`raw.json`](raw.json). The benchmark source fingerprint is `bae6ab17de4fa1db2566606fe3060e356669082039f6d96cb047f2729789425e`; baseline and candidate executable hashes, per-file source hashes, commands, and raw driver JSON lines are retained there. The final source fingerprint after the rejected candidate was removed is `82886dd958e024f8fd74d86a547f410592ad2cd9487168ac9466e3aeeb2ffcf5`.
The rerun command installs the rejected patch only for the experiment and reverses it with an `EXIT` trap. If the shell is interrupted before cleanup, run `git apply --reverse artifacts/18-streaming-benchmark-20261012T161920Z/candidate.patch` from the repository root.

```sh
set -e
candidate_patch=artifacts/18-streaming-benchmark-20261012T161920Z/candidate.patch
git apply "$candidate_patch"
trap 'git apply --reverse "$candidate_patch"' EXIT
make stream-benchmark-build BUILD_DIR=.build CPPFLAGS=-Isrc
make stream-benchmark-build BUILD_DIR=.build/stream-adain 'CPPFLAGS=-Isrc -DKOKORO_ADAIN_FUSED_AFFINE_EXPERIMENT'
KOKORO_STREAM_CPPFLAGS=-Isrc KOKORO_STREAM_CANDIDATE_FLAGS=-DKOKORO_ADAIN_FUSED_AFFINE_EXPERIMENT UV_CACHE_DIR=/private/tmp/kokoro-uv-cache uv run --project reference --locked --no-sync python benchmarks/stream.py --runs 5 --optimization-pairs 5 --binary .build/kokoro-stream-benchmark --candidate-binary .build/stream-adain/kokoro-stream-benchmark --candidate-patch artifacts/18-streaming-benchmark-20261012T161920Z/candidate.patch --candidate-check-log artifacts/18-streaming-benchmark-20261012T161920Z/candidate-check.log --profile-record artifacts/18-streaming-benchmark-20261012T161920Z/profile.txt --profile-command '/private/tmp/18-profile.sh launched baseline --seed 104729 --warmups 1 --repetitions 12; sample "$benchmark_pid" 4 100 > /private/tmp/18-native-profile.txt' --sandbox-probe-osstatus -66680 --optimization-description 'Profile-guided neural-generator AdaIN affine fusion candidate (float32). The unchanged fixed parity budget rejected it: 1 of 72,611 spectral phase elements exceeded tolerance.' --optimization-disposition 'Candidate rejected before eligibility; no tolerance changed. Paired timings are diagnostic only and do not qualify as a safe gain. Candidate source code is removed from the final tree, and the optimization criterion remains open.' --output artifacts/18-streaming-benchmark-rerun
```

## Verification

- `make format` — passed.
- `make check` — passed.
- `make stream-benchmark-build BUILD_DIR=.build/ticket18-final` — passed.
- `make test-one TEST=tests/test_neural_generator.c TEST_CFLAGS=-O2` — passed.
- `make test-one TEST=tests/test_inference.c TEST_CFLAGS=-O2` — passed.
- `UV_CACHE_DIR=/private/tmp/kokoro-uv-cache uv run --project reference --offline --no-sync --locked python -m unittest discover -s reference/tests -p test_stream_report.py` — passed.
- `git diff --check` — passed.
Independent review: approved; no outstanding findings across Correctness, Standards, Spec, Reuse, Quality, and Efficiency.

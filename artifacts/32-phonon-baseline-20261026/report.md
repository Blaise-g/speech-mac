# Phonon 40M native baseline

Source: `0f7ea8eab3e52f3b388ed810de0a3f71218935e9`. All times are seconds; Darwin RSS high water is bytes.

| Workload | Mode | First PCM median (range) | Synthesis median (range) | RTF median |
|---|---|---|---|---|
| short | fresh_process | 0.0432 (0.0430–0.0487) | 0.7933 (0.7706–0.8071) | 0.2479 |
| short | warm_resident | 0.0422 (0.0420–0.0454) | 0.7767 (0.7691–0.7872) | 0.2427 |
| structured | fresh_process | 0.0490 (0.0489–0.0496) | 3.9706 (3.9625–3.9863) | 0.2421 |
| structured | warm_resident | 0.0479 (0.0474–0.0482) | 3.9615 (3.9587–3.9642) | 0.2416 |
| long | fresh_process | 0.0488 (0.0486–0.0492) | 7.8100 (7.8005–7.8470) | 0.2422 |
| long | warm_resident | 0.0478 (0.0477–0.0480) | 7.8278 (7.8188–7.8428) | 0.2428 |

Fresh process means a new child, with no OS cache purge. First PCM and synthesis start at the native request call, after separately recorded engine/voice preparation. Voice projection and request state initialization are inside synthesis. CPU is process user+system delta over synthesis. RSS is process lifetime high water, not current memory or a request allocation delta. The resident child excludes one recorded warmup and reuses engine/voice while each request resets state and seed. Callback copies PCM; no playback occurs.

Top-level stage wall scopes are disjoint producer-thread intervals, separately compiled and measured with instrumentation. They include callee work and exclude callback/output I/O; prefill, flow head, Mimi/PCM and incremental backbone are not operator-exclusive CPU samples. Additional linear_expand_bf16, linear_expand_q8 and linear_sgemm operator scopes are nested within the top-level stages (and request preparation); never add them to stage totals. Expansion scopes include allocation and finite-value conversion, but not GEMM; GEMM scopes time only the Accelerate call. Unattributed synthesis work includes frontend/planning, voice projection and request setup/cleanup. Profile warmup is retained and excluded. Profile timing is not used in baseline medians.

Audition WAVs have not been played; listening remains pending. Acoustic onset is unavailable without independent capture. Playback queue events are a separate artifact and are never inferred from PCM or fake consumer timings.

## Separate stage profile

| Workload | Prefill s | Flow head s | Mimi/PCM s | Incremental backbone s |
|---|---|---|---|---|
| short | 0.0282 | 0.1147 | 0.4972 | 0.1446 |
| structured | 0.0665 | 0.5927 | 2.6019 | 0.7534 |
| long | 0.1346 | 1.1398 | 5.0098 | 1.4618 |

Nested linear operators (overlap stages above):

| Workload | BF16 expansion s | Q8 expansion s | SGEMM s |
|---|---|---|---|
| short | 0.1784 | 0.1124 | 0.0500 |
| structured | 0.9206 | 0.5650 | 0.2360 |
| long | 1.7709 | 1.1078 | 0.4574 |

## Host, preparation and memory

Measured on Apple M5, 16 GiB RAM, macOS 27.0.1 (build and compiler recorded in `raw.json`). Native compiler flags are C17, warnings-as-errors and `-O2`, with Accelerate and AudioToolbox frameworks. The source pack contains 36 Q8_0 and 173 BF16 tensors. The raw default voice is projected once inside each request; opening the voice file is timed separately. Source artifact hashes were recomputed against the pinned local verification manifest before measurement.

| Workload | Output seconds / frames / chunks | Engine preparation median s | Voice open median s | Synthesis CPU median s | RSS high water range bytes |
|---|---|---|---|---|---|
| short | 3.2 / 40 / 1 | 0.031386 | 0.000215 | 0.779789 | 118308864–121978880 |
| structured | 16.4 / 205 / 2 | 0.032954 | 0.000219 | 3.955050 | 127238144–128827392 |
| long | 32.24 / 403 / 4 | 0.032974 | 0.000231 | 7.817378 | 151781376–159285248 |

These preparation/CPU medians combine both modes only for a compact overview; raw repetitions retain their own mode and values. The first-PCM boundary excludes model/voice preparation, file input, child startup and parent process-launch overhead. Request clocks never cross process epochs. No current-memory measurement is claimed. Retained PCM capacity is bounded at the API maximum and only generated samples are touched by the consumer.

## Profile interpretation for ticket 33

Mimi/PCM takes about 63–65% of each separately profiled request. The shared linear expansion scopes together take about 37% of synthesis, while the SGEMM calls take about 6%. Expansion is therefore a measured candidate for a bounded CPU change in ticket 33. These nested counts cover all shared linears, including Flow and request voice projection; they do not attribute every expansion to Mimi or prove expansion is the only bottleneck. No optimization has been applied.

There is one retained measured profile request per workload after one excluded warmup. Its synthesis wall time differs from the unprofiled resident median by approximately +1.1%, +1.4%, and -1.0% for short, structured and long input. These unpaired samples do not estimate profiling overhead or prove a speed change; profile totals are excluded from baseline statistics. Ticket 33 must use repeated same-host comparisons and preserve numerical budgets.

## Playback and listening evidence

`fake-stream.log` records an actual native engine with a fake AudioQueue consumer for the separate `Hello.` / seed 123 regression workload: 30720 samples, exact batch/stream F32 equality, first PCM 0.039986 s, submission 0.039995 s, fake running observation 0.039996 s, synthesis end 0.336642 s and fake drain 0.336643 s. These events are not hardware timings and are not comparable to the three-workload release benchmark as a playback speed claim.

`queue-agent-attempt.json` retains the one bounded queue-muted SDK attempt for the short baseline workload / seed 0. The tool-context attempt failed at AudioQueueStart with status -66681 (child exit 4); it produced no successful queue timing. Ticket 30's prior user-Terminal functional result remains historical evidence, not a ticket 32 timing measurement. A separately reviewed private Terminal helper is prepared for the current silent timing binary. Actual running/reclamation/drain timing remains pending until that helper succeeds. Acoustic onset remains unavailable even if it succeeds.

`short.wav`, `structured.wav`, and `long.wav` are mono 24 kHz PCM16 audition files, lasting 3.2, 16.4 and 32.24 seconds. Their texts are adjacent `.txt` files; `raw.json` records settings, sample/frame/chunk counts and SHA-256. Each corresponding `-profile.wav` is byte-identical, and native F32 hashes also match across every fresh/resident repetition and the separately instrumented requests. No file has been played; user listening remains pending.

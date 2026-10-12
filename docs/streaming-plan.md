# Native streaming playback plan

Status: playback and timing implemented; optimization criterion remains open · 2026-10-12

## Decision and evidence

Add opt-in sentence-chunk playback to the existing CLI. Here, streaming means that playback of the first fully synthesized chunk begins while the engine is producing later chunks. Each chunk remains one complete `kokoro_engine_synthesize` request; this does not stream samples from inside a model request.

Ticket 15's matched measurements support this small first slice but do not set a first-audio target. On `hello from the terminal`, fresh native completion was 0.492 s and resident waveform-ready time was 0.263 s. On the representative paragraph, fresh native completion was 1.729 s; resident time was 1.500 s native and 1.406 s CoreML, with overlapping resident audio-throughput ranges. Native resident throughput was 6.760× for the short request and 6.551× for the paragraph. Median fresh-process native peak RSS was 0.569 GiB for the short request and 0.667 GiB for the paragraph. Those process-wide figures do not isolate waveform storage, so the 48 KiB playback queue is a bounded playback choice, not evidence of lower peak RSS. The measurements also lack sentence-level underruns, per-chunk timings, or playback submission, so they cannot set a numeric streaming-latency target. See the [comparison report](../artifacts/15-coreml-comparison-20261012T113654Z/report.md) and the [English frontend contract](../.scratch/kokoro-native/issues/11-english-espeak-frontend.md).

The first slice reuses the current full-input read, English frontend, sentence-aware IPA chunking, resident engine, and ordered synthesis loop in [the CLI](../src/cli.c) and [tokenizer](../src/ipa_tokenizer.c). It replaces only the final combine-to-WAV-and-`afplay` path when the user explicitly passes `--stream`. Keep the completed-WAV playback behavior as the default. Accept the existing text, file, stdin, `--ipa`, `-v/--voice`, and `-s/--speed` choices with `--stream`; reject `--stream` combined with `-o/--output` for this first slice. Continue reading and validating the whole source and building all frontend chunks before synthesis starts. Incremental stdin or frontend work is deferred.

## C interfaces and ownership

Keep synthesis and platform playback separate. A small orchestration interface can expose the following shape; names may follow the implementation's existing header conventions:

```c
typedef struct KokoroStream KokoroStream;

KokoroStatus kokoro_stream_start(KokoroEngine *engine, const KokoroIpaChunks *chunks,
                                 float speed, uint64_t random_seed, KokoroStream **stream,
                                 KokoroError *error);
KokoroStatus kokoro_stream_wait(KokoroStream *stream, KokoroError *error);
void kokoro_stream_cancel(KokoroStream *stream);
void kokoro_stream_free(KokoroStream *stream);
```

Append a distinct `KOKORO_CANCELLED` value to the existing status enum so current numeric values stay stable. `wait` returns `KOKORO_OK` only after normal queue drain, `KOKORO_CANCELLED` if cancellation wins, or the first synthesis/device error otherwise.

The call uses the selected voice's existing resident engine (`af_heart` by default or `am_adam`) and derives each chunk's style row as the current CLI does. IPA input bypasses eSpeak; ordinary text uses the current eSpeak settings. Speed applies unchanged to every chunk. Chunks are submitted in their original order as mono 24 kHz float samples.

`start` copies scalar options but borrows the engine and immutable chunk array. The caller keeps both alive and does not use the engine for another request until `wait` returns, including after cancellation. The CLI passes its configured seed (currently `104729` by default); for chunk `i`, preserve `random_seed + i`. Derive its style row from `token_count - 3` and reuse the selected voice and speed. Compare streamed and batch float-sample output using identical chunks, style rows, speed, and per-chunk seeds, before PCM16 WAV quantization, against the existing numerical budget. Each synthesis result belongs to the producer until copied into fixed playback buffers, then is released with `kokoro_synthesis_output_free`. The playback callback never retains a synthesis-result pointer. On end-of-stream, the control path calls `AudioQueueStop(queue, false)` and waits for the `kAudioQueueProperty_IsRunning` listener to report stopped before `wait` returns `KOKORO_OK`; Apple's API documents that this stop mode waits for queued playback buffers to play. A returned-buffer callback alone does not establish playback completion. A synthesis or device error is returned once, stops playback immediately, discards pending audio, and releases all owned buffers. `free` cancels an active stream, joins its worker, and is safe after success, error, or cancellation.

Put macOS output in a narrow C adapter around [Audio Queue Services](https://developer.apple.com/documentation/audiotoolbox/audio-queue-services), isolated from inference and portable audio logic. Use AudioQueue's fixed mono 24 kHz float output buffers; an initial on-device configuration of three 4,096-frame buffers holds at most 48 KiB (about 512 ms) of queued PCM. This is a bounded starting point, not a proven optimum: ticket 15 measures whole-request throughput, not sentence-level underruns. The producer splits each completed waveform across these buffers. If all buffers are in use, the producer waits for a completed-buffer signal before submitting more. The [AudioQueue output callback](https://developer.apple.com/documentation/audiotoolbox/audioqueueoutputcallback) only marks a buffer reusable and signals the producer; Apple explicitly warns that this callback does not prove its audio has played. The [running-state property listener](https://developer.apple.com/documentation/audiotoolbox/audioqueuepropertylistenerproc) reports graceful drain completion after the producer requests asynchronous stop. Neither callback performs inference, file I/O, or a wait. Total playback memory is bounded by those fixed buffers plus at most the one complete current model-chunk result required by chunk-level synthesis. Queue pressure therefore slows production instead of accumulating audio for the whole input.

Start AudioQueue after enqueueing the first buffer from the first completed model chunk. The producer then submits remaining samples and moves through later chunks while the device drains queued buffers. Natural device underflow may leave a gap at sentence boundaries if synthesis falls behind; record that outcome during playback acceptance rather than hiding it with dropped audio. End-of-stream is declared after the final chunk has been synthesized and every sample enqueued; call asynchronous `AudioQueueStop(queue, false)` and return success only after the queue's running-state listener reports stopped. The callback that makes buffers reusable is not the drain signal.

Cancellation records `KOKORO_CANCELLED` unless a terminal error or normal completion has already been recorded, then stops AudioQueue immediately. The existing engine call has no cancellation seam, so the producer checks the flag between chunks and frees the current result when that call returns. Cancellation while waiting for a playback buffer wakes the producer. The control path joins the producer before freeing the borrowed chunks or engine. If cancellation and an error race, the first terminal result recorded under the stream state lock wins; later cleanup errors do not replace it.

## Timing and acceptance

Measure from a monotonic timestamp captured at CLI entry:

- **First audio available:** after the first complete `kokoro_engine_synthesize` call returns. This includes source reading, full frontend preparation, model loading, and that chunk's synthesis.
- **First buffer submitted:** when AudioQueue accepts the first buffer containing synthesized samples.
- **Playback start requested:** when the first `AudioQueueStart` call succeeds after that buffer is submitted. This is an API event, not proof of audible output.
- **First buffer reclaimed:** when AudioQueue's callback reports the first buffer available for reuse. This records queue acquisition, not proof that its audio was consumed or rendered. Report first-render time only if a device clock or output loopback can establish it; otherwise mark it unavailable.

Report medians and observed ranges for these four events on the ticket 15 short and paragraph inputs, both at `af_heart`, speed `1.0`; repeat for `am_adam` as an option-coverage check. Do not claim a TTFA target from existing data or infer acoustic onset from an API event or callback. For a multi-chunk request, the fake backend must consume and reclaim first-chunk PCM while the test deliberately blocks synthesis of the final chunk; recording an early submit or `AudioQueueStart` call alone does not pass. The single-chunk `hello from the terminal` workload validates timing and parity but cannot demonstrate overlap or reduce time within that model request. Compare complete stream samples with current batch output under the existing fixed numerical budget.

Implemented defect-revealing tests cover:

- chunk order, completeness, and no duplicate or missing samples;
- streamed versus combined batch output within the existing numerical budget;
- a full playback buffer pool blocking the producer, then resuming when the callback releases a buffer;
- cancellation during synthesis (observed after the active call returns) and while waiting for a buffer, with `wait` returning `KOKORO_CANCELLED` unless an earlier terminal error won;
- successful `wait` only after graceful AudioQueue stop reports the queue is no longer running;
- AudioQueue start requested after first-chunk PCM is submitted, and a fake backend consumes/reclaims that PCM before a deliberately blocked final chunk completes on multi-chunk input;
- cleanup on synthesis, AudioQueue creation, enqueue, and playback errors, including queued-buffer release;
- repeated stream requests through one resident engine;
- accepted CLI option combinations and clear rejection of `--stream` with `--output`.

## Delivery stages and later work

1. **Completed minimal delivery:** the bounded AudioQueue adapter and producer use the existing prepared chunk list; opt-in `--stream` playback keeps completed-WAV playback as the default and rejects `--stream -o`. Ticket 18 records repeated muted AudioQueue timings for the ticket 15 short and paragraph inputs at `af_heart` and `am_adam`; see the [streaming timing and optimization report](../artifacts/18-streaming-benchmark-20261012T161920Z/report.md). The report marks acoustic onset unavailable and queue-starvation counts as observations, not confirmed device underruns or audible gaps.
2. **Measured Apple Silicon optimization:** ticket 18 records a profile-guided float32 AdaIN affine-fusion attempt. The unchanged fixed parity budget rejected it, and paired gains did not clear contemporaneous variation. No optimization is accepted, so the criterion remains open. A future bounded attempt must profile the ticket 15 workloads, check the fixed numerical budgets, and use paired before/after results; prefer supported Accelerate/NEON work or layout/scratch reuse. Consider an isolated Metal path only if profiling identifies a suitable bottleneck and measurements support the cost; assume no gain in advance. Ticket 13's unsuccessful accuracy/performance attempt remains prior evidence. Incremental input/frontend or within-request decoding, daemon mode, quantization, and wider language support need their own evidence and scope.

Keep the inference architecture readable and modular C, following the structure of [antirez's qwen-asr](https://github.com/antirez/qwen-asr) and [voxtral.c](https://github.com/antirez/voxtral.c). The corresponding Kokoro boundaries are already visible in `english_frontend.c`/`ipa_tokenizer.c` (frontend), `model_pack.c`/`voice_artifact.c` (weights), `tensor_utils.c` and the stage modules (numerical operations and model graph), `synthesis.c` (engine orchestration), and `cli.c` (command interface); the AudioQueue adapter should stay a separate output seam. The ASR models are architectural references only; do not import their decoding, KV-cache, or rollback behavior into Kokoro. Core ML remains outside the native inference path.

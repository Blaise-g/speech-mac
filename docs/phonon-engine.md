# Native resident speech engine

`speech.h` provides synchronous native Phonon 40M English synthesis. Create a
`SpeechEngine` from a verified native pack once, open a compatible `PhononVoice`,
then call `speech_synthesize` or `speech_collect` for each request. Destroy the
engine after all requests finish. Close voices after calls that borrow them finish.
The pack mapping, Flow tensor views/head and expanded SEANet model stay resident;
request KV caches, RNG, decoder state and scratch are private to each invocation.
The engine must stay at its allocated address because layer views point into it.

The producer normalizes English with the pinned Rust default policy port, prepares
and groups text toward 50 tokens, and fits every chunk to the source's 200-token
hard limit. It projects the entire voice in one batch and prefills each complete
voice/text/BOS prefix in one batch. After each native flow-head step it denormalizes
one latent, decodes 1920 mono samples at 24 kHz, and immediately calls the consumer
before generating the next latent. The 80 ms cadence describes audio duration,
not a wall-clock latency guarantee. Batch collection calls this same producer.
There is no inference thread or internal PCM queue.

Effective generation policy follows the approved migration spec and reference19:
temperature **0.8**, one LSD Euler step, raw EOS threshold **4**, and minimum EOS
frame **0**. Temperature 0.8 is an explicit runtime policy override; the downloaded
`generation_config.json` says 0.3. Neither value is silently relabelled. The EOS
frame is retained, then three tail frames for source chunks of at most four words
or one otherwise. A frame budget can also end a chunk. Finish adds no codec frames.
BOS and previous generated Flow latents are normalized; Mimi receives denormalized
latents exactly once. The request RNG is seeded once and continues across chunks.
Flow/KV and Mimi state reset at every source chunk, as pinned Rust `run_inline`
does. Equal native seeds repeat exactly; matching Torch's seed is not a contract.

Inputs must be nonempty ordinary NUL-free UTF-8 English, at most 1 MiB before and
after normalization. Empty normalized text is invalid. Only `en` is accepted.
`duration_delta` must be finite and uses native conditioning units. The API accepts
raw compatible voices with 1..4096 frames. Every actual voice/text/BOS prefix plus
planned generation must fit 8192 positions; the planner's 512-position voice
headroom is not used as the actual voice size. Generation is limited to 4096
frames per planned chunk and 8192 frames total per request. An exhausted total
request limit returns `SPEECH_LIMIT`, preserving the partial progress result.
The collector has an additional explicit sample capacity, capped at 8192*1920.
It returns an owned allocation only on success and frees partial data on errors.

Callbacks borrow the frame buffer for their duration. Return zero to continue,
one to cancel, or another value for a consumer error. A request cancellation poll
is latched and checked before and after synchronous normalization/planning calls
and between numerical operations,
including Flow layers, prefill attention blocks and sampler blocks. It does not
interrupt a BLAS call. All exits release private state; a new request starts clean.
Progress counts describe generated/delivered callback frames, including a frame
whose callback rejects it; `chunks_completed` counts fully finished chunks only.
Destroying an engine or voice from a callback is invalid. Callers synchronize any
shared callback state and never destroy assets while requests use them.

Decoder caches are bounded and SEANet processing storage is preallocated, but
shared Mimi/Flow linear kernels still allocate bounded temporary expanded weights.
This API makes no allocation-free or real-time callback-safety promise.

The independent integration oracle is
`reference/phonon/export_engine_fixture.py`. It uses pinned Rust normalization and
prompt preparation, the independent source-checked planner reference, HF raw
Unigram tokenization and all 209 GGUF tensors (36 Q8, 173 BF16, including Mimi).
It never obtains expected frontend IDs or PCM from the C engine. Its generated
bridge records exact raw Gaussian draws, chunk counts, text and PCM, with a
provenance manifest in `.build/phonon-engine-v1.json`. The C gate compares final
PCM against the unchanged NRMSE 0.001/max-absolute 0.002 budgets, and requires exact
sample counts and native batch/callback equality. A test-only observation seam
counts actual head evaluations to prove the first callback occurs before later
frames exist; it cannot replace model math and is absent from ordinary builds.

Reproduce after preparing the pinned Rust tools and native pack:

```sh
UV_CACHE_DIR=/private/tmp/speech-mac-uv-cache uv run --project reference/phonon --offline --no-sync --locked python reference/phonon/export_engine_fixture.py
PHONON_DEFAULT_VOICE=/path/to/default-voice.safetensors PHONON_NATIVE_PACK=.build/phonon-native-f6acc8f43cb296be-v1.kpack PHONON_ENGINE_FIXTURE=.build/phonon-engine-v1.bin make test-one TEST=tests/test_speech.c
```

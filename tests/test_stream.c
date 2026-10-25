#include "audio_queue_playback.h"

#include "ipa_tokenizer.h"
#include "kokoro_internal.h"
#include "stream_internal.h"

#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define MODEL_PACK_PATH "models/kokoro-v1_0-af_heart.kpack"
#define FAILURE_STATUS ((OSStatus) - 71)

#include "audio_queue_fake.h"

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    bool block_second;
    bool second_entered;
    bool release_second;
    size_t sample_counts[2];
    int fail_chunk;
    size_t call_count;
    size_t style_indices[2];
    uint64_t seeds[2];
    float speeds[2];
} SyntheticSynthesis;

typedef struct {
    KokoroStream *stream;
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    bool entered;
    bool finished;
} AsyncCancel;

typedef struct {
    KokoroStream *stream;
    const volatile sig_atomic_t *interrupted;
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    bool entered;
    bool finished;
    KokoroStatus status;
    KokoroError error;
} AsyncWait;

static void synthetic_init(SyntheticSynthesis *synthesis, const size_t sample_counts[2]) {
    memset(synthesis, 0, sizeof(*synthesis));
    (void)pthread_mutex_init(&synthesis->mutex, NULL);
    (void)pthread_cond_init(&synthesis->condition, NULL);
    synthesis->sample_counts[0] = sample_counts[0];
    synthesis->sample_counts[1] = sample_counts[1];
    synthesis->fail_chunk = -1;
}

static void synthetic_destroy(SyntheticSynthesis *synthesis) {
    pthread_cond_destroy(&synthesis->condition);
    pthread_mutex_destroy(&synthesis->mutex);
}

static void release_second_synthesis(SyntheticSynthesis *synthesis) {
    pthread_mutex_lock(&synthesis->mutex);
    synthesis->release_second = true;
    pthread_cond_broadcast(&synthesis->condition);
    pthread_mutex_unlock(&synthesis->mutex);
}

static void *cancel_stream_thread(void *argument) {
    AsyncCancel *cancel = argument;
    pthread_mutex_lock(&cancel->mutex);
    cancel->entered = true;
    pthread_cond_broadcast(&cancel->condition);
    pthread_mutex_unlock(&cancel->mutex);
    kokoro_stream_cancel(cancel->stream);
    pthread_mutex_lock(&cancel->mutex);
    cancel->finished = true;
    pthread_cond_broadcast(&cancel->condition);
    pthread_mutex_unlock(&cancel->mutex);
    return NULL;
}

static bool start_async_cancel(AsyncCancel *cancel, KokoroStream *stream, pthread_t *thread) {
    memset(cancel, 0, sizeof(*cancel));
    cancel->stream = stream;
    if (pthread_mutex_init(&cancel->mutex, NULL) != 0) {
        return false;
    }
    if (pthread_cond_init(&cancel->condition, NULL) != 0) {
        pthread_mutex_destroy(&cancel->mutex);
        return false;
    }
    if (pthread_create(thread, NULL, cancel_stream_thread, cancel) != 0) {
        pthread_cond_destroy(&cancel->condition);
        pthread_mutex_destroy(&cancel->mutex);
        return false;
    }
    return true;
}

static bool wait_cancel_flag(AsyncCancel *cancel, bool wait_for_finished, long timeout_ms) {
    struct timespec deadline;
    deadline_after_ms(&deadline, timeout_ms);
    pthread_mutex_lock(&cancel->mutex);
    int wait_error = 0;
    while (!(wait_for_finished ? cancel->finished : cancel->entered) && wait_error == 0) {
        wait_error = pthread_cond_timedwait(&cancel->condition, &cancel->mutex, &deadline);
    }
    bool reached = wait_for_finished ? cancel->finished : cancel->entered;
    pthread_mutex_unlock(&cancel->mutex);
    return reached;
}

static bool finish_async_cancel(AsyncCancel *cancel, pthread_t thread) {
    int join_error = pthread_join(thread, NULL);
    bool finished = wait_cancel_flag(cancel, true, 1);
    pthread_cond_destroy(&cancel->condition);
    pthread_mutex_destroy(&cancel->mutex);
    return join_error == 0 && finished;
}

static void *wait_stream_thread(void *argument) {
    AsyncWait *wait = argument;
    pthread_mutex_lock(&wait->mutex);
    wait->entered = true;
    pthread_cond_broadcast(&wait->condition);
    pthread_mutex_unlock(&wait->mutex);
    wait->status = kokoro_stream_wait(wait->stream, wait->interrupted, &wait->error);
    pthread_mutex_lock(&wait->mutex);
    wait->finished = true;
    pthread_cond_broadcast(&wait->condition);
    pthread_mutex_unlock(&wait->mutex);
    return NULL;
}

static bool start_async_wait(AsyncWait *wait, KokoroStream *stream,
                             const volatile sig_atomic_t *interrupted, pthread_t *thread) {
    memset(wait, 0, sizeof(*wait));
    wait->stream = stream;
    wait->interrupted = interrupted;
    if (pthread_mutex_init(&wait->mutex, NULL) != 0) {
        return false;
    }
    if (pthread_cond_init(&wait->condition, NULL) != 0) {
        pthread_mutex_destroy(&wait->mutex);
        return false;
    }
    if (pthread_create(thread, NULL, wait_stream_thread, wait) != 0) {
        pthread_cond_destroy(&wait->condition);
        pthread_mutex_destroy(&wait->mutex);
        return false;
    }
    return true;
}

static bool async_wait_flag(AsyncWait *wait, bool finished, long timeout_ms) {
    struct timespec deadline;
    deadline_after_ms(&deadline, timeout_ms);
    pthread_mutex_lock(&wait->mutex);
    int wait_error = 0;
    while (!(finished ? wait->finished : wait->entered) && wait_error == 0) {
        wait_error = pthread_cond_timedwait(&wait->condition, &wait->mutex, &deadline);
    }
    bool reached = finished ? wait->finished : wait->entered;
    pthread_mutex_unlock(&wait->mutex);
    return reached;
}

static bool finish_async_wait(AsyncWait *wait, pthread_t thread) {
    int join_error = pthread_join(thread, NULL);
    bool finished = async_wait_flag(wait, true, 1);
    pthread_cond_destroy(&wait->condition);
    pthread_mutex_destroy(&wait->mutex);
    return join_error == 0 && finished;
}

static KokoroStatus synthesize_deterministic(void *context, KokoroEngine *engine,
                                             const int64_t *token_ids, size_t token_count,
                                             const KokoroSynthesisOptions *options,
                                             KokoroSynthesisOutput *output, KokoroError *error) {
    (void)engine;
    if (token_ids == NULL || token_count < 3) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT, "bad synthetic chunk");
        return KOKORO_INVALID_ARGUMENT;
    }
    SyntheticSynthesis *synthesis = context;
    size_t index = token_ids[1] == 101 ? 1 : 0;
    pthread_mutex_lock(&synthesis->mutex);
    synthesis->call_count++;
    synthesis->style_indices[index] = options->voice_style_index;
    synthesis->seeds[index] = options->random_seed;
    synthesis->speeds[index] = options->speed;
    if (index == 1 && synthesis->block_second) {
        synthesis->second_entered = true;
        pthread_cond_broadcast(&synthesis->condition);
        while (!synthesis->release_second) {
            pthread_cond_wait(&synthesis->condition, &synthesis->mutex);
        }
    }
    int fail_chunk = synthesis->fail_chunk == (int)index;
    pthread_mutex_unlock(&synthesis->mutex);
    if (fail_chunk) {
        kokoro_error_set(error, KOKORO_INVALID_TOKEN_ID, "synthetic synthesis failure");
        return KOKORO_INVALID_TOKEN_ID;
    }
    size_t count = synthesis->sample_counts[index];
    output->samples = malloc(count * sizeof(*output->samples));
    if (output->samples == NULL) {
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY, "cannot allocate synthetic waveform");
        return KOKORO_OUT_OF_MEMORY;
    }
    output->sample_count = count;
    for (size_t sample = 0; sample < count; sample++) {
        output->samples[sample] = (float)(index * 10000 + sample + 1) * 0.001f;
    }
    return KOKORO_OK;
}

static KokoroStatus synthesize_silence(void *context, KokoroEngine *engine,
                                       const int64_t *token_ids, size_t token_count,
                                       const KokoroSynthesisOptions *options,
                                       KokoroSynthesisOutput *output, KokoroError *error) {
    (void)context;
    (void)engine;
    (void)token_ids;
    (void)token_count;
    (void)options;
    const size_t count = KOKORO_STREAM_BUFFER_FRAMES * 2U;
    output->samples = calloc(count, sizeof(*output->samples));
    if (output->samples == NULL) {
        if (error != NULL) {
            error->code = KOKORO_OUT_OF_MEMORY;
            (void)snprintf(error->message, sizeof(error->message), "cannot allocate silence");
        }
        return KOKORO_OUT_OF_MEMORY;
    }
    output->sample_count = count;
    return KOKORO_OK;
}

static int wait_second_synthesis(SyntheticSynthesis *synthesis, long timeout_ms) {
    struct timespec deadline;
    deadline_after_ms(&deadline, timeout_ms);
    pthread_mutex_lock(&synthesis->mutex);
    int wait_error = 0;
    while (!synthesis->second_entered && wait_error == 0) {
        wait_error = pthread_cond_timedwait(&synthesis->condition, &synthesis->mutex, &deadline);
    }
    bool entered = synthesis->second_entered;
    pthread_mutex_unlock(&synthesis->mutex);
    return entered;
}

static void make_two_chunks(int64_t token_ids[2][3], KokoroIpaChunks *chunks) {
    token_ids[0][0] = 0;
    token_ids[0][1] = 100;
    token_ids[0][2] = 0;
    token_ids[1][0] = 0;
    token_ids[1][1] = 101;
    token_ids[1][2] = 0;
    static KokoroIpaChunk items[2];
    items[0] = (KokoroIpaChunk){.token_ids = token_ids[0], .token_count = 3};
    items[1] = (KokoroIpaChunk){.token_ids = token_ids[1], .token_count = 3};
    chunks->items = items;
    chunks->count = 2;
}

static KokoroStatus start_synthetic_stream(MockAudio *audio, SyntheticSynthesis *synthesis,
                                           const KokoroIpaChunks *chunks, KokoroStream **stream) {
    active_mock_audio = audio;
    return kokoro_stream_start_with((KokoroEngine *)(uintptr_t)1, chunks, 1.25f, 701,
                                    kokoro_audio_queue_playback_ops(), (void *)&mock_api,
                                    synthesize_deterministic, synthesis, stream, NULL);
}

static int expected_synthetic_samples(float *samples, const size_t counts[2], size_t count) {
    if (count != counts[0] + counts[1]) {
        return 0;
    }
    size_t offset = 0;
    for (size_t chunk = 0; chunk < 2; chunk++) {
        for (size_t sample = 0; sample < counts[chunk]; sample++) {
            samples[offset++] = (float)(chunk * 10000 + sample + 1) * 0.001f;
        }
    }
    return 1;
}

static int test_first_chunk_consumed_before_final_synthesis(void) {
    MockAudio audio;
    const size_t sample_counts[] = {9000, 5000};
    SyntheticSynthesis synthesis;
    synthetic_init(&synthesis, sample_counts);
    synthesis.block_second = true;
    if (!mock_audio_init(&audio, true)) {
        synthetic_destroy(&synthesis);
        return fail("could not initialize fake consumer");
    }
    int64_t token_ids[2][3];
    KokoroIpaChunks chunks = {0};
    make_two_chunks(token_ids, &chunks);
    KokoroStream *stream = NULL;
    KokoroStatus status = start_synthetic_stream(&audio, &synthesis, &chunks, &stream);
    int passed = status == KOKORO_OK && wait_second_synthesis(&synthesis, 3000) &&
                 wait_audio(&audio, ready_consumed_first_chunk, 3000);
    pthread_mutex_lock(&synthesis.mutex);
    passed = passed && synthesis.second_entered && !synthesis.release_second;
    pthread_mutex_unlock(&synthesis.mutex);
    pthread_mutex_lock(&audio.mutex);
    passed = passed && audio.format_valid && audio.start_count == 1 &&
             audio.consumed_count == sample_counts[0];
    pthread_mutex_unlock(&audio.mutex);
    release_second_synthesis(&synthesis);
    KokoroError error = {0};
    if (stream != NULL) {
        status = kokoro_stream_wait(stream, NULL, &error);
    }
    passed = passed && status == KOKORO_OK;
    float *expected = malloc((sample_counts[0] + sample_counts[1]) * sizeof(*expected));
    if (expected == NULL) {
        passed = 0;
    } else {
        expected_synthetic_samples(expected, sample_counts, sample_counts[0] + sample_counts[1]);
        pthread_mutex_lock(&audio.mutex);
        passed = passed && audio.consumed_count == sample_counts[0] + sample_counts[1] &&
                 memcmp(audio.consumed_samples, expected,
                        audio.consumed_count * sizeof(*expected)) == 0 &&
                 audio.graceful_stop_count == 1 && audio.immediate_stop_count == 0;
        pthread_mutex_unlock(&audio.mutex);
        free(expected);
    }
    kokoro_stream_free(stream);
    pthread_mutex_lock(&audio.mutex);
    passed = passed && audio.dispose_count == 1 && audio.remove_listener_count == 1;
    pthread_mutex_unlock(&audio.mutex);
    mock_audio_destroy(&audio);
    synthetic_destroy(&synthesis);
    return passed ? 1
                  : fail("first chunk was not consumed before final synthesis or samples changed");
}

static void enable_consumer(MockAudio *audio) {
    pthread_mutex_lock(&audio->mutex);
    audio->consume_enabled = true;
    pthread_cond_broadcast(&audio->condition);
    pthread_mutex_unlock(&audio->mutex);
}

static int test_full_buffer_backpressure_and_resume(void) {
    MockAudio audio;
    const size_t sample_counts[] = {KOKORO_STREAM_BUFFER_FRAMES * 3U + 73U, 0};
    SyntheticSynthesis synthesis;
    synthetic_init(&synthesis, sample_counts);
    if (!mock_audio_init(&audio, false)) {
        synthetic_destroy(&synthesis);
        return fail("could not initialize backpressure consumer");
    }
    int64_t token_ids[2][3];
    KokoroIpaChunks chunks = {0};
    make_two_chunks(token_ids, &chunks);
    chunks.count = 1;
    KokoroStream *stream = NULL;
    KokoroStatus status = start_synthetic_stream(&audio, &synthesis, &chunks, &stream);
    int passed = status == KOKORO_OK && wait_audio(&audio, ready_enqueued_three, 3000);
    pthread_mutex_lock(&audio.mutex);
    passed = passed && audio.enqueue_count == KOKORO_STREAM_BUFFER_COUNT &&
             audio.consumed_count == 0 && audio.graceful_stop_count == 0;
    pthread_mutex_unlock(&audio.mutex);
    enable_consumer(&audio);
    KokoroError error = {0};
    if (stream != NULL) {
        status = kokoro_stream_wait(stream, NULL, &error);
    }
    passed = passed && status == KOKORO_OK;
    pthread_mutex_lock(&audio.mutex);
    passed = passed && audio.consumed_count == sample_counts[0] && audio.graceful_stop_count == 1 &&
             audio.start_count == 1;
    pthread_mutex_unlock(&audio.mutex);
    kokoro_stream_free(stream);
    mock_audio_destroy(&audio);
    synthetic_destroy(&synthesis);
    return passed ? 1 : fail("producer did not wait at three buffers and resume after reclamation");
}

static int test_cancel_stops_playback_during_synthesis(void) {
    MockAudio audio;
    const size_t sample_counts[] = {256, 128};
    SyntheticSynthesis synthesis;
    synthetic_init(&synthesis, sample_counts);
    synthesis.block_second = true;
    if (!mock_audio_init(&audio, false)) {
        synthetic_destroy(&synthesis);
        return fail("could not initialize cancellation consumer");
    }
    int64_t token_ids[2][3];
    KokoroIpaChunks chunks = {0};
    make_two_chunks(token_ids, &chunks);
    KokoroStream *stream = NULL;
    KokoroStatus status = start_synthetic_stream(&audio, &synthesis, &chunks, &stream);
    int passed = status == KOKORO_OK && wait_second_synthesis(&synthesis, 3000);
    kokoro_stream_cancel(stream);
    pthread_mutex_lock(&audio.mutex);
    passed = passed && audio.immediate_stop_count > 0 && !audio.queue->running &&
             audio.synchronous_output_callbacks > 0 && audio.synchronous_property_callbacks > 0;
    pthread_mutex_unlock(&audio.mutex);
    pthread_mutex_lock(&synthesis.mutex);
    passed = passed && !synthesis.release_second;
    pthread_mutex_unlock(&synthesis.mutex);
    release_second_synthesis(&synthesis);
    KokoroError error = {0};
    status = kokoro_stream_wait(stream, NULL, &error);
    passed = passed && status == KOKORO_CANCELLED;
    kokoro_stream_free(stream);
    mock_audio_destroy(&audio);
    synthetic_destroy(&synthesis);
    return passed ? 1 : fail("cancel did not stop the queue before joining blocked synthesis");
}

static int test_signal_wait_cancels_once_while_synthesis_is_blocked(void) {
    MockAudio audio;
    const size_t sample_counts[] = {64, 32};
    SyntheticSynthesis synthesis;
    synthetic_init(&synthesis, sample_counts);
    synthesis.block_second = true;
    if (!mock_audio_init(&audio, false)) {
        synthetic_destroy(&synthesis);
        return fail("could not initialize signal cancellation fixture");
    }
    int64_t token_ids[2][3];
    KokoroIpaChunks chunks = {0};
    make_two_chunks(token_ids, &chunks);
    KokoroStream *stream = NULL;
    KokoroStatus status = start_synthetic_stream(&audio, &synthesis, &chunks, &stream);
    int passed = status == KOKORO_OK && wait_second_synthesis(&synthesis, 3000);
    volatile sig_atomic_t interrupted = 1;
    AsyncWait wait;
    pthread_t wait_thread;
    bool wait_started =
        stream != NULL && start_async_wait(&wait, stream, &interrupted, &wait_thread);
    passed = passed && wait_started && async_wait_flag(&wait, false, 3000) &&
             wait_audio(&audio, ready_immediate_stop, 3000);
    bool wait_returned_while_synthesizing = wait_started && async_wait_flag(&wait, true, 100);
    pthread_mutex_lock(&audio.mutex);
    passed = passed && audio.immediate_stop_count == 1;
    pthread_mutex_unlock(&audio.mutex);
    release_second_synthesis(&synthesis);
    if (wait_started) {
        bool wait_finished = finish_async_wait(&wait, wait_thread);
        passed = passed && wait_finished && wait.status == KOKORO_CANCELLED;
    } else if (stream != NULL) {
        (void)kokoro_stream_wait(stream, NULL, NULL);
    }
    passed = passed && !wait_returned_while_synthesizing;
    pthread_mutex_lock(&audio.mutex);
    passed = passed && audio.immediate_stop_count == 2;
    pthread_mutex_unlock(&audio.mutex);
    kokoro_stream_free(stream);
    mock_audio_destroy(&audio);
    synthetic_destroy(&synthesis);
    return passed ? 1 : fail("latched SIGINT repeatedly aborted or spun during active synthesis");
}

static int test_cancel_wakes_full_buffer_wait(void) {
    MockAudio audio;
    const size_t sample_counts[] = {KOKORO_STREAM_BUFFER_FRAMES * 3U + 1U, 0};
    SyntheticSynthesis synthesis;
    synthetic_init(&synthesis, sample_counts);
    if (!mock_audio_init(&audio, false)) {
        synthetic_destroy(&synthesis);
        return fail("could not initialize full-queue cancellation fixture");
    }
    int64_t token_ids[2][3];
    KokoroIpaChunks chunks = {0};
    make_two_chunks(token_ids, &chunks);
    chunks.count = 1;
    KokoroStream *stream = NULL;
    KokoroStatus status = start_synthetic_stream(&audio, &synthesis, &chunks, &stream);
    int passed = status == KOKORO_OK && wait_audio(&audio, ready_enqueued_three, 3000);
    kokoro_stream_cancel(stream);
    KokoroError error = {0};
    if (stream != NULL) {
        status = kokoro_stream_wait(stream, NULL, &error);
    }
    passed = passed && status == KOKORO_CANCELLED;
    pthread_mutex_lock(&audio.mutex);
    passed = passed && audio.immediate_stop_count > 0 && audio.pending_on_dispose == 0 &&
             audio.synchronous_output_callbacks == KOKORO_STREAM_BUFFER_COUNT;
    pthread_mutex_unlock(&audio.mutex);
    kokoro_stream_free(stream);
    mock_audio_destroy(&audio);
    synthetic_destroy(&synthesis);
    return passed ? 1 : fail("cancellation did not release queued buffers or wake backpressure");
}

static int test_cancel_wakes_graceful_drain(void) {
    MockAudio audio;
    const size_t sample_counts[] = {64, 0};
    SyntheticSynthesis synthesis;
    synthetic_init(&synthesis, sample_counts);
    if (!mock_audio_init(&audio, false)) {
        synthetic_destroy(&synthesis);
        return fail("could not initialize drain cancellation fixture");
    }
    int64_t token_ids[2][3];
    KokoroIpaChunks chunks = {0};
    make_two_chunks(token_ids, &chunks);
    chunks.count = 1;
    KokoroStream *stream = NULL;
    KokoroStatus status = start_synthetic_stream(&audio, &synthesis, &chunks, &stream);
    int passed = status == KOKORO_OK && wait_audio(&audio, ready_graceful_stop, 3000);
    kokoro_stream_cancel(stream);
    KokoroError error = {0};
    if (stream != NULL) {
        status = kokoro_stream_wait(stream, NULL, &error);
    }
    passed = passed && status == KOKORO_CANCELLED;
    pthread_mutex_lock(&audio.mutex);
    passed = passed && audio.graceful_stop_count == 1 && audio.immediate_stop_count > 0 &&
             audio.pending_on_dispose == 0;
    pthread_mutex_unlock(&audio.mutex);
    kokoro_stream_free(stream);
    mock_audio_destroy(&audio);
    synthetic_destroy(&synthesis);
    return passed ? 1 : fail("cancellation during drain deadlocked or returned normal completion");
}

static int test_cancel_during_enqueue_does_not_start_later(void) {
    MockAudio audio;
    const size_t sample_counts[] = {32, 0};
    SyntheticSynthesis synthesis;
    synthetic_init(&synthesis, sample_counts);
    if (!mock_audio_init(&audio, false)) {
        synthetic_destroy(&synthesis);
        return fail("could not initialize enqueue race fixture");
    }
    audio.enqueue_blocked = true;
    int64_t token_ids[2][3];
    KokoroIpaChunks chunks = {0};
    make_two_chunks(token_ids, &chunks);
    chunks.count = 1;
    KokoroStream *stream = NULL;
    KokoroStatus status = start_synthetic_stream(&audio, &synthesis, &chunks, &stream);
    int passed = status == KOKORO_OK && wait_audio(&audio, ready_enqueue_entered, 3000);
    AsyncCancel cancel;
    pthread_t cancel_thread;
    bool cancel_started = stream != NULL && start_async_cancel(&cancel, stream, &cancel_thread);
    passed = passed && cancel_started && wait_cancel_flag(&cancel, false, 3000);
    bool cancel_returned_before_enqueue = cancel_started && wait_cancel_flag(&cancel, true, 100);
    pthread_mutex_lock(&audio.mutex);
    audio.release_blocked_api = true;
    pthread_cond_broadcast(&audio.condition);
    pthread_mutex_unlock(&audio.mutex);
    bool cancel_finished = cancel_started && finish_async_cancel(&cancel, cancel_thread);
    KokoroError error = {0};
    if (stream != NULL) {
        status = kokoro_stream_wait(stream, NULL, &error);
    }
    passed =
        passed && cancel_finished && !cancel_returned_before_enqueue && status == KOKORO_CANCELLED;
    pthread_mutex_lock(&audio.mutex);
    passed = passed && audio.start_count == 0 && audio.pending_on_dispose == 0 &&
             !audio.running_on_dispose && audio.immediate_stop_count >= 1;
    pthread_mutex_unlock(&audio.mutex);
    kokoro_stream_free(stream);
    mock_audio_destroy(&audio);
    synthetic_destroy(&synthesis);
    return passed ? 1 : fail("cancellation raced enqueue and left later audio queued");
}

static int test_cancel_during_start_does_not_resume_queue(void) {
    MockAudio audio;
    const size_t sample_counts[] = {32, 0};
    SyntheticSynthesis synthesis;
    synthetic_init(&synthesis, sample_counts);
    if (!mock_audio_init(&audio, false)) {
        synthetic_destroy(&synthesis);
        return fail("could not initialize start race fixture");
    }
    audio.start_blocked = true;
    int64_t token_ids[2][3];
    KokoroIpaChunks chunks = {0};
    make_two_chunks(token_ids, &chunks);
    chunks.count = 1;
    KokoroStream *stream = NULL;
    KokoroStatus status = start_synthetic_stream(&audio, &synthesis, &chunks, &stream);
    int passed = status == KOKORO_OK && wait_audio(&audio, ready_start_entered, 3000);
    AsyncCancel cancel;
    pthread_t cancel_thread;
    bool cancel_started = stream != NULL && start_async_cancel(&cancel, stream, &cancel_thread);
    passed = passed && cancel_started && wait_cancel_flag(&cancel, false, 3000);
    bool cancel_returned_before_start = cancel_started && wait_cancel_flag(&cancel, true, 100);
    pthread_mutex_lock(&audio.mutex);
    audio.release_blocked_api = true;
    pthread_cond_broadcast(&audio.condition);
    pthread_mutex_unlock(&audio.mutex);
    bool cancel_finished = cancel_started && finish_async_cancel(&cancel, cancel_thread);
    KokoroError error = {0};
    if (stream != NULL) {
        status = kokoro_stream_wait(stream, NULL, &error);
    }
    passed =
        passed && cancel_finished && !cancel_returned_before_start && status == KOKORO_CANCELLED;
    pthread_mutex_lock(&audio.mutex);
    passed = passed && audio.start_count == 1 && !audio.running_on_dispose &&
             audio.immediate_stop_count >= 2 && audio.graceful_stop_count == 0;
    pthread_mutex_unlock(&audio.mutex);
    kokoro_stream_free(stream);
    mock_audio_destroy(&audio);
    synthetic_destroy(&synthesis);
    return passed ? 1 : fail("cancellation raced AudioQueueStart and left playback running");
}

static int run_error_case(bool create_error, bool listener_error, size_t allocation_error_at,
                          size_t enqueue_error_at, bool start_error, bool drain_error,
                          bool property_error, bool dispose_error, bool synthesis_error,
                          KokoroStatus expected_status, const char *expected_message) {
    MockAudio audio;
    const size_t sample_counts[] = {128, 64};
    SyntheticSynthesis synthesis;
    synthetic_init(&synthesis, sample_counts);
    synthesis.fail_chunk = synthesis_error ? 1 : -1;
    if (!mock_audio_init(&audio, true)) {
        synthetic_destroy(&synthesis);
        return 0;
    }
    audio.create_error = create_error ? FAILURE_STATUS : 0;
    audio.listener_error = listener_error ? FAILURE_STATUS : 0;
    audio.allocation_error_at = allocation_error_at;
    audio.enqueue_error_at = enqueue_error_at;
    audio.start_error = start_error;
    audio.graceful_stop_error = drain_error;
    audio.property_error = property_error;
    audio.dispose_error = dispose_error;
    int64_t token_ids[2][3];
    KokoroIpaChunks chunks = {0};
    make_two_chunks(token_ids, &chunks);
    KokoroStream *stream = NULL;
    KokoroStatus status = start_synthetic_stream(&audio, &synthesis, &chunks, &stream);
    KokoroError error = {0};
    if (status == KOKORO_OK) {
        status = kokoro_stream_wait(stream, NULL, &error);
    }
    int passed = status == expected_status &&
                 (expected_message == NULL || strstr(error.message, expected_message) != NULL);
    if (stream != NULL) {
        kokoro_stream_free(stream);
    }
    pthread_mutex_lock(&audio.mutex);
    if (audio.queue_created) {
        passed = passed && audio.dispose_count == 1 &&
                 audio.remove_listener_count == (listener_error ? 0U : 1U);
    }
    pthread_mutex_unlock(&audio.mutex);
    mock_audio_destroy(&audio);
    synthetic_destroy(&synthesis);
    return passed;
}

static int test_errors_stop_and_clean_up(void) {
    int passed = run_error_case(true, false, 0, 0, false, false, false, false, false,
                                KOKORO_IO_ERROR, "AudioQueue create") &&
                 run_error_case(false, true, 0, 0, false, false, false, false, false,
                                KOKORO_IO_ERROR, "running-state listener") &&
                 run_error_case(false, false, 2, 0, false, false, false, false, false,
                                KOKORO_IO_ERROR, "allocate buffer") &&
                 run_error_case(false, false, 0, 1, false, false, false, false, false,
                                KOKORO_IO_ERROR, "enqueue buffer") &&
                 run_error_case(false, false, 0, 0, true, false, false, false, false,
                                KOKORO_IO_ERROR, "AudioQueue start") &&
                 run_error_case(false, false, 0, 0, false, true, false, false, false,
                                KOKORO_IO_ERROR, "graceful stop") &&
                 run_error_case(false, false, 0, 0, false, false, true, false, false,
                                KOKORO_IO_ERROR, "running state") &&
                 run_error_case(false, false, 0, 0, false, false, false, true, false,
                                KOKORO_IO_ERROR, "dispose") &&
                 run_error_case(false, false, 0, 0, false, false, false, true, true,
                                KOKORO_INVALID_TOKEN_ID, "synthetic synthesis failure");
    return passed ? 1 : fail("AudioQueue or synthesis errors were not preserved through cleanup");
}

static int test_invalid_stream_request(void) {
    int64_t token_ids[3] = {0, 100, 0};
    KokoroIpaChunk chunk = {.token_ids = token_ids, .token_count = 3};
    KokoroIpaChunks chunks = {.items = &chunk, .count = 1};
    const KokoroStreamPlaybackOps invalid_ops = {0};
    KokoroStream *stream = NULL;
    KokoroError error = {0};
    KokoroStatus status =
        kokoro_stream_start_with((KokoroEngine *)(uintptr_t)1, &chunks, 2.1f, 1, &invalid_ops, NULL,
                                 synthesize_deterministic, NULL, &stream, &error);
    return status == KOKORO_INVALID_ARGUMENT && stream == NULL
               ? 1
               : fail("invalid speed or playback ops were accepted");
}

static int test_float_parity_and_repeated_engine_requests(void) {
    if (access(MODEL_PACK_PATH, R_OK) != 0) {
        puts("SKIP streamed float parity: convert the pinned model pack first");
        return 1;
    }
    char *phonemes = "həlˈoʊ\nən";
    KokoroIpaChunks chunks = {0};
    KokoroError error = {0};
    KokoroStatus status = kokoro_ipa_encode_chunks(phonemes, true, &chunks, &error);
    if (status != KOKORO_OK || chunks.count != 2) {
        kokoro_ipa_chunks_free(&chunks);
        return fail("could not prepare two sentence chunks for model parity");
    }
    KokoroEngine *engine = NULL;
    status = kokoro_engine_create(MODEL_PACK_PATH, "af_heart", NULL, &engine, &error);
    if (status != KOKORO_OK) {
        kokoro_ipa_chunks_free(&chunks);
        return fail("could not create resident engine for stream parity");
    }
    int passed = 1;
    for (size_t repeat = 0; passed && repeat < 2; repeat++) {
        float *expected = NULL;
        size_t expected_count = 0;
        size_t expected_capacity = 0;
        for (size_t index = 0; index < chunks.count; index++) {
            KokoroSynthesisOptions options = {
                .voice_style_index = chunks.items[index].token_count - 3,
                .speed = 1.0f,
                .random_seed = 104729 + (uint64_t)index,
            };
            KokoroSynthesisOutput output = {0};
            status = kokoro_engine_synthesize(engine, chunks.items[index].token_ids,
                                              chunks.items[index].token_count, &options, &output,
                                              &error);
            if (status != KOKORO_OK || output.samples == NULL || output.sample_count == 0) {
                passed = 0;
                kokoro_synthesis_output_free(&output);
                break;
            }
            size_t required = expected_count + output.sample_count;
            if (required > expected_capacity) {
                size_t grown = required;
                float *next = realloc(expected, grown * sizeof(*next));
                if (next == NULL) {
                    passed = 0;
                    kokoro_synthesis_output_free(&output);
                    break;
                }
                expected = next;
                expected_capacity = grown;
            }
            memcpy(expected + expected_count, output.samples,
                   output.sample_count * sizeof(*output.samples));
            expected_count = required;
            kokoro_synthesis_output_free(&output);
        }
        MockAudio audio;
        if (!mock_audio_init(&audio, true)) {
            passed = 0;
            free(expected);
            break;
        }
        active_mock_audio = &audio;
        KokoroStream *stream = NULL;
        if (passed) {
            status = kokoro_stream_start_with(engine, &chunks, 1.0f, 104729,
                                              kokoro_audio_queue_playback_ops(), (void *)&mock_api,
                                              NULL, NULL, &stream, &error);
            passed = status == KOKORO_OK;
        }
        if (passed) {
            status = kokoro_stream_wait(stream, NULL, &error);
            passed = status == KOKORO_OK;
        }
        pthread_mutex_lock(&audio.mutex);
        passed =
            passed && audio.consumed_count == expected_count &&
            memcmp(audio.consumed_samples, expected, expected_count * sizeof(*expected)) == 0 &&
            audio.graceful_stop_count == 1;
        pthread_mutex_unlock(&audio.mutex);
        kokoro_stream_free(stream);
        mock_audio_destroy(&audio);
        free(expected);
    }
    kokoro_engine_free(engine);
    kokoro_ipa_chunks_free(&chunks);
    return passed ? 1 : fail("streamed float samples differed from repeated batch engine requests");
}

static int test_audio_queue_silence_smoke(void) {
    if (getenv("KOKORO_AUDIO_QUEUE_SMOKE") == NULL) {
        puts("SKIP AudioQueue device smoke: set KOKORO_AUDIO_QUEUE_SMOKE=1 to play silence");
        return 1;
    }
    int64_t token_ids[] = {0, 100, 0};
    KokoroIpaChunk chunk = {.token_ids = token_ids, .token_count = 3};
    KokoroIpaChunks chunks = {.items = &chunk, .count = 1};
    KokoroStream *stream = NULL;
    KokoroError error = {0};
    KokoroStatus status = kokoro_stream_start_with((KokoroEngine *)(uintptr_t)1, &chunks, 1.0f, 701,
                                                   kokoro_audio_queue_playback_ops(), NULL,
                                                   synthesize_silence, NULL, &stream, &error);
    if (status == KOKORO_OK) {
        status = kokoro_stream_wait(stream, NULL, &error);
    }
    kokoro_stream_free(stream);
    return status == KOKORO_OK
               ? 1
               : fail(error.message[0] == '\0' ? "real AudioQueue silence lifecycle failed"
                                               : error.message);
}

int main(void) {
    alarm(90);
    if (!test_first_chunk_consumed_before_final_synthesis() ||
        !test_full_buffer_backpressure_and_resume() ||
        !test_cancel_stops_playback_during_synthesis() ||
        !test_signal_wait_cancels_once_while_synthesis_is_blocked() ||
        !test_cancel_wakes_full_buffer_wait() || !test_cancel_wakes_graceful_drain() ||
        !test_cancel_during_enqueue_does_not_start_later() ||
        !test_cancel_during_start_does_not_resume_queue() || !test_errors_stop_and_clean_up() ||
        !test_invalid_stream_request() || !test_float_parity_and_repeated_engine_requests()) {
        return 1;
    }
    if (!test_audio_queue_silence_smoke()) {
        return 1;
    }
    puts("stream producer, AudioQueue lifecycle, cancellation, and float parity checks passed");
    return 0;
}

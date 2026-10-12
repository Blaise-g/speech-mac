#include "audio_queue_playback.h"

#include "kokoro_internal.h"
#include "stream_benchmark.h"

#include <dispatch/dispatch.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define AUDIO_QUEUE_FRAME_BYTES ((UInt32)sizeof(float))
#define AUDIO_QUEUE_BUFFER_BYTES (KOKORO_STREAM_BUFFER_FRAMES * AUDIO_QUEUE_FRAME_BYTES)

_Static_assert(ATOMIC_BOOL_LOCK_FREE == 2, "AudioQueue callbacks require lock-free buffer flags");
_Static_assert(ATOMIC_INT_LOCK_FREE == 2, "AudioQueue callbacks require lock-free property state");

typedef struct {
    const KokoroAudioQueueApi *api;
    AudioQueueRef queue;
    AudioQueueBufferRef buffers[KOKORO_STREAM_BUFFER_COUNT];
    atomic_bool used[KOKORO_STREAM_BUFFER_COUNT];
    atomic_bool cancelled;
    atomic_bool progress_pending;
    atomic_uint property_generation;
#ifdef KOKORO_STREAM_BENCHMARK_TIMING
    atomic_uint queued_buffers;
    atomic_bool submissions_complete;
    atomic_bool started;
#endif
    bool listener_installed;
    bool running_observed;
    dispatch_semaphore_t progress;
    /* Orders SDK calls; callbacks never acquire this mutex. */
    pthread_mutex_t operation_mutex;
} AudioQueuePlayback;

static const KokoroAudioQueueApi default_api = {
    .new_output = AudioQueueNewOutput,
    .add_property_listener = AudioQueueAddPropertyListener,
    .remove_property_listener = AudioQueueRemovePropertyListener,
    .allocate_buffer = AudioQueueAllocateBuffer,
    .enqueue_buffer = AudioQueueEnqueueBuffer,
    .start = AudioQueueStart,
    .stop = AudioQueueStop,
    .get_property = AudioQueueGetProperty,
    .dispose = AudioQueueDispose,
};

const KokoroAudioQueueApi *kokoro_audio_queue_default_api(void) { return &default_api; }

static void set_audio_error(KokoroError *error, const char *operation, OSStatus status) {
    kokoro_error_set(error, KOKORO_IO_ERROR, "AudioQueue %s failed with status %d", operation,
                     (int)status);
}

static void signal_progress(AudioQueuePlayback *playback) {
    bool expected = false;
    if (atomic_compare_exchange_strong_explicit(&playback->progress_pending, &expected, true,
                                                memory_order_acq_rel, memory_order_relaxed)) {
        (void)dispatch_semaphore_signal(playback->progress);
    }
}

static void wait_for_progress(AudioQueuePlayback *playback) {
    (void)dispatch_semaphore_wait(playback->progress, DISPATCH_TIME_FOREVER);
    atomic_store_explicit(&playback->progress_pending, false, memory_order_release);
}

static void buffer_available(void *user_data, AudioQueueRef queue, AudioQueueBufferRef buffer) {
    (void)queue;
    AudioQueuePlayback *playback = user_data;
#ifdef KOKORO_STREAM_BENCHMARK_TIMING
    unsigned int queued =
        atomic_fetch_sub_explicit(&playback->queued_buffers, 1, memory_order_acq_rel);
    if (queued == 1 && atomic_load_explicit(&playback->started, memory_order_acquire) &&
        !atomic_load_explicit(&playback->submissions_complete, memory_order_acquire)) {
        KOKORO_STREAM_BENCHMARK_RECORD(KOKORO_STREAM_BENCHMARK_QUEUE_STARVED, 0);
    }
    KOKORO_STREAM_BENCHMARK_RECORD(KOKORO_STREAM_BENCHMARK_BUFFER_RECLAIMED, 0);
#endif
    for (size_t index = 0; index < KOKORO_STREAM_BUFFER_COUNT; index++) {
        if (playback->buffers[index] == buffer) {
            atomic_store_explicit(&playback->used[index], false, memory_order_release);
            break;
        }
    }
    signal_progress(playback);
}

static void property_changed(void *user_data, AudioQueueRef queue, AudioQueuePropertyID property) {
    (void)queue;
    if (property != kAudioQueueProperty_IsRunning) {
        return;
    }
    AudioQueuePlayback *playback = user_data;
    atomic_fetch_add_explicit(&playback->property_generation, 1, memory_order_release);
    signal_progress(playback);
}

static KokoroStatus audio_queue_close(void *handle, KokoroError *error) {
    AudioQueuePlayback *playback = handle;
    if (playback == NULL) {
        return KOKORO_OK;
    }
    OSStatus status = noErr;
    const char *failed_operation = NULL;
    if (playback->listener_installed && playback->queue != NULL) {
        status = playback->api->remove_property_listener(
            playback->queue, kAudioQueueProperty_IsRunning, property_changed, playback);
        if (status != noErr) {
            failed_operation = "remove running-state listener";
        }
        playback->listener_installed = false;
    }
    if (playback->queue != NULL) {
        OSStatus dispose_status = playback->api->dispose(playback->queue, true);
        if (dispose_status != noErr && status == noErr) {
            status = dispose_status;
            failed_operation = "dispose";
        }
    }
    dispatch_release(playback->progress);
    pthread_mutex_destroy(&playback->operation_mutex);
    free(playback);
    if (status != noErr) {
        set_audio_error(error, failed_operation, status);
        return KOKORO_IO_ERROR;
    }
    return KOKORO_OK;
}

static KokoroStatus audio_queue_open(void *context, void **handle, KokoroError *error) {
    KOKORO_STREAM_BENCHMARK_RECORD(KOKORO_STREAM_BENCHMARK_AUDIOQUEUE_OPEN_BEGIN, 0);
    if (handle == NULL) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT, "playback output pointer is required");
        return KOKORO_INVALID_ARGUMENT;
    }
    *handle = NULL;
    const KokoroAudioQueueApi *api = context == NULL ? &default_api : context;
    if (api->new_output == NULL || api->add_property_listener == NULL ||
        api->remove_property_listener == NULL || api->allocate_buffer == NULL ||
        api->enqueue_buffer == NULL || api->start == NULL || api->stop == NULL ||
        api->get_property == NULL || api->dispose == NULL) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT, "incomplete AudioQueue API");
        return KOKORO_INVALID_ARGUMENT;
    }
    AudioQueuePlayback *playback = calloc(1, sizeof(*playback));
    if (playback == NULL) {
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY, "cannot allocate AudioQueue state");
        return KOKORO_OUT_OF_MEMORY;
    }
    playback->api = api;
    atomic_init(&playback->cancelled, false);
    atomic_init(&playback->progress_pending, false);
    atomic_init(&playback->property_generation, 0);
#ifdef KOKORO_STREAM_BENCHMARK_TIMING
    atomic_init(&playback->queued_buffers, 0);
    atomic_init(&playback->submissions_complete, false);
    atomic_init(&playback->started, false);
#endif
    for (size_t index = 0; index < KOKORO_STREAM_BUFFER_COUNT; index++) {
        atomic_init(&playback->used[index], false);
    }
    int operation_mutex_error = pthread_mutex_init(&playback->operation_mutex, NULL);
    if (operation_mutex_error != 0) {
        free(playback);
        kokoro_error_set(error, KOKORO_IO_ERROR, "cannot initialize AudioQueue operation mutex: %s",
                         strerror(operation_mutex_error));
        return KOKORO_IO_ERROR;
    }
    playback->progress = dispatch_semaphore_create(0);
    if (playback->progress == NULL) {
        pthread_mutex_destroy(&playback->operation_mutex);
        free(playback);
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY,
                         "cannot create AudioQueue progress semaphore");
        return KOKORO_OUT_OF_MEMORY;
    }
    AudioStreamBasicDescription format = {
        .mSampleRate = 24000.0,
        .mFormatID = kAudioFormatLinearPCM,
        .mFormatFlags = kAudioFormatFlagsNativeFloatPacked,
        .mBytesPerPacket = AUDIO_QUEUE_FRAME_BYTES,
        .mFramesPerPacket = 1,
        .mBytesPerFrame = AUDIO_QUEUE_FRAME_BYTES,
        .mChannelsPerFrame = 1,
        .mBitsPerChannel = 32,
        .mReserved = 0,
    };
    OSStatus status =
        api->new_output(&format, buffer_available, playback, NULL, NULL, 0, &playback->queue);
    if (status != noErr) {
        set_audio_error(error, "create", status);
        audio_queue_close(playback, NULL);
        return KOKORO_IO_ERROR;
    }
#ifdef KOKORO_STREAM_SILENT_OUTPUT
    status = AudioQueueSetParameter(playback->queue, kAudioQueueParam_Volume, 0.0f);
    if (status != noErr) {
        set_audio_error(error, "set benchmark queue volume to zero", status);
        audio_queue_close(playback, NULL);
        return KOKORO_IO_ERROR;
    }
#endif
    status = api->add_property_listener(playback->queue, kAudioQueueProperty_IsRunning,
                                        property_changed, playback);
    if (status != noErr) {
        set_audio_error(error, "add running-state listener", status);
        audio_queue_close(playback, NULL);
        return KOKORO_IO_ERROR;
    }
    playback->listener_installed = true;
    for (size_t index = 0; index < KOKORO_STREAM_BUFFER_COUNT; index++) {
        status = api->allocate_buffer(playback->queue, AUDIO_QUEUE_BUFFER_BYTES,
                                      &playback->buffers[index]);
        if (status != noErr) {
            set_audio_error(error, "allocate buffer", status);
            audio_queue_close(playback, NULL);
            return KOKORO_IO_ERROR;
        }
    }
    *handle = playback;
    KOKORO_STREAM_BENCHMARK_RECORD(KOKORO_STREAM_BENCHMARK_AUDIOQUEUE_OPEN_END, 0);
    kokoro_error_set(error, KOKORO_OK, NULL);
    return KOKORO_OK;
}

static size_t claim_free_buffer(AudioQueuePlayback *playback) {
    for (size_t index = 0; index < KOKORO_STREAM_BUFFER_COUNT; index++) {
        bool expected = false;
        if (atomic_compare_exchange_strong_explicit(&playback->used[index], &expected, true,
                                                    memory_order_acquire, memory_order_relaxed)) {
            return index;
        }
    }
    return KOKORO_STREAM_BUFFER_COUNT;
}

static void release_buffer(AudioQueuePlayback *playback, size_t index) {
    atomic_store_explicit(&playback->used[index], false, memory_order_release);
    signal_progress(playback);
}

static KokoroStatus audio_queue_submit(void *handle, const float *samples, size_t sample_count,
                                       KokoroError *error) {
    AudioQueuePlayback *playback = handle;
    if (playback == NULL || samples == NULL || sample_count == 0 ||
        sample_count > KOKORO_STREAM_BUFFER_FRAMES) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT, "invalid AudioQueue buffer submission");
        return KOKORO_INVALID_ARGUMENT;
    }
    size_t index;
    for (;;) {
        if (atomic_load_explicit(&playback->cancelled, memory_order_acquire)) {
            kokoro_error_set(error, KOKORO_CANCELLED, "stream cancelled");
            return KOKORO_CANCELLED;
        }
        index = claim_free_buffer(playback);
        if (index < KOKORO_STREAM_BUFFER_COUNT) {
            break;
        }
        KOKORO_STREAM_BENCHMARK_RECORD(KOKORO_STREAM_BENCHMARK_BACKPRESSURE_BEGIN, 0);
        wait_for_progress(playback);
        KOKORO_STREAM_BENCHMARK_RECORD(KOKORO_STREAM_BENCHMARK_BACKPRESSURE_END, 0);
    }
    AudioQueueBufferRef buffer = playback->buffers[index];

    size_t byte_count = sample_count * sizeof(*samples);
    memcpy(buffer->mAudioData, samples, byte_count);
    buffer->mAudioDataByteSize = (UInt32)byte_count;
#ifdef KOKORO_STREAM_BENCHMARK_TIMING
    (void)atomic_fetch_add_explicit(&playback->queued_buffers, 1, memory_order_acq_rel);
#endif
    pthread_mutex_lock(&playback->operation_mutex);
    if (atomic_load_explicit(&playback->cancelled, memory_order_acquire)) {
        pthread_mutex_unlock(&playback->operation_mutex);
#ifdef KOKORO_STREAM_BENCHMARK_TIMING
        (void)atomic_fetch_sub_explicit(&playback->queued_buffers, 1, memory_order_acq_rel);
#endif
        release_buffer(playback, index);
        kokoro_error_set(error, KOKORO_CANCELLED, "stream cancelled");
        return KOKORO_CANCELLED;
    }
    OSStatus status = playback->api->enqueue_buffer(playback->queue, buffer, 0, NULL);
    pthread_mutex_unlock(&playback->operation_mutex);
    if (status != noErr) {
#ifdef KOKORO_STREAM_BENCHMARK_TIMING
        (void)atomic_fetch_sub_explicit(&playback->queued_buffers, 1, memory_order_acq_rel);
#endif
        release_buffer(playback, index);
        set_audio_error(error, "enqueue buffer", status);
        return KOKORO_IO_ERROR;
    }
    if (atomic_load_explicit(&playback->cancelled, memory_order_acquire)) {
        kokoro_error_set(error, KOKORO_CANCELLED, "stream cancelled");
        return KOKORO_CANCELLED;
    }
    return KOKORO_OK;
}

static KokoroStatus wait_for_running(AudioQueuePlayback *playback, bool running,
                                     KokoroError *error) {
    for (;;) {
        if (atomic_load_explicit(&playback->cancelled, memory_order_acquire)) {
            kokoro_error_set(error, KOKORO_CANCELLED, "stream cancelled");
            return KOKORO_CANCELLED;
        }
        unsigned generation =
            atomic_load_explicit(&playback->property_generation, memory_order_acquire);
        UInt32 is_running = 0;
        UInt32 size = sizeof(is_running);
        OSStatus status = playback->api->get_property(
            playback->queue, kAudioQueueProperty_IsRunning, &is_running, &size);
        if (status != noErr) {
            set_audio_error(error, "read running state", status);
            return KOKORO_IO_ERROR;
        }
        if ((is_running != 0) == running) {
            if (running) {
                playback->running_observed = true;
            }
            return KOKORO_OK;
        }
        if (atomic_load_explicit(&playback->cancelled, memory_order_acquire)) {
            kokoro_error_set(error, KOKORO_CANCELLED, "stream cancelled");
            return KOKORO_CANCELLED;
        }
        if (atomic_load_explicit(&playback->property_generation, memory_order_acquire) !=
            generation) {
            continue;
        }
        wait_for_progress(playback);
    }
}

static KokoroStatus audio_queue_start(void *handle, KokoroError *error) {
    AudioQueuePlayback *playback = handle;
    pthread_mutex_lock(&playback->operation_mutex);
    if (atomic_load_explicit(&playback->cancelled, memory_order_acquire)) {
        pthread_mutex_unlock(&playback->operation_mutex);
        kokoro_error_set(error, KOKORO_CANCELLED, "stream cancelled");
        return KOKORO_CANCELLED;
    }

    KOKORO_STREAM_BENCHMARK_RECORD(KOKORO_STREAM_BENCHMARK_AUDIOQUEUE_START_BEGIN, 0);
    OSStatus status = playback->api->start(playback->queue, NULL);
#ifdef KOKORO_STREAM_BENCHMARK_TIMING
    if (status == noErr) {
        atomic_store_explicit(&playback->started, true, memory_order_release);
    }
#endif
    pthread_mutex_unlock(&playback->operation_mutex);
    if (status != noErr) {
        set_audio_error(error, "start", status);
        return KOKORO_IO_ERROR;
    }
    KOKORO_STREAM_BENCHMARK_RECORD(KOKORO_STREAM_BENCHMARK_START_REQUESTED, 0);
    return wait_for_running(playback, true, error);
}

static KokoroStatus audio_queue_drain(void *handle, KokoroError *error) {
    AudioQueuePlayback *playback = handle;
#ifdef KOKORO_STREAM_BENCHMARK_TIMING
    atomic_store_explicit(&playback->submissions_complete, true, memory_order_release);
#endif
    pthread_mutex_lock(&playback->operation_mutex);
    bool running_observed = playback->running_observed;
    if (atomic_load_explicit(&playback->cancelled, memory_order_acquire)) {
        pthread_mutex_unlock(&playback->operation_mutex);
        kokoro_error_set(error, KOKORO_CANCELLED, "stream cancelled");
        return KOKORO_CANCELLED;
    }
    if (!running_observed) {
        pthread_mutex_unlock(&playback->operation_mutex);
        kokoro_error_set(error, KOKORO_IO_ERROR,
                         "AudioQueue drain started before running state was observed");
        return KOKORO_IO_ERROR;
    }
    OSStatus status = playback->api->stop(playback->queue, false);
    pthread_mutex_unlock(&playback->operation_mutex);
    if (status != noErr) {
        set_audio_error(error, "graceful stop", status);
        return KOKORO_IO_ERROR;
    }
    return wait_for_running(playback, false, error);
}

static void audio_queue_wake(void *handle) {
    AudioQueuePlayback *playback = handle;
    atomic_store_explicit(&playback->cancelled, true, memory_order_release);
    signal_progress(playback);
}

static void audio_queue_abort(void *handle) {
    AudioQueuePlayback *playback = handle;
    if (playback != NULL && playback->queue != NULL) {
        pthread_mutex_lock(&playback->operation_mutex);
        (void)playback->api->stop(playback->queue, true);
        pthread_mutex_unlock(&playback->operation_mutex);
    }
}

static const KokoroStreamPlaybackOps playback_ops = {
    .open = audio_queue_open,
    .submit = audio_queue_submit,
    .start = audio_queue_start,
    .drain = audio_queue_drain,
    .wake = audio_queue_wake,
    .abort = audio_queue_abort,
    .close = audio_queue_close,
};

const KokoroStreamPlaybackOps *kokoro_audio_queue_playback_ops(void) { return &playback_ops; }

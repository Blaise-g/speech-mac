#include "speech_stream.h"

#include "audio_queue_playback.h"
#include "speech_error_internal.h"
#include "stream_benchmark.h"
#ifdef PHONON_PACK_TEST_API
#include "speech_stream_test.h"
#endif

#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

struct SpeechStream {
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    pthread_t producer;
    bool producer_started;
    bool producer_joined;
    bool producer_finished;
    bool cancelled;
    bool terminal;
    SpeechIOStatus status;
    SpeechError error;
    size_t playback_users;
    const SpeechEngine *engine;
    const PhononVoice *voice;
    const char *text;
    size_t length;
    SpeechOptions options;
    SpeechResult progress;
    SpeechStreamMetrics metrics;
#ifdef PHONON_PACK_TEST_API
    SpeechStreamTestProducer test_producer;
#endif
    bool started;
    SpeechIOStatus playback_status;
    const SpeechStreamPlaybackOps *playback_ops;
    void *playback_context;
    void *playback;
};

static bool valid_playback_ops(const SpeechStreamPlaybackOps *ops) {
    return ops != NULL && ops->open != NULL && ops->submit != NULL && ops->start != NULL &&
           ops->drain != NULL && ops->wake != NULL && ops->abort != NULL && ops->close != NULL;
}

static void set_terminal_locked(SpeechStream *stream, SpeechIOStatus status,
                                const SpeechError *error) {
    if (stream->terminal) {
        return;
    }
    stream->terminal = true;
    stream->status = status;
    if (error != NULL) {
        stream->error = *error;
        stream->error.code = status;
    } else {
        speech_error_set(&stream->error, status,
                         status == SPEECH_IO_CANCELLED ? "stream cancelled" : NULL);
    }
}

static void set_terminal(SpeechStream *stream, SpeechIOStatus status, const SpeechError *error) {
    pthread_mutex_lock(&stream->mutex);
    set_terminal_locked(stream, status, error);
    pthread_cond_broadcast(&stream->condition);
    pthread_mutex_unlock(&stream->mutex);
}

static bool is_cancelled(SpeechStream *stream) {
    pthread_mutex_lock(&stream->mutex);
    bool cancelled = stream->cancelled;
    pthread_mutex_unlock(&stream->mutex);
    return cancelled;
}

static void publish_playback(SpeechStream *stream, void *playback) {
    pthread_mutex_lock(&stream->mutex);
    stream->playback = playback;
    pthread_mutex_unlock(&stream->mutex);
}

static void clear_playback(SpeechStream *stream, void *playback) {
    pthread_mutex_lock(&stream->mutex);
    if (stream->playback == playback) {
        stream->playback = NULL;
    }
    while (stream->playback_users > 0) {
        pthread_cond_wait(&stream->condition, &stream->mutex);
    }
    pthread_mutex_unlock(&stream->mutex);
}

static double timestamp(void) {
    struct timespec now;
    (void)clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec / 1e9;
}

static int request_cancel(void *context) {
    SpeechStream *stream = context;
    if (is_cancelled(stream))
        return 1;
    if (stream->options.cancel && stream->options.cancel(stream->options.cancel_context)) {
        speech_stream_cancel(stream);
        return 1;
    }
    return 0;
}

static int submit_pcm(void *context, const float *pcm, size_t count, const SpeechResult *progress) {
    SpeechStream *stream = context;
    (void)progress;
    if (request_cancel(stream))
        return 1;
    if (!stream->metrics.first_pcm)
        stream->metrics.first_pcm = timestamp();
    if (count != SPEECH_FRAME_SAMPLES)
        return 2;
    SpeechError error = {0};
    stream->playback_status = stream->playback_ops->submit(stream->playback, pcm, count, &error);
    if (stream->playback_status != SPEECH_IO_OK) {
        set_terminal(stream, stream->playback_status, &error);
        return 2;
    }
    if (!stream->metrics.first_submission)
        stream->metrics.first_submission = timestamp();
    stream->metrics.submitted_samples += count;
    SPEECH_STREAM_BENCHMARK_RECORD(SPEECH_STREAM_BENCHMARK_BUFFER_ACCEPTED, count);
    if (!stream->started) {
        stream->playback_status = stream->playback_ops->start(stream->playback, &error);
        if (stream->playback_status != SPEECH_IO_OK) {
            set_terminal(stream, stream->playback_status, &error);
            return 2;
        }
        stream->started = true;
        stream->metrics.started = timestamp();
    }
    return 0;
}

static void *stream_producer(void *argument) {
    SpeechStream *stream = argument;
    SpeechError error = {0};
    void *playback = NULL;
    SpeechIOStatus result = stream->playback_ops->open(stream->playback_context, &playback, &error);
    if (result != SPEECH_IO_OK) {
        set_terminal(stream, result, &error);
        goto done;
    }
    publish_playback(stream, playback);
    SpeechOptions options = stream->options;
    options.cancel = request_cancel;
    options.cancel_context = stream;
    SPEECH_STREAM_BENCHMARK_RECORD(SPEECH_STREAM_BENCHMARK_SYNTHESIS_BEGIN, 0);
    stream->metrics.synthesis_begin = timestamp();
    SpeechStatus status;
#ifdef PHONON_PACK_TEST_API
    if (stream->test_producer)
        status = stream->test_producer(stream->engine, stream->voice, stream->text, stream->length,
                                       &options, submit_pcm, stream, &stream->progress);
    else
#endif
        status = speech_synthesize(stream->engine, stream->voice, stream->text, stream->length,
                                   &options, submit_pcm, stream, &stream->progress);
    stream->metrics.synthesis_end = timestamp();
    SPEECH_STREAM_BENCHMARK_RECORD(SPEECH_STREAM_BENCHMARK_SYNTHESIS_END, 0);
    if (status == SPEECH_CANCELLED)
        result = SPEECH_IO_CANCELLED;
    else if (status != SPEECH_OK) {
        result = stream->playback_status != SPEECH_IO_OK ? stream->playback_status
                 : status == SPEECH_INVALID              ? SPEECH_IO_INVALID_ARGUMENT
                 : status == SPEECH_LIMIT                ? SPEECH_IO_CONTEXT_TOO_LONG
                                                         : SPEECH_IO_ERROR;
        speech_error_set(&error, result, "stream synthesis failed (speech status %d)", status);
        set_terminal(stream, result, &error);
    }
    if (result == SPEECH_IO_OK && !stream->started)
        result = SPEECH_IO_ERROR;
    if (result == SPEECH_IO_OK) {
        result = stream->playback_ops->drain(playback, &error);
        if (result == SPEECH_IO_OK) {
            /* Publish drain only if cancellation has not won the terminal state.
             * A later cancellation may still win before close completes. */
            pthread_mutex_lock(&stream->mutex);
            if (stream->cancelled)
                result = SPEECH_IO_CANCELLED;
            else {
                stream->metrics.drained = timestamp();
                SPEECH_STREAM_BENCHMARK_RECORD(SPEECH_STREAM_BENCHMARK_DRAIN_COMPLETE, 0);
            }
            pthread_mutex_unlock(&stream->mutex);
        }
    }
    if (result != SPEECH_IO_OK)
        stream->playback_ops->abort(playback);
    if (stream->playback_ops == speech_audio_queue_playback_ops())
        speech_audio_queue_reclaimed(playback, &stream->metrics.reclaimed_buffers,
                                     &stream->metrics.first_reclaimed);
    clear_playback(stream, playback);
    SpeechIOStatus closed = stream->playback_ops->close(playback, &error);
    if (result == SPEECH_IO_OK)
        result = closed;
    set_terminal(stream, result, &error);
done:
    pthread_mutex_lock(&stream->mutex);
    stream->producer_finished = true;
    pthread_cond_broadcast(&stream->condition);
    pthread_mutex_unlock(&stream->mutex);
    return NULL;
}

static SpeechIOStatus start_request(const SpeechEngine *engine, const PhononVoice *voice,
                                    const char *text, size_t length, const SpeechOptions *options,
                                    const SpeechStreamPlaybackOps *playback_ops,
                                    void *playback_context, SpeechStream **stream_out,
                                    SpeechError *error
#ifdef PHONON_PACK_TEST_API
                                    ,
                                    SpeechStreamTestProducer test_producer
#endif
) {
    if (!stream_out)
        return SPEECH_IO_INVALID_ARGUMENT;
    *stream_out = NULL;
    if (!engine || !voice || !text || !length || length > SPEECH_MAX_INPUT_BYTES ||
        !valid_playback_ops(playback_ops))
        return SPEECH_IO_INVALID_ARGUMENT;
    SpeechStream *stream = calloc(1, sizeof(*stream));
    if (!stream)
        return SPEECH_IO_OUT_OF_MEMORY;
    if (pthread_mutex_init(&stream->mutex, NULL)) {
        free(stream);
        return SPEECH_IO_ERROR;
    }
    if (pthread_cond_init(&stream->condition, NULL)) {
        pthread_mutex_destroy(&stream->mutex);
        free(stream);
        return SPEECH_IO_ERROR;
    }
#ifdef PHONON_PACK_TEST_API
    stream->test_producer = test_producer;
#endif
    stream->engine = engine;
    stream->voice = voice;
    stream->text = text;
    stream->length = length;
    stream->options = options ? *options : speech_options_default();
    stream->playback_ops = playback_ops;
    stream->playback_context = playback_context;
    int code = pthread_create(&stream->producer, NULL, stream_producer, stream);
    if (code) {
        pthread_cond_destroy(&stream->condition);
        pthread_mutex_destroy(&stream->mutex);
        free(stream);
        speech_error_set(error, SPEECH_IO_ERROR, "cannot start stream producer: %s",
                         strerror(code));
        return SPEECH_IO_ERROR;
    }
    stream->producer_started = true;
    *stream_out = stream;
    return SPEECH_IO_OK;
}

SpeechIOStatus speech_stream_start_with(const SpeechEngine *engine, const PhononVoice *voice,
                                        const char *text, size_t length,
                                        const SpeechOptions *options,
                                        const SpeechStreamPlaybackOps *ops, void *context,
                                        SpeechStream **stream, SpeechError *error) {
    return start_request(engine, voice, text, length, options, ops, context, stream, error
#ifdef PHONON_PACK_TEST_API
                         ,
                         NULL
#endif
    );
}
#ifdef PHONON_PACK_TEST_API
SpeechIOStatus speech_stream_test_start(const SpeechOptions *options,
                                        const SpeechStreamPlaybackOps *ops, void *context,
                                        SpeechStream **stream, SpeechStreamTestProducer producer) {
    if (!producer)
        return SPEECH_IO_INVALID_ARGUMENT;
    /* Test handles are never dereferenced by the controlled producer. */
    return start_request((const SpeechEngine *)(uintptr_t)1, (const PhononVoice *)(uintptr_t)1,
                         "synthetic", 9, options, ops, context, stream, NULL, producer);
}
#endif

SpeechIOStatus speech_stream_start(const SpeechEngine *engine, const PhononVoice *voice,
                                   const char *text, size_t length, const SpeechOptions *options,
                                   SpeechStream **stream, SpeechError *error) {
    return speech_stream_start_with(engine, voice, text, length, options,
                                    speech_audio_queue_playback_ops(), NULL, stream, error);
}

void speech_stream_cancel(SpeechStream *stream) {
    if (stream == NULL) {
        return;
    }
    pthread_mutex_lock(&stream->mutex);
    stream->cancelled = true;
    if (!stream->terminal) {
        speech_error_set(&stream->error, SPEECH_IO_CANCELLED, "stream cancelled");
        set_terminal_locked(stream, SPEECH_IO_CANCELLED, &stream->error);
    }
    void *playback = stream->playback;
    if (playback != NULL) {
        stream->playback_users++;
    }
    pthread_cond_broadcast(&stream->condition);
    pthread_mutex_unlock(&stream->mutex);
    if (playback != NULL) {
        stream->playback_ops->wake(playback);
        stream->playback_ops->abort(playback);
        pthread_mutex_lock(&stream->mutex);
        stream->playback_users--;
        pthread_cond_broadcast(&stream->condition);
        pthread_mutex_unlock(&stream->mutex);
    }
}

static struct timespec wait_deadline(void) {
    struct timespec deadline;
    (void)clock_gettime(CLOCK_REALTIME, &deadline);
    deadline.tv_nsec += 50000000L;
    if (deadline.tv_nsec >= 1000000000L) {
        deadline.tv_sec++;
        deadline.tv_nsec -= 1000000000L;
    }
    return deadline;
}

SpeechIOStatus speech_stream_wait(SpeechStream *stream, const volatile sig_atomic_t *interrupted,
                                  SpeechError *error) {
    if (stream == NULL) {
        speech_error_set(error, SPEECH_IO_INVALID_ARGUMENT, "stream is required");
        return SPEECH_IO_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&stream->mutex);
    while (!stream->producer_finished) {
        if (interrupted != NULL && *interrupted != 0 && !stream->cancelled) {
            pthread_mutex_unlock(&stream->mutex);
            speech_stream_cancel(stream);
            pthread_mutex_lock(&stream->mutex);
            continue;
        }
        struct timespec deadline = wait_deadline();
        int wait_error = pthread_cond_timedwait(&stream->condition, &stream->mutex, &deadline);
        if (wait_error != 0 && wait_error != ETIMEDOUT) {
            SpeechError wait_failure = {0};
            speech_error_set(&wait_failure, SPEECH_IO_ERROR, "cannot wait for stream producer: %s",
                             strerror(wait_error));
            set_terminal_locked(stream, SPEECH_IO_ERROR, &wait_failure);
            stream->cancelled = true;
            pthread_mutex_unlock(&stream->mutex);
            speech_stream_cancel(stream);
            pthread_mutex_lock(&stream->mutex);
        }
    }
    SpeechIOStatus status = stream->status;
    SpeechError stream_error = stream->error;
    pthread_mutex_unlock(&stream->mutex);

    int join_error = 0;
    pthread_mutex_lock(&stream->mutex);
    bool needs_join = stream->producer_started && !stream->producer_joined;
    pthread_mutex_unlock(&stream->mutex);
    if (needs_join) {
        join_error = pthread_join(stream->producer, NULL);
        if (join_error == 0) {
            pthread_mutex_lock(&stream->mutex);
            stream->producer_joined = true;
            pthread_mutex_unlock(&stream->mutex);
        }
    }
    if (join_error != 0 && status == SPEECH_IO_OK) {
        status = SPEECH_IO_ERROR;
        speech_error_set(&stream_error, status, "cannot join stream producer: %s",
                         strerror(join_error));
    }
    if (error != NULL) {
        *error = stream_error;
    }
    return status;
}

void speech_stream_free(SpeechStream *stream) {
    if (stream == NULL) {
        return;
    }
    speech_stream_cancel(stream);
    pthread_mutex_lock(&stream->mutex);
    bool needs_join = stream->producer_started && !stream->producer_joined;
    pthread_mutex_unlock(&stream->mutex);
    if (needs_join && pthread_join(stream->producer, NULL) != 0) {
        return;
    }
    pthread_cond_destroy(&stream->condition);
    pthread_mutex_destroy(&stream->mutex);
    free(stream);
}

void speech_stream_result(const SpeechStream *stream, SpeechResult *result,
                          SpeechStreamMetrics *metrics) {
    if (result)
        *result = stream->progress;
    if (metrics)
        *metrics = stream->metrics;
}

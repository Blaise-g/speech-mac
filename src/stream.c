#include "stream_internal.h"

#include "audio_queue_playback.h"
#include "kokoro_internal.h"
#include "stream_benchmark.h"

#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

struct KokoroStream {
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    pthread_t producer;
    bool producer_started;
    bool producer_joined;
    bool producer_finished;
    bool cancelled;
    bool terminal;
    KokoroStatus status;
    KokoroError error;
    size_t playback_users;
    KokoroEngine *engine;
    const KokoroIpaChunks *chunks;
    float speed;
    uint64_t random_seed;
    const KokoroStreamPlaybackOps *playback_ops;
    void *playback_context;
    void *playback;
    KokoroStreamSynthesizeFn synthesize;
    void *synthesize_context;
};

static KokoroStatus synthesize_engine(void *context, KokoroEngine *engine, const int64_t *token_ids,
                                      size_t token_count, const KokoroSynthesisOptions *options,
                                      KokoroSynthesisOutput *output, KokoroError *error) {
    (void)context;
    return kokoro_engine_synthesize(engine, token_ids, token_count, options, output, error);
}

static bool valid_playback_ops(const KokoroStreamPlaybackOps *ops) {
    return ops != NULL && ops->open != NULL && ops->submit != NULL && ops->start != NULL &&
           ops->drain != NULL && ops->wake != NULL && ops->abort != NULL && ops->close != NULL;
}

static void set_terminal_locked(KokoroStream *stream, KokoroStatus status,
                                const KokoroError *error) {
    if (stream->terminal) {
        return;
    }
    stream->terminal = true;
    stream->status = status;
    if (error != NULL) {
        stream->error = *error;
        stream->error.code = status;
    } else {
        kokoro_error_set(&stream->error, status,
                         status == KOKORO_CANCELLED ? "stream cancelled" : NULL);
    }
}

static void set_terminal(KokoroStream *stream, KokoroStatus status, const KokoroError *error) {
    pthread_mutex_lock(&stream->mutex);
    set_terminal_locked(stream, status, error);
    pthread_cond_broadcast(&stream->condition);
    pthread_mutex_unlock(&stream->mutex);
}

static bool is_cancelled(KokoroStream *stream) {
    pthread_mutex_lock(&stream->mutex);
    bool cancelled = stream->cancelled;
    pthread_mutex_unlock(&stream->mutex);
    return cancelled;
}

static void publish_playback(KokoroStream *stream, void *playback) {
    pthread_mutex_lock(&stream->mutex);
    stream->playback = playback;
    pthread_mutex_unlock(&stream->mutex);
}

static void clear_playback(KokoroStream *stream, void *playback) {
    pthread_mutex_lock(&stream->mutex);
    if (stream->playback == playback) {
        stream->playback = NULL;
    }
    while (stream->playback_users > 0) {
        pthread_cond_wait(&stream->condition, &stream->mutex);
    }
    pthread_mutex_unlock(&stream->mutex);
}

static void *stream_producer(void *argument) {
    KokoroStream *stream = argument;
    KokoroError error = {0};
    void *playback = NULL;
    bool opened = false;
    bool started = false;
    KokoroStatus result = stream->playback_ops->open(stream->playback_context, &playback, &error);
    if (result != KOKORO_OK) {
        set_terminal(stream, result, &error);
        goto done;
    }
    opened = true;
    publish_playback(stream, playback);
    if (is_cancelled(stream)) {
        result = KOKORO_CANCELLED;
        goto stop;
    }

    for (size_t chunk_index = 0; chunk_index < stream->chunks->count; chunk_index++) {
        if (is_cancelled(stream)) {
            result = KOKORO_CANCELLED;
            goto stop;
        }
        const KokoroIpaChunk *chunk = &stream->chunks->items[chunk_index];
        KOKORO_STREAM_BENCHMARK_RECORD(KOKORO_STREAM_BENCHMARK_CHUNK_BEGIN, 1);
        KokoroSynthesisOptions options = {
            .voice_style_index = chunk->token_count - 3,
            .speed = stream->speed,
            .random_seed = stream->random_seed + (uint64_t)chunk_index,
        };
        KokoroSynthesisOutput output = {0};
        memset(&error, 0, sizeof(error));
#ifdef KOKORO_STREAM_BENCHMARK_TIMING
        KOKORO_STREAM_BENCHMARK_RECORD(KOKORO_STREAM_BENCHMARK_SYNTHESIS_BEGIN, 0);
#endif
        result = stream->synthesize(stream->synthesize_context, stream->engine, chunk->token_ids,
                                    chunk->token_count, &options, &output, &error);
#ifdef KOKORO_STREAM_BENCHMARK_TIMING
        KOKORO_STREAM_BENCHMARK_RECORD(KOKORO_STREAM_BENCHMARK_SYNTHESIS_END, 0);
#endif
        if (result != KOKORO_OK) {
            kokoro_synthesis_output_free(&output);
            set_terminal(stream, result, &error);
            goto stop;
        }
        if (chunk_index == 0) {
            KOKORO_STREAM_BENCHMARK_RECORD(KOKORO_STREAM_BENCHMARK_FIRST_CHUNK_READY, 0);
        }
        if (is_cancelled(stream)) {
            kokoro_synthesis_output_free(&output);
            result = KOKORO_CANCELLED;
            goto stop;
        }
        if (output.samples == NULL || output.sample_count == 0) {
            kokoro_error_set(&error, KOKORO_NUMERIC_ERROR, "synthesis returned an empty waveform");
            kokoro_synthesis_output_free(&output);
            result = KOKORO_NUMERIC_ERROR;
            set_terminal(stream, result, &error);
            goto stop;
        }
        for (size_t offset = 0; offset < output.sample_count;) {
            if (is_cancelled(stream)) {
                kokoro_synthesis_output_free(&output);
                result = KOKORO_CANCELLED;
                goto stop;
            }
            size_t count = output.sample_count - offset;
            if (count > KOKORO_STREAM_BUFFER_FRAMES) {
                count = KOKORO_STREAM_BUFFER_FRAMES;
            }
            memset(&error, 0, sizeof(error));
            result = stream->playback_ops->submit(playback, output.samples + offset, count, &error);
            if (result != KOKORO_OK) {
                kokoro_synthesis_output_free(&output);
                if (result == KOKORO_CANCELLED) {
                    result = KOKORO_CANCELLED;
                } else {
                    set_terminal(stream, result, &error);
                }
                goto stop;
            }
            KOKORO_STREAM_BENCHMARK_RECORD(KOKORO_STREAM_BENCHMARK_BUFFER_ACCEPTED, count);
            offset += count;
            if (!started) {
                memset(&error, 0, sizeof(error));
                result = stream->playback_ops->start(playback, &error);
                if (result != KOKORO_OK) {
                    kokoro_synthesis_output_free(&output);
                    if (result != KOKORO_CANCELLED) {
                        set_terminal(stream, result, &error);
                    }
                    goto stop;
                }
                started = true;
            }
        }
        kokoro_synthesis_output_free(&output);
    }

    if (!started) {
        kokoro_error_set(&error, KOKORO_NUMERIC_ERROR, "stream produced no audio buffers");
        result = KOKORO_NUMERIC_ERROR;
        set_terminal(stream, result, &error);
        goto stop;
    }
    memset(&error, 0, sizeof(error));
    result = stream->playback_ops->drain(playback, &error);
#ifdef KOKORO_STREAM_BENCHMARK_TIMING
    if (result == KOKORO_OK) {
        KOKORO_STREAM_BENCHMARK_RECORD(KOKORO_STREAM_BENCHMARK_DRAIN_COMPLETE, 0);
    }
#endif
    if (result != KOKORO_OK) {
        if (result != KOKORO_CANCELLED) {
            set_terminal(stream, result, &error);
        }
        goto stop;
    }
    goto close;

stop:
    if (opened) {
        stream->playback_ops->abort(playback);
    }
close:
    if (opened) {
        clear_playback(stream, playback);
        memset(&error, 0, sizeof(error));
        KokoroStatus close_status = stream->playback_ops->close(playback, &error);
        if (result == KOKORO_OK && close_status != KOKORO_OK) {
            result = close_status;
        }
    }
    if (result == KOKORO_OK || result == KOKORO_CANCELLED) {
        set_terminal(stream, result, NULL);
    } else {
        pthread_mutex_lock(&stream->mutex);
        if (!stream->terminal) {
            if (error.code != result || error.message[0] == '\0') {
                kokoro_error_set(&error, result, "stream playback failed");
            }
            set_terminal_locked(stream, result, &error);
        }
        pthread_mutex_unlock(&stream->mutex);
    }
done:
    pthread_mutex_lock(&stream->mutex);
    stream->producer_finished = true;
    pthread_cond_broadcast(&stream->condition);
    pthread_mutex_unlock(&stream->mutex);
    return NULL;
}

static KokoroStatus stream_start(KokoroEngine *engine, const KokoroIpaChunks *chunks, float speed,
                                 uint64_t random_seed, const KokoroStreamPlaybackOps *playback_ops,
                                 void *playback_context, KokoroStreamSynthesizeFn synthesize,
                                 void *synthesize_context, KokoroStream **stream_out,
                                 KokoroError *error) {
    if (stream_out == NULL) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT, "stream output pointer is required");
        return KOKORO_INVALID_ARGUMENT;
    }
    *stream_out = NULL;
    if (engine == NULL || chunks == NULL || chunks->items == NULL || chunks->count == 0 ||
        !isfinite(speed) || speed < 0.5f || speed > 2.0f || !valid_playback_ops(playback_ops) ||
        (synthesize == NULL && synthesize_context != NULL)) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT, "invalid streaming request");
        return KOKORO_INVALID_ARGUMENT;
    }
    for (size_t index = 0; index < chunks->count; index++) {
        if (chunks->items[index].token_ids == NULL || chunks->items[index].token_count < 3) {
            kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                             "stream chunks require token IDs with BOS and EOS");
            return KOKORO_INVALID_ARGUMENT;
        }
    }
    KokoroStream *stream = calloc(1, sizeof(*stream));
    if (stream == NULL) {
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY, "cannot allocate stream state");
        return KOKORO_OUT_OF_MEMORY;
    }
    int pthread_error = pthread_mutex_init(&stream->mutex, NULL);
    if (pthread_error != 0) {
        free(stream);
        kokoro_error_set(error, KOKORO_IO_ERROR, "cannot initialize stream mutex: %s",
                         strerror(pthread_error));
        return KOKORO_IO_ERROR;
    }
    pthread_error = pthread_cond_init(&stream->condition, NULL);
    if (pthread_error != 0) {
        pthread_mutex_destroy(&stream->mutex);
        free(stream);
        kokoro_error_set(error, KOKORO_IO_ERROR, "cannot initialize stream condition: %s",
                         strerror(pthread_error));
        return KOKORO_IO_ERROR;
    }
    stream->engine = engine;
    stream->chunks = chunks;
    stream->speed = speed;
    stream->random_seed = random_seed;
    stream->playback_ops = playback_ops;
    stream->playback_context = playback_context;
    stream->synthesize = synthesize == NULL ? synthesize_engine : synthesize;
    stream->synthesize_context = synthesize_context;
    pthread_error = pthread_create(&stream->producer, NULL, stream_producer, stream);
    if (pthread_error != 0) {
        pthread_cond_destroy(&stream->condition);
        pthread_mutex_destroy(&stream->mutex);
        free(stream);
        kokoro_error_set(error, KOKORO_IO_ERROR, "cannot start stream producer: %s",
                         strerror(pthread_error));
        return KOKORO_IO_ERROR;
    }
    stream->producer_started = true;
    *stream_out = stream;
    kokoro_error_set(error, KOKORO_OK, NULL);
    return KOKORO_OK;
}

KokoroStatus kokoro_stream_start_with(KokoroEngine *engine, const KokoroIpaChunks *chunks,
                                      float speed, uint64_t random_seed,
                                      const KokoroStreamPlaybackOps *playback_ops,
                                      void *playback_context, KokoroStreamSynthesizeFn synthesize,
                                      void *synthesize_context, KokoroStream **stream,
                                      KokoroError *error) {
    return stream_start(engine, chunks, speed, random_seed, playback_ops, playback_context,
                        synthesize, synthesize_context, stream, error);
}

KokoroStatus kokoro_stream_start(KokoroEngine *engine, const KokoroIpaChunks *chunks, float speed,
                                 uint64_t random_seed, KokoroStream **stream, KokoroError *error) {
    return stream_start(engine, chunks, speed, random_seed, kokoro_audio_queue_playback_ops(), NULL,
                        synthesize_engine, NULL, stream, error);
}

void kokoro_stream_cancel(KokoroStream *stream) {
    if (stream == NULL) {
        return;
    }
    pthread_mutex_lock(&stream->mutex);
    stream->cancelled = true;
    if (!stream->terminal) {
        kokoro_error_set(&stream->error, KOKORO_CANCELLED, "stream cancelled");
        set_terminal_locked(stream, KOKORO_CANCELLED, &stream->error);
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

KokoroStatus kokoro_stream_wait(KokoroStream *stream, const volatile sig_atomic_t *interrupted,
                                KokoroError *error) {
    if (stream == NULL) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT, "stream is required");
        return KOKORO_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&stream->mutex);
    while (!stream->producer_finished) {
        if (interrupted != NULL && *interrupted != 0 && !stream->cancelled) {
            pthread_mutex_unlock(&stream->mutex);
            kokoro_stream_cancel(stream);
            pthread_mutex_lock(&stream->mutex);
            continue;
        }
        struct timespec deadline = wait_deadline();
        int wait_error = pthread_cond_timedwait(&stream->condition, &stream->mutex, &deadline);
        if (wait_error != 0 && wait_error != ETIMEDOUT) {
            KokoroError wait_failure = {0};
            kokoro_error_set(&wait_failure, KOKORO_IO_ERROR, "cannot wait for stream producer: %s",
                             strerror(wait_error));
            set_terminal_locked(stream, KOKORO_IO_ERROR, &wait_failure);
            stream->cancelled = true;
            pthread_mutex_unlock(&stream->mutex);
            kokoro_stream_cancel(stream);
            pthread_mutex_lock(&stream->mutex);
        }
    }
    KokoroStatus status = stream->status;
    KokoroError stream_error = stream->error;
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
    if (join_error != 0 && status == KOKORO_OK) {
        status = KOKORO_IO_ERROR;
        kokoro_error_set(&stream_error, status, "cannot join stream producer: %s",
                         strerror(join_error));
    }
    if (error != NULL) {
        *error = stream_error;
    }
    return status;
}

void kokoro_stream_free(KokoroStream *stream) {
    if (stream == NULL) {
        return;
    }
    kokoro_stream_cancel(stream);
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

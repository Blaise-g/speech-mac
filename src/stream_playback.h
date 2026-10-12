#ifndef KOKORO_STREAM_PLAYBACK_H
#define KOKORO_STREAM_PLAYBACK_H

#include "kokoro.h"

#include <stddef.h>

#define KOKORO_STREAM_BUFFER_FRAMES 4096U
#define KOKORO_STREAM_BUFFER_COUNT 3U

/* Playback owns every submitted copy. Calls may block for buffer reuse. */
typedef struct {
    KokoroStatus (*open)(void *context, void **playback, KokoroError *error);
    KokoroStatus (*submit)(void *playback, const float *samples, size_t sample_count,
                           KokoroError *error);
    KokoroStatus (*start)(void *playback, KokoroError *error);
    KokoroStatus (*drain)(void *playback, KokoroError *error);
    /* Wake a blocked submit/drain without calling the platform audio API. */
    void (*wake)(void *playback);
    /* Stop immediately. May be called by control flow while synthesis is active. */
    void (*abort)(void *playback);
    KokoroStatus (*close)(void *playback, KokoroError *error);
} KokoroStreamPlaybackOps;

#endif

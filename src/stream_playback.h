#ifndef SPEECH_STREAM_PLAYBACK_H
#define SPEECH_STREAM_PLAYBACK_H

#include "speech_error.h"

#include <stddef.h>

#define SPEECH_STREAM_BUFFER_FRAMES 4096U
#define SPEECH_STREAM_BUFFER_COUNT 3U

/* Playback owns every submitted copy. Calls may block for buffer reuse. */
typedef struct {
    SpeechIOStatus (*open)(void *context, void **playback, SpeechError *error);
    SpeechIOStatus (*submit)(void *playback, const float *samples, size_t sample_count,
                             SpeechError *error);
    SpeechIOStatus (*start)(void *playback, SpeechError *error);
    SpeechIOStatus (*drain)(void *playback, SpeechError *error);
    /* Wake a blocked submit/drain without calling the platform audio API. */
    void (*wake)(void *playback);
    /* Stop immediately. May be called by control flow while synthesis is active. */
    void (*abort)(void *playback);
    SpeechIOStatus (*close)(void *playback, SpeechError *error);
} SpeechStreamPlaybackOps;

#endif

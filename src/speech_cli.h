#ifndef SPEECH_CLI_H
#define SPEECH_CLI_H
#include "stream_playback.h"
#include <signal.h>
#include <stdio.h>
typedef struct {
    const char *pack_path;
    const char *voices_directory;
    FILE *input;
    FILE *output;
    FILE *error;
    /* Called synchronously with a completed WAV. Path borrowed during call. */
    int (*play)(void *context, const char *path);
    void *play_context;
    const volatile sig_atomic_t *interrupted;
    /* Optional playback-only seam; NULL uses the native AudioQueue adapter. */
    const KokoroStreamPlaybackOps *stream_playback;
    void *stream_context;
} SpeechCliConfig;
int speech_cli_run(int argc, char **argv, const SpeechCliConfig *config);
#endif

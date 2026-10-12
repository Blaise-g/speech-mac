#ifndef KOKORO_CLI_H
#define KOKORO_CLI_H

#include <stdint.h>
#include <stdio.h>

typedef struct {
    const char *model_pack_path;
    const char *voices_directory;
    const char *player_path;
    const char *espeak_path;
    uint64_t random_seed;
    FILE *stdin_stream;
    FILE *stdout_stream;
    FILE *stderr_stream;
} KokoroCliConfig;

int kokoro_cli_run(int argc, char **argv, const KokoroCliConfig *config);

#endif

#define KOKORO_STREAM_BENCHMARK_TIMING
#define KOKORO_STREAM_SILENT_OUTPUT
#define kokoro_cli_run stream_driver_test_cli
#define main kokoro_stream_driver_entry
#include "../benchmarks/stream_driver.c"
#undef main
#undef kokoro_cli_run

#include <stdio.h>
#include <string.h>

static int saw_fail_closed_nonstream_call;

int stream_driver_test_cli(int argc, char **argv, const KokoroCliConfig *config) {
    int is_streaming = 0;
    for (int index = 0; index < argc; index++) {
        if (strcmp(argv[index], "--stream") == 0) {
            is_streaming = 1;
        }
    }
    if (!is_streaming && config != NULL && config->player_path != NULL &&
        strcmp(config->player_path, "/usr/bin/false") == 0) {
        saw_fail_closed_nonstream_call = 1;
        return 1;
    }
    return 0;
}

int main(void) {
    char *args[] = {"stream-benchmark", "say", "hello"};
    int status = kokoro_stream_driver_entry(3, args);
    if (status != 1 || !saw_fail_closed_nonstream_call) {
        fputs("stream benchmark did not fail closed for completed-WAV playback\n", stderr);
        return 1;
    }
    puts("stream benchmark passes the disabled player to nonstream CLI calls");
    return 0;
}

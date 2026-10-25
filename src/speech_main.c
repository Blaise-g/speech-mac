#define _POSIX_C_SOURCE 200809L
#include "speech_cli.h"
#include <signal.h>
#include <string.h>
static volatile sig_atomic_t interrupted;
static void interrupt_request(int signal_number) {
    (void)signal_number;
    interrupted = 1;
}
int main(int argc, char **argv) {
    int streaming = 0;
    for (int i = 2; i < argc && strcmp(argv[i], "--"); ++i) {
        if (!strcmp(argv[i], "--stream"))
            streaming = 1;
        else if (argv[i][0] == '-' && strcmp(argv[i], "--stdin"))
            ++i;
    }
    if (!streaming)
        return speech_cli_run(argc, argv, NULL);
    struct sigaction action;
    memset(&action, 0, sizeof(action));
    action.sa_handler = interrupt_request;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGINT, &action, NULL) || sigaction(SIGTERM, &action, NULL))
        return 1;
    SpeechCliConfig config = {.interrupted = &interrupted};
    return speech_cli_run(argc, argv, &config);
}

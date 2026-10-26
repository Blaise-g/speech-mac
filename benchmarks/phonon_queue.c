#define _DARWIN_C_SOURCE
#include "speech_stream.h"
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
/* This binary is only useful compiled with SPEECH_STREAM_SILENT_OUTPUT:
 * the adapter sets this new queue's volume to zero before its first start. */
static void timeout_handler(int signal_number) {
    (void)signal_number;
    static const char message[] = "QUEUE_TIMEOUT=1\n";
    (void)write(STDERR_FILENO, message, sizeof(message) - 1);
    _exit(124);
}
int main(int argc, char **argv) {
#ifndef SPEECH_STREAM_SILENT_OUTPUT
    fputs("Rebuild with SPEECH_STREAM_SILENT_OUTPUT; refusing device access.\n", stderr);
    return 2;
#endif
    if (argc != 3)
        return 2;
    signal(SIGALRM, timeout_handler);
    alarm(60);
    SpeechEngine *engine = NULL;
    PhononVoice *voice = NULL;
    if (speech_engine_create(argv[1], &engine) != SPEECH_OK || !phonon_voice_open(argv[2], &voice))
        return 3;
    SpeechOptions options = speech_options_default();
    SpeechStream *stream = NULL;
    SpeechError error = {0};
    SpeechIOStatus status = speech_stream_start(
        engine, voice, "Hello there. I am ready to help you today.",
        strlen("Hello there. I am ready to help you today."), &options, &stream, &error);
    if (status == SPEECH_IO_OK)
        status = speech_stream_wait(stream, NULL, &error);
    fprintf(stderr, "muted AudioQueue status=%d detail=%s\n", status, error.message);
    if (status != SPEECH_IO_OK)
        return 4;
    SpeechResult result;
    SpeechStreamMetrics metrics;
    speech_stream_result(stream, &result, &metrics);
    if (!metrics.started || metrics.first_pcm < metrics.synthesis_begin ||
        metrics.first_submission < metrics.first_pcm ||
        metrics.started < metrics.first_submission || metrics.synthesis_end < metrics.started ||
        metrics.drained < metrics.synthesis_end ||
        metrics.submitted_samples != result.samples_generated ||
        metrics.reclaimed_buffers != result.frames_generated ||
        metrics.first_reclaimed < metrics.first_submission)
        return 5;
    printf("{\"consumer\":\"queue_volume_zero_AudioQueue\",\"first_pcm_s\":%.9f,\"first_submission_"
           "s\":%.9f,\"running_observed_s\":%.9f,\"first_reclaimed_s\":%.9f,\"synthesis_end_s\":%."
           "9f,\"drain_s\":%.9f,\"samples\":%zu,\"frames\":%zu,\"reclaimed_buffers\":%zu,"
           "\"acoustic_onset_s\":null}\n",
           metrics.first_pcm - metrics.synthesis_begin,
           metrics.first_submission - metrics.synthesis_begin,
           metrics.started - metrics.synthesis_begin,
           metrics.first_reclaimed - metrics.synthesis_begin,
           metrics.synthesis_end - metrics.synthesis_begin,
           metrics.drained - metrics.synthesis_begin, result.samples_generated,
           result.frames_generated, metrics.reclaimed_buffers);
    speech_stream_free(stream);
    phonon_voice_close(voice);
    speech_engine_destroy(engine);
    alarm(0);
    puts("QUEUE_TIMING_EXIT=0");
    return 0;
}

#define _DARWIN_C_SOURCE
#include "sha256.h"
#include "speech.h"
#include "speech_profile.h"
#include "wav_writer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

typedef struct {
    const char *name;
    double total, begin;
    size_t calls;
} Stage;
static Stage stages[] = {{"prefill", 0, 0, 0},
                         {"flow_head", 0, 0, 0},
                         {"mimi_pcm", 0, 0, 0},
                         {"incremental_backbone", 0, 0, 0},
                         {"linear_expand_bf16", 0, 0, 0},
                         {"linear_expand_q8", 0, 0, 0},
                         {"linear_sgemm", 0, 0, 0}};
static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
void speech_profile_begin(const char *name) {
    for (size_t i = 0; i < sizeof(stages) / sizeof(stages[0]); i++)
        if (!strcmp(name, stages[i].name))
            stages[i].begin = now();
}
void speech_profile_end(const char *name) {
    for (size_t i = 0; i < sizeof(stages) / sizeof(stages[0]); i++)
        if (!strcmp(name, stages[i].name)) {
            stages[i].total += now() - stages[i].begin;
            stages[i].calls++;
        }
}
typedef struct {
    double begin, first;
    float *pcm;
    size_t count, capacity;
} Sink;
static int consume(void *context, const float *pcm, size_t count, const SpeechResult *result) {
    (void)result;
    Sink *s = context;
    if (!s->count)
        s->first = now() - s->begin;
    if (count > s->capacity - s->count)
        return 2;
    memcpy(s->pcm + s->count, pcm, count * sizeof(float));
    s->count += count;
    return 0;
}
static double cpu(const struct rusage *r) {
    return r->ru_utime.tv_sec + r->ru_utime.tv_usec * 1e-6 + r->ru_stime.tv_sec +
           r->ru_stime.tv_usec * 1e-6;
}
static int release(char *text, SpeechEngine *engine, PhononVoice *voice, float *pcm, int code) {
    free(pcm);
    free(text);
    phonon_voice_close(voice);
    speech_engine_destroy(engine);
    return code;
}
int main(int argc, char **argv) {
    if (argc != 6)
        return 2;
    int repeats = atoi(argv[4]);
    if (repeats < 1 || repeats > 20)
        return 2;
    FILE *f = fopen(argv[3], "rb");
    if (!f)
        return 2;
    if (fseek(f, 0, SEEK_END)) {
        fclose(f);
        return 2;
    }
    long length = ftell(f);
    if (length <= 0 || length > SPEECH_MAX_INPUT_BYTES || fseek(f, 0, SEEK_SET)) {
        fclose(f);
        return 2;
    }
    char *text = calloc((size_t)length + 1, 1);
    if (!text || fread(text, 1, (size_t)length, f) != (size_t)length) {
        free(text);
        fclose(f);
        return 2;
    }
    fclose(f);
    double begin = now();
    SpeechEngine *engine = NULL;
    PhononVoice *voice = NULL;
    Sink sink = {0};
    if (speech_engine_create(argv[1], &engine) != SPEECH_OK)
        return release(text, engine, voice, sink.pcm, 3);
    double engine_s = now() - begin;
    begin = now();
    if (!phonon_voice_open(argv[2], &voice))
        return release(text, engine, voice, sink.pcm, 3);
    double voice_s = now() - begin;
    size_t capacity = SPEECH_MAX_REQUEST_FRAMES * SPEECH_FRAME_SAMPLES;
    sink.capacity = capacity;
    sink.pcm = malloc(capacity * sizeof(float));
    if (!sink.pcm)
        return release(text, engine, voice, sink.pcm, 3);
    for (int run = 0; run < repeats; run++) {
        for (size_t i = 0; i < sizeof(stages) / sizeof(stages[0]); i++) {
            stages[i].total = 0;
            stages[i].calls = 0;
        }
        struct rusage before, after;
        getrusage(RUSAGE_SELF, &before);
        sink.count = 0;
        sink.begin = now();
        SpeechResult result;
        SpeechOptions options = speech_options_default();
        SpeechStatus status = speech_synthesize(engine, voice, text, (size_t)length, &options,
                                                consume, &sink, &result);
        double synthesis = now() - sink.begin;
        getrusage(RUSAGE_SELF, &after);
        if (status != SPEECH_OK) {
            fprintf(stderr, "synthesis status=%d\n", status);
            return release(text, engine, voice, sink.pcm, 4);
        }
        char pcm_hash[65];
        if (!speech_sha256_hex((const unsigned char *)sink.pcm, sink.count * sizeof(float),
                               pcm_hash))
            return release(text, engine, voice, sink.pcm, 6);
        printf("{\"pcm_f32_sha256\":\"%s\",\"run\":%d,\"engine_prepare_s\":%.9f,\"voice_open_s\":%."
               "9f,\"first_pcm_s\":%.9f,"
               "\"synthesis_s\":%.9f,\"cpu_s\":%.9f,\"rss_high_water_bytes\":%ld,\"samples\":%zu,"
               "\"frames\":%zu,\"chunks\":%zu,\"stages\":{",
               pcm_hash, run, engine_s, voice_s, sink.first, synthesis, cpu(&after) - cpu(&before),
               after.ru_maxrss, sink.count, result.frames_generated, result.chunks_completed);
        for (size_t i = 0; i < sizeof(stages) / sizeof(stages[0]); i++)
            printf("%s\"%s\":{\"seconds\":%.9f,\"calls\":%zu}", i ? "," : "", stages[i].name,
                   stages[i].total, stages[i].calls);
        puts("}}");
        fflush(stdout);
        if (run == repeats - 1 && strcmp(argv[5], "-")) {
            SpeechError error;
            if (speech_wav_write_pcm16(argv[5], sink.pcm, sink.count, 24000, &error) !=
                SPEECH_IO_OK)
                return release(text, engine, voice, sink.pcm, 5);
        }
    }
    return release(text, engine, voice, sink.pcm, 0);
}

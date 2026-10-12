#include "english_frontend.h"
#include "ipa_tokenizer.h"
#include "synthesis.h"
#include "wav_writer.h"

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    float *samples;
    size_t count;
    size_t capacity;
} SampleBuffer;

typedef struct {
    double context_encoder;
    double duration_prediction;
    double pitch_noise_prediction;
    double text_encoder;
    double decoder_conditioning;
    double neural_generator;
    double istft;
} StageTiming;

typedef struct {
    double frontend_seconds;
    double synthesis_seconds;
    double request_seconds;
    size_t sample_count;
    StageTiming stages;
} RequestTiming;

static StageTiming current_stage_timing;

void kokoro_benchmark_stage_record(const char *name, double seconds) {
    if (strcmp(name, "context_encoder") == 0) {
        current_stage_timing.context_encoder += seconds;
    } else if (strcmp(name, "duration_prediction") == 0) {
        current_stage_timing.duration_prediction += seconds;
    } else if (strcmp(name, "pitch_noise_prediction") == 0) {
        current_stage_timing.pitch_noise_prediction += seconds;
    } else if (strcmp(name, "text_encoder") == 0) {
        current_stage_timing.text_encoder += seconds;
    } else if (strcmp(name, "decoder_conditioning") == 0) {
        current_stage_timing.decoder_conditioning += seconds;
    } else if (strcmp(name, "neural_generator") == 0) {
        current_stage_timing.neural_generator += seconds;
    } else if (strcmp(name, "istft") == 0) {
        current_stage_timing.istft += seconds;
    }
}

static void reset_stage_timings(void) { current_stage_timing = (StageTiming){0}; }

static double stage_total(const StageTiming *stages) {
    return stages->context_encoder + stages->duration_prediction + stages->pitch_noise_prediction +
           stages->text_encoder + stages->decoder_conditioning + stages->neural_generator +
           stages->istft;
}

static void print_stage_json(const StageTiming *stages, double synthesis_seconds) {
    double overhead = synthesis_seconds - stage_total(stages);
    if (overhead < 0.0) {
        overhead = 0.0;
    }
    printf("\"stage_seconds\":{\"context_encoder\":%.9f,\"duration_prediction\":%.9f,"
           "\"pitch_noise_prediction\":%.9f,\"text_encoder\":%.9f,"
           "\"decoder_conditioning\":%.9f,\"neural_generator\":%.9f,\"istft\":%.9f,"
           "\"other_synthesis_overhead\":%.9f}",
           stages->context_encoder, stages->duration_prediction, stages->pitch_noise_prediction,
           stages->text_encoder, stages->decoder_conditioning, stages->neural_generator,
           stages->istft, overhead);
}

static double monotonic_seconds(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        perror("clock_gettime");
        exit(2);
    }
    return (double)value.tv_sec + (double)value.tv_nsec / 1000000000.0;
}

static bool append_samples(SampleBuffer *buffer, const float *samples, size_t count) {
    if (samples == NULL || count == 0 || count > SIZE_MAX - buffer->count) {
        return false;
    }
    size_t required = buffer->count + count;
    if (required > buffer->capacity) {
        size_t capacity = buffer->capacity == 0 ? count : buffer->capacity;
        while (capacity < required) {
            if (capacity > SIZE_MAX / 2) {
                capacity = required;
                break;
            }
            capacity *= 2;
        }
        if (capacity > SIZE_MAX / sizeof(*buffer->samples)) {
            return false;
        }
        float *grown = realloc(buffer->samples, capacity * sizeof(*grown));
        if (grown == NULL) {
            return false;
        }
        buffer->samples = grown;
        buffer->capacity = capacity;
    }
    memcpy(buffer->samples + buffer->count, samples, count * sizeof(*samples));
    buffer->count = required;
    return true;
}

static bool prepare_chunks(const char *mode, const char *text, KokoroIpaChunks *chunks,
                           double *frontend_seconds, KokoroError *error) {
    char *phonemes = NULL;
    size_t phoneme_length = 0;
    double started = monotonic_seconds();
    KokoroStatus status = KOKORO_OK;
    bool sentence_lines = strcmp(mode, "english") == 0;
    if (sentence_lines) {
        status = kokoro_english_frontend("espeak-ng", text, strlen(text), &phonemes,
                                         &phoneme_length, error);
        if (status != KOKORO_OK) {
            *frontend_seconds = monotonic_seconds() - started;
            return false;
        }
    } else {
        phonemes = (char *)text;
        phoneme_length = strlen(text);
    }
    (void)phoneme_length;
    status = kokoro_ipa_encode_chunks(phonemes, sentence_lines, chunks, error);
    if (sentence_lines) {
        free(phonemes);
    }
    *frontend_seconds = monotonic_seconds() - started;
    return status == KOKORO_OK;
}

static bool synthesize_chunks(KokoroEngine *engine, KokoroIpaChunks *chunks, float speed,
                              uint64_t seed, SampleBuffer *combined, double *synthesis_seconds,
                              KokoroError *error) {
    double started = monotonic_seconds();
    for (size_t index = 0; index < chunks->count; index++) {
        KokoroSynthesisOptions options = {
            .voice_style_index = chunks->items[index].token_count - 3,
            .speed = speed,
            .random_seed = seed + (uint64_t)index,
        };
        KokoroSynthesisOutput output = {0};
        KokoroStatus status =
            kokoro_engine_synthesize(engine, chunks->items[index].token_ids,
                                     chunks->items[index].token_count, &options, &output, error);
        if (status != KOKORO_OK || !append_samples(combined, output.samples, output.sample_count)) {
            if (status == KOKORO_OK) {
                (void)snprintf(error->message, sizeof(error->message),
                               "cannot append synthesized samples");
            }
            kokoro_synthesis_output_free(&output);
            *synthesis_seconds = monotonic_seconds() - started;
            return false;
        }
        kokoro_synthesis_output_free(&output);
    }
    *synthesis_seconds = monotonic_seconds() - started;
    return combined->count > 0;
}

static bool run_request(KokoroEngine *engine, const char *mode, const char *text, float speed,
                        uint64_t seed, SampleBuffer *combined, RequestTiming *timing,
                        KokoroError *error) {
    double request_started = monotonic_seconds();
    KokoroIpaChunks chunks = {0};
    if (!prepare_chunks(mode, text, &chunks, &timing->frontend_seconds, error)) {
        kokoro_ipa_chunks_free(&chunks);
        return false;
    }
    bool success = synthesize_chunks(engine, &chunks, speed, seed, combined,
                                     &timing->synthesis_seconds, error);
    timing->stages = current_stage_timing;
    kokoro_ipa_chunks_free(&chunks);
    timing->sample_count = combined->count;
    timing->request_seconds = monotonic_seconds() - request_started;
    return success;
}

static bool write_wav(const char *path, const SampleBuffer *samples, double *write_seconds,
                      KokoroError *error) {
    double started = monotonic_seconds();
    KokoroStatus status =
        kokoro_wav_write_pcm16(path, samples->samples, samples->count, 24000, error);
    *write_seconds = monotonic_seconds() - started;
    return status == KOKORO_OK;
}

static void print_request(const char *kind, size_t index, const RequestTiming *timing,
                          double write_seconds) {
    double audio_seconds = (double)timing->sample_count / 24000.0;
    printf("{\"kind\":\"%s\",\"index\":%zu,\"frontend_seconds\":%.9f,"
           "\"synthesis_seconds\":%.9f,\"request_to_waveform_seconds\":%.9f,"
           "\"wav_write_completion_seconds\":%.9f,\"sample_count\":%zu,"
           "\"audio_seconds\":%.9f,",
           kind, index, timing->frontend_seconds, timing->synthesis_seconds,
           timing->request_seconds, write_seconds, timing->sample_count, audio_seconds);
    print_stage_json(&timing->stages, timing->synthesis_seconds);
    puts("}");
    (void)fflush(stdout);
}

static bool parse_positive_integer(const char *text, size_t *value) {
    if (text == NULL || text[0] == '\0') {
        return false;
    }
    errno = 0;
    char *end = NULL;
    unsigned long long parsed = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed == 0 || parsed > SIZE_MAX) {
        return false;
    }
    *value = (size_t)parsed;
    return true;
}

static bool parse_speed(const char *text, float *speed) {
    if (text == NULL || text[0] == '\0') {
        return false;
    }
    errno = 0;
    char *end = NULL;
    float value = strtof(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !isfinite(value) || value < 0.5f ||
        value > 2.0f) {
        return false;
    }
    *speed = value;
    return true;
}

static int probe_pack(const char *path) {
    double started = monotonic_seconds();
    KokoroPack *pack = NULL;
    KokoroError error = {0};
    KokoroStatus status = kokoro_pack_open(path, &pack, &error);
    double elapsed = monotonic_seconds() - started;
    kokoro_pack_close(pack);
    if (status != KOKORO_OK) {
        fprintf(stderr, "kokoro-benchmark: model pack open failed: %s\n", error.message);
        return 1;
    }
    printf("{\"kind\":\"pack_open\",\"seconds\":%.9f}\n", elapsed);
    return 0;
}

int main(int argc, char **argv) {
    double process_started = monotonic_seconds();
    if (argc == 3 && strcmp(argv[1], "--probe-pack") == 0) {
        return probe_pack(argv[2]);
    }

    const char *mode = NULL;
    const char *text = NULL;
    const char *model_path = "models/kokoro-v1_0-af_heart.kpack";
    const char *voice = "af_heart";
    const char *voice_path = NULL;
    const char *wav_path = NULL;
    float speed = 1.0f;
    uint64_t seed = 104729;
    size_t warmups = 1;
    size_t repetitions = 5;
    for (int index = 1; index < argc; index++) {
        const char *option = argv[index];
        if (index + 1 >= argc) {
            fprintf(stderr, "kokoro-benchmark: option requires a value: %s\n", option);
            return 2;
        }
        const char *value = argv[++index];
        if (strcmp(option, "--mode") == 0) {
            mode = value;
        } else if (strcmp(option, "--text") == 0) {
            text = value;
        } else if (strcmp(option, "--model") == 0) {
            model_path = value;
        } else if (strcmp(option, "--voice") == 0) {
            voice = value;
        } else if (strcmp(option, "--voice-path") == 0) {
            voice_path = value;
        } else if (strcmp(option, "--wav") == 0) {
            wav_path = value;
        } else if (strcmp(option, "--speed") == 0) {
            if (!parse_speed(value, &speed)) {
                fprintf(stderr, "kokoro-benchmark: invalid speed\n");
                return 2;
            }
        } else if (strcmp(option, "--seed") == 0) {
            size_t parsed = 0;
            if (!parse_positive_integer(value, &parsed)) {
                fprintf(stderr, "kokoro-benchmark: invalid seed\n");
                return 2;
            }
            seed = (uint64_t)parsed;
        } else if (strcmp(option, "--warmups") == 0) {
            if (!parse_positive_integer(value, &warmups)) {
                fprintf(stderr, "kokoro-benchmark: invalid warmup count\n");
                return 2;
            }
        } else if (strcmp(option, "--repetitions") == 0) {
            if (!parse_positive_integer(value, &repetitions)) {
                fprintf(stderr, "kokoro-benchmark: invalid repetition count\n");
                return 2;
            }
        } else {
            fprintf(stderr, "kokoro-benchmark: unknown option: %s\n", option);
            return 2;
        }
    }
    if ((mode == NULL || (strcmp(mode, "english") != 0 && strcmp(mode, "ipa") != 0)) ||
        text == NULL || text[0] == '\0' || wav_path == NULL ||
        (strcmp(voice, "af_heart") != 0 && strcmp(voice, "am_adam") != 0)) {
        fprintf(stderr, "kokoro-benchmark: mode, text, voice, and WAV path are required\n");
        return 2;
    }

    KokoroError error = {0};
    KokoroIpaChunks initial_chunks = {0};
    double first_frontend_seconds = 0.0;
    if (!prepare_chunks(mode, text, &initial_chunks, &first_frontend_seconds, &error)) {
        fprintf(stderr, "kokoro-benchmark: frontend failed: %s\n", error.message);
        return 1;
    }

    KokoroEngine *engine = NULL;
    double engine_started = monotonic_seconds();
    KokoroStatus status = kokoro_engine_create(model_path, voice, voice_path, &engine, &error);
    double engine_create_seconds = monotonic_seconds() - engine_started;
    if (status != KOKORO_OK) {
        fprintf(stderr, "kokoro-benchmark: engine creation failed: %s\n", error.message);
        kokoro_ipa_chunks_free(&initial_chunks);
        return 1;
    }

    SampleBuffer cold_samples = {0};
    double cold_synthesis_seconds = 0.0;
    reset_stage_timings();
    if (!synthesize_chunks(engine, &initial_chunks, speed, seed, &cold_samples,
                           &cold_synthesis_seconds, &error)) {
        fprintf(stderr, "kokoro-benchmark: first synthesis failed: %s\n", error.message);
        kokoro_ipa_chunks_free(&initial_chunks);
        kokoro_engine_free(engine);
        free(cold_samples.samples);
        return 1;
    }
    kokoro_ipa_chunks_free(&initial_chunks);
    double cold_audio_seconds = (double)cold_samples.count / 24000.0;
    printf("{\"kind\":\"cold_audio\",\"frontend_seconds\":%.9f,"
           "\"engine_create_seconds\":%.9f,\"synthesis_seconds\":%.9f,"
           "\"driver_start_to_waveform_seconds\":%.9f,\"sample_count\":%zu,"
           "\"audio_seconds\":%.9f,",
           first_frontend_seconds, engine_create_seconds, cold_synthesis_seconds,
           monotonic_seconds() - process_started, cold_samples.count, cold_audio_seconds);
    print_stage_json(&current_stage_timing, cold_synthesis_seconds);
    puts("}");
    (void)fflush(stdout);

    RequestTiming cold_timing = {
        .frontend_seconds = first_frontend_seconds,
        .synthesis_seconds = cold_synthesis_seconds,
        .request_seconds = first_frontend_seconds + cold_synthesis_seconds,
        .sample_count = cold_samples.count,
        .stages = current_stage_timing,
    };
    double cold_wav_seconds = 0.0;
    if (!write_wav(wav_path, &cold_samples, &cold_wav_seconds, &error)) {
        fprintf(stderr, "kokoro-benchmark: WAV write failed: %s\n", error.message);
        kokoro_engine_free(engine);
        free(cold_samples.samples);
        return 1;
    }
    print_request("cold_wav", 0, &cold_timing, cold_wav_seconds);
    free(cold_samples.samples);

    for (size_t index = 0; index < warmups; index++) {
        SampleBuffer warmup = {0};
        RequestTiming timing = {0};
        reset_stage_timings();
        if (!run_request(engine, mode, text, speed, seed, &warmup, &timing, &error)) {
            fprintf(stderr, "kokoro-benchmark: warmup failed: %s\n", error.message);
            kokoro_engine_free(engine);
            free(warmup.samples);
            return 1;
        }
        free(warmup.samples);
    }

    for (size_t index = 0; index < repetitions; index++) {
        SampleBuffer warm = {0};
        RequestTiming timing = {0};
        reset_stage_timings();
        if (!run_request(engine, mode, text, speed, seed, &warm, &timing, &error)) {
            fprintf(stderr, "kokoro-benchmark: warm synthesis failed: %s\n", error.message);
            kokoro_engine_free(engine);
            free(warm.samples);
            return 1;
        }
        double write_seconds = 0.0;
        if (!write_wav(wav_path, &warm, &write_seconds, &error)) {
            fprintf(stderr, "kokoro-benchmark: WAV write failed: %s\n", error.message);
            kokoro_engine_free(engine);
            free(warm.samples);
            return 1;
        }
        print_request("warm_sample", index, &timing, write_seconds);
        free(warm.samples);
    }

    kokoro_engine_free(engine);
    return 0;
}

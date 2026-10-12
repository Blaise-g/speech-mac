#include "synthesis.h"
#include "wav_writer.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MODEL_PACK_PATH "models/kokoro-v1_0-af_heart.kpack"
#define TOKEN_COUNT 8U
#define DURATION_FRAME_COUNT 55U
#define FIXTURE_SAMPLE_COUNT 33000U
#define FIXTURE_SOURCE_FRAME_COUNT 6601U
#define SPECTRAL_VALUE_COUNT (KOKORO_NEURAL_GENERATOR_SPECTRAL_BINS * FIXTURE_SOURCE_FRAME_COUNT)
#define EXCITATION_SAMPLE_COUNT FIXTURE_SAMPLE_COUNT
#define WAVEFORM_NORMALIZED_RMSE_MAX 1.0e-3
#define WAVEFORM_MAX_ABS_ERROR 2.0e-3

static int fail(const char *message) {
    fprintf(stderr, "test_inference: %s\n", message);
    return 0;
}

static int expect_status(KokoroStatus actual, KokoroStatus expected, const KokoroError *error,
                         const char *operation) {
    if (actual == expected) {
        return 1;
    }
    fprintf(stderr, "test_inference: %s returned %s (%s), expected %s\n", operation,
            kokoro_status_name(actual), error->message, kokoro_status_name(expected));
    return 0;
}

static int read_fixture(const char *path, void *output, size_t byte_count) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "test_inference: cannot open fixture %s\n", path);
        return 0;
    }
    int valid = fread(output, 1, byte_count, file) == byte_count && fgetc(file) == EOF;
    fclose(file);
    if (!valid) {
        fprintf(stderr, "test_inference: fixture has the wrong size: %s\n", path);
    }
    return valid;
}

static uint32_t little_u32(const unsigned char *bytes) {
    return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8U) | ((uint32_t)bytes[2] << 16U) |
           ((uint32_t)bytes[3] << 24U);
}

static int verify_wav(const char *path, size_t sample_count) {
    unsigned char header[44];
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "test_inference: cannot open generated WAV %s\n", path);
        return 0;
    }
    int valid = fread(header, 1, sizeof(header), file) == sizeof(header) &&
                memcmp(header, "RIFF", 4) == 0 && memcmp(header + 8, "WAVEfmt ", 8) == 0 &&
                little_u32(header + 4) == 36U + sample_count * 2U &&
                little_u32(header + 24) == KOKORO_ISTFT_SAMPLE_RATE_HZ && header[22] == 1 &&
                header[23] == 0 && header[34] == 16 && header[35] == 0 &&
                memcmp(header + 36, "data", 4) == 0 && little_u32(header + 40) == sample_count * 2U;
    fclose(file);
    if (!valid) {
        fprintf(stderr,
                "test_inference: generated WAV has incorrect PCM16 mono 24 kHz metadata: %s\n",
                path);
    }
    return valid;
}

static int synthesis_output_is_empty(const KokoroSynthesisOutput *output) {
    return output->token_count == 0 && output->frame_count == 0 && output->sample_count == 0 &&
           output->durations == NULL && output->samples == NULL;
}

static int compare_reference_waveform(const float *actual, const float *expected, size_t count) {
    double squared_error = 0.0;
    double squared_reference = 0.0;
    double maximum_error = 0.0;
    for (size_t sample = 0; sample < count; sample++) {
        double difference = (double)actual[sample] - expected[sample];
        double absolute_error = fabs(difference);
        if (!isfinite(actual[sample])) {
            return fail("controlled waveform contains a non-finite sample");
        }
        if (absolute_error > maximum_error) {
            maximum_error = absolute_error;
        }
        squared_error += difference * difference;
        squared_reference += (double)expected[sample] * expected[sample];
    }
    double normalized_rmse = sqrt(squared_error / count) / sqrt(squared_reference / count);
    if (!isfinite(normalized_rmse) || normalized_rmse > WAVEFORM_NORMALIZED_RMSE_MAX ||
        maximum_error > WAVEFORM_MAX_ABS_ERROR) {
        fprintf(stderr,
                "test_inference: assembled controlled waveform misses the fixed budget "
                "(normalized RMSE %.9g, max abs %.9g; limits %.1g and %.1g)\n",
                normalized_rmse, maximum_error, WAVEFORM_NORMALIZED_RMSE_MAX,
                WAVEFORM_MAX_ABS_ERROR);
        return 0;
    }
    printf("assembled controlled waveform passed (normalized RMSE %.9g, max abs %.9g)\n",
           normalized_rmse, maximum_error);
    return 1;
}

static int test_invalid_inputs(void) {
    KokoroError error = {0};
    KokoroSynthesisOptions options = {.voice_style_index = 5, .speed = 1.0f};
    KokoroSynthesisOutput output = {0};
    KokoroStatus status = kokoro_synthesize(NULL, NULL, 0, &options, &output, &error);
    if (!expect_status(status, KOKORO_INVALID_ARGUMENT, &error, "reject missing synthesis input") ||
        !synthesis_output_is_empty(&output)) {
        return fail("invalid synthesis request changed the output");
    }

    const char *invalid_path = ".build/test-inference-invalid.wav";
    (void)remove(invalid_path);
    float invalid_samples[2] = {0.25f, NAN};
    status = kokoro_wav_write_pcm16(invalid_path, invalid_samples, 2, 24000, &error);
    if (!expect_status(status, KOKORO_NUMERIC_ERROR, &error, "reject non-finite WAV sample")) {
        return 0;
    }
    FILE *unexpected = fopen(invalid_path, "rb");
    if (unexpected != NULL) {
        fclose(unexpected);
        (void)remove(invalid_path);
        return fail("invalid WAV input left an output file");
    }
    const char *valid_path = ".build/test-inference-valid.wav";
    (void)remove(valid_path);
    const float valid_samples[4] = {0.0f, 0.5f, -1.0f, 1.0f};
    status = kokoro_wav_write_pcm16(valid_path, valid_samples, 4, 24000, &error);
    int valid = expect_status(status, KOKORO_OK, &error, "write finite PCM16 samples");
    if (valid && (error.code != KOKORO_OK || error.message[0] != '\0')) {
        valid = fail("successful WAV write left a stale error");
    }
    if (valid) {
        valid = verify_wav(valid_path, 4);
    }
    (void)remove(valid_path);
    return valid;
}

static int test_synthesis_errors(const KokoroPack *pack) {
    const int64_t token_ids[TOKEN_COUNT] = {0, 50, 83, 156, 54, 57, 135, 0};
    const int64_t invalid_ids[2] = {0, INT64_MAX};
    const int64_t placeholder_id = 0;
    KokoroSynthesisOptions options = {.voice_style_index = 5, .speed = 1.0f};
    KokoroSynthesisOutput output = {0};
    KokoroError error = {0};

    KokoroStatus status =
        kokoro_synthesize(pack, &placeholder_id, SIZE_MAX, &options, &output, &error);
    if (!expect_status(status, KOKORO_CONTEXT_TOO_LONG, &error,
                       "reject overlong synthesis input") ||
        !synthesis_output_is_empty(&output)) {
        return fail("overlong synthesis request returned partial output");
    }
    status = kokoro_synthesize(pack, invalid_ids, 2, &options, &output, &error);
    if (!expect_status(status, KOKORO_INVALID_TOKEN_ID, &error, "reject invalid phoneme ID") ||
        !synthesis_output_is_empty(&output)) {
        return fail("invalid-token synthesis request returned partial output");
    }
    options.voice_style_index = SIZE_MAX;
    status = kokoro_synthesize(pack, token_ids, TOKEN_COUNT, &options, &output, &error);
    if (!expect_status(status, KOKORO_INVALID_ARGUMENT, &error, "reject invalid voice style row") ||
        !synthesis_output_is_empty(&output)) {
        return fail("invalid-voice synthesis request returned partial output");
    }
    return 1;
}

int main(void) {
    if (!test_invalid_inputs()) {
        return 1;
    }
    FILE *model_file = fopen(MODEL_PACK_PATH, "rb");
    if (model_file == NULL) {
        puts("SKIP assembled inference: convert the pinned model pack to run the weight-backed "
             "test");
        return 0;
    }
    fclose(model_file);

    const int64_t token_ids[TOKEN_COUNT] = {0, 50, 83, 156, 54, 57, 135, 0};
    int64_t expected_durations[TOKEN_COUNT];
    float *expected_waveform = malloc(FIXTURE_SAMPLE_COUNT * sizeof(float));
    float *excitation_harmonic = malloc(EXCITATION_SAMPLE_COUNT * sizeof(float));
    float *excitation_noise = malloc(EXCITATION_SAMPLE_COUNT * sizeof(float));
    float *excitation_voiced = malloc(EXCITATION_SAMPLE_COUNT * sizeof(float));
    float *source_magnitude = malloc(SPECTRAL_VALUE_COUNT * sizeof(float));
    float *source_phase = malloc(SPECTRAL_VALUE_COUNT * sizeof(float));
    if (expected_waveform == NULL || excitation_harmonic == NULL || excitation_noise == NULL ||
        excitation_voiced == NULL || source_magnitude == NULL || source_phase == NULL) {
        free(expected_waveform);
        free(excitation_harmonic);
        free(excitation_noise);
        free(excitation_voiced);
        free(source_magnitude);
        free(source_phase);
        return fail("cannot allocate reference fixtures");
    }
    int valid = read_fixture("reference/fixtures/durations.i64le", expected_durations,
                             sizeof(expected_durations)) &&
                read_fixture("reference/fixtures/waveform.f32le", expected_waveform,
                             FIXTURE_SAMPLE_COUNT * sizeof(float)) &&
                read_fixture("reference/fixtures/excitation_harmonic.f32le", excitation_harmonic,
                             EXCITATION_SAMPLE_COUNT * sizeof(float)) &&
                read_fixture("reference/fixtures/excitation_noise.f32le", excitation_noise,
                             EXCITATION_SAMPLE_COUNT * sizeof(float)) &&
                read_fixture("reference/fixtures/excitation_voiced.f32le", excitation_voiced,
                             EXCITATION_SAMPLE_COUNT * sizeof(float)) &&
                read_fixture("reference/fixtures/source_spectrogram_magnitude.f32le",
                             source_magnitude, SPECTRAL_VALUE_COUNT * sizeof(float)) &&
                read_fixture("reference/fixtures/source_spectrogram_phase.f32le", source_phase,
                             SPECTRAL_VALUE_COUNT * sizeof(float));
    if (!valid) {
        free(expected_waveform);
        free(excitation_harmonic);
        free(excitation_noise);
        free(excitation_voiced);
        free(source_magnitude);
        free(source_phase);
        return 1;
    }

    KokoroError error = {0};
    KokoroPack *pack = NULL;
    KokoroStatus status = kokoro_pack_open(MODEL_PACK_PATH, &pack, &error);
    if (!expect_status(status, KOKORO_OK, &error, "open pinned model pack")) {
        valid = 0;
        goto cleanup_fixtures;
    }

    if (!test_synthesis_errors(pack)) {
        kokoro_pack_close(pack);
        valid = 0;
        goto cleanup_fixtures;
    }

    KokoroNeuralGeneratorExcitation excitation = {
        .sample_count = EXCITATION_SAMPLE_COUNT,
        .harmonic = excitation_harmonic,
        .noise = excitation_noise,
        .voiced = excitation_voiced,
    };
    KokoroNeuralGeneratorSourceSpectrogram source_spectrogram = {
        .frame_count = FIXTURE_SOURCE_FRAME_COUNT,
        .magnitude = source_magnitude,
        .phase = source_phase,
    };
    KokoroSynthesisOptions controlled_options = {
        .voice_style_index = 5,
        .speed = 1.0f,
        .random_seed = 104729,
        .reference_excitation = &excitation,
        .reference_source_spectrogram = &source_spectrogram,
    };
    KokoroSynthesisOutput controlled = {0};
    status =
        kokoro_synthesize(pack, token_ids, TOKEN_COUNT, &controlled_options, &controlled, &error);
    valid =
        expect_status(status, KOKORO_OK, &error, "run controlled phoneme-to-waveform inference");
    if (valid) {
        if (controlled.token_count != TOKEN_COUNT || controlled.durations == NULL ||
            controlled.samples == NULL) {
            valid = fail("controlled synthesis returned incomplete output buffers");
        }
    }
    if (valid) {
        uint64_t expected_frame_count = 0;
        for (size_t token = 0; token < TOKEN_COUNT; token++) {
            if (expected_durations[token] <= 0 ||
                UINT64_MAX - expected_frame_count < (uint64_t)expected_durations[token] ||
                controlled.durations[token] != (uint64_t)expected_durations[token]) {
                valid =
                    fail("assembled inference durations differ from the exact reference values");
                break;
            }
            expected_frame_count += (uint64_t)expected_durations[token];
        }
        if (valid && (controlled.token_count != TOKEN_COUNT ||
                      expected_frame_count != DURATION_FRAME_COUNT ||
                      controlled.frame_count != DURATION_FRAME_COUNT ||
                      controlled.sample_count != FIXTURE_SAMPLE_COUNT)) {
            fprintf(stderr,
                    "test_inference: assembled output shape mismatch (tokens %zu, frames %zu, "
                    "samples %zu; expected %u, %u, %u)\n",
                    controlled.token_count, controlled.frame_count, controlled.sample_count,
                    TOKEN_COUNT, DURATION_FRAME_COUNT, FIXTURE_SAMPLE_COUNT);
            valid = 0;
        }
        if (valid) {
            valid = compare_reference_waveform(controlled.samples, expected_waveform,
                                               FIXTURE_SAMPLE_COUNT);
        }
    }
    if (valid) {
        status =
            kokoro_wav_write_pcm16("artifacts/09-af-heart-hello-controlled.wav", controlled.samples,
                                   controlled.sample_count, KOKORO_ISTFT_SAMPLE_RATE_HZ, &error);
        valid = expect_status(status, KOKORO_OK, &error, "write controlled audition WAV") &&
                verify_wav("artifacts/09-af-heart-hello-controlled.wav", controlled.sample_count);
    }
    kokoro_synthesis_output_free(&controlled);
    if (!valid) {
        kokoro_pack_close(pack);
        goto cleanup_fixtures;
    }

    KokoroSynthesisOptions native_options = {
        .voice_style_index = 5,
        .speed = 1.0f,
        .random_seed = 104729,
    };
    KokoroSynthesisOutput native_first = {0};
    KokoroSynthesisOutput native_repeat = {0};
    status =
        kokoro_synthesize(pack, token_ids, TOKEN_COUNT, &native_options, &native_first, &error);
    valid =
        expect_status(status, KOKORO_OK, &error, "run fully native phoneme-to-waveform inference");
    if (valid) {
        status = kokoro_synthesize(pack, token_ids, TOKEN_COUNT, &native_options, &native_repeat,
                                   &error);
        valid = expect_status(status, KOKORO_OK, &error,
                              "repeat fully native inference with the same seed");
    }
    if (valid) {
        valid = native_first.token_count == TOKEN_COUNT && native_first.durations != NULL &&
                native_first.samples != NULL && native_repeat.token_count == TOKEN_COUNT &&
                native_repeat.durations != NULL && native_repeat.samples != NULL &&
                native_first.frame_count == DURATION_FRAME_COUNT &&
                native_first.sample_count == FIXTURE_SAMPLE_COUNT &&
                native_repeat.frame_count == native_first.frame_count &&
                native_repeat.sample_count == native_first.sample_count;
        if (valid) {
            valid = memcmp(native_first.durations, native_repeat.durations,
                           TOKEN_COUNT * sizeof(*native_first.durations)) == 0 &&
                    memcmp(native_first.samples, native_repeat.samples,
                           FIXTURE_SAMPLE_COUNT * sizeof(*native_first.samples)) == 0;
        }
        for (size_t token = 0; token < TOKEN_COUNT && valid; token++) {
            valid = expected_durations[token] > 0 &&
                    native_first.durations[token] == (uint64_t)expected_durations[token];
        }
        for (size_t sample = 0; sample < native_first.sample_count && valid; sample++) {
            valid = isfinite(native_first.samples[sample]);
        }
        if (!valid) {
            valid = fail("fully native output is non-finite, unstable, or has the wrong shape");
        }
    }
    if (valid) {
        status =
            kokoro_wav_write_pcm16("artifacts/09-af-heart-hello-native.wav", native_first.samples,
                                   native_first.sample_count, KOKORO_ISTFT_SAMPLE_RATE_HZ, &error);
        valid = expect_status(status, KOKORO_OK, &error, "write fully native audition WAV") &&
                verify_wav("artifacts/09-af-heart-hello-native.wav", native_first.sample_count);
    }
    if (valid) {
        puts("fully native synthesis passed shape, finite-value, and seeded determinism checks");
    }
    kokoro_synthesis_output_free(&native_repeat);
    kokoro_synthesis_output_free(&native_first);
    kokoro_pack_close(pack);

cleanup_fixtures:
    free(expected_waveform);
    free(excitation_harmonic);
    free(excitation_noise);
    free(excitation_voiced);
    free(source_magnitude);
    free(source_phase);
    return valid ? 0 : 1;
}

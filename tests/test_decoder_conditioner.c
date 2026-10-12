#include "decoder_conditioner.h"
#include "kokoro.h"
#include "model_pack.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MODEL_PACK_PATH "models/kokoro-v1_0-af_heart.kpack"
#define TOKEN_COUNT 8U
#define DURATION_FRAME_COUNT 55U
#define FEATURE_FRAME_COUNT 110U
#define TEXT_FEATURE_COUNT (512U * DURATION_FRAME_COUNT)
#define DECODER_FEATURE_COUNT (512U * FEATURE_FRAME_COUNT)
#define STAGE_ATOL 1.0e-5f
#define STAGE_RTOL 1.0e-3f

static int fail(const char *message) {
    fprintf(stderr, "test_decoder_conditioner: %s\n", message);
    return 0;
}

static int expect_status(KokoroStatus actual, KokoroStatus expected, const KokoroError *error,
                         const char *operation) {
    if (actual == expected) {
        return 1;
    }
    fprintf(stderr, "test_decoder_conditioner: %s returned %s (%s), expected %s\n", operation,
            kokoro_status_name(actual), error->message, kokoro_status_name(expected));
    return 0;
}

static int read_fixture(const char *path, void *output, size_t byte_count) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "test_decoder_conditioner: cannot open fixture %s\n", path);
        return 0;
    }
    int valid = fread(output, 1, byte_count, file) == byte_count && fgetc(file) == EOF;
    fclose(file);
    if (!valid) {
        fprintf(stderr, "test_decoder_conditioner: fixture has the wrong size: %s\n", path);
    }
    return valid;
}

static int compare_features(const float *actual, const float *expected, size_t count) {
    float maximum = 0.0f;
    for (size_t index = 0; index < count; index++) {
        float difference = fabsf(actual[index] - expected[index]);
        if (difference > maximum) {
            maximum = difference;
        }
        float tolerance = STAGE_ATOL + STAGE_RTOL * fabsf(expected[index]);
        if (!isfinite(difference) || difference > tolerance) {
            fprintf(stderr,
                    "test_decoder_conditioner: decoder fixture miss at element %zu; difference "
                    "%.9g exceeds %.9g (atol %.1g, rtol %.1g)\n",
                    index, difference, tolerance, STAGE_ATOL, STAGE_RTOL);
            return 0;
        }
    }
    printf("decoder features fixture passed (max abs %.9g; atol %.1g, rtol %.1g)\n", maximum,
           STAGE_ATOL, STAGE_RTOL);
    return 1;
}

static int test_mismatched_lengths(const KokoroDecoderConditioner *conditioner,
                                   const KokoroTextEncoderOutput *text,
                                   const KokoroPitchNoiseOutput *pitch_noise) {
    KokoroDecoderOutput output = {0};
    KokoroError error = {0};
    KokoroTextEncoderOutput short_text = *text;
    short_text.frame_count--;
    KokoroStatus status =
        kokoro_decoder_condition(conditioner, &short_text, pitch_noise, 5, &output, &error);
    if (!expect_status(status, KOKORO_INVALID_ARGUMENT, &error,
                       "reject text and duration frame mismatch") ||
        strstr(error.message, "frame counts") == NULL || output.features != NULL ||
        output.feature_frame_count != 0 || output.duration_frame_count != 0) {
        return fail("text/pitch frame mismatch did not return a clear error without output");
    }

    KokoroPitchNoiseOutput short_pitch = *pitch_noise;
    short_pitch.prediction_frame_count--;
    memset(&error, 0, sizeof(error));
    status = kokoro_decoder_condition(conditioner, text, &short_pitch, 5, &output, &error);
    if (!expect_status(status, KOKORO_INVALID_ARGUMENT, &error, "reject F0/noise rate mismatch") ||
        strstr(error.message, "twice") == NULL || output.features != NULL ||
        output.feature_frame_count != 0 || output.duration_frame_count != 0) {
        return fail("F0/noise frame mismatch did not return a clear error without output");
    }
    puts("mismatched text and F0/noise lengths reject cleanly without partial output");
    return 1;
}

static int test_short_sequence(const KokoroDecoderConditioner *conditioner) {
    float text_features[512] = {0};
    float f0[2] = {0};
    float noise[2] = {0};
    KokoroTextEncoderOutput text = {
        .token_count = 1, .frame_count = 1, .aligned_features = text_features};
    KokoroPitchNoiseOutput pitch_noise = {
        .duration_frame_count = 1, .prediction_frame_count = 2, .f0 = f0, .noise = noise};
    KokoroDecoderOutput output = {0};
    KokoroError error = {0};
    KokoroStatus status =
        kokoro_decoder_condition(conditioner, &text, &pitch_noise, 0, &output, &error);
    int valid = expect_status(status, KOKORO_OK, &error, "condition one-frame sequence") &&
                output.duration_frame_count == 1 && output.feature_frame_count == 2 &&
                output.features != NULL;
    if (valid) {
        for (size_t index = 0; index < 512U * 2U; index++) {
            if (!isfinite(output.features[index])) {
                valid = fail("short decoder sequence contains a non-finite feature");
                break;
            }
        }
    }
    kokoro_decoder_output_free(&output);
    if (valid) {
        puts("one duration frame produces two finite decoder frames");
    }
    return valid;
}

static int test_reference_parity(const KokoroDecoderConditioner *conditioner) {
    KokoroTextEncoderOutput text = {.token_count = TOKEN_COUNT,
                                    .frame_count = DURATION_FRAME_COUNT};
    KokoroPitchNoiseOutput pitch_noise = {.duration_frame_count = DURATION_FRAME_COUNT,
                                          .prediction_frame_count = FEATURE_FRAME_COUNT};
    float expected[DECODER_FEATURE_COUNT];
    text.aligned_features = malloc(TEXT_FEATURE_COUNT * sizeof(float));
    pitch_noise.f0 = malloc(FEATURE_FRAME_COUNT * sizeof(float));
    pitch_noise.noise = malloc(FEATURE_FRAME_COUNT * sizeof(float));
    if (text.aligned_features == NULL || pitch_noise.f0 == NULL || pitch_noise.noise == NULL ||
        !read_fixture("reference/fixtures/aligned_text.f32le", text.aligned_features,
                      TEXT_FEATURE_COUNT * sizeof(float)) ||
        !read_fixture("reference/fixtures/f0.f32le", pitch_noise.f0,
                      FEATURE_FRAME_COUNT * sizeof(float)) ||
        !read_fixture("reference/fixtures/noise.f32le", pitch_noise.noise,
                      FEATURE_FRAME_COUNT * sizeof(float)) ||
        !read_fixture("reference/fixtures/decoder_features.f32le", expected,
                      DECODER_FEATURE_COUNT * sizeof(float))) {
        free(text.aligned_features);
        free(pitch_noise.f0);
        free(pitch_noise.noise);
        return 0;
    }

    KokoroDecoderOutput output = {0};
    KokoroError error = {0};
    KokoroStatus status =
        kokoro_decoder_condition(conditioner, &text, &pitch_noise, 5, &output, &error);
    free(text.aligned_features);
    free(pitch_noise.f0);
    free(pitch_noise.noise);
    if (!expect_status(status, KOKORO_OK, &error, "condition reference text/prosody")) {
        kokoro_decoder_output_free(&output);
        return 0;
    }
    int valid = output.duration_frame_count == DURATION_FRAME_COUNT &&
                output.feature_frame_count == FEATURE_FRAME_COUNT && output.features != NULL &&
                compare_features(output.features, expected, DECODER_FEATURE_COUNT);
    kokoro_decoder_output_free(&output);
    if (valid) {
        puts("model-backed decoder conditioning parity passed");
    }
    return valid;
}

int main(void) {
    KokoroError error = {0};
    KokoroDecoderConditioner *conditioner = NULL;
    KokoroPack *pack = NULL;
    KokoroStatus status = kokoro_decoder_conditioner_create(NULL, NULL, &conditioner, &error);
    if (!expect_status(status, KOKORO_INVALID_ARGUMENT, &error,
                       "reject null decoder conditioner pack") ||
        conditioner != NULL) {
        fail("null model pack was accepted");
        return 1;
    }
    if (access(MODEL_PACK_PATH, R_OK) != 0) {
        puts("SKIP model-backed decoder conditioning parity: ignored model pack is unavailable");
        return 0;
    }
    status = kokoro_pack_open(MODEL_PACK_PATH, &pack, &error);
    if (!expect_status(status, KOKORO_OK, &error, "open pinned model pack")) {
        return 1;
    }
    status = kokoro_decoder_conditioner_create(pack, NULL, &conditioner, &error);
    if (!expect_status(status, KOKORO_OK, &error, "bind decoder conditioning weights")) {
        kokoro_pack_close(pack);
        return 1;
    }
    float *text_buffer = calloc(TEXT_FEATURE_COUNT, sizeof(float));
    float *f0_buffer = calloc(FEATURE_FRAME_COUNT, sizeof(float));
    float *noise_buffer = calloc(FEATURE_FRAME_COUNT, sizeof(float));
    if (text_buffer == NULL || f0_buffer == NULL || noise_buffer == NULL) {
        free(text_buffer);
        free(f0_buffer);
        free(noise_buffer);
        kokoro_decoder_conditioner_free(conditioner);
        kokoro_pack_close(pack);
        return fail("cannot allocate mismatch test inputs");
    }
    KokoroTextEncoderOutput text = {.token_count = TOKEN_COUNT,
                                    .frame_count = DURATION_FRAME_COUNT,
                                    .aligned_features = text_buffer};
    KokoroPitchNoiseOutput pitch_noise = {.duration_frame_count = DURATION_FRAME_COUNT,
                                          .prediction_frame_count = FEATURE_FRAME_COUNT,
                                          .f0 = f0_buffer,
                                          .noise = noise_buffer};
    int valid = test_mismatched_lengths(conditioner, &text, &pitch_noise) &&
                test_short_sequence(conditioner) && test_reference_parity(conditioner);
    free(text_buffer);
    free(f0_buffer);
    free(noise_buffer);
    kokoro_decoder_conditioner_free(conditioner);
    kokoro_pack_close(pack);
    return valid ? 0 : 1;
}

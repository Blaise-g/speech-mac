#include "duration_alignment.h"
#include "kokoro.h"
#include "model_pack.h"
#include "pitch_noise_predictor.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MODEL_PACK_PATH "models/kokoro-v1_0-af_heart.kpack"
#define TOKEN_COUNT 8U
#define DURATION_FRAME_COUNT 55U
#define PREDICTION_FRAME_COUNT 110U
#define FEATURE_COUNT 640U
#define STAGE_ATOL 1.0e-5f
#define STAGE_RTOL 1.0e-3f

static int fail(const char *message) {
    fprintf(stderr, "test_pitch_noise_predictor: %s\n", message);
    return 0;
}

static int expect_status(KokoroStatus actual, KokoroStatus expected, const KokoroError *error,
                         const char *operation) {
    if (actual == expected) {
        return 1;
    }
    fprintf(stderr, "test_pitch_noise_predictor: %s returned %s (%s), expected %s\n", operation,
            kokoro_status_name(actual), error->message, kokoro_status_name(expected));
    return 0;
}

static int read_fixture(const char *path, void *output, size_t byte_count) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "test_pitch_noise_predictor: cannot open fixture %s\n", path);
        return 0;
    }
    int valid = fread(output, 1, byte_count, file) == byte_count && fgetc(file) == EOF;
    fclose(file);
    if (!valid) {
        fprintf(stderr, "test_pitch_noise_predictor: fixture has the wrong size: %s\n", path);
    }
    return valid;
}

static int compare_floats(const char *name, const float *actual, const float *expected,
                          size_t count) {
    float maximum = 0.0f;
    for (size_t index = 0; index < count; index++) {
        float difference = fabsf(actual[index] - expected[index]);
        if (difference > maximum) {
            maximum = difference;
        }
        float tolerance = STAGE_ATOL + STAGE_RTOL * fabsf(expected[index]);
        if (!isfinite(difference) || difference > tolerance) {
            fprintf(stderr,
                    "test_pitch_noise_predictor: %s fixture miss at element %zu; difference "
                    "%.9g exceeds %.9g (atol %.1g, rtol %.1g)\n",
                    name, index, difference, tolerance, STAGE_ATOL, STAGE_RTOL);
            return 0;
        }
    }
    printf("%s fixture passed (max abs %.9g; atol %.1g, rtol %.1g)\n", name, maximum, STAGE_ATOL,
           STAGE_RTOL);
    return 1;
}

static int check_finite(const char *name, const float *values, size_t count) {
    for (size_t index = 0; index < count; index++) {
        if (!isfinite(values[index])) {
            fprintf(stderr, "test_pitch_noise_predictor: %s contains a non-finite value at %zu\n",
                    name, index);
            return 0;
        }
    }
    return 1;
}

static int test_output_lengths(const KokoroPitchNoisePredictor *predictor) {
    static const size_t frame_counts[] = {1, 33};
    float features[FEATURE_COUNT] = {0};
    for (size_t scenario = 0; scenario < sizeof(frame_counts) / sizeof(frame_counts[0]);
         scenario++) {
        size_t frame_count = frame_counts[scenario];
        float *alignment = malloc(frame_count * sizeof(float));
        if (alignment == NULL) {
            return fail("cannot allocate synthetic alignment");
        }
        for (size_t frame = 0; frame < frame_count; frame++) {
            alignment[frame] = 1.0f;
        }
        uint64_t duration_value = frame_count;
        KokoroDurationOutput duration = {.token_count = 1,
                                         .frame_count = frame_count,
                                         .duration_features = features,
                                         .durations = &duration_value,
                                         .alignment = alignment};
        KokoroPitchNoiseOutput output = {0};
        KokoroError error = {0};
        KokoroStatus status = kokoro_pitch_noise_predict(predictor, &duration, 0, &output, &error);
        if (!expect_status(status, KOKORO_OK, &error, "predict synthetic sequence")) {
            free(alignment);
            kokoro_pitch_noise_output_free(&output);
            return 0;
        }
        if (output.duration_frame_count != frame_count ||
            output.prediction_frame_count != frame_count * 2 || output.f0 == NULL ||
            output.noise == NULL ||
            !check_finite("short/long F0", output.f0, output.prediction_frame_count) ||
            !check_finite("short/long noise", output.noise, output.prediction_frame_count)) {
            free(alignment);
            kokoro_pitch_noise_output_free(&output);
            return fail("prediction length or finite-value invariant failed");
        }
        float *f0_before_reuse = output.f0;
        float *noise_before_reuse = output.noise;
        KokoroError reuse_error = {0};
        status = kokoro_pitch_noise_predict(predictor, &duration, 0, &output, &reuse_error);
        if (!expect_status(status, KOKORO_INVALID_ARGUMENT, &reuse_error,
                           "reject reuse of owned output") ||
            output.f0 != f0_before_reuse || output.noise != noise_before_reuse ||
            output.duration_frame_count != frame_count ||
            output.prediction_frame_count != frame_count * 2) {
            free(alignment);
            kokoro_pitch_noise_output_free(&output);
            return fail("rejected output reuse changed or released existing buffers");
        }
        free(alignment);
        kokoro_pitch_noise_output_free(&output);
    }
    puts("short and longer sequences preserve the official 2x prediction rate and finite values");
    return 1;
}

static int test_invalid_inputs(const KokoroPitchNoisePredictor *predictor) {
    float features[FEATURE_COUNT] = {0};
    float alignment[1] = {1.0f};
    uint64_t duration_value = 1;
    KokoroDurationOutput valid_duration = {.token_count = 1,
                                           .frame_count = 1,
                                           .duration_features = features,
                                           .durations = &duration_value,
                                           .alignment = alignment};
    KokoroPitchNoiseOutput output = {0};
    KokoroError error = {0};
    KokoroStatus status = kokoro_pitch_noise_predict(NULL, &valid_duration, 0, &output, &error);
    if (!expect_status(status, KOKORO_INVALID_ARGUMENT, &error, "reject null predictor")) {
        return 0;
    }
    status = kokoro_pitch_noise_predict(predictor, &valid_duration, SIZE_MAX, &output, &error);
    if (!expect_status(status, KOKORO_INVALID_ARGUMENT, &error, "reject unknown voice style")) {
        return 0;
    }
    KokoroDurationOutput malformed = valid_duration;
    malformed.frame_count = 2;
    status = kokoro_pitch_noise_predict(predictor, &malformed, 0, &output, &error);
    if (!expect_status(status, KOKORO_INVALID_ARGUMENT, &error,
                       "reject inconsistent duration shape")) {
        return 0;
    }
    features[0] = NAN;
    status = kokoro_pitch_noise_predict(predictor, &valid_duration, 0, &output, &error);
    features[0] = 0.0f;
    if (!expect_status(status, KOKORO_NUMERIC_ERROR, &error, "reject non-finite features") ||
        output.f0 != NULL || output.noise != NULL || output.prediction_frame_count != 0) {
        kokoro_pitch_noise_output_free(&output);
        return fail("invalid input changed the output");
    }
    puts("invalid voice, duration shape, and non-finite feature inputs were rejected");
    return 1;
}

static int test_reference_parity(const KokoroPitchNoisePredictor *predictor) {
    KokoroDurationOutput duration = {0};
    duration.token_count = TOKEN_COUNT;
    duration.frame_count = DURATION_FRAME_COUNT;
    duration.duration_features = malloc(TOKEN_COUNT * FEATURE_COUNT * sizeof(float));
    duration.durations = malloc(TOKEN_COUNT * sizeof(uint64_t));
    duration.alignment = malloc(TOKEN_COUNT * DURATION_FRAME_COUNT * sizeof(float));
    float expected_f0[PREDICTION_FRAME_COUNT];
    float expected_noise[PREDICTION_FRAME_COUNT];
    int64_t fixture_durations[TOKEN_COUNT];
    if (duration.duration_features == NULL || duration.durations == NULL ||
        duration.alignment == NULL ||
        !read_fixture("reference/fixtures/duration_features.f32le", duration.duration_features,
                      TOKEN_COUNT * FEATURE_COUNT * sizeof(float)) ||
        !read_fixture("reference/fixtures/durations.i64le", fixture_durations,
                      sizeof(fixture_durations)) ||
        !read_fixture("reference/fixtures/alignment.f32le", duration.alignment,
                      TOKEN_COUNT * DURATION_FRAME_COUNT * sizeof(float)) ||
        !read_fixture("reference/fixtures/f0.f32le", expected_f0, sizeof(expected_f0)) ||
        !read_fixture("reference/fixtures/noise.f32le", expected_noise, sizeof(expected_noise))) {
        kokoro_duration_output_free(&duration);
        return 0;
    }
    size_t frame_sum = 0;
    for (size_t token = 0; token < TOKEN_COUNT; token++) {
        if (fixture_durations[token] <= 0) {
            kokoro_duration_output_free(&duration);
            return fail("reference duration fixture contains a non-positive value");
        }
        duration.durations[token] = (uint64_t)fixture_durations[token];
        frame_sum += (size_t)duration.durations[token];
    }
    if (frame_sum != DURATION_FRAME_COUNT) {
        kokoro_duration_output_free(&duration);
        return fail("reference duration fixture has an unexpected frame sum");
    }
    KokoroPitchNoiseOutput output = {0};
    KokoroError error = {0};
    KokoroStatus status = kokoro_pitch_noise_predict(predictor, &duration, 5, &output, &error);
    kokoro_duration_output_free(&duration);
    if (!expect_status(status, KOKORO_OK, &error, "predict reference pitch/noise")) {
        kokoro_pitch_noise_output_free(&output);
        return 0;
    }
    int valid = output.duration_frame_count == DURATION_FRAME_COUNT &&
                output.prediction_frame_count == PREDICTION_FRAME_COUNT &&
                check_finite("reference F0", output.f0, PREDICTION_FRAME_COUNT) &&
                check_finite("reference noise", output.noise, PREDICTION_FRAME_COUNT) &&
                compare_floats("F0", output.f0, expected_f0, PREDICTION_FRAME_COUNT) &&
                compare_floats("noise", output.noise, expected_noise, PREDICTION_FRAME_COUNT);
    kokoro_pitch_noise_output_free(&output);
    if (valid) {
        puts("model-backed F0/noise predictor parity passed");
    }
    return valid;
}

int main(void) {
    KokoroError error = {0};
    KokoroPitchNoisePredictor *predictor = NULL;
    KokoroPack *pack = NULL;
    KokoroStatus status = kokoro_pitch_noise_predictor_create(NULL, &predictor, &error);
    if (!expect_status(status, KOKORO_INVALID_ARGUMENT, &error,
                       "reject null pitch/noise predictor pack") ||
        predictor != NULL) {
        fail("null model pack was accepted");
        return 1;
    }
    if (access(MODEL_PACK_PATH, R_OK) != 0) {
        puts("SKIP model-backed pitch/noise parity: ignored model pack is unavailable");
        return 0;
    }
    status = kokoro_pack_open(MODEL_PACK_PATH, &pack, &error);
    if (!expect_status(status, KOKORO_OK, &error, "open pinned model pack")) {
        return 1;
    }
    status = kokoro_pitch_noise_predictor_create(pack, &predictor, &error);
    if (!expect_status(status, KOKORO_OK, &error, "bind pitch/noise predictor weights")) {
        kokoro_pack_close(pack);
        return 1;
    }
    int valid = test_invalid_inputs(predictor) && test_output_lengths(predictor) &&
                test_reference_parity(predictor);
    kokoro_pitch_noise_predictor_free(predictor);
    kokoro_pack_close(pack);
    return valid ? 0 : 1;
}

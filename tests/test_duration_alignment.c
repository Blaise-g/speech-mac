#include "duration_alignment.h"
#include "kokoro.h"
#include "model_pack.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MODEL_PACK_PATH "models/kokoro-v1_0-af_heart.kpack"
#define CONTEXT_FIXTURE_PATH "reference/fixtures/contextual_features.f32le"
#define TOKEN_COUNT 8U
#define CONTEXT_FEATURES 512U
#define STAGE_ATOL 1.0e-5f
#define STAGE_RTOL 1.0e-3f

static int fail(const char *message) {
    fprintf(stderr, "test_duration_alignment: %s\n", message);
    return 1;
}

static int expect_status(KokoroStatus actual, KokoroStatus expected, const KokoroError *error,
                         const char *operation) {
    if (actual == expected) {
        return 1;
    }
    fprintf(stderr, "test_duration_alignment: %s returned %s (%s), expected %s\n", operation,
            kokoro_status_name(actual), error->message, kokoro_status_name(expected));
    return 0;
}

static int read_fixture(const char *path, void *output, size_t byte_count) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "test_duration_alignment: cannot open fixture %s\n", path);
        return 0;
    }
    int valid = fread(output, 1, byte_count, file) == byte_count && fgetc(file) == EOF;
    fclose(file);
    if (!valid) {
        fprintf(stderr, "test_duration_alignment: fixture has the wrong size: %s\n", path);
    }
    return valid;
}

static int compare_floats(const char *name, const float *actual, const float *expected,
                          size_t count) {
    float max_difference = 0.0f;
    size_t worst_index = 0;
    for (size_t index = 0; index < count; index++) {
        float difference = fabsf(actual[index] - expected[index]);
        if (difference > max_difference) {
            max_difference = difference;
            worst_index = index;
        }
        float tolerance = STAGE_ATOL + STAGE_RTOL * fabsf(expected[index]);
        if (!isfinite(difference) || difference > tolerance) {
            fprintf(stderr,
                    "test_duration_alignment: %s fixture miss at element %zu; max abs %.9g "
                    "(atol %.1g, rtol %.1g)\n",
                    name, index, max_difference, STAGE_ATOL, STAGE_RTOL);
            return 0;
        }
    }
    printf("%s fixture passed (max abs %.9g; atol %.1g, rtol %.1g)\n", name, max_difference,
           STAGE_ATOL, STAGE_RTOL);
    (void)worst_index;
    return 1;
}

static int check_alignment(const KokoroDurationOutput *output) {
    size_t first_frame = 0;
    for (size_t token = 0; token < output->token_count; token++) {
        size_t last_frame = first_frame + (size_t)output->durations[token];
        for (size_t frame = 0; frame < output->frame_count; frame++) {
            float expected = frame >= first_frame && frame < last_frame ? 1.0f : 0.0f;
            if (output->alignment[token * output->frame_count + frame] != expected) {
                return 0;
            }
        }
        first_frame = last_frame;
    }
    if (first_frame != output->frame_count) {
        return 0;
    }
    for (size_t frame = 0; frame < output->frame_count; frame++) {
        size_t mapped_tokens = 0;
        for (size_t token = 0; token < output->token_count; token++) {
            mapped_tokens += output->alignment[token * output->frame_count + frame] == 1.0f;
        }
        if (mapped_tokens != 1) {
            return 0;
        }
    }
    return 1;
}

static int test_single_token_predictor(const KokoroDurationPredictor *predictor) {
    float contextual_features[CONTEXT_FEATURES] = {0};
    KokoroDurationOutput output = {0};
    KokoroError error = {0};
    KokoroStatus status =
        kokoro_duration_predict(predictor, contextual_features, 1, 5, 1.0f, &output, &error);
    if (!expect_status(status, KOKORO_OK, &error, "predict one-token duration")) {
        return 0;
    }
    int valid = output.token_count == 1 && output.duration_features != NULL &&
                output.duration_rnn != NULL && output.duration_logits != NULL &&
                output.duration_values != NULL && output.durations != NULL &&
                output.alignment != NULL && output.frame_count == output.durations[0] &&
                output.durations[0] > 0 && isfinite(output.duration_values[0]) &&
                check_alignment(&output);
    kokoro_duration_output_free(&output);
    if (valid) {
        puts("one-token duration predictor shape and alignment passed");
    }
    return valid;
}

static int test_rounding_speed_and_alignment(void) {
    const float raw_values[] = {0.0f, 0.5f, 1.5f, 2.5f, 3.5f, 4.5f};
    const uint64_t expected_durations[][6] = {
        {1, 1, 2, 2, 4, 4},
        {1, 1, 3, 5, 7, 9},
        {1, 1, 1, 1, 2, 2},
    };
    const float speeds[] = {1.0f, 0.5f, 2.0f};
    for (size_t scenario = 0; scenario < sizeof(speeds) / sizeof(speeds[0]); scenario++) {
        KokoroDurationOutput output = {0};
        KokoroError error = {0};
        KokoroStatus status =
            kokoro_duration_align_values(raw_values, 6, speeds[scenario], &output, &error);
        if (!expect_status(status, KOKORO_OK, &error, "apply duration speed and build alignment")) {
            return 0;
        }
        size_t expected_frames = 0;
        for (size_t token = 0; token < 6; token++) {
            if (output.durations[token] != expected_durations[scenario][token]) {
                kokoro_duration_output_free(&output);
                return 0;
            }
            expected_frames += (size_t)expected_durations[scenario][token];
            if (output.duration_values[token] != raw_values[token] / speeds[scenario]) {
                kokoro_duration_output_free(&output);
                return 0;
            }
        }
        if (output.frame_count != expected_frames || !check_alignment(&output)) {
            kokoro_duration_output_free(&output);
            return 0;
        }
        kokoro_duration_output_free(&output);
    }
    puts("speed boundaries, ties-to-even rounding, minimum duration, and alignment passed");
    return 1;
}

static int test_invalid_duration_inputs(void) {
    KokoroError error = {0};
    KokoroDurationOutput output = {0};
    float value = 1.0f;
    const float invalid_speeds[] = {0.499f, 2.001f, NAN, INFINITY};
    for (size_t index = 0; index < sizeof(invalid_speeds) / sizeof(invalid_speeds[0]); index++) {
        KokoroStatus status =
            kokoro_duration_align_values(&value, 1, invalid_speeds[index], &output, &error);
        if (!expect_status(status, KOKORO_INVALID_ARGUMENT, &error, "reject invalid speed") ||
            output.duration_values != NULL || output.durations != NULL ||
            output.alignment != NULL) {
            kokoro_duration_output_free(&output);
            return 0;
        }
    }
    value = 50.01f;
    KokoroStatus status = kokoro_duration_align_values(&value, 1, 1.0f, &output, &error);
    if (!expect_status(status, KOKORO_NUMERIC_ERROR, &error,
                       "reject duration outside sigmoid sum") ||
        output.duration_values != NULL) {
        kokoro_duration_output_free(&output);
        return 0;
    }
    if (kokoro_duration_align_values(&value, KOKORO_DURATION_MAX_TOKENS + 1, 1.0f, &output,
                                     &error) != KOKORO_INVALID_ARGUMENT) {
        kokoro_duration_output_free(&output);
        return 0;
    }
    kokoro_duration_output_free(&output);
    puts("invalid speed, duration, and sequence inputs were rejected without output");
    return 1;
}

static int model_backed_parity(void) {
    FILE *model_file = fopen(MODEL_PACK_PATH, "rb");
    if (model_file == NULL) {
        puts("SKIP model-backed duration parity: convert the pinned model pack to run weight "
             "parity");
        return 1;
    }
    fclose(model_file);

    float contextual_features[CONTEXT_FEATURES * TOKEN_COUNT];
    if (!read_fixture(CONTEXT_FIXTURE_PATH, contextual_features, sizeof(contextual_features))) {
        return 0;
    }
    KokoroError error = {0};
    KokoroPack *pack = NULL;
    KokoroStatus status = kokoro_pack_open(MODEL_PACK_PATH, &pack, &error);
    if (!expect_status(status, KOKORO_OK, &error, "open model pack")) {
        return 0;
    }
    KokoroDurationPredictor *predictor = NULL;
    status = kokoro_duration_predictor_create(pack, &predictor, &error);
    if (!expect_status(status, KOKORO_OK, &error, "bind duration predictor weights")) {
        kokoro_pack_close(pack);
        return 0;
    }
    if (!test_single_token_predictor(predictor)) {
        kokoro_duration_predictor_free(predictor);
        kokoro_pack_close(pack);
        return 0;
    }

    float expected_features[TOKEN_COUNT * KOKORO_DURATION_FEATURES];
    float expected_rnn[TOKEN_COUNT * KOKORO_DURATION_RNN_FEATURES];
    float expected_logits[TOKEN_COUNT * KOKORO_DURATION_LOGIT_CHANNELS];
    float expected_values[TOKEN_COUNT];
    int64_t expected_durations[TOKEN_COUNT];
    const char *fixture_names[] = {"duration_features", "duration_rnn", "duration_logits",
                                   "duration_values", "durations"};
    void *fixture_outputs[] = {expected_features, expected_rnn, expected_logits, expected_values,
                               expected_durations};
    const size_t fixture_sizes[] = {sizeof(expected_features), sizeof(expected_rnn),
                                    sizeof(expected_logits), sizeof(expected_values),
                                    sizeof(expected_durations)};
    const char *fixture_suffixes[] = {"f32le", "f32le", "f32le", "f32le", "i64le"};
    for (size_t index = 0; index < sizeof(fixture_names) / sizeof(fixture_names[0]); index++) {
        char path[160];
        (void)snprintf(path, sizeof(path), "reference/fixtures/%s.%s", fixture_names[index],
                       fixture_suffixes[index]);
        if (!read_fixture(path, fixture_outputs[index], fixture_sizes[index])) {
            kokoro_duration_predictor_free(predictor);
            kokoro_pack_close(pack);
            return 0;
        }
    }

    KokoroDurationOutput output = {0};
    status = kokoro_duration_predict(predictor, contextual_features, TOKEN_COUNT, 5, 1.0f, &output,
                                     &error);
    if (!expect_status(status, KOKORO_OK, &error, "predict reference hello durations")) {
        kokoro_duration_predictor_free(predictor);
        kokoro_pack_close(pack);
        return 0;
    }
    int valid =
        compare_floats("duration features", output.duration_features, expected_features,
                       TOKEN_COUNT * KOKORO_DURATION_FEATURES) &&
        compare_floats("duration recurrent output", output.duration_rnn, expected_rnn,
                       TOKEN_COUNT * KOKORO_DURATION_RNN_FEATURES) &&
        compare_floats("duration logits", output.duration_logits, expected_logits,
                       TOKEN_COUNT * KOKORO_DURATION_LOGIT_CHANNELS) &&
        compare_floats("duration values", output.duration_values, expected_values, TOKEN_COUNT);
    size_t expected_frame_count = 0;
    for (size_t token = 0; token < TOKEN_COUNT; token++) {
        if (expected_durations[token] <= 0 ||
            output.durations[token] != (uint64_t)expected_durations[token]) {
            valid = 0;
        }
        expected_frame_count += (size_t)expected_durations[token];
    }
    float *expected_alignment = NULL;
    if (expected_frame_count > SIZE_MAX / TOKEN_COUNT / sizeof(float)) {
        valid = 0;
    } else {
        expected_alignment = malloc(TOKEN_COUNT * expected_frame_count * sizeof(float));
    }
    if (expected_alignment == NULL ||
        !read_fixture("reference/fixtures/alignment.f32le", expected_alignment,
                      TOKEN_COUNT * expected_frame_count * sizeof(float)) ||
        output.frame_count != expected_frame_count || !check_alignment(&output) ||
        !compare_floats("duration alignment", output.alignment, expected_alignment,
                        TOKEN_COUNT * expected_frame_count)) {
        valid = 0;
    }
    free(expected_alignment);
    kokoro_duration_output_free(&output);

    const float speeds[] = {0.5f, 2.0f};
    const char *speed_tags[] = {"0_5", "2_0"};
    for (size_t scenario = 0; scenario < 2; scenario++) {
        char values_path[160];
        char durations_path[160];
        char alignment_path[160];
        (void)snprintf(values_path, sizeof(values_path),
                       "reference/fixtures/duration_values_speed_%s.f32le", speed_tags[scenario]);
        (void)snprintf(durations_path, sizeof(durations_path),
                       "reference/fixtures/durations_speed_%s.i64le", speed_tags[scenario]);
        (void)snprintf(alignment_path, sizeof(alignment_path),
                       "reference/fixtures/alignment_speed_%s.f32le", speed_tags[scenario]);
        float expected_speed_values[TOKEN_COUNT];
        int64_t expected_speed_durations[TOKEN_COUNT];
        if (!read_fixture(values_path, expected_speed_values, sizeof(expected_speed_values)) ||
            !read_fixture(durations_path, expected_speed_durations,
                          sizeof(expected_speed_durations))) {
            valid = 0;
            continue;
        }
        expected_frame_count = 0;
        for (size_t token = 0; token < TOKEN_COUNT; token++) {
            if (expected_speed_durations[token] <= 0) {
                valid = 0;
            } else {
                expected_frame_count += (size_t)expected_speed_durations[token];
            }
        }
        expected_alignment = malloc(TOKEN_COUNT * expected_frame_count * sizeof(float));
        if (expected_alignment == NULL ||
            !read_fixture(alignment_path, expected_alignment,
                          TOKEN_COUNT * expected_frame_count * sizeof(float))) {
            free(expected_alignment);
            valid = 0;
            continue;
        }
        status = kokoro_duration_predict(predictor, contextual_features, TOKEN_COUNT, 5,
                                         speeds[scenario], &output, &error);
        if (!expect_status(status, KOKORO_OK, &error, "predict speed boundary durations")) {
            free(expected_alignment);
            valid = 0;
            continue;
        }
        if (!compare_floats("speed duration values", output.duration_values, expected_speed_values,
                            TOKEN_COUNT) ||
            output.frame_count != expected_frame_count || !check_alignment(&output) ||
            !compare_floats("speed duration alignment", output.alignment, expected_alignment,
                            TOKEN_COUNT * expected_frame_count)) {
            valid = 0;
        }
        for (size_t token = 0; token < TOKEN_COUNT; token++) {
            if (output.durations[token] != (uint64_t)expected_speed_durations[token]) {
                valid = 0;
            }
        }
        free(expected_alignment);
        kokoro_duration_output_free(&output);
    }

    kokoro_duration_predictor_free(predictor);
    kokoro_pack_close(pack);
    if (valid) {
        puts("model-backed predictor, duration, and speed alignment parity passed");
    }
    return valid;
}

int main(void) {
    if (!test_rounding_speed_and_alignment()) {
        return fail("duration rounding, speed scaling, or alignment invariant failed");
    }
    if (!test_invalid_duration_inputs()) {
        return fail("invalid duration input was accepted or changed output");
    }
    if (!model_backed_parity()) {
        return fail("duration predictor did not match reference fixtures");
    }
    return 0;
}

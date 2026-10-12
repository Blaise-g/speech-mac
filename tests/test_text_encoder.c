#include "duration_alignment.h"
#include "kokoro.h"
#include "model_pack.h"
#include "text_encoder.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MODEL_PACK_PATH "models/kokoro-v1_0-af_heart.kpack"
#define TOKEN_COUNT 8U
#define FRAME_COUNT 55U
#define STAGE_ATOL 1.0e-5f
#define STAGE_RTOL 1.0e-3f

static int fail(const char *message) {
    fprintf(stderr, "test_text_encoder: %s\n", message);
    return 0;
}

static int expect_status(KokoroStatus actual, KokoroStatus expected, const KokoroError *error,
                         const char *operation) {
    if (actual == expected) {
        return 1;
    }
    fprintf(stderr, "test_text_encoder: %s returned %s (%s), expected %s\n", operation,
            kokoro_status_name(actual), error->message, kokoro_status_name(expected));
    return 0;
}

static int read_fixture(const char *path, void *output, size_t byte_count) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "test_text_encoder: cannot open fixture %s\n", path);
        return 0;
    }
    int valid = fread(output, 1, byte_count, file) == byte_count && fgetc(file) == EOF;
    fclose(file);
    if (!valid) {
        fprintf(stderr, "test_text_encoder: fixture has the wrong size: %s\n", path);
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
                    "test_text_encoder: %s fixture miss at element %zu; difference %.9g "
                    "exceeds %.9g (atol %.1g, rtol %.1g)\n",
                    name, index, difference, tolerance, STAGE_ATOL, STAGE_RTOL);
            return 0;
        }
    }
    printf("%s fixture passed (max abs %.9g; atol %.1g, rtol %.1g)\n", name, maximum, STAGE_ATOL,
           STAGE_RTOL);
    return 1;
}

static int check_finite(const float *values, size_t count) {
    for (size_t index = 0; index < count; index++) {
        if (!isfinite(values[index])) {
            return 0;
        }
    }
    return 1;
}

static int test_reference_parity(const KokoroTextEncoder *encoder) {
    const int64_t token_ids[TOKEN_COUNT] = {0, 50, 83, 156, 54, 57, 135, 0};
    float expected_text[KOKORO_TEXT_ENCODER_FEATURES * TOKEN_COUNT];
    float expected_aligned[KOKORO_TEXT_ENCODER_FEATURES * FRAME_COUNT];
    float alignment_values[TOKEN_COUNT * FRAME_COUNT];
    if (!read_fixture("reference/fixtures/text_features.f32le", expected_text,
                      sizeof(expected_text)) ||
        !read_fixture("reference/fixtures/aligned_text.f32le", expected_aligned,
                      sizeof(expected_aligned)) ||
        !read_fixture("reference/fixtures/alignment.f32le", alignment_values,
                      sizeof(alignment_values))) {
        return 0;
    }
    KokoroDurationOutput duration = {
        .token_count = TOKEN_COUNT, .frame_count = FRAME_COUNT, .alignment = alignment_values};
    KokoroTextEncoderOutput output = {0};
    KokoroError error = {0};
    KokoroStatus status = kokoro_text_encoder_run(encoder, token_ids, TOKEN_COUNT, TOKEN_COUNT,
                                                  &duration, &output, &error);
    if (!expect_status(status, KOKORO_OK, &error, "encode and align reference tokens")) {
        kokoro_text_encoder_output_free(&output);
        return 0;
    }
    int valid = output.token_count == TOKEN_COUNT && output.frame_count == FRAME_COUNT &&
                output.text_features != NULL && output.aligned_features != NULL &&
                check_finite(output.text_features, TOKEN_COUNT * KOKORO_TEXT_ENCODER_FEATURES) &&
                check_finite(output.aligned_features, FRAME_COUNT * KOKORO_TEXT_ENCODER_FEATURES) &&
                compare_floats("text features", output.text_features, expected_text,
                               TOKEN_COUNT * KOKORO_TEXT_ENCODER_FEATURES) &&
                compare_floats("aligned text", output.aligned_features, expected_aligned,
                               FRAME_COUNT * KOKORO_TEXT_ENCODER_FEATURES);
    kokoro_text_encoder_output_free(&output);
    return valid;
}

static int test_padding_and_single_token(const KokoroTextEncoder *encoder) {
    int64_t first_ids[TOKEN_COUNT] = {0, 50, 83, 156, 54, 57, 135, 0};
    int64_t changed_padding[TOKEN_COUNT] = {0, 50, 83, 156, 54, 57, 4, 177};
    float padding_alignment[TOKEN_COUNT * 2] = {0};
    padding_alignment[6 * 2] = 1.0f;
    padding_alignment[7 * 2 + 1] = 1.0f;
    KokoroDurationOutput duration = {
        .token_count = TOKEN_COUNT, .frame_count = 2, .alignment = padding_alignment};
    KokoroTextEncoderOutput first = {0};
    KokoroTextEncoderOutput second = {0};
    KokoroError error = {0};
    KokoroStatus status =
        kokoro_text_encoder_run(encoder, first_ids, TOKEN_COUNT, 6, &duration, &first, &error);
    if (!expect_status(status, KOKORO_OK, &error, "encode a padded sequence")) {
        kokoro_text_encoder_output_free(&first);
        return 0;
    }
    status = kokoro_text_encoder_run(encoder, changed_padding, TOKEN_COUNT, 6, &duration, &second,
                                     &error);
    if (!expect_status(status, KOKORO_OK, &error, "change masked token IDs")) {
        kokoro_text_encoder_output_free(&first);
        kokoro_text_encoder_output_free(&second);
        return 0;
    }
    int valid = first.token_count == TOKEN_COUNT && first.frame_count == 2 &&
                memcmp(first.text_features, second.text_features,
                       TOKEN_COUNT * KOKORO_TEXT_ENCODER_FEATURES * sizeof(float)) == 0 &&
                memcmp(first.aligned_features, second.aligned_features,
                       KOKORO_TEXT_ENCODER_FEATURES * 2 * sizeof(float)) == 0 &&
                check_finite(first.text_features, TOKEN_COUNT * KOKORO_TEXT_ENCODER_FEATURES);
    for (size_t channel = 0; channel < KOKORO_TEXT_ENCODER_FEATURES && valid; channel++) {
        valid = first.text_features[channel * TOKEN_COUNT + 6] == 0.0f &&
                first.text_features[channel * TOKEN_COUNT + 7] == 0.0f &&
                first.aligned_features[channel * 2] == 0.0f &&
                first.aligned_features[channel * 2 + 1] == 0.0f;
    }
    kokoro_text_encoder_output_free(&first);
    kokoro_text_encoder_output_free(&second);
    if (!valid) {
        return fail("masked IDs changed output or contributed features");
    }

    const int64_t one_id[] = {0};
    const float one_alignment[] = {1.0f};
    KokoroDurationOutput one_duration = {
        .token_count = 1, .frame_count = 1, .alignment = (float *)one_alignment};
    KokoroTextEncoderOutput one = {0};
    status = kokoro_text_encoder_run(encoder, one_id, 1, 1, &one_duration, &one, &error);
    valid = expect_status(status, KOKORO_OK, &error, "encode one token") && one.token_count == 1 &&
            one.frame_count == 1 && check_finite(one.text_features, KOKORO_TEXT_ENCODER_FEATURES) &&
            memcmp(one.text_features, one.aligned_features,
                   KOKORO_TEXT_ENCODER_FEATURES * sizeof(float)) == 0;
    kokoro_text_encoder_output_free(&one);
    return valid ? 1 : fail("one-token output shape or alignment failed");
}

static int test_invalid_inputs(const KokoroTextEncoder *encoder) {
    const int64_t token_ids[] = {0, 50};
    const float valid_alignment[] = {1.0f, 0.0f, 0.0f, 1.0f};
    KokoroDurationOutput duration = {
        .token_count = 2, .frame_count = 2, .alignment = (float *)valid_alignment};
    KokoroTextEncoderOutput output = {0};
    KokoroError error = {0};
    KokoroStatus status =
        kokoro_text_encoder_run(encoder, token_ids, 2, 0, &duration, &output, &error);
    if (!expect_status(status, KOKORO_INVALID_ARGUMENT, &error, "reject zero valid length")) {
        return 0;
    }
    status = kokoro_text_encoder_run(encoder, token_ids, 2, 3, &duration, &output, &error);
    if (!expect_status(status, KOKORO_INVALID_ARGUMENT, &error, "reject excessive valid length")) {
        return 0;
    }
    int64_t invalid_ids[] = {0, 178};
    status = kokoro_text_encoder_run(encoder, invalid_ids, 2, 2, &duration, &output, &error);
    if (!expect_status(status, KOKORO_INVALID_TOKEN_ID, &error, "reject invalid token ID")) {
        return 0;
    }
    status = kokoro_text_encoder_run(encoder, token_ids, KOKORO_TEXT_ENCODER_MAX_TOKENS + 1, 2,
                                     &duration, &output, &error);
    if (!expect_status(status, KOKORO_INVALID_ARGUMENT, &error,
                       "reject excessive sequence length")) {
        return 0;
    }
    KokoroDurationOutput mismatched = duration;
    mismatched.token_count = 1;
    status = kokoro_text_encoder_run(encoder, token_ids, 2, 2, &mismatched, &output, &error);
    if (!expect_status(status, KOKORO_INVALID_ARGUMENT, &error,
                       "reject mismatched token alignment")) {
        return 0;
    }
    float invalid_alignment[] = {1.0f, 0.0f, NAN, 1.0f};
    KokoroDurationOutput non_finite = duration;
    non_finite.alignment = invalid_alignment;
    status = kokoro_text_encoder_run(encoder, token_ids, 2, 2, &non_finite, &output, &error);
    if (!expect_status(status, KOKORO_NUMERIC_ERROR, &error, "reject non-finite alignment")) {
        return 0;
    }
    float partial_alignment[] = {0.5f, 0.5f, 0.0f, 1.0f};
    KokoroDurationOutput non_one_hot = duration;
    non_one_hot.alignment = partial_alignment;
    status = kokoro_text_encoder_run(encoder, token_ids, 2, 2, &non_one_hot, &output, &error);
    if (!expect_status(status, KOKORO_INVALID_ARGUMENT, &error, "reject non one-hot alignment")) {
        return 0;
    }
    if (output.text_features != NULL || output.aligned_features != NULL ||
        output.token_count != 0 || output.frame_count != 0) {
        kokoro_text_encoder_output_free(&output);
        return fail("rejected input changed the output");
    }
    puts("invalid lengths, IDs, mismatched and malformed alignment were rejected");
    return 1;
}

static int model_backed_tests(void) {
    errno = 0;
    FILE *model_file = fopen(MODEL_PACK_PATH, "rb");
    if (model_file == NULL) {
        if (errno == ENOENT) {
            puts("SKIP model-backed text encoder checks: convert the pinned model pack to run "
                 "weight parity");
            return 1;
        }
        fprintf(stderr, "test_text_encoder: cannot open model pack %s: %s\n", MODEL_PACK_PATH,
                strerror(errno));
        return 0;
    }
    fclose(model_file);

    KokoroError error = {0};
    KokoroPack *pack = NULL;
    KokoroStatus status = kokoro_pack_open(MODEL_PACK_PATH, &pack, &error);
    if (!expect_status(status, KOKORO_OK, &error, "open model pack")) {
        return 0;
    }
    KokoroTextEncoder *encoder = NULL;
    status = kokoro_text_encoder_create(pack, &encoder, &error);
    if (!expect_status(status, KOKORO_OK, &error, "bind text encoder weights")) {
        kokoro_pack_close(pack);
        return 0;
    }
    int valid = test_reference_parity(encoder) && test_padding_and_single_token(encoder) &&
                test_invalid_inputs(encoder);
    kokoro_text_encoder_free(encoder);
    kokoro_pack_close(pack);
    return valid;
}

int main(void) { return model_backed_tests() ? 0 : 1; }

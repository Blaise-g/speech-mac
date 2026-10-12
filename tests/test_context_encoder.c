#include "context_encoder.h"
#include "kokoro.h"
#include "model_pack.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MODEL_PACK_PATH "models/kokoro-v1_0-af_heart.kpack"
#define CONTEXT_FIXTURE_PATH "reference/fixtures/contextual_features.f32le"
#define HELLO_LENGTH 8U
#define PADDED_LENGTH 10U
#define FEATURE_COUNT 512U
#define STAGE_ATOL 1.0e-5f
#define STAGE_RTOL 1.0e-3f

static int fail(const char *message) {
    fprintf(stderr, "test_context_encoder: %s\n", message);
    return 1;
}

static int compare_fixture(const float *actual, size_t stride, const float *expected,
                           size_t *worst_channel, size_t *worst_token, float *max_difference) {
    *max_difference = 0.0f;
    for (size_t channel = 0; channel < FEATURE_COUNT; channel++) {
        for (size_t token = 0; token < HELLO_LENGTH; token++) {
            float reference = expected[channel * HELLO_LENGTH + token];
            float difference = fabsf(actual[channel * stride + token] - reference);
            float tolerance = STAGE_ATOL + STAGE_RTOL * fabsf(reference);
            if (difference > *max_difference) {
                *max_difference = difference;
                *worst_channel = channel;
                *worst_token = token;
            }
            if (!isfinite(difference) || difference > tolerance) {
                return 0;
            }
        }
    }
    return 1;
}

static int expect_status(KokoroStatus actual, KokoroStatus expected, const KokoroError *error,
                         const char *operation) {
    if (actual == expected) {
        return 1;
    }
    fprintf(stderr, "test_context_encoder: %s returned %s (%s), expected %s\n", operation,
            kokoro_status_name(actual), error->message, kokoro_status_name(expected));
    return 0;
}

int main(void) {
    FILE *model_file = fopen(MODEL_PACK_PATH, "rb");
    if (model_file == NULL) {
        puts("SKIP context parity: convert the pinned model pack to run the weight-backed fixture "
             "test");
        return 0;
    }
    fclose(model_file);

    FILE *fixture_file = fopen(CONTEXT_FIXTURE_PATH, "rb");
    if (fixture_file == NULL) {
        return fail("could not open the raw contextual feature fixture");
    }
    float expected[FEATURE_COUNT * HELLO_LENGTH];
    size_t expected_bytes = sizeof(expected);
    if (fread(expected, 1, expected_bytes, fixture_file) != expected_bytes ||
        fgetc(fixture_file) != EOF) {
        fclose(fixture_file);
        return fail("contextual fixture has the wrong byte length");
    }
    fclose(fixture_file);

    KokoroError error = {0};
    KokoroPack *pack = NULL;
    KokoroStatus status = kokoro_pack_open(MODEL_PACK_PATH, &pack, &error);
    if (!expect_status(status, KOKORO_OK, &error, "open pinned model pack")) {
        return 1;
    }
    KokoroContextEncoder *encoder = NULL;
    status = kokoro_context_encoder_create(pack, &encoder, &error);
    if (!expect_status(status, KOKORO_OK, &error, "load ALBERT context weights")) {
        kokoro_pack_close(pack);
        return 1;
    }

    const int64_t hello_ids[HELLO_LENGTH] = {0, 50, 83, 156, 54, 57, 135, 0};
    float hello_output[FEATURE_COUNT * HELLO_LENGTH];
    status = kokoro_context_encode(encoder, hello_ids, HELLO_LENGTH, HELLO_LENGTH, hello_output,
                                   FEATURE_COUNT * HELLO_LENGTH, &error);
    if (!expect_status(status, KOKORO_OK, &error, "encode hello fixture IDs")) {
        kokoro_context_encoder_free(encoder);
        kokoro_pack_close(pack);
        return 1;
    }
    size_t worst_channel = 0;
    size_t worst_token = 0;
    float max_difference = 0.0f;
    if (!compare_fixture(hello_output, HELLO_LENGTH, expected, &worst_channel, &worst_token,
                         &max_difference)) {
        fprintf(stderr,
                "test_context_encoder: stage fixture miss at channel %zu token %zu; max abs %.9g "
                "(atol %.1g, rtol %.1g)\n",
                worst_channel, worst_token, max_difference, STAGE_ATOL, STAGE_RTOL);
        kokoro_context_encoder_free(encoder);
        kokoro_pack_close(pack);
        return 1;
    }

    int64_t padded_ids_a[PADDED_LENGTH] = {0, 50, 83, 156, 54, 57, 135, 0, 1, 2};
    int64_t padded_ids_b[PADDED_LENGTH] = {0, 50, 83, 156, 54, 57, 135, 0, 3, 177};
    float padded_output[FEATURE_COUNT * PADDED_LENGTH];
    status = kokoro_context_encode(encoder, padded_ids_a, PADDED_LENGTH, HELLO_LENGTH,
                                   padded_output, FEATURE_COUNT * PADDED_LENGTH, &error);
    if (!expect_status(status, KOKORO_OK, &error, "encode padded sequence")) {
        kokoro_context_encoder_free(encoder);
        kokoro_pack_close(pack);
        return 1;
    }
    if (!compare_fixture(padded_output, PADDED_LENGTH, expected, &worst_channel, &worst_token,
                         &max_difference)) {
        fprintf(stderr, "test_context_encoder: padding mask changed valid token %zu:%zu\n",
                worst_channel, worst_token);
        kokoro_context_encoder_free(encoder);
        kokoro_pack_close(pack);
        return 1;
    }
    status = kokoro_context_encode(encoder, padded_ids_b, PADDED_LENGTH, HELLO_LENGTH,
                                   padded_output, FEATURE_COUNT * PADDED_LENGTH, &error);
    if (!expect_status(status, KOKORO_OK, &error, "encode alternate padded sequence") ||
        !compare_fixture(padded_output, PADDED_LENGTH, expected, &worst_channel, &worst_token,
                         &max_difference)) {
        fprintf(stderr, "test_context_encoder: masked padding IDs affected valid token %zu:%zu\n",
                worst_channel, worst_token);
        kokoro_context_encoder_free(encoder);
        kokoro_pack_close(pack);
        return 1;
    }

    float sentinel_output[FEATURE_COUNT * HELLO_LENGTH];
    for (size_t index = 0; index < FEATURE_COUNT * HELLO_LENGTH; index++) {
        sentinel_output[index] = 123.5f;
    }
    int64_t invalid_ids[HELLO_LENGTH] = {0, 50, 83, 178, 54, 57, 135, 0};
    status = kokoro_context_encode(encoder, invalid_ids, HELLO_LENGTH, HELLO_LENGTH,
                                   sentinel_output, FEATURE_COUNT * HELLO_LENGTH, &error);
    if (!expect_status(status, KOKORO_INVALID_TOKEN_ID, &error, "reject token ID 178") ||
        sentinel_output[0] != 123.5f) {
        kokoro_context_encoder_free(encoder);
        kokoro_pack_close(pack);
        return fail("invalid token ID was not rejected without modifying output");
    }
    invalid_ids[3] = -1;
    status = kokoro_context_encode(encoder, invalid_ids, HELLO_LENGTH, HELLO_LENGTH,
                                   sentinel_output, FEATURE_COUNT * HELLO_LENGTH, &error);
    if (!expect_status(status, KOKORO_INVALID_TOKEN_ID, &error, "reject negative token ID")) {
        kokoro_context_encoder_free(encoder);
        kokoro_pack_close(pack);
        return 1;
    }

    int64_t overlong_ids[KOKORO_CONTEXT_MAX_TOKENS + 1] = {0};
    status = kokoro_context_encode(encoder, overlong_ids, KOKORO_CONTEXT_MAX_TOKENS + 1,
                                   KOKORO_CONTEXT_MAX_TOKENS + 1, sentinel_output,
                                   FEATURE_COUNT * HELLO_LENGTH, &error);
    if (!expect_status(status, KOKORO_CONTEXT_TOO_LONG, &error, "reject context length 513")) {
        kokoro_context_encoder_free(encoder);
        kokoro_pack_close(pack);
        return 1;
    }
    status = kokoro_context_encode(encoder, hello_ids, HELLO_LENGTH, HELLO_LENGTH, sentinel_output,
                                   FEATURE_COUNT * HELLO_LENGTH - 1, &error);
    if (!expect_status(status, KOKORO_BUFFER_TOO_SMALL, &error, "reject short output buffer")) {
        kokoro_context_encoder_free(encoder);
        kokoro_pack_close(pack);
        return 1;
    }
    status = kokoro_context_encode(encoder, hello_ids, HELLO_LENGTH, HELLO_LENGTH + 1,
                                   sentinel_output, FEATURE_COUNT * HELLO_LENGTH, &error);
    if (!expect_status(status, KOKORO_CONTEXT_TOO_LONG, &error,
                       "reject valid length past sequence")) {
        kokoro_context_encoder_free(encoder);
        kokoro_pack_close(pack);
        return 1;
    }

    kokoro_context_encoder_free(encoder);
    kokoro_pack_close(pack);
    printf("context fixture passed (max abs %.9g; atol %.1g, rtol %.1g)\n", max_difference,
           STAGE_ATOL, STAGE_RTOL);
    return 0;
}

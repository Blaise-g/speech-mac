#include "synthesis.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MODEL_PACK_PATH "models/kokoro-v1_0-af_heart.kpack"
#define AM_ADAM_PATH "models/voices/am_adam.kvoice"

static int fail(const char *message) {
    fprintf(stderr, "test_engine: %s\n", message);
    return 0;
}

static int expect_status(KokoroStatus actual, KokoroStatus expected, const KokoroError *error,
                         const char *operation) {
    if (actual == expected) {
        return 1;
    }
    fprintf(stderr, "test_engine: %s returned %s (%s), expected %s\n", operation,
            kokoro_status_name(actual), error->message, kokoro_status_name(expected));
    return 0;
}

static int same_output(const KokoroSynthesisOutput *left, const KokoroSynthesisOutput *right) {
    return left->token_count == right->token_count && left->frame_count == right->frame_count &&
           left->sample_count == right->sample_count && left->durations != NULL &&
           right->durations != NULL && left->samples != NULL && right->samples != NULL &&
           memcmp(left->durations, right->durations,
                  left->token_count * sizeof(*left->durations)) == 0 &&
           memcmp(left->samples, right->samples, left->sample_count * sizeof(*left->samples)) == 0;
}

static int waveforms_differ(const KokoroSynthesisOutput *left, const KokoroSynthesisOutput *right) {
    return left->sample_count != right->sample_count ||
           memcmp(left->samples, right->samples, left->sample_count * sizeof(*left->samples)) != 0;
}

static int empty_output(const KokoroSynthesisOutput *output) {
    return output->token_count == 0 && output->frame_count == 0 && output->sample_count == 0 &&
           output->durations == NULL && output->samples == NULL;
}

int main(void) {
    if (access(MODEL_PACK_PATH, R_OK) != 0) {
        puts("SKIP resident engine checks: convert the pinned model pack first");
        return 0;
    }

    KokoroError error = {0};
    KokoroEngine *engine = NULL;
    KokoroStatus status = kokoro_engine_create(MODEL_PACK_PATH, "unknown", NULL, &engine, &error);
    if (!expect_status(status, KOKORO_INVALID_ARGUMENT, &error, "reject unknown engine voice") ||
        engine != NULL) {
        return fail("unknown engine voice returned an engine");
    }
    status = kokoro_engine_create(MODEL_PACK_PATH, "am_adam", "/no/such/am_adam.kvoice", &engine,
                                  &error);
    if (!expect_status(status, KOKORO_IO_ERROR, &error, "reject missing voice artifact") ||
        engine != NULL) {
        return fail("missing voice artifact returned an engine");
    }

    KokoroPack *pack = NULL;
    status = kokoro_pack_open(MODEL_PACK_PATH, &pack, &error);
    if (!expect_status(status, KOKORO_OK, &error, "open model pack")) {
        return 1;
    }
    status = kokoro_engine_create(MODEL_PACK_PATH, "af_heart", NULL, &engine, &error);
    if (!expect_status(status, KOKORO_OK, &error, "create af_heart resident engine")) {
        kokoro_pack_close(pack);
        return 1;
    }

    const int64_t hello_ids[] = {0, 50, 83, 156, 54, 57, 135, 0};
    const int64_t longer_ids[] = {0, 50, 83, 156, 54, 57, 135, 16, 50, 83, 156, 54, 57, 135, 0};
    const int64_t invalid_ids[] = {0, INT64_MAX};
    KokoroSynthesisOptions hello_options = {
        .voice_style_index = 5,
        .speed = 1.0f,
        .random_seed = 104729,
    };
    KokoroSynthesisOptions longer_options = {
        .voice_style_index = 12,
        .speed = 1.0f,
        .random_seed = 104729,
    };
    KokoroSynthesisOutput engine_first = {0};
    KokoroSynthesisOutput engine_longer = {0};
    KokoroSynthesisOutput engine_after_error = {0};
    KokoroSynthesisOutput direct_hello = {0};
    KokoroSynthesisOutput direct_longer = {0};
    status = kokoro_engine_synthesize(engine, hello_ids, sizeof(hello_ids) / sizeof(hello_ids[0]),
                                      &hello_options, &engine_first, &error);
    int passed = expect_status(status, KOKORO_OK, &error, "synthesize hello with resident engine");
    if (passed) {
        status = kokoro_synthesize(pack, hello_ids, sizeof(hello_ids) / sizeof(hello_ids[0]),
                                   &hello_options, &direct_hello, &error);
        passed = expect_status(status, KOKORO_OK, &error, "synthesize hello through direct seam") &&
                 same_output(&engine_first, &direct_hello);
        if (!passed) {
            fail("resident engine hello differs from the direct inference seam");
        }
    }
    if (passed) {
        status =
            kokoro_engine_synthesize(engine, longer_ids, sizeof(longer_ids) / sizeof(longer_ids[0]),
                                     &longer_options, &engine_longer, &error);
        passed = expect_status(status, KOKORO_OK, &error, "synthesize a second request") &&
                 engine_longer.sample_count != engine_first.sample_count;
        if (passed) {
            status = kokoro_synthesize(pack, longer_ids, sizeof(longer_ids) / sizeof(longer_ids[0]),
                                       &longer_options, &direct_longer, &error);
            passed = expect_status(status, KOKORO_OK, &error,
                                   "synthesize second request through direct seam") &&
                     same_output(&engine_longer, &direct_longer);
        }
        if (!passed) {
            fail("second resident request differed from direct inference");
        }
    }
    KokoroSynthesisOutput failed_output = {0};
    if (passed) {
        status = kokoro_engine_synthesize(engine, invalid_ids,
                                          sizeof(invalid_ids) / sizeof(invalid_ids[0]),
                                          &hello_options, &failed_output, &error);
        passed = expect_status(status, KOKORO_INVALID_TOKEN_ID, &error,
                               "reject invalid resident request") &&
                 empty_output(&failed_output);
        if (!passed) {
            fail("failed resident request returned partial output");
        }
    }
    if (passed) {
        status =
            kokoro_engine_synthesize(engine, hello_ids, sizeof(hello_ids) / sizeof(hello_ids[0]),
                                     &hello_options, &engine_after_error, &error);
        passed = expect_status(status, KOKORO_OK, &error,
                               "synthesize after a failed resident request") &&
                 same_output(&engine_first, &engine_after_error);
        if (!passed) {
            fail("failed request left state that changed the following output");
        }
    }

    KokoroEngine *adam_engine = NULL;
    KokoroSynthesisOutput adam_first = {0};
    KokoroSynthesisOutput adam_repeat = {0};
    if (passed && access(AM_ADAM_PATH, R_OK) == 0) {
        status =
            kokoro_engine_create(MODEL_PACK_PATH, "am_adam", AM_ADAM_PATH, &adam_engine, &error);
        passed = expect_status(status, KOKORO_OK, &error, "create am_adam resident engine");
        if (passed) {
            status = kokoro_engine_synthesize(adam_engine, hello_ids,
                                              sizeof(hello_ids) / sizeof(hello_ids[0]),
                                              &hello_options, &adam_first, &error);
            passed = expect_status(status, KOKORO_OK, &error, "synthesize with am_adam");
        }
        if (passed) {
            status = kokoro_engine_synthesize(adam_engine, hello_ids,
                                              sizeof(hello_ids) / sizeof(hello_ids[0]),
                                              &hello_options, &adam_repeat, &error);
            passed = expect_status(status, KOKORO_OK, &error, "repeat am_adam synthesis") &&
                     same_output(&adam_first, &adam_repeat) &&
                     waveforms_differ(&adam_first, &engine_first);
            if (!passed) {
                fail("am_adam output was unstable or reused af_heart style data");
            }
        }
    } else if (passed) {
        puts("SKIP am_adam engine checks: convert the pinned am_adam voice artifact first");
    }

    kokoro_synthesis_output_free(&adam_repeat);
    kokoro_synthesis_output_free(&adam_first);
    kokoro_engine_free(adam_engine);
    kokoro_synthesis_output_free(&engine_after_error);
    kokoro_synthesis_output_free(&failed_output);
    kokoro_synthesis_output_free(&direct_longer);
    kokoro_synthesis_output_free(&direct_hello);
    kokoro_synthesis_output_free(&engine_longer);
    kokoro_synthesis_output_free(&engine_first);
    kokoro_engine_free(engine);
    kokoro_pack_close(pack);
    if (passed) {
        puts("resident engine repeated-request parity and voice checks passed");
    }
    return passed ? 0 : 1;
}

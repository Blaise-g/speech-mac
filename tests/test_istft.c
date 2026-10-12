#include "istft.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define FIXTURE_FRAMES 6601U
#define FIXTURE_SAMPLES 33000U
#define FIXTURE_SPECTRUM_COUNT (KOKORO_ISTFT_SPECTRAL_BINS * FIXTURE_FRAMES)
/* Fixed at ticket 01 from reference repeatability and float32 precision, in budgets.json. */
#define WAVEFORM_NORMALIZED_RMSE_MAX 1.0e-3
#define WAVEFORM_MAX_ABS_ERROR 2.0e-3

static int fail(const char *message) {
    fprintf(stderr, "test_istft: %s\n", message);
    return 0;
}

static int expect_status(KokoroStatus actual, KokoroStatus expected, const KokoroError *error,
                         const char *operation) {
    if (actual == expected) {
        return 1;
    }
    fprintf(stderr, "test_istft: %s returned %s (%s), expected %s\n", operation,
            kokoro_status_name(actual), error->message, kokoro_status_name(expected));
    return 0;
}

static int read_fixture(const char *path, float *output, size_t count) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "test_istft: cannot open fixture %s\n", path);
        return 0;
    }
    int valid = fread(output, sizeof(float), count, file) == count && fgetc(file) == EOF;
    fclose(file);
    if (!valid) {
        fprintf(stderr, "test_istft: fixture has the wrong size: %s\n", path);
    }
    return valid;
}

static int compare_waveform(const float *actual, const float *expected, size_t count) {
    double squared_error = 0.0;
    double squared_reference = 0.0;
    double maximum_error = 0.0;
    for (size_t sample = 0; sample < count; sample++) {
        double difference = (double)actual[sample] - expected[sample];
        double absolute_error = fabs(difference);
        if (!isfinite(actual[sample]) || absolute_error > maximum_error) {
            maximum_error = absolute_error;
        }
        squared_error += difference * difference;
        squared_reference += (double)expected[sample] * expected[sample];
    }
    double normalized_rmse = sqrt(squared_error / count) / sqrt(squared_reference / count);
    if (!isfinite(normalized_rmse) || normalized_rmse > WAVEFORM_NORMALIZED_RMSE_MAX ||
        maximum_error > WAVEFORM_MAX_ABS_ERROR) {
        fprintf(stderr,
                "test_istft: final waveform misses fixed budget (normalized RMSE %.9g, max "
                "absolute error %.9g; limits %.1g and %.1g)\n",
                normalized_rmse, maximum_error, WAVEFORM_NORMALIZED_RMSE_MAX,
                WAVEFORM_MAX_ABS_ERROR);
        return 0;
    }
    printf("official spectrum-to-waveform fixture passed (normalized RMSE %.9g, max absolute "
           "error %.9g)\n",
           normalized_rmse, maximum_error);
    return 1;
}

static int test_official_fixture(void) {
    float *magnitude = malloc(FIXTURE_SPECTRUM_COUNT * sizeof(float));
    float *phase = malloc(FIXTURE_SPECTRUM_COUNT * sizeof(float));
    float *expected = malloc(FIXTURE_SAMPLES * sizeof(float));
    if (magnitude == NULL || phase == NULL || expected == NULL) {
        free(magnitude);
        free(phase);
        free(expected);
        return fail("cannot allocate fixed ISTFT fixture inputs");
    }
    int valid =
        read_fixture("reference/fixtures/spectral_magnitude.f32le", magnitude,
                     FIXTURE_SPECTRUM_COUNT) &&
        read_fixture("reference/fixtures/spectral_phase.f32le", phase, FIXTURE_SPECTRUM_COUNT) &&
        read_fixture("reference/fixtures/waveform.f32le", expected, FIXTURE_SAMPLES);
    if (!valid) {
        free(magnitude);
        free(phase);
        free(expected);
        return 0;
    }

    KokoroWaveformOutput output = {0};
    KokoroError error = {0};
    KokoroStatus status =
        kokoro_istft_reconstruct(magnitude, phase, FIXTURE_FRAMES, &output, &error);
    valid = expect_status(status, KOKORO_OK, &error, "reconstruct the official spectrum") &&
            output.sample_count == FIXTURE_SAMPLES && output.samples != NULL;
    for (size_t sample = 0; sample < output.sample_count && valid; sample++) {
        valid = isfinite(output.samples[sample]);
    }
    if (valid) {
        valid = KOKORO_ISTFT_SAMPLE_RATE_HZ == 24000U &&
                compare_waveform(output.samples, expected, FIXTURE_SAMPLES);
    }
    kokoro_waveform_output_free(&output);
    free(magnitude);
    free(phase);
    free(expected);
    return valid ? 1 : fail("official output shape, rate, or finite-value check failed");
}

static int test_two_frame_edge(void) {
    float magnitude[KOKORO_ISTFT_SPECTRAL_BINS * 2] = {0};
    float phase[KOKORO_ISTFT_SPECTRAL_BINS * 2] = {0};
    KokoroWaveformOutput output = {0};
    KokoroError error = {0};
    KokoroStatus status = kokoro_istft_reconstruct(magnitude, phase, 2, &output, &error);
    if (!expect_status(status, KOKORO_OK, &error, "reconstruct two edge frames")) {
        kokoro_waveform_output_free(&output);
        return 0;
    }
    int valid = output.sample_count == KOKORO_ISTFT_HOP_SIZE && output.samples != NULL;
    for (size_t sample = 0; sample < output.sample_count && valid; sample++) {
        valid = isfinite(output.samples[sample]) && output.samples[sample] == 0.0f;
    }
    kokoro_waveform_output_free(&output);
    return valid ? 1 : fail("two-frame output length or finite sample check failed");
}

static int test_invalid_inputs(void) {
    float magnitude[KOKORO_ISTFT_SPECTRAL_BINS * 2] = {0};
    float phase[KOKORO_ISTFT_SPECTRAL_BINS * 2] = {0};
    KokoroWaveformOutput output = {0};
    KokoroError error = {0};
    KokoroStatus status = kokoro_istft_reconstruct(magnitude, phase, 0, &output, &error);
    if (!expect_status(status, KOKORO_INVALID_ARGUMENT, &error, "reject zero frames")) {
        return 0;
    }
    status = kokoro_istft_reconstruct(magnitude, phase, 1, &output, &error);
    if (!expect_status(status, KOKORO_INVALID_ARGUMENT, &error, "reject one empty-output frame")) {
        return 0;
    }
    status = kokoro_istft_reconstruct(NULL, phase, 2, &output, &error);
    if (!expect_status(status, KOKORO_INVALID_ARGUMENT, &error, "reject a missing magnitude")) {
        return 0;
    }
    status = kokoro_istft_reconstruct(magnitude, NULL, 2, &output, &error);
    if (!expect_status(status, KOKORO_INVALID_ARGUMENT, &error, "reject a missing phase")) {
        return 0;
    }
    status = kokoro_istft_reconstruct(magnitude, phase, SIZE_MAX, &output, &error);
    if (!expect_status(status, KOKORO_INVALID_ARGUMENT, &error, "reject overflowing frame count")) {
        return 0;
    }
    magnitude[0] = -1.0f;
    status = kokoro_istft_reconstruct(magnitude, phase, 2, &output, &error);
    if (!expect_status(status, KOKORO_INVALID_ARGUMENT, &error, "reject negative magnitude")) {
        return 0;
    }
    magnitude[0] = NAN;
    status = kokoro_istft_reconstruct(magnitude, phase, 2, &output, &error);
    return expect_status(status, KOKORO_NUMERIC_ERROR, &error, "reject non-finite magnitude") ? 1
                                                                                              : 0;
}

static int test_success_clears_prior_error(void) {
    float magnitude[KOKORO_ISTFT_SPECTRAL_BINS * 2] = {0};
    float phase[KOKORO_ISTFT_SPECTRAL_BINS * 2] = {0};
    KokoroWaveformOutput output = {0};
    KokoroError error = {0};
    KokoroStatus status = kokoro_istft_reconstruct(magnitude, phase, 1, &output, &error);
    if (!expect_status(status, KOKORO_INVALID_ARGUMENT, &error, "seed a prior ISTFT error")) {
        return 0;
    }

    status = kokoro_istft_reconstruct(magnitude, phase, 2, &output, &error);
    int valid = expect_status(status, KOKORO_OK, &error, "reconstruct after an error") &&
                error.code == KOKORO_OK && error.message[0] == '\0';
    kokoro_waveform_output_free(&output);
    return valid ? 1 : fail("successful reconstruction left a stale error");
}

int main(void) {
    return test_official_fixture() && test_two_frame_edge() && test_invalid_inputs() &&
                   test_success_clears_prior_error()
               ? 0
               : 1;
}

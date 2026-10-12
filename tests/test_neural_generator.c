#include "decoder_conditioner.h"
#include "kokoro.h"
#include "model_pack.h"
#include "neural_generator.h"
#include "pitch_noise_predictor.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MODEL_PACK_PATH "models/kokoro-v1_0-af_heart.kpack"
#define DURATION_FRAMES 55U
#define FEATURE_FRAMES (DURATION_FRAMES * 2U)
#define FEATURE_COUNT (512U * FEATURE_FRAMES)
#define EXCITATION_SAMPLES (300U * FEATURE_FRAMES)
#define SPECTRAL_FRAMES (60U * FEATURE_FRAMES + 1U)
#define SPECTRAL_COUNT (KOKORO_NEURAL_GENERATOR_SPECTRAL_BINS * SPECTRAL_FRAMES)
#define STAGE_ATOL 1.0e-5f
#define STAGE_RTOL 1.0e-3f

static int fail(const char *message) {
    fprintf(stderr, "test_neural_generator: %s\n", message);
    return 0;
}

static int expect_status(KokoroStatus actual, KokoroStatus expected, const KokoroError *error,
                         const char *operation) {
    if (actual == expected) {
        return 1;
    }
    fprintf(stderr, "test_neural_generator: %s returned %s (%s), expected %s\n", operation,
            kokoro_status_name(actual), error->message, kokoro_status_name(expected));
    return 0;
}

static int read_fixture(const char *path, float *output, size_t count) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "test_neural_generator: cannot open fixture %s\n", path);
        return 0;
    }
    int valid = fread(output, sizeof(float), count, file) == count && fgetc(file) == EOF;
    fclose(file);
    if (!valid) {
        fprintf(stderr, "test_neural_generator: fixture has the wrong size: %s\n", path);
    }
    return valid;
}

static int compare_spectrum(const char *name, const float *actual, const float *expected,
                            size_t count) {
    float maximum = 0.0f;
    size_t failing_count = 0;
    size_t first_failure = 0;
    for (size_t index = 0; index < count; index++) {
        float difference = fabsf(actual[index] - expected[index]);
        if (difference > maximum) {
            maximum = difference;
        }
        float tolerance = STAGE_ATOL + STAGE_RTOL * fabsf(expected[index]);
        if (!isfinite(difference) || difference > tolerance) {
            if (failing_count == 0) {
                first_failure = index;
            }
            failing_count++;
        }
    }
    if (failing_count != 0) {
        float difference = fabsf(actual[first_failure] - expected[first_failure]);
        float tolerance = STAGE_ATOL + STAGE_RTOL * fabsf(expected[first_failure]);
        fprintf(stderr,
                "test_neural_generator: %s fixture miss at %zu of %zu elements; first at %zu: "
                "actual %.9g, expected %.9g, difference %.9g exceeds %.9g; max abs %.9g\n",
                name, failing_count, count, first_failure, actual[first_failure],
                expected[first_failure], difference, tolerance, maximum);
        return 0;
    }
    printf("%s fixture passed (max abs %.9g; atol %.1g, rtol %.1g)\n", name, maximum, STAGE_ATOL,
           STAGE_RTOL);
    return 1;
}

static int compare_complex_spectrogram(const KokoroNeuralGeneratorSourceOutput *actual,
                                       const float *expected_magnitude, const float *expected_phase,
                                       size_t count) {
    size_t failures = 0;
    size_t first_failure = 0;
    double maximum = 0.0;
    for (size_t index = 0; index < count; index++) {
        double actual_real = actual->magnitude[index] * cos((double)actual->phase[index]);
        double actual_imaginary = actual->magnitude[index] * sin((double)actual->phase[index]);
        double expected_real = expected_magnitude[index] * cos((double)expected_phase[index]);
        double expected_imaginary = expected_magnitude[index] * sin((double)expected_phase[index]);
        double difference =
            hypot(actual_real - expected_real, actual_imaginary - expected_imaginary);
        double tolerance = STAGE_ATOL + STAGE_RTOL * expected_magnitude[index];
        if (difference > maximum) {
            maximum = difference;
        }
        if (!isfinite(difference) || difference > tolerance) {
            if (failures == 0) {
                first_failure = index;
            }
            failures++;
        }
    }
    if (failures != 0) {
        fprintf(stderr,
                "test_neural_generator: native source complex STFT misses %zu/%zu bins; first "
                "at %zu (complex error %.9g exceeds %.9g); max error %.9g\n",
                failures, count, first_failure,
                hypot((double)actual->magnitude[first_failure] *
                              cos((double)actual->phase[first_failure]) -
                          (double)expected_magnitude[first_failure] *
                              cos((double)expected_phase[first_failure]),
                      (double)actual->magnitude[first_failure] *
                              sin((double)actual->phase[first_failure]) -
                          (double)expected_magnitude[first_failure] *
                              sin((double)expected_phase[first_failure])),
                STAGE_ATOL + STAGE_RTOL * expected_magnitude[first_failure], maximum);
        return 0;
    }
    printf("native source complex STFT passed (max complex error %.9g)\n", maximum);
    return 1;
}

static int check_finite_spectrum(const KokoroSpectralOutput *output, size_t count) {
    for (size_t index = 0; index < count; index++) {
        if (!isfinite(output->magnitude[index]) || output->magnitude[index] < 0.0f ||
            !isfinite(output->phase[index])) {
            fprintf(stderr, "test_neural_generator: invalid spectral value at element %zu\n",
                    index);
            return 0;
        }
    }
    return 1;
}

static int output_is_empty(const KokoroSpectralOutput *output) {
    return output->input_frame_count == 0 && output->frame_count == 0 &&
           output->magnitude == NULL && output->phase == NULL;
}

static int test_fixture_parity(const KokoroNeuralGenerator *generator) {
    float *features = malloc(FEATURE_COUNT * sizeof(float));
    float *f0 = malloc(FEATURE_FRAMES * sizeof(float));
    float *noise = malloc(FEATURE_FRAMES * sizeof(float));
    float *harmonic = malloc(EXCITATION_SAMPLES * sizeof(float));
    float *source_noise = malloc(EXCITATION_SAMPLES * sizeof(float));
    float *voiced = malloc(EXCITATION_SAMPLES * sizeof(float));
    float *source_magnitude = malloc(SPECTRAL_COUNT * sizeof(float));
    float *source_phase = malloc(SPECTRAL_COUNT * sizeof(float));
    float *expected_magnitude = malloc(SPECTRAL_COUNT * sizeof(float));
    float *expected_phase = malloc(SPECTRAL_COUNT * sizeof(float));
    if (features == NULL || f0 == NULL || noise == NULL || harmonic == NULL ||
        source_noise == NULL || voiced == NULL || source_magnitude == NULL ||
        source_phase == NULL || expected_magnitude == NULL || expected_phase == NULL) {
        free(features);
        free(f0);
        free(noise);
        free(harmonic);
        free(source_noise);
        free(voiced);
        free(source_magnitude);
        free(source_phase);
        free(expected_magnitude);
        free(expected_phase);
        return fail("cannot allocate neural generator fixture inputs");
    }
    int valid =
        read_fixture("reference/fixtures/decoder_features.f32le", features, FEATURE_COUNT) &&
        read_fixture("reference/fixtures/f0.f32le", f0, FEATURE_FRAMES) &&
        read_fixture("reference/fixtures/noise.f32le", noise, FEATURE_FRAMES) &&
        read_fixture("reference/fixtures/excitation_harmonic.f32le", harmonic,
                     EXCITATION_SAMPLES) &&
        read_fixture("reference/fixtures/excitation_noise.f32le", source_noise,
                     EXCITATION_SAMPLES) &&
        read_fixture("reference/fixtures/excitation_voiced.f32le", voiced, EXCITATION_SAMPLES) &&
        read_fixture("reference/fixtures/source_spectrogram_magnitude.f32le", source_magnitude,
                     SPECTRAL_COUNT) &&
        read_fixture("reference/fixtures/source_spectrogram_phase.f32le", source_phase,
                     SPECTRAL_COUNT) &&
        read_fixture("reference/fixtures/spectral_magnitude.f32le", expected_magnitude,
                     SPECTRAL_COUNT) &&
        read_fixture("reference/fixtures/spectral_phase.f32le", expected_phase, SPECTRAL_COUNT);
    if (!valid) {
        free(features);
        free(f0);
        free(noise);
        free(harmonic);
        free(source_noise);
        free(voiced);
        free(source_magnitude);
        free(source_phase);
        free(expected_magnitude);
        free(expected_phase);
        return 0;
    }

    KokoroDecoderOutput decoder = {.duration_frame_count = DURATION_FRAMES,
                                   .feature_frame_count = FEATURE_FRAMES,
                                   .features = features};
    KokoroPitchNoiseOutput pitch_noise = {.duration_frame_count = DURATION_FRAMES,
                                          .prediction_frame_count = FEATURE_FRAMES,
                                          .f0 = f0,
                                          .noise = noise};
    KokoroNeuralGeneratorExcitation excitation = {.sample_count = EXCITATION_SAMPLES,
                                                  .harmonic = harmonic,
                                                  .noise = source_noise,
                                                  .voiced = voiced};
    /* Compare learned stages with identical source-STFT input; native STFT is tested separately
       by magnitude and complex-bin equivalence because angle branches at ±pi are equivalent. */
    KokoroNeuralGeneratorSourceSpectrogram source_spectrogram = {
        .frame_count = SPECTRAL_FRAMES, .magnitude = source_magnitude, .phase = source_phase};
    KokoroNeuralGeneratorOptions options = {
        .random_seed = 0, .excitation = &excitation, .source_spectrogram = &source_spectrogram};
    KokoroSpectralOutput output = {0};
    KokoroError error = {0};
    KokoroStatus status =
        kokoro_neural_generate(generator, &decoder, &pitch_noise, 5, &options, &output, &error);
    valid = expect_status(status, KOKORO_OK, &error, "generate pinned spectral fixture") &&
            output.input_frame_count == FEATURE_FRAMES && output.frame_count == SPECTRAL_FRAMES &&
            output.magnitude != NULL && output.phase != NULL;
    if (valid) {
        valid = check_finite_spectrum(&output, SPECTRAL_COUNT);
        int magnitude_match = compare_spectrum("spectral magnitude", output.magnitude,
                                               expected_magnitude, SPECTRAL_COUNT);
        int phase_match =
            compare_spectrum("spectral phase", output.phase, expected_phase, SPECTRAL_COUNT);
        valid = valid && magnitude_match && phase_match;
    }
    kokoro_spectral_output_free(&output);

    size_t original_sample_count = excitation.sample_count;
    excitation.sample_count--;
    memset(&error, 0, sizeof(error));
    status =
        kokoro_neural_generate(generator, &decoder, &pitch_noise, 5, &options, &output, &error);
    valid = valid &&
            expect_status(status, KOKORO_INVALID_ARGUMENT, &error,
                          "reject captured excitation length mismatch") &&
            output_is_empty(&output);
    excitation.sample_count = original_sample_count;

    source_spectrogram.frame_count--;
    memset(&error, 0, sizeof(error));
    status =
        kokoro_neural_generate(generator, &decoder, &pitch_noise, 5, &options, &output, &error);
    valid = valid &&
            expect_status(status, KOKORO_INVALID_ARGUMENT, &error,
                          "reject captured source spectrogram length mismatch") &&
            output_is_empty(&output);
    source_spectrogram.frame_count++;

    float saved_source_magnitude = source_magnitude[0];
    source_magnitude[0] = NAN;
    memset(&error, 0, sizeof(error));
    status =
        kokoro_neural_generate(generator, &decoder, &pitch_noise, 5, &options, &output, &error);
    valid = valid &&
            expect_status(status, KOKORO_NUMERIC_ERROR, &error,
                          "reject non-finite captured source spectrogram") &&
            output_is_empty(&output);
    source_magnitude[0] = saved_source_magnitude;

    float saved_feature = decoder.features[0];
    decoder.features[0] = NAN;
    memset(&error, 0, sizeof(error));
    status =
        kokoro_neural_generate(generator, &decoder, &pitch_noise, 5, &options, &output, &error);
    valid =
        valid &&
        expect_status(status, KOKORO_NUMERIC_ERROR, &error, "reject non-finite decoder feature") &&
        output_is_empty(&output);
    decoder.features[0] = saved_feature;

    free(features);
    free(f0);
    free(noise);
    free(harmonic);
    free(source_noise);
    free(voiced);
    free(source_magnitude);
    free(source_phase);
    free(expected_magnitude);
    free(expected_phase);
    if (valid) {
        puts("captured-excitation model-backed generator parity and input boundaries passed");
    }
    return valid;
}

static int test_native_source_spectrogram(void) {
    float *harmonic = malloc(EXCITATION_SAMPLES * sizeof(float));
    float *expected_magnitude = malloc(SPECTRAL_COUNT * sizeof(float));
    float *expected_phase = malloc(SPECTRAL_COUNT * sizeof(float));
    if (harmonic == NULL || expected_magnitude == NULL || expected_phase == NULL) {
        free(harmonic);
        free(expected_magnitude);
        free(expected_phase);
        return fail("cannot allocate source STFT fixture inputs");
    }
    int valid = read_fixture("reference/fixtures/excitation_harmonic.f32le", harmonic,
                             EXCITATION_SAMPLES) &&
                read_fixture("reference/fixtures/source_spectrogram_magnitude.f32le",
                             expected_magnitude, SPECTRAL_COUNT) &&
                read_fixture("reference/fixtures/source_spectrogram_phase.f32le", expected_phase,
                             SPECTRAL_COUNT);
    if (!valid) {
        free(harmonic);
        free(expected_magnitude);
        free(expected_phase);
        return 0;
    }
    KokoroNeuralGeneratorSourceOutput output = {0};
    KokoroError error = {0};
    KokoroStatus status =
        kokoro_neural_generator_source_spectrogram(harmonic, EXCITATION_SAMPLES, &output, &error);
    valid =
        expect_status(status, KOKORO_OK, &error, "compute native source STFT") &&
        output.frame_count == SPECTRAL_FRAMES && output.magnitude != NULL && output.phase != NULL &&
        compare_spectrum("native source magnitude", output.magnitude, expected_magnitude,
                         SPECTRAL_COUNT) &&
        compare_complex_spectrogram(&output, expected_magnitude, expected_phase, SPECTRAL_COUNT);

    kokoro_neural_generator_source_output_free(&output);
    float saved_harmonic = harmonic[0];
    harmonic[0] = NAN;
    memset(&error, 0, sizeof(error));
    status =
        kokoro_neural_generator_source_spectrogram(harmonic, EXCITATION_SAMPLES, &output, &error);
    valid = valid &&
            expect_status(status, KOKORO_NUMERIC_ERROR, &error,
                          "reject non-finite source excitation") &&
            output.frame_count == 0 && output.magnitude == NULL && output.phase == NULL;
    harmonic[0] = saved_harmonic;
    for (size_t sample = 0; sample < EXCITATION_SAMPLES; sample++) {
        harmonic[sample] = 1.0e38f;
    }
    memset(&error, 0, sizeof(error));
    status =
        kokoro_neural_generator_source_spectrogram(harmonic, EXCITATION_SAMPLES, &output, &error);
    valid = valid &&
            expect_status(status, KOKORO_NUMERIC_ERROR, &error,
                          "reject source STFT values outside float range") &&
            output.frame_count == 0 && output.magnitude == NULL && output.phase == NULL;
    free(harmonic);
    free(expected_magnitude);
    free(expected_phase);
    return valid;
}

static int test_production_excitation(const KokoroNeuralGenerator *generator) {
    float features[512U * 2U] = {0};
    float f0[2] = {221.0f, 246.0f};
    float noise[2] = {0.0f, 0.0f};
    KokoroDecoderOutput decoder = {
        .duration_frame_count = 1, .feature_frame_count = 2, .features = features};
    KokoroPitchNoiseOutput pitch_noise = {
        .duration_frame_count = 1, .prediction_frame_count = 2, .f0 = f0, .noise = noise};
    KokoroNeuralGeneratorOptions options = {.random_seed = 71, .excitation = NULL};
    KokoroSpectralOutput first = {0};
    KokoroSpectralOutput repeated = {0};
    KokoroSpectralOutput changed_seed = {0};
    KokoroError error = {0};
    KokoroStatus status =
        kokoro_neural_generate(generator, &decoder, &pitch_noise, 0, &options, &first, &error);
    if (!expect_status(status, KOKORO_OK, &error, "generate from production F0 excitation")) {
        return 0;
    }
    options.random_seed = 71;
    status =
        kokoro_neural_generate(generator, &decoder, &pitch_noise, 0, &options, &repeated, &error);
    if (!expect_status(status, KOKORO_OK, &error, "repeat seeded production excitation")) {
        kokoro_spectral_output_free(&first);
        return 0;
    }
    options.random_seed = 72;
    status = kokoro_neural_generate(generator, &decoder, &pitch_noise, 0, &options, &changed_seed,
                                    &error);
    int valid =
        expect_status(status, KOKORO_OK, &error, "vary production excitation seed") &&
        first.input_frame_count == 2 && first.frame_count == 121 &&
        repeated.frame_count == first.frame_count &&
        changed_seed.frame_count == first.frame_count &&
        check_finite_spectrum(&first, 11U * first.frame_count) &&
        memcmp(first.magnitude, repeated.magnitude, 11U * first.frame_count * sizeof(float)) == 0 &&
        memcmp(first.phase, repeated.phase, 11U * first.frame_count * sizeof(float)) == 0;
    bool differs = false;
    if (valid) {
        for (size_t index = 0; index < 11U * first.frame_count; index++) {
            if (first.magnitude[index] != changed_seed.magnitude[index] ||
                first.phase[index] != changed_seed.phase[index]) {
                differs = true;
                break;
            }
        }
        valid = differs;
    }
    kokoro_spectral_output_free(&first);
    kokoro_spectral_output_free(&repeated);
    kokoro_spectral_output_free(&changed_seed);
    if (valid) {
        puts("native F0 excitation is finite, deterministic by seed, and seed-sensitive");
    } else {
        fail("production F0 excitation output shape or deterministic seed behavior is invalid");
    }
    return valid;
}

int main(void) {
    KokoroError error = {0};
    KokoroPack *pack = NULL;
    KokoroNeuralGenerator *generator = NULL;
    KokoroStatus status = kokoro_neural_generator_create(NULL, &generator, &error);
    if (!expect_status(status, KOKORO_INVALID_ARGUMENT, &error, "reject null model pack") ||
        generator != NULL) {
        return fail("null model pack was accepted");
    }
    if (access(MODEL_PACK_PATH, R_OK) != 0) {
        puts("SKIP model-backed neural generator parity: ignored model pack is unavailable");
        return 0;
    }
    status = kokoro_pack_open(MODEL_PACK_PATH, &pack, &error);
    if (!expect_status(status, KOKORO_OK, &error, "open pinned model pack")) {
        return 1;
    }
    status = kokoro_neural_generator_create(pack, &generator, &error);
    if (!expect_status(status, KOKORO_OK, &error, "bind neural generator weights")) {
        kokoro_pack_close(pack);
        return 1;
    }
    int valid = test_native_source_spectrogram() && test_fixture_parity(generator) &&
                test_production_excitation(generator);
    kokoro_neural_generator_free(generator);
    kokoro_pack_close(pack);
    return valid ? 0 : 1;
}

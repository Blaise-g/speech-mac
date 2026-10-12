#include "istft.h"

#include "kokoro_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#define ISTFT_PI 3.14159265358979323846264338327950288

static KokoroStatus invalid_argument(KokoroError *error, const char *message) {
    kokoro_error_set(error, KOKORO_INVALID_ARGUMENT, "%s", message);
    return KOKORO_INVALID_ARGUMENT;
}

KokoroStatus kokoro_istft_reconstruct(const float *magnitude, const float *phase,
                                      size_t frame_count, KokoroWaveformOutput *output,
                                      KokoroError *error) {
    if (magnitude == NULL || phase == NULL || output == NULL) {
        return invalid_argument(error, "ISTFT input and output buffers are required");
    }
    if (output->samples != NULL || output->sample_count != 0) {
        return invalid_argument(error, "ISTFT output must be empty before reconstruction");
    }
    if (frame_count < 2 || frame_count > SIZE_MAX / KOKORO_ISTFT_SPECTRAL_BINS ||
        frame_count - 1 > (SIZE_MAX - KOKORO_ISTFT_FFT_SIZE) / KOKORO_ISTFT_HOP_SIZE) {
        return invalid_argument(error, "ISTFT frame count is empty, too short, or too large");
    }

    size_t sample_count = (frame_count - 1) * KOKORO_ISTFT_HOP_SIZE;
    if (sample_count == 0 || sample_count > SIZE_MAX / sizeof(float) ||
        sample_count > SIZE_MAX / sizeof(double)) {
        return invalid_argument(error, "ISTFT output length exceeds addressable memory");
    }

    size_t spectrum_count = KOKORO_ISTFT_SPECTRAL_BINS * frame_count;
    for (size_t index = 0; index < spectrum_count; index++) {
        if (!isfinite(magnitude[index]) || !isfinite(phase[index])) {
            kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                             "ISTFT spectrum contains a non-finite value");
            return KOKORO_NUMERIC_ERROR;
        }
        if (magnitude[index] < 0.0f) {
            return invalid_argument(error, "ISTFT magnitude cannot be negative");
        }
    }

    double *overlap_sum = calloc(sample_count, sizeof(double));
    double *window_sum = calloc(sample_count, sizeof(double));
    float *samples = malloc(sample_count * sizeof(float));
    if (overlap_sum == NULL || window_sum == NULL || samples == NULL) {
        free(overlap_sum);
        free(window_sum);
        free(samples);
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY, "cannot allocate ISTFT output buffers");
        return KOKORO_OUT_OF_MEMORY;
    }

    double window[KOKORO_ISTFT_FFT_SIZE];
    double basis_cos[KOKORO_ISTFT_SPECTRAL_BINS][KOKORO_ISTFT_FFT_SIZE];
    double basis_sin[KOKORO_ISTFT_SPECTRAL_BINS][KOKORO_ISTFT_FFT_SIZE];
    for (size_t sample = 0; sample < KOKORO_ISTFT_FFT_SIZE; sample++) {
        double angle = 2.0 * ISTFT_PI * (double)sample / KOKORO_ISTFT_FFT_SIZE;
        window[sample] = 0.5 * (1.0 - cos(angle));
        for (size_t bin = 0; bin < KOKORO_ISTFT_SPECTRAL_BINS; bin++) {
            angle = 2.0 * ISTFT_PI * (double)bin * (double)sample / KOKORO_ISTFT_FFT_SIZE;
            basis_cos[bin][sample] = cos(angle);
            basis_sin[bin][sample] = sin(angle);
        }
    }

    const size_t trim = KOKORO_ISTFT_FFT_SIZE / 2U;
    for (size_t frame = 0; frame < frame_count; frame++) {
        double real[KOKORO_ISTFT_SPECTRAL_BINS];
        double imaginary[KOKORO_ISTFT_SPECTRAL_BINS];
        for (size_t bin = 0; bin < KOKORO_ISTFT_SPECTRAL_BINS; bin++) {
            size_t index = bin * frame_count + frame;
            real[bin] = (double)magnitude[index] * cos((double)phase[index]);
            imaginary[bin] = (double)magnitude[index] * sin((double)phase[index]);
        }

        size_t frame_start = frame * KOKORO_ISTFT_HOP_SIZE;
        for (size_t sample = 0; sample < KOKORO_ISTFT_FFT_SIZE; sample++) {
            size_t absolute_sample = frame_start + sample;
            if (absolute_sample < trim || absolute_sample - trim >= sample_count) {
                continue;
            }

            double inverse_value = 0.0;
            for (size_t bin = 0; bin < KOKORO_ISTFT_SPECTRAL_BINS; bin++) {
                double contribution = real[bin] * basis_cos[bin][sample];
                if (bin != 0 && bin != KOKORO_ISTFT_FFT_SIZE / 2U) {
                    contribution = 2.0 * (contribution - imaginary[bin] * basis_sin[bin][sample]);
                }
                inverse_value += contribution;
            }
            inverse_value /= KOKORO_ISTFT_FFT_SIZE;

            size_t output_sample = absolute_sample - trim;
            overlap_sum[output_sample] += inverse_value * window[sample];
            window_sum[output_sample] += window[sample] * window[sample];
        }
    }

    for (size_t sample = 0; sample < sample_count; sample++) {
        if (!(window_sum[sample] > 0.0) || !isfinite(window_sum[sample])) {
            free(overlap_sum);
            free(window_sum);
            free(samples);
            kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                             "ISTFT overlap window has an uncovered output sample");
            return KOKORO_NUMERIC_ERROR;
        }
        samples[sample] = (float)(overlap_sum[sample] / window_sum[sample]);
        if (!isfinite(samples[sample])) {
            free(overlap_sum);
            free(window_sum);
            free(samples);
            kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                             "ISTFT produced a non-finite waveform sample");
            return KOKORO_NUMERIC_ERROR;
        }
    }

    free(overlap_sum);
    free(window_sum);
    output->sample_count = sample_count;
    output->samples = samples;
    kokoro_error_set(error, KOKORO_OK, NULL);
    return KOKORO_OK;
}

void kokoro_waveform_output_free(KokoroWaveformOutput *output) {
    if (output == NULL) {
        return;
    }
    free(output->samples);
    output->samples = NULL;
    output->sample_count = 0;
}

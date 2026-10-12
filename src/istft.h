#ifndef KOKORO_ISTFT_H
#define KOKORO_ISTFT_H

#include "kokoro.h"

#include <stddef.h>

#define KOKORO_ISTFT_SAMPLE_RATE_HZ 24000U
#define KOKORO_ISTFT_FFT_SIZE 20U
#define KOKORO_ISTFT_HOP_SIZE 5U
#define KOKORO_ISTFT_SPECTRAL_BINS (KOKORO_ISTFT_FFT_SIZE / 2U + 1U)

typedef struct {
    size_t sample_count;
    /* Owned mono float32 samples, released with kokoro_waveform_output_free. */
    float *samples;
} KokoroWaveformOutput;

/*
 * Reconstruct channel-major [11, frame_count] magnitudes and phases into 24 kHz mono audio.
 * Initialize output to {0}; on success its samples are owned by the caller, and on failure it
 * remains unchanged.
 */
KokoroStatus kokoro_istft_reconstruct(const float *magnitude, const float *phase,
                                      size_t frame_count, KokoroWaveformOutput *output,
                                      KokoroError *error);
void kokoro_waveform_output_free(KokoroWaveformOutput *output);

#endif

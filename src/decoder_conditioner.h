#ifndef KOKORO_DECODER_CONDITIONER_H
#define KOKORO_DECODER_CONDITIONER_H

#include "kokoro.h"
#include "model_pack.h"
#include "pitch_noise_predictor.h"
#include "text_encoder.h"

#include <stddef.h>

#define KOKORO_DECODER_FEATURES 512U

typedef struct KokoroDecoderConditioner KokoroDecoderConditioner;

typedef struct {
    size_t duration_frame_count;
    size_t feature_frame_count;
    /* Owned channel-major [512, feature_frame_count] features for the waveform generator. */
    float *features;
} KokoroDecoderOutput;

/* The pack must outlive the conditioner; it borrows the voice table and AdaIN weights. */
KokoroStatus kokoro_decoder_conditioner_create(const KokoroPack *pack,
                                               KokoroDecoderConditioner **conditioner,
                                               KokoroError *error);
void kokoro_decoder_conditioner_free(KokoroDecoderConditioner *conditioner);

/*
 * Combine duration-aligned text and 2x-rate F0/noise using the decoder half of the selected
 * 256-value voice style. Output must be initialized to {0}; on success its feature buffer is
 * owned by the caller and released with kokoro_decoder_output_free. On failure output is unchanged.
 */
KokoroStatus kokoro_decoder_condition(const KokoroDecoderConditioner *conditioner,
                                      const KokoroTextEncoderOutput *text,
                                      const KokoroPitchNoiseOutput *pitch_noise,
                                      size_t voice_style_index, KokoroDecoderOutput *output,
                                      KokoroError *error);
void kokoro_decoder_output_free(KokoroDecoderOutput *output);

#endif

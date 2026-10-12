#ifndef PITCH_NOISE_PREDICTOR_H
#define PITCH_NOISE_PREDICTOR_H

#include "duration_alignment.h"
#include "kokoro.h"
#include "model_pack.h"

typedef struct KokoroPitchNoisePredictor KokoroPitchNoisePredictor;

typedef struct {
    size_t duration_frame_count;
    size_t prediction_frame_count;
    /* Owned [prediction_frame] outputs, released with kokoro_pitch_noise_output_free. */
    float *f0;
    float *noise;
} KokoroPitchNoiseOutput;

/* The pack must outlive the predictor; the predictor borrows its voice and projection weights. */
KokoroStatus kokoro_pitch_noise_predictor_create(const KokoroPack *pack,
                                                 const KokoroTensorView *voice_style,
                                                 KokoroPitchNoisePredictor **predictor,
                                                 KokoroError *error);
void kokoro_pitch_noise_predictor_free(KokoroPitchNoisePredictor *predictor);

/*
 * Expand the duration predictor's token-major [token, 640] features through its one-hot alignment,
 * then predict F0 and noise. The pinned v1.0 blocks upsample once, so prediction_frame_count is
 * twice duration->frame_count. The predictor borrows the duration output for this call. Initialize
 * output to {0}; it must be empty and its buffers must be freed before reuse. On failure, output is
 * left unchanged.
 */
KokoroStatus kokoro_pitch_noise_predict(const KokoroPitchNoisePredictor *predictor,
                                        const KokoroDurationOutput *duration,
                                        size_t voice_style_index, KokoroPitchNoiseOutput *output,
                                        KokoroError *error);
void kokoro_pitch_noise_output_free(KokoroPitchNoiseOutput *output);

#endif

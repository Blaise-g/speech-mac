#ifndef DURATION_ALIGNMENT_H
#define DURATION_ALIGNMENT_H

#include "kokoro.h"
#include "model_pack.h"

#include <stddef.h>
#include <stdint.h>

#define KOKORO_DURATION_FEATURES 640U
#define KOKORO_DURATION_RNN_FEATURES 512U
#define KOKORO_DURATION_LOGIT_CHANNELS 50U
#define KOKORO_DURATION_MAX_TOKENS 512U

typedef struct KokoroDurationPredictor KokoroDurationPredictor;

typedef struct {
    size_t token_count;
    size_t frame_count;
    /* Owned, token-major arrays: [token, 640], [token, 512], [token, 50], and [token]. */
    float *duration_features;
    float *duration_rnn;
    float *duration_logits;
    float *duration_values;
    uint64_t *durations;
    /* Owned dense, token-major [token, frame] one-hot alignment. */
    float *alignment;
} KokoroDurationOutput;

/* The pack must outlive the predictor; the predictor borrows its read-only weights and voice rows.
 */
KokoroStatus kokoro_duration_predictor_create(const KokoroPack *pack,
                                              const KokoroTensorView *voice_style,
                                              KokoroDurationPredictor **predictor,
                                              KokoroError *error);
void kokoro_duration_predictor_free(KokoroDurationPredictor *predictor);

/*
 * Context features use the channel-major [512, token] layout returned by
 * kokoro_context_encode. `voice_style_index` selects a row of the packed af_heart table.
 * Output must be initialized to {0}; all returned buffers are owned by the caller and are
 * released with kokoro_duration_output_free. `speed` follows the v1.0 native CLI range [0.5, 2].
 */
KokoroStatus kokoro_duration_predict(const KokoroDurationPredictor *predictor,
                                     const float *contextual_features, size_t token_count,
                                     size_t voice_style_index, float speed,
                                     KokoroDurationOutput *output, KokoroError *error);

/* Apply upstream speed scaling, ties-to-even rounding, minimum-one clamp, and alignment expansion.
 */
KokoroStatus kokoro_duration_align_values(const float *unscaled_duration_values, size_t token_count,
                                          float speed, KokoroDurationOutput *output,
                                          KokoroError *error);

void kokoro_duration_output_free(KokoroDurationOutput *output);

#endif

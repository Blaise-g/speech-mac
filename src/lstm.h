#ifndef KOKORO_LSTM_H
#define KOKORO_LSTM_H

#include "model_pack.h"

typedef struct {
    const float *weight_ih;
    const float *weight_hh;
    const float *bias_ih;
    const float *bias_hh;
    const float *weight_ih_reverse;
    const float *weight_hh_reverse;
    const float *bias_ih_reverse;
    const float *bias_hh_reverse;
} KokoroBiLstmWeights;

/* `weights` borrows pack data, so the pack must outlive every bound use. */
KokoroStatus kokoro_bilstm_bind(const KokoroPack *pack, const char *prefix, size_t input_width,
                                size_t hidden_width, KokoroBiLstmWeights *weights,
                                KokoroError *error);
KokoroStatus kokoro_bilstm_run(const KokoroBiLstmWeights *weights, const float *input,
                               size_t sequence_length, size_t input_width, size_t hidden_width,
                               float *output, KokoroError *error);

#endif

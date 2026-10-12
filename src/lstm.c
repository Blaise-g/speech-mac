#include "lstm.h"

#include "kokoro_internal.h"
#include "tensor_utils.h"

#define ACCELERATE_NEW_LAPACK 1
#define ACCELERATE_LAPACK_ILP64 1
#include <Accelerate/Accelerate.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

KokoroStatus kokoro_bilstm_bind(const KokoroPack *pack, const char *prefix, size_t input_width,
                                size_t hidden_width, KokoroBiLstmWeights *weights,
                                KokoroError *error) {
    if (pack == NULL || prefix == NULL || weights == NULL || input_width == 0 ||
        hidden_width == 0 || hidden_width > SIZE_MAX / 4 || hidden_width * 4 > UINT64_MAX ||
        input_width > UINT64_MAX || input_width > INT_MAX || hidden_width > INT_MAX / 4) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT, "invalid bidirectional LSTM binding");
        return KOKORO_INVALID_ARGUMENT;
    }

    static const char *const suffixes[] = {
        "weight_ih_l0",         "weight_hh_l0",         "bias_ih_l0",         "bias_hh_l0",
        "weight_ih_l0_reverse", "weight_hh_l0_reverse", "bias_ih_l0_reverse", "bias_hh_l0_reverse",
    };
    const uint64_t gate_width = (uint64_t)(hidden_width * 4);
    const uint64_t shapes[8][2] = {
        {gate_width, input_width}, {gate_width, hidden_width}, {gate_width, 0}, {gate_width, 0},
        {gate_width, input_width}, {gate_width, hidden_width}, {gate_width, 0}, {gate_width, 0},
    };
    const float **destinations[] = {
        &weights->weight_ih,       &weights->weight_hh,         &weights->bias_ih,
        &weights->bias_hh,         &weights->weight_ih_reverse, &weights->weight_hh_reverse,
        &weights->bias_ih_reverse, &weights->bias_hh_reverse,
    };
    char name[160];
    for (size_t index = 0; index < sizeof(suffixes) / sizeof(suffixes[0]); index++) {
        int length = snprintf(name, sizeof(name), "%s.%s", prefix, suffixes[index]);
        if (length < 0 || (size_t)length >= sizeof(name)) {
            kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "LSTM tensor name is too long");
            return KOKORO_INCOMPATIBLE_PACK;
        }
        unsigned int rank = index == 0 || index == 1 || index == 4 || index == 5 ? 2U : 1U;
        KokoroStatus status =
            kokoro_tensor_bind_f32(pack, name, rank, shapes[index], destinations[index], error);
        if (status != KOKORO_OK) {
            return status;
        }
    }
    return KOKORO_OK;
}

static bool allocate_floats(size_t count, float **values) {
    if (count > SIZE_MAX / sizeof(float)) {
        return false;
    }
    *values = malloc(count * sizeof(float));
    return *values != NULL;
}

static KokoroStatus run_direction(const float *input, size_t sequence_length, size_t input_width,
                                  size_t hidden_width, const float *weight_ih,
                                  const float *weight_hh, const float *bias_ih,
                                  const float *bias_hh, bool reverse, float *output,
                                  KokoroError *error) {
    const size_t gate_width = hidden_width * 4;
    if (sequence_length > SIZE_MAX / gate_width || sequence_length > INT_MAX) {
        kokoro_error_set(error, KOKORO_NUMERIC_ERROR, "LSTM sequence is too large");
        return KOKORO_NUMERIC_ERROR;
    }
    size_t gate_count = sequence_length * gate_width;
    float *input_gates = NULL;
    if (!allocate_floats(gate_count, &input_gates)) {
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY, "cannot allocate LSTM workspace");
        return KOKORO_OUT_OF_MEMORY;
    }
    float *combined_bias = malloc(gate_width * sizeof(float));
    if (combined_bias == NULL) {
        free(input_gates);
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY, "cannot allocate LSTM bias workspace");
        return KOKORO_OUT_OF_MEMORY;
    }
    for (size_t gate = 0; gate < gate_width; gate++) {
        combined_bias[gate] = bias_ih[gate] + bias_hh[gate];
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)sequence_length, (int)gate_width,
                (int)input_width, 1.0f, input, (int)input_width, weight_ih, (int)input_width, 0.0f,
                input_gates, (int)gate_width);
    for (size_t row = 0; row < sequence_length; row++) {
        float *gates = input_gates + row * gate_width;
        for (size_t gate = 0; gate < gate_width; gate++) {
            gates[gate] += combined_bias[gate];
        }
    }
    free(combined_bias);

    float *state = calloc(hidden_width * 2, sizeof(float));
    float *cell = calloc(hidden_width, sizeof(float));
    if (state == NULL || cell == NULL) {
        free(input_gates);
        free(state);
        free(cell);
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY, "cannot allocate LSTM state");
        return KOKORO_OUT_OF_MEMORY;
    }
    float *hidden = state;
    float *next_hidden = state + hidden_width;
    for (size_t step = 0; step < sequence_length; step++) {
        size_t sequence_index = reverse ? sequence_length - 1 - step : step;
        float *gates = input_gates + sequence_index * gate_width;
        cblas_sgemv(CblasRowMajor, CblasNoTrans, (int)gate_width, (int)hidden_width, 1.0f,
                    weight_hh, (int)hidden_width, hidden, 1, 1.0f, gates, 1);
        for (size_t unit = 0; unit < hidden_width; unit++) {
            float input_gate = 1.0f / (1.0f + expf(-gates[unit]));
            float forget_gate = 1.0f / (1.0f + expf(-gates[hidden_width + unit]));
            float candidate = tanhf(gates[2 * hidden_width + unit]);
            float output_gate = 1.0f / (1.0f + expf(-gates[3 * hidden_width + unit]));
            cell[unit] = forget_gate * cell[unit] + input_gate * candidate;
            next_hidden[unit] = output_gate * tanhf(cell[unit]);
        }
        float *target = output + sequence_index * hidden_width * 2;
        memcpy(target + (reverse ? hidden_width : 0), next_hidden, hidden_width * sizeof(float));
        float *previous_hidden = hidden;
        hidden = next_hidden;
        next_hidden = previous_hidden;
    }
    free(input_gates);
    free(state);
    free(cell);
    return KOKORO_OK;
}

KokoroStatus kokoro_bilstm_run(const KokoroBiLstmWeights *weights, const float *input,
                               size_t sequence_length, size_t input_width, size_t hidden_width,
                               float *output, KokoroError *error) {
    if (weights == NULL || input == NULL || output == NULL || sequence_length == 0 ||
        input_width == 0 || hidden_width == 0 || sequence_length > INT_MAX ||
        input_width > INT_MAX || hidden_width > INT_MAX / 4 ||
        sequence_length > SIZE_MAX / (hidden_width * 2)) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT, "invalid bidirectional LSTM input");
        return KOKORO_INVALID_ARGUMENT;
    }
    KokoroStatus status =
        run_direction(input, sequence_length, input_width, hidden_width, weights->weight_ih,
                      weights->weight_hh, weights->bias_ih, weights->bias_hh, false, output, error);
    if (status != KOKORO_OK) {
        return status;
    }
    return run_direction(input, sequence_length, input_width, hidden_width,
                         weights->weight_ih_reverse, weights->weight_hh_reverse,
                         weights->bias_ih_reverse, weights->bias_hh_reverse, true, output, error);
}

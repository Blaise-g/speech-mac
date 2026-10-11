#include "duration_alignment.h"

#include "kokoro_internal.h"
#include "lstm.h"
#include "tensor_utils.h"

#define ACCELERATE_NEW_LAPACK 1
#define ACCELERATE_LAPACK_ILP64 1
#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DURATION_STYLE_SIZE 128U
#define DURATION_VOICE_SIZE 256U
#define DURATION_HIDDEN_SIZE 512U
#define DURATION_LSTM_WIDTH 256U
#define DURATION_GATE_COUNT 4U
#define DURATION_LAYER_COUNT 3U
#define DURATION_LSTM_GATE_SIZE (DURATION_LSTM_WIDTH * DURATION_GATE_COUNT)
#define DURATION_ADANORM_EPSILON 1.0e-5f
#define DURATION_MIN_SPEED 0.5f
#define DURATION_MAX_SPEED 2.0f
#define DURATION_MAX_RAW_VALUE ((float)KOKORO_DURATION_LOGIT_CHANNELS)

typedef struct {
    KokoroBiLstmWeights lstm;
    const float *norm_weight;
    const float *norm_bias;
} DurationLayerWeights;

struct KokoroDurationPredictor {
    const float *voice_styles;
    size_t voice_style_count;
    DurationLayerWeights layers[DURATION_LAYER_COUNT];
    KokoroBiLstmWeights projection_lstm;
    const float *duration_projection_weight;
    const float *duration_projection_bias;
};

static KokoroStatus bind_norm(const KokoroPack *pack, const char *prefix,
                              DurationLayerWeights *weights, KokoroError *error) {
    const uint64_t matrix_shape[2] = {DURATION_LSTM_GATE_SIZE, DURATION_STYLE_SIZE};
    const uint64_t vector_shape[1] = {DURATION_LSTM_GATE_SIZE};
    char name[160];
    int length = snprintf(name, sizeof(name), "%s.weight", prefix);
    if (length < 0 || (size_t)length >= sizeof(name)) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "predictor tensor name is too long");
        return KOKORO_INCOMPATIBLE_PACK;
    }
    KokoroStatus status =
        kokoro_tensor_bind_f32(pack, name, 2, matrix_shape, &weights->norm_weight, error);
    if (status != KOKORO_OK) {
        return status;
    }
    length = snprintf(name, sizeof(name), "%s.bias", prefix);
    if (length < 0 || (size_t)length >= sizeof(name)) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "predictor tensor name is too long");
        return KOKORO_INCOMPATIBLE_PACK;
    }
    return kokoro_tensor_bind_f32(pack, name, 1, vector_shape, &weights->norm_bias, error);
}

KokoroStatus kokoro_duration_predictor_create(const KokoroPack *pack,
                                              KokoroDurationPredictor **result,
                                              KokoroError *error) {
    if (result != NULL) {
        *result = NULL;
    }
    if (pack == NULL || result == NULL) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "model pack and duration predictor output are required");
        return KOKORO_INVALID_ARGUMENT;
    }
    KokoroDurationPredictor *predictor = calloc(1, sizeof(*predictor));
    if (predictor == NULL) {
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY, "cannot allocate duration predictor");
        return KOKORO_OUT_OF_MEMORY;
    }

    KokoroTensorView voice_view;
    KokoroStatus status = kokoro_pack_tensor(pack, "voice.af_heart.style", &voice_view, error);
    if (status != KOKORO_OK) {
        free(predictor);
        return status;
    }
    if (voice_view.type != KOKORO_TENSOR_F32 || voice_view.rank != 3 || voice_view.shape[0] == 0 ||
        voice_view.shape[1] != 1 || voice_view.shape[2] != DURATION_VOICE_SIZE ||
        voice_view.shape[0] > SIZE_MAX) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK,
                         "af_heart style tensor must have shape [rows, 1, 256]");
        free(predictor);
        return KOKORO_INCOMPATIBLE_PACK;
    }
    predictor->voice_styles = voice_view.data;
    predictor->voice_style_count = (size_t)voice_view.shape[0];
    if (predictor->voice_style_count > SIZE_MAX / DURATION_VOICE_SIZE) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK,
                         "af_heart style tensor has too many rows");
        free(predictor);
        return KOKORO_INCOMPATIBLE_PACK;
    }
    for (size_t index = 0; index < predictor->voice_style_count * DURATION_VOICE_SIZE; index++) {
        if (!isfinite(predictor->voice_styles[index])) {
            kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK,
                             "af_heart style tensor contains a non-finite value");
            free(predictor);
            return KOKORO_INCOMPATIBLE_PACK;
        }
    }

    for (size_t layer = 0; layer < DURATION_LAYER_COUNT; layer++) {
        char prefix[128];
        int length =
            snprintf(prefix, sizeof(prefix), "predictor.module.text_encoder.lstms.%zu", layer * 2);
        if (length < 0 || (size_t)length >= sizeof(prefix)) {
            kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "predictor tensor name is too long");
            free(predictor);
            return KOKORO_INCOMPATIBLE_PACK;
        }
        status = kokoro_bilstm_bind(pack, prefix, KOKORO_DURATION_FEATURES, DURATION_LSTM_WIDTH,
                                    &predictor->layers[layer].lstm, error);
        if (status != KOKORO_OK) {
            free(predictor);
            return status;
        }
        length = snprintf(prefix, sizeof(prefix), "predictor.module.text_encoder.lstms.%zu.fc",
                          layer * 2 + 1);
        if (length < 0 || (size_t)length >= sizeof(prefix)) {
            kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "predictor tensor name is too long");
            free(predictor);
            return KOKORO_INCOMPATIBLE_PACK;
        }
        status = bind_norm(pack, prefix, &predictor->layers[layer], error);
        if (status != KOKORO_OK) {
            free(predictor);
            return status;
        }
    }

    status = kokoro_bilstm_bind(pack, "predictor.module.lstm", KOKORO_DURATION_FEATURES,
                                DURATION_LSTM_WIDTH, &predictor->projection_lstm, error);
    if (status != KOKORO_OK) {
        free(predictor);
        return status;
    }
    const uint64_t projection_shape[2] = {KOKORO_DURATION_LOGIT_CHANNELS,
                                          KOKORO_DURATION_RNN_FEATURES};
    const uint64_t projection_bias_shape[1] = {KOKORO_DURATION_LOGIT_CHANNELS};
    status =
        kokoro_tensor_bind_f32(pack, "predictor.module.duration_proj.linear_layer.weight", 2,
                               projection_shape, &predictor->duration_projection_weight, error);
    if (status == KOKORO_OK) {
        status = kokoro_tensor_bind_f32(pack, "predictor.module.duration_proj.linear_layer.bias", 1,
                                        projection_bias_shape, &predictor->duration_projection_bias,
                                        error);
    }
    if (status != KOKORO_OK) {
        free(predictor);
        return status;
    }

    *result = predictor;
    kokoro_error_set(error, KOKORO_OK, NULL);
    return KOKORO_OK;
}

void kokoro_duration_predictor_free(KokoroDurationPredictor *predictor) { free(predictor); }

static bool allocate_floats(size_t count, float **values) {
    if (count > SIZE_MAX / sizeof(float)) {
        return false;
    }
    *values = malloc(count * sizeof(float));
    return *values != NULL;
}

static void linear(const float *input, const float *weight, const float *bias, size_t rows,
                   size_t input_width, size_t output_width, float *output) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)rows, (int)output_width,
                (int)input_width, 1.0f, input, (int)input_width, weight, (int)input_width, 0.0f,
                output, (int)output_width);
    for (size_t row = 0; row < rows; row++) {
        float *values = output + row * output_width;
        for (size_t column = 0; column < output_width; column++) {
            values[column] += bias[column];
        }
    }
}

static float sigmoid(float value) { return 1.0f / (1.0f + expf(-value)); }

static void apply_adaptive_layer_norm(const DurationLayerWeights *weights, const float *input,
                                      const float *style, size_t token_count, float *output) {
    float affine[DURATION_LSTM_GATE_SIZE];
    cblas_sgemv(CblasRowMajor, CblasNoTrans, (int)DURATION_LSTM_GATE_SIZE, (int)DURATION_STYLE_SIZE,
                1.0f, weights->norm_weight, (int)DURATION_STYLE_SIZE, style, 1, 0.0f, affine, 1);
    for (size_t index = 0; index < DURATION_LSTM_GATE_SIZE; index++) {
        affine[index] += weights->norm_bias[index];
    }

    for (size_t token = 0; token < token_count; token++) {
        const float *source = input + token * DURATION_HIDDEN_SIZE;
        float mean = 0.0f;
        for (size_t channel = 0; channel < DURATION_HIDDEN_SIZE; channel++) {
            mean += source[channel];
        }
        mean /= (float)DURATION_HIDDEN_SIZE;
        float variance = 0.0f;
        for (size_t channel = 0; channel < DURATION_HIDDEN_SIZE; channel++) {
            float centered = source[channel] - mean;
            variance += centered * centered;
        }
        variance /= (float)DURATION_HIDDEN_SIZE;
        float inverse_scale = 1.0f / sqrtf(variance + DURATION_ADANORM_EPSILON);
        float *target = output + token * KOKORO_DURATION_FEATURES;
        for (size_t channel = 0; channel < DURATION_HIDDEN_SIZE; channel++) {
            float normalized = (source[channel] - mean) * inverse_scale;
            target[channel] =
                (1.0f + affine[channel]) * normalized + affine[DURATION_HIDDEN_SIZE + channel];
            target[DURATION_HIDDEN_SIZE + channel] = style[channel % DURATION_STYLE_SIZE];
        }
    }
}

static KokoroStatus run_duration_encoder(const KokoroDurationPredictor *predictor,
                                         const float *contextual_features, size_t token_count,
                                         const float *style, float **result, KokoroError *error) {
    size_t initial_count = token_count * KOKORO_DURATION_FEATURES;
    float *current = NULL;
    if (!allocate_floats(initial_count, &current)) {
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY, "cannot allocate duration feature workspace");
        return KOKORO_OUT_OF_MEMORY;
    }
    for (size_t token = 0; token < token_count; token++) {
        float *row = current + token * KOKORO_DURATION_FEATURES;
        for (size_t channel = 0; channel < DURATION_HIDDEN_SIZE; channel++) {
            row[channel] = contextual_features[channel * token_count + token];
        }
        memcpy(row + DURATION_HIDDEN_SIZE, style, DURATION_STYLE_SIZE * sizeof(float));
    }

    for (size_t layer = 0; layer < DURATION_LAYER_COUNT; layer++) {
        float *recurrent = NULL;
        float *next = NULL;
        if (!allocate_floats(token_count * KOKORO_DURATION_RNN_FEATURES, &recurrent) ||
            !allocate_floats(initial_count, &next)) {
            free(current);
            free(recurrent);
            free(next);
            kokoro_error_set(error, KOKORO_OUT_OF_MEMORY,
                             "cannot allocate duration encoder layer workspace");
            return KOKORO_OUT_OF_MEMORY;
        }
        KokoroStatus status =
            kokoro_bilstm_run(&predictor->layers[layer].lstm, current, token_count,
                              KOKORO_DURATION_FEATURES, DURATION_LSTM_WIDTH, recurrent, error);
        if (status != KOKORO_OK) {
            free(current);
            free(recurrent);
            free(next);
            return status;
        }
        apply_adaptive_layer_norm(&predictor->layers[layer], recurrent, style, token_count, next);
        free(current);
        free(recurrent);
        current = next;
    }
    *result = current;
    return KOKORO_OK;
}

static double round_ties_to_even(float value) {
    double lower = floor((double)value);
    double fraction = (double)value - lower;
    if (fraction < 0.5) {
        return lower;
    }
    if (fraction > 0.5) {
        return lower + 1.0;
    }
    return fmod(lower, 2.0) == 0.0 ? lower : lower + 1.0;
}

void kokoro_duration_output_free(KokoroDurationOutput *output) {
    if (output == NULL) {
        return;
    }
    free(output->duration_features);
    free(output->duration_rnn);
    free(output->duration_logits);
    free(output->duration_values);
    free(output->durations);
    free(output->alignment);
    memset(output, 0, sizeof(*output));
}

KokoroStatus kokoro_duration_align_values(const float *unscaled_duration_values, size_t token_count,
                                          float speed, KokoroDurationOutput *output,
                                          KokoroError *error) {
    if (unscaled_duration_values == NULL || output == NULL) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT, "duration values and output are required");
        return KOKORO_INVALID_ARGUMENT;
    }
    if (token_count == 0 || token_count > KOKORO_DURATION_MAX_TOKENS || !isfinite(speed) ||
        speed < DURATION_MIN_SPEED || speed > DURATION_MAX_SPEED) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "token count must be within 1..%u and speed within [0.5, 2.0]",
                         KOKORO_DURATION_MAX_TOKENS);
        return KOKORO_INVALID_ARGUMENT;
    }

    KokoroDurationOutput aligned = {0};
    aligned.duration_values = malloc(token_count * sizeof(float));
    aligned.durations = malloc(token_count * sizeof(uint64_t));
    if (aligned.duration_values == NULL || aligned.durations == NULL) {
        kokoro_duration_output_free(&aligned);
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY, "cannot allocate duration outputs");
        return KOKORO_OUT_OF_MEMORY;
    }

    size_t frame_count = 0;
    for (size_t token = 0; token < token_count; token++) {
        float value = unscaled_duration_values[token];
        if (!isfinite(value) || value < 0.0f || value > DURATION_MAX_RAW_VALUE) {
            kokoro_duration_output_free(&aligned);
            kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                             "unscaled duration at token %zu is outside [0, 50]", token);
            return KOKORO_NUMERIC_ERROR;
        }
        float scaled = value / speed;
        double rounded = round_ties_to_even(scaled);
        if (rounded < 1.0) {
            rounded = 1.0;
        }
        if (rounded > (double)(SIZE_MAX - frame_count)) {
            kokoro_duration_output_free(&aligned);
            kokoro_error_set(error, KOKORO_NUMERIC_ERROR, "duration sum exceeds addressable size");
            return KOKORO_NUMERIC_ERROR;
        }
        aligned.duration_values[token] = scaled;
        aligned.durations[token] = (uint64_t)rounded;
        frame_count += (size_t)rounded;
    }
    if (frame_count != 0 && token_count > SIZE_MAX / frame_count) {
        kokoro_duration_output_free(&aligned);
        kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                         "alignment dimensions exceed addressable size");
        return KOKORO_NUMERIC_ERROR;
    }
    size_t alignment_count = token_count * frame_count;
    if (alignment_count > SIZE_MAX / sizeof(float)) {
        kokoro_duration_output_free(&aligned);
        kokoro_error_set(error, KOKORO_NUMERIC_ERROR, "alignment size exceeds addressable memory");
        return KOKORO_NUMERIC_ERROR;
    }
    aligned.alignment = calloc(alignment_count, sizeof(float));
    if (aligned.alignment == NULL) {
        kokoro_duration_output_free(&aligned);
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY, "cannot allocate duration alignment");
        return KOKORO_OUT_OF_MEMORY;
    }
    size_t first_frame = 0;
    for (size_t token = 0; token < token_count; token++) {
        size_t last_frame = first_frame + (size_t)aligned.durations[token];
        for (size_t frame = first_frame; frame < last_frame; frame++) {
            aligned.alignment[token * frame_count + frame] = 1.0f;
        }
        first_frame = last_frame;
    }
    aligned.token_count = token_count;
    aligned.frame_count = frame_count;
    *output = aligned;
    kokoro_error_set(error, KOKORO_OK, NULL);
    return KOKORO_OK;
}

static bool all_finite(const float *values, size_t count) {
    for (size_t index = 0; index < count; index++) {
        if (!isfinite(values[index])) {
            return false;
        }
    }
    return true;
}

KokoroStatus kokoro_duration_predict(const KokoroDurationPredictor *predictor,
                                     const float *contextual_features, size_t token_count,
                                     size_t voice_style_index, float speed,
                                     KokoroDurationOutput *output, KokoroError *error) {
    if (predictor == NULL || contextual_features == NULL || output == NULL) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "predictor, contextual features, and output are required");
        return KOKORO_INVALID_ARGUMENT;
    }
    if (token_count == 0 || token_count > KOKORO_DURATION_MAX_TOKENS ||
        voice_style_index >= predictor->voice_style_count || !isfinite(speed) ||
        speed < DURATION_MIN_SPEED || speed > DURATION_MAX_SPEED) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "token count, voice style index, or speed is outside the supported range");
        return KOKORO_INVALID_ARGUMENT;
    }
    if (token_count > SIZE_MAX / DURATION_HIDDEN_SIZE ||
        !all_finite(contextual_features, DURATION_HIDDEN_SIZE * token_count)) {
        kokoro_error_set(
            error, KOKORO_NUMERIC_ERROR,
            "contextual features contain non-finite values or exceed addressable size");
        return KOKORO_NUMERIC_ERROR;
    }

    const float *voice_style = predictor->voice_styles + voice_style_index * DURATION_VOICE_SIZE;
    const float *style = voice_style + DURATION_STYLE_SIZE;
    KokoroDurationOutput result = {0};
    KokoroStatus status = run_duration_encoder(predictor, contextual_features, token_count, style,
                                               &result.duration_features, error);
    if (status != KOKORO_OK) {
        return status;
    }
    result.duration_rnn = NULL;
    result.duration_logits = NULL;
    if (!allocate_floats(token_count * KOKORO_DURATION_RNN_FEATURES, &result.duration_rnn) ||
        !allocate_floats(token_count * KOKORO_DURATION_LOGIT_CHANNELS, &result.duration_logits)) {
        kokoro_duration_output_free(&result);
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY,
                         "cannot allocate duration projection workspace");
        return KOKORO_OUT_OF_MEMORY;
    }
    status = kokoro_bilstm_run(&predictor->projection_lstm, result.duration_features, token_count,
                               KOKORO_DURATION_FEATURES, DURATION_LSTM_WIDTH, result.duration_rnn,
                               error);
    if (status != KOKORO_OK) {
        kokoro_duration_output_free(&result);
        return status;
    }
    linear(result.duration_rnn, predictor->duration_projection_weight,
           predictor->duration_projection_bias, token_count, KOKORO_DURATION_RNN_FEATURES,
           KOKORO_DURATION_LOGIT_CHANNELS, result.duration_logits);
    size_t logit_count = token_count * KOKORO_DURATION_LOGIT_CHANNELS;
    if (!all_finite(result.duration_features, token_count * KOKORO_DURATION_FEATURES) ||
        !all_finite(result.duration_rnn, token_count * KOKORO_DURATION_RNN_FEATURES) ||
        !all_finite(result.duration_logits, logit_count)) {
        kokoro_duration_output_free(&result);
        kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                         "duration predictor produced a non-finite intermediate value");
        return KOKORO_NUMERIC_ERROR;
    }

    float unscaled_values[KOKORO_DURATION_MAX_TOKENS];
    for (size_t token = 0; token < token_count; token++) {
        float total = 0.0f;
        for (size_t channel = 0; channel < KOKORO_DURATION_LOGIT_CHANNELS; channel++) {
            total +=
                sigmoid(result.duration_logits[token * KOKORO_DURATION_LOGIT_CHANNELS + channel]);
        }
        unscaled_values[token] = total;
    }
    KokoroDurationOutput aligned = {0};
    status = kokoro_duration_align_values(unscaled_values, token_count, speed, &aligned, error);
    if (status != KOKORO_OK) {
        kokoro_duration_output_free(&result);
        return status;
    }
    result.duration_values = aligned.duration_values;
    result.durations = aligned.durations;
    result.alignment = aligned.alignment;
    result.token_count = aligned.token_count;
    result.frame_count = aligned.frame_count;
    aligned.duration_values = NULL;
    aligned.durations = NULL;
    aligned.alignment = NULL;
    kokoro_duration_output_free(&aligned);
    *output = result;
    kokoro_error_set(error, KOKORO_OK, NULL);
    return KOKORO_OK;
}

#include "context_encoder.h"

#include "kokoro_internal.h"

#define ACCELERATE_NEW_LAPACK 1
#define ACCELERATE_LAPACK_ILP64 1
#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#define ALBERT_VOCAB_SIZE 178U
#define ALBERT_EMBEDDING_SIZE 128U
#define ALBERT_HIDDEN_SIZE 768U
#define ALBERT_HEADS 12U
#define ALBERT_HEAD_SIZE 64U
#define ALBERT_INTERMEDIATE_SIZE 2048U
#define ALBERT_LAYERS 12U
#define ALBERT_LAYER_NORM_EPSILON 1.0e-12f
#define ALBERT_GELU_COEFFICIENT 0.044715f
#define ALBERT_GELU_SCALE 0.7978845608028654f

struct KokoroContextEncoder {
    const float *word_embeddings;
    const float *position_embeddings;
    const float *token_type_embeddings;
    const float *embedding_norm_weight;
    const float *embedding_norm_bias;
    const float *embedding_mapping_weight;
    const float *embedding_mapping_bias;
    const float *attention_query_weight;
    const float *attention_query_bias;
    const float *attention_key_weight;
    const float *attention_key_bias;
    const float *attention_value_weight;
    const float *attention_value_bias;
    const float *attention_dense_weight;
    const float *attention_dense_bias;
    const float *attention_norm_weight;
    const float *attention_norm_bias;
    const float *full_norm_weight;
    const float *full_norm_bias;
    const float *ffn_weight;
    const float *ffn_bias;
    const float *ffn_output_weight;
    const float *ffn_output_bias;
    const float *projection_weight;
    const float *projection_bias;
};

typedef struct {
    const char *name;
    const float **destination;
    unsigned int rank;
    uint64_t shape[2];
} RequiredTensor;

static bool shape_matches(const KokoroTensorView *view, unsigned int rank, const uint64_t *shape) {
    if (view->type != KOKORO_TENSOR_F32 || view->rank != rank) {
        return false;
    }
    for (unsigned int dimension = 0; dimension < rank; dimension++) {
        if (view->shape[dimension] != shape[dimension]) {
            return false;
        }
    }
    return true;
}

static KokoroStatus bind_tensor(const KokoroPack *pack, const RequiredTensor *required,
                                KokoroError *error) {
    KokoroTensorView view;
    KokoroStatus status = kokoro_pack_tensor(pack, required->name, &view, error);
    if (status != KOKORO_OK) {
        return status;
    }
    if (!shape_matches(&view, required->rank, required->shape)) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "tensor has an incompatible shape: %s",
                         required->name);
        return KOKORO_INCOMPATIBLE_PACK;
    }
    size_t element_count = 1;
    for (unsigned int dimension = 0; dimension < required->rank; dimension++) {
        element_count *= (size_t)required->shape[dimension];
    }
    const float *values = view.data;
    for (size_t index = 0; index < element_count; index++) {
        if (!isfinite(values[index])) {
            kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK,
                             "tensor contains a non-finite value: %s", required->name);
            return KOKORO_INCOMPATIBLE_PACK;
        }
    }
    *required->destination = values;
    return KOKORO_OK;
}

KokoroStatus kokoro_context_encoder_create(const KokoroPack *pack, KokoroContextEncoder **result,
                                           KokoroError *error) {
    if (result != NULL) {
        *result = NULL;
    }
    if (pack == NULL || result == NULL) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "model pack and encoder output are required");
        return KOKORO_INVALID_ARGUMENT;
    }
    KokoroContextEncoder *encoder = calloc(1, sizeof(*encoder));
    if (encoder == NULL) {
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY, "cannot allocate context encoder");
        return KOKORO_OUT_OF_MEMORY;
    }
    RequiredTensor tensors[] = {
        {"bert.module.embeddings.word_embeddings.weight", &encoder->word_embeddings, 2, {178, 128}},
        {"bert.module.embeddings.position_embeddings.weight",
         &encoder->position_embeddings,
         2,
         {512, 128}},
        {"bert.module.embeddings.token_type_embeddings.weight",
         &encoder->token_type_embeddings,
         2,
         {2, 128}},
        {"bert.module.embeddings.LayerNorm.weight", &encoder->embedding_norm_weight, 1, {128, 0}},
        {"bert.module.embeddings.LayerNorm.bias", &encoder->embedding_norm_bias, 1, {128, 0}},
        {"bert.module.encoder.embedding_hidden_mapping_in.weight",
         &encoder->embedding_mapping_weight,
         2,
         {768, 128}},
        {"bert.module.encoder.embedding_hidden_mapping_in.bias",
         &encoder->embedding_mapping_bias,
         1,
         {768, 0}},
        {"bert.module.encoder.albert_layer_groups.0.albert_layers.0.attention.query.weight",
         &encoder->attention_query_weight,
         2,
         {768, 768}},
        {"bert.module.encoder.albert_layer_groups.0.albert_layers.0.attention.query.bias",
         &encoder->attention_query_bias,
         1,
         {768, 0}},
        {"bert.module.encoder.albert_layer_groups.0.albert_layers.0.attention.key.weight",
         &encoder->attention_key_weight,
         2,
         {768, 768}},
        {"bert.module.encoder.albert_layer_groups.0.albert_layers.0.attention.key.bias",
         &encoder->attention_key_bias,
         1,
         {768, 0}},
        {"bert.module.encoder.albert_layer_groups.0.albert_layers.0.attention.value.weight",
         &encoder->attention_value_weight,
         2,
         {768, 768}},
        {"bert.module.encoder.albert_layer_groups.0.albert_layers.0.attention.value.bias",
         &encoder->attention_value_bias,
         1,
         {768, 0}},
        {"bert.module.encoder.albert_layer_groups.0.albert_layers.0.attention.dense.weight",
         &encoder->attention_dense_weight,
         2,
         {768, 768}},
        {"bert.module.encoder.albert_layer_groups.0.albert_layers.0.attention.dense.bias",
         &encoder->attention_dense_bias,
         1,
         {768, 0}},
        {"bert.module.encoder.albert_layer_groups.0.albert_layers.0.attention.LayerNorm.weight",
         &encoder->attention_norm_weight,
         1,
         {768, 0}},
        {"bert.module.encoder.albert_layer_groups.0.albert_layers.0.attention.LayerNorm.bias",
         &encoder->attention_norm_bias,
         1,
         {768, 0}},
        {"bert.module.encoder.albert_layer_groups.0.albert_layers.0.full_layer_layer_norm.weight",
         &encoder->full_norm_weight,
         1,
         {768, 0}},
        {"bert.module.encoder.albert_layer_groups.0.albert_layers.0.full_layer_layer_norm.bias",
         &encoder->full_norm_bias,
         1,
         {768, 0}},
        {"bert.module.encoder.albert_layer_groups.0.albert_layers.0.ffn.weight",
         &encoder->ffn_weight,
         2,
         {2048, 768}},
        {"bert.module.encoder.albert_layer_groups.0.albert_layers.0.ffn.bias",
         &encoder->ffn_bias,
         1,
         {2048, 0}},
        {"bert.module.encoder.albert_layer_groups.0.albert_layers.0.ffn_output.weight",
         &encoder->ffn_output_weight,
         2,
         {768, 2048}},
        {"bert.module.encoder.albert_layer_groups.0.albert_layers.0.ffn_output.bias",
         &encoder->ffn_output_bias,
         1,
         {768, 0}},
        {"bert_encoder.module.weight", &encoder->projection_weight, 2, {512, 768}},
        {"bert_encoder.module.bias", &encoder->projection_bias, 1, {512, 0}},
    };
    for (size_t index = 0; index < sizeof(tensors) / sizeof(tensors[0]); index++) {
        KokoroStatus status = bind_tensor(pack, &tensors[index], error);
        if (status != KOKORO_OK) {
            free(encoder);
            return status;
        }
    }
    *result = encoder;
    kokoro_error_set(error, KOKORO_OK, NULL);
    return KOKORO_OK;
}

void kokoro_context_encoder_free(KokoroContextEncoder *encoder) { free(encoder); }

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

static void layer_norm(float *values, const float *weight, const float *bias, size_t rows,
                       size_t width) {
    for (size_t row = 0; row < rows; row++) {
        float *sample = values + row * width;
        float mean = 0.0f;
        for (size_t column = 0; column < width; column++) {
            mean += sample[column];
        }
        mean /= (float)width;
        float variance = 0.0f;
        for (size_t column = 0; column < width; column++) {
            float centered = sample[column] - mean;
            variance += centered * centered;
        }
        variance /= (float)width;
        float scale = 1.0f / sqrtf(variance + ALBERT_LAYER_NORM_EPSILON);
        for (size_t column = 0; column < width; column++) {
            sample[column] = (sample[column] - mean) * scale * weight[column] + bias[column];
        }
    }
}

static void apply_gelu_new(float *values, size_t count) {
    for (size_t index = 0; index < count; index++) {
        float value = values[index];
        float cubic = value * value * value;
        float inner = ALBERT_GELU_SCALE * (value + ALBERT_GELU_COEFFICIENT * cubic);
        values[index] = 0.5f * value * (1.0f + tanhf(inner));
    }
}

static void self_attention(const float *query, const float *key, const float *value, size_t length,
                           size_t valid_length, float *scores, float *attention) {
    const float scale = 1.0f / sqrtf((float)ALBERT_HEAD_SIZE);
    for (size_t head = 0; head < ALBERT_HEADS; head++) {
        const float *head_query = query + head * ALBERT_HEAD_SIZE;
        const float *head_key = key + head * ALBERT_HEAD_SIZE;
        const float *head_value = value + head * ALBERT_HEAD_SIZE;
        float *head_output = attention + head * ALBERT_HEAD_SIZE;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)length, (int)length,
                    (int)ALBERT_HEAD_SIZE, scale, head_query, (int)ALBERT_HIDDEN_SIZE, head_key,
                    (int)ALBERT_HIDDEN_SIZE, 0.0f, scores, (int)length);
        for (size_t row = 0; row < length; row++) {
            float *row_scores = scores + row * length;
            float maximum = -INFINITY;
            for (size_t column = 0; column < valid_length; column++) {
                if (row_scores[column] > maximum) {
                    maximum = row_scores[column];
                }
            }
            float total = 0.0f;
            for (size_t column = 0; column < length; column++) {
                if (column >= valid_length) {
                    row_scores[column] = 0.0f;
                } else {
                    row_scores[column] = expf(row_scores[column] - maximum);
                    total += row_scores[column];
                }
            }
            float reciprocal = 1.0f / total;
            for (size_t column = 0; column < valid_length; column++) {
                row_scores[column] *= reciprocal;
            }
        }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)length, (int)ALBERT_HEAD_SIZE,
                    (int)length, 1.0f, scores, (int)length, head_value, (int)ALBERT_HIDDEN_SIZE,
                    0.0f, head_output, (int)ALBERT_HIDDEN_SIZE);
    }
}

static bool all_finite(const float *values, size_t count) {
    for (size_t index = 0; index < count; index++) {
        if (!isfinite(values[index])) {
            return false;
        }
    }
    return true;
}

static bool allocate_scratch(size_t count, float **buffer) {
    if (count > SIZE_MAX / sizeof(float)) {
        return false;
    }
    *buffer = malloc(count * sizeof(float));
    return *buffer != NULL;
}

KokoroStatus kokoro_context_encode(const KokoroContextEncoder *encoder, const int64_t *token_ids,
                                   size_t sequence_length, size_t valid_length, float *output,
                                   size_t output_capacity, KokoroError *error) {
    if (encoder == NULL || token_ids == NULL || output == NULL) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "encoder, token IDs, and output buffer are required");
        return KOKORO_INVALID_ARGUMENT;
    }
    if (sequence_length == 0 || sequence_length > KOKORO_CONTEXT_MAX_TOKENS || valid_length == 0 ||
        valid_length > sequence_length) {
        kokoro_error_set(error, KOKORO_CONTEXT_TOO_LONG,
                         "sequence and valid lengths must be within 1..%u and valid length cannot "
                         "exceed sequence length",
                         KOKORO_CONTEXT_MAX_TOKENS);
        return KOKORO_CONTEXT_TOO_LONG;
    }
    if (sequence_length > SIZE_MAX / KOKORO_CONTEXT_FEATURES ||
        output_capacity < KOKORO_CONTEXT_FEATURES * sequence_length) {
        kokoro_error_set(error, KOKORO_BUFFER_TOO_SMALL, "output needs at least %zu float elements",
                         KOKORO_CONTEXT_FEATURES * sequence_length);
        return KOKORO_BUFFER_TOO_SMALL;
    }
    for (size_t index = 0; index < sequence_length; index++) {
        if (token_ids[index] < 0 || token_ids[index] >= ALBERT_VOCAB_SIZE) {
            kokoro_error_set(error, KOKORO_INVALID_TOKEN_ID,
                             "token ID at index %zu is outside [0, %u)", index, ALBERT_VOCAB_SIZE);
            return KOKORO_INVALID_TOKEN_ID;
        }
    }

    size_t hidden_count = sequence_length * ALBERT_HIDDEN_SIZE;
    size_t intermediate_count = sequence_length * ALBERT_INTERMEDIATE_SIZE;
    size_t score_count = sequence_length * sequence_length;
    size_t output_count = sequence_length * KOKORO_CONTEXT_FEATURES;
    float *hidden = NULL;
    float *query = NULL;
    float *key = NULL;
    float *value = NULL;
    float *attention = NULL;
    float *projected = NULL;
    float *intermediate = NULL;
    float *feed_forward = NULL;
    float *scores = NULL;
    float *features = NULL;
    bool allocated =
        allocate_scratch(hidden_count, &hidden) && allocate_scratch(hidden_count, &query) &&
        allocate_scratch(hidden_count, &key) && allocate_scratch(hidden_count, &value) &&
        allocate_scratch(hidden_count, &attention) && allocate_scratch(hidden_count, &projected) &&
        allocate_scratch(intermediate_count, &intermediate) &&
        allocate_scratch(hidden_count, &feed_forward) && allocate_scratch(score_count, &scores) &&
        allocate_scratch(output_count, &features);
    if (!allocated) {
        free(hidden);
        free(query);
        free(key);
        free(value);
        free(attention);
        free(projected);
        free(intermediate);
        free(feed_forward);
        free(scores);
        free(features);
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY, "cannot allocate context encoder workspace");
        return KOKORO_OUT_OF_MEMORY;
    }

    for (size_t token = 0; token < sequence_length; token++) {
        const float *word =
            encoder->word_embeddings + (size_t)token_ids[token] * ALBERT_EMBEDDING_SIZE;
        const float *position = encoder->position_embeddings + token * ALBERT_EMBEDDING_SIZE;
        const float *token_type = encoder->token_type_embeddings;
        float *embedding = hidden + token * ALBERT_EMBEDDING_SIZE;
        for (size_t column = 0; column < ALBERT_EMBEDDING_SIZE; column++) {
            embedding[column] = word[column] + position[column] + token_type[column];
        }
    }
    layer_norm(hidden, encoder->embedding_norm_weight, encoder->embedding_norm_bias,
               sequence_length, ALBERT_EMBEDDING_SIZE);
    linear(hidden, encoder->embedding_mapping_weight, encoder->embedding_mapping_bias,
           sequence_length, ALBERT_EMBEDDING_SIZE, ALBERT_HIDDEN_SIZE, projected);
    memcpy(hidden, projected, hidden_count * sizeof(float));

    for (size_t layer = 0; layer < ALBERT_LAYERS; layer++) {
        linear(hidden, encoder->attention_query_weight, encoder->attention_query_bias,
               sequence_length, ALBERT_HIDDEN_SIZE, ALBERT_HIDDEN_SIZE, query);
        linear(hidden, encoder->attention_key_weight, encoder->attention_key_bias, sequence_length,
               ALBERT_HIDDEN_SIZE, ALBERT_HIDDEN_SIZE, key);
        linear(hidden, encoder->attention_value_weight, encoder->attention_value_bias,
               sequence_length, ALBERT_HIDDEN_SIZE, ALBERT_HIDDEN_SIZE, value);
        self_attention(query, key, value, sequence_length, valid_length, scores, attention);
        linear(attention, encoder->attention_dense_weight, encoder->attention_dense_bias,
               sequence_length, ALBERT_HIDDEN_SIZE, ALBERT_HIDDEN_SIZE, projected);
        for (size_t index = 0; index < hidden_count; index++) {
            hidden[index] += projected[index];
        }
        layer_norm(hidden, encoder->attention_norm_weight, encoder->attention_norm_bias,
                   sequence_length, ALBERT_HIDDEN_SIZE);
        linear(hidden, encoder->ffn_weight, encoder->ffn_bias, sequence_length, ALBERT_HIDDEN_SIZE,
               ALBERT_INTERMEDIATE_SIZE, intermediate);
        apply_gelu_new(intermediate, intermediate_count);
        linear(intermediate, encoder->ffn_output_weight, encoder->ffn_output_bias, sequence_length,
               ALBERT_INTERMEDIATE_SIZE, ALBERT_HIDDEN_SIZE, feed_forward);
        for (size_t index = 0; index < hidden_count; index++) {
            hidden[index] += feed_forward[index];
        }
        layer_norm(hidden, encoder->full_norm_weight, encoder->full_norm_bias, sequence_length,
                   ALBERT_HIDDEN_SIZE);
    }
    linear(hidden, encoder->projection_weight, encoder->projection_bias, sequence_length,
           ALBERT_HIDDEN_SIZE, KOKORO_CONTEXT_FEATURES, features);
    if (!all_finite(features, output_count)) {
        free(hidden);
        free(query);
        free(key);
        free(value);
        free(attention);
        free(projected);
        free(intermediate);
        free(feed_forward);
        free(scores);
        free(features);
        kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                         "context encoder produced a non-finite feature value");
        return KOKORO_NUMERIC_ERROR;
    }
    for (size_t token = 0; token < sequence_length; token++) {
        for (size_t channel = 0; channel < KOKORO_CONTEXT_FEATURES; channel++) {
            output[channel * sequence_length + token] =
                features[token * KOKORO_CONTEXT_FEATURES + channel];
        }
    }
    free(hidden);
    free(query);
    free(key);
    free(value);
    free(attention);
    free(projected);
    free(intermediate);
    free(feed_forward);
    free(scores);
    free(features);
    kokoro_error_set(error, KOKORO_OK, NULL);
    return KOKORO_OK;
}

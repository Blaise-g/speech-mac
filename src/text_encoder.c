#include "text_encoder.h"

#include "kokoro_internal.h"
#include "lstm.h"
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

#define TEXT_ENCODER_VOCAB_SIZE 178U
#define TEXT_ENCODER_KERNEL_SIZE 5U
#define TEXT_ENCODER_BLOCK_COUNT 3U
#define TEXT_ENCODER_LSTM_WIDTH (KOKORO_TEXT_ENCODER_FEATURES / 2)
#define TEXT_ENCODER_LAYER_NORM_EPSILON 1.0e-5f
#define TEXT_ENCODER_LEAKY_RELU_SLOPE 0.2f
#define TEXT_ENCODER_CONV_REDUCTION (KOKORO_TEXT_ENCODER_FEATURES * TEXT_ENCODER_KERNEL_SIZE)

typedef struct {
    float *weight;
    const float *bias;
    const float *norm_weight;
    const float *norm_bias;
} TextEncoderBlock;

struct KokoroTextEncoder {
    const float *embedding;
    TextEncoderBlock blocks[TEXT_ENCODER_BLOCK_COUNT];
    KokoroBiLstmWeights lstm;
};

static bool allocate_floats(size_t count, float **values, bool clear) {
    if (count == 0 || count > SIZE_MAX / sizeof(float)) {
        return false;
    }
    *values = clear ? calloc(count, sizeof(float)) : malloc(count * sizeof(float));
    return *values != NULL;
}

static KokoroStatus bind_block(const KokoroPack *pack, size_t block_index, TextEncoderBlock *block,
                               KokoroError *error) {
    char name[160];
    int length =
        snprintf(name, sizeof(name), "text_encoder.module.cnn.%zu.0.weight_g", block_index);
    if (length < 0 || (size_t)length >= sizeof(name)) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "text encoder tensor name is too long");
        return KOKORO_INCOMPATIBLE_PACK;
    }
    const uint64_t norm_shape[3] = {KOKORO_TEXT_ENCODER_FEATURES, 1, 1};
    const float *norm = NULL;
    KokoroStatus status = kokoro_tensor_bind_f32(pack, name, 3, norm_shape, &norm, error);
    if (status != KOKORO_OK) {
        return status;
    }

    length = snprintf(name, sizeof(name), "text_encoder.module.cnn.%zu.0.weight_v", block_index);
    if (length < 0 || (size_t)length >= sizeof(name)) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "text encoder tensor name is too long");
        return KOKORO_INCOMPATIBLE_PACK;
    }
    const uint64_t weight_shape[3] = {KOKORO_TEXT_ENCODER_FEATURES, KOKORO_TEXT_ENCODER_FEATURES,
                                      TEXT_ENCODER_KERNEL_SIZE};
    const float *raw_weight = NULL;
    status = kokoro_tensor_bind_f32(pack, name, 3, weight_shape, &raw_weight, error);
    if (status != KOKORO_OK) {
        return status;
    }

    length = snprintf(name, sizeof(name), "text_encoder.module.cnn.%zu.0.bias", block_index);
    if (length < 0 || (size_t)length >= sizeof(name)) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "text encoder tensor name is too long");
        return KOKORO_INCOMPATIBLE_PACK;
    }
    const uint64_t channel_shape[1] = {KOKORO_TEXT_ENCODER_FEATURES};
    status = kokoro_tensor_bind_f32(pack, name, 1, channel_shape, &block->bias, error);
    if (status != KOKORO_OK) {
        return status;
    }

    length = snprintf(name, sizeof(name), "text_encoder.module.cnn.%zu.1.gamma", block_index);
    if (length < 0 || (size_t)length >= sizeof(name)) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "text encoder tensor name is too long");
        return KOKORO_INCOMPATIBLE_PACK;
    }
    status = kokoro_tensor_bind_f32(pack, name, 1, channel_shape, &block->norm_weight, error);
    if (status != KOKORO_OK) {
        return status;
    }
    length = snprintf(name, sizeof(name), "text_encoder.module.cnn.%zu.1.beta", block_index);
    if (length < 0 || (size_t)length >= sizeof(name)) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "text encoder tensor name is too long");
        return KOKORO_INCOMPATIBLE_PACK;
    }
    status = kokoro_tensor_bind_f32(pack, name, 1, channel_shape, &block->norm_bias, error);
    if (status != KOKORO_OK) {
        return status;
    }

    float *effective_weight = NULL;
    if (!allocate_floats(KOKORO_TEXT_ENCODER_FEATURES * TEXT_ENCODER_CONV_REDUCTION,
                         &effective_weight, false)) {
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY,
                         "cannot allocate text encoder convolution weights");
        return KOKORO_OUT_OF_MEMORY;
    }
    for (size_t output = 0; output < KOKORO_TEXT_ENCODER_FEATURES; output++) {
        const float *source = raw_weight + output * TEXT_ENCODER_CONV_REDUCTION;
        double squared_norm = 0.0;
        for (size_t index = 0; index < TEXT_ENCODER_CONV_REDUCTION; index++) {
            squared_norm += (double)source[index] * source[index];
        }
        double magnitude = sqrt(squared_norm);
        if (!(magnitude > 0.0) || !isfinite(magnitude)) {
            free(effective_weight);
            kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK,
                             "text encoder convolution has a zero or invalid weight norm");
            return KOKORO_INCOMPATIBLE_PACK;
        }
        float scale = (float)((double)norm[output] / magnitude);
        float *target = effective_weight + output * TEXT_ENCODER_CONV_REDUCTION;
        for (size_t kernel = 0; kernel < TEXT_ENCODER_KERNEL_SIZE; kernel++) {
            for (size_t input = 0; input < KOKORO_TEXT_ENCODER_FEATURES; input++) {
                size_t source_index = input * TEXT_ENCODER_KERNEL_SIZE + kernel;
                size_t target_index = kernel * KOKORO_TEXT_ENCODER_FEATURES + input;
                target[target_index] = source[source_index] * scale;
            }
        }
    }
    block->weight = effective_weight;
    return KOKORO_OK;
}

KokoroStatus kokoro_text_encoder_create(const KokoroPack *pack, KokoroTextEncoder **result,
                                        KokoroError *error) {
    if (result != NULL) {
        *result = NULL;
    }
    if (pack == NULL || result == NULL) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "model pack and text encoder output are required");
        return KOKORO_INVALID_ARGUMENT;
    }
    KokoroTextEncoder *encoder = calloc(1, sizeof(*encoder));
    if (encoder == NULL) {
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY, "cannot allocate text encoder");
        return KOKORO_OUT_OF_MEMORY;
    }

    const uint64_t embedding_shape[2] = {TEXT_ENCODER_VOCAB_SIZE, KOKORO_TEXT_ENCODER_FEATURES};
    KokoroStatus status = kokoro_tensor_bind_f32(pack, "text_encoder.module.embedding.weight", 2,
                                                 embedding_shape, &encoder->embedding, error);
    for (size_t block = 0; status == KOKORO_OK && block < TEXT_ENCODER_BLOCK_COUNT; block++) {
        status = bind_block(pack, block, &encoder->blocks[block], error);
    }
    if (status == KOKORO_OK) {
        status = kokoro_bilstm_bind(pack, "text_encoder.module.lstm", KOKORO_TEXT_ENCODER_FEATURES,
                                    TEXT_ENCODER_LSTM_WIDTH, &encoder->lstm, error);
    }
    if (status != KOKORO_OK) {
        kokoro_text_encoder_free(encoder);
        return status;
    }
    *result = encoder;
    kokoro_error_set(error, KOKORO_OK, NULL);
    return KOKORO_OK;
}

void kokoro_text_encoder_free(KokoroTextEncoder *encoder) {
    if (encoder == NULL) {
        return;
    }
    for (size_t block = 0; block < TEXT_ENCODER_BLOCK_COUNT; block++) {
        free(encoder->blocks[block].weight);
    }
    free(encoder);
}

static void run_convolution(const TextEncoderBlock *block, const float *input, size_t token_count,
                            float *patches, float *output) {
    const size_t patch_width = TEXT_ENCODER_CONV_REDUCTION;
    for (size_t token = 0; token < token_count; token++) {
        float *patch = patches + token * patch_width;
        for (size_t kernel = 0; kernel < TEXT_ENCODER_KERNEL_SIZE; kernel++) {
            int64_t source_token = (int64_t)token + (int64_t)kernel - 2;
            float *patch_row = patch + kernel * KOKORO_TEXT_ENCODER_FEATURES;
            if (source_token < 0 || (uint64_t)source_token >= token_count) {
                memset(patch_row, 0, KOKORO_TEXT_ENCODER_FEATURES * sizeof(float));
            } else {
                memcpy(patch_row, input + (size_t)source_token * KOKORO_TEXT_ENCODER_FEATURES,
                       KOKORO_TEXT_ENCODER_FEATURES * sizeof(float));
            }
        }
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)token_count,
                (int)KOKORO_TEXT_ENCODER_FEATURES, (int)patch_width, 1.0f, patches,
                (int)patch_width, block->weight, (int)patch_width, 0.0f, output,
                (int)KOKORO_TEXT_ENCODER_FEATURES);
    for (size_t token = 0; token < token_count; token++) {
        float *values = output + token * KOKORO_TEXT_ENCODER_FEATURES;
        for (size_t channel = 0; channel < KOKORO_TEXT_ENCODER_FEATURES; channel++) {
            values[channel] += block->bias[channel];
        }
        float mean = 0.0f;
        for (size_t channel = 0; channel < KOKORO_TEXT_ENCODER_FEATURES; channel++) {
            mean += values[channel];
        }
        mean /= (float)KOKORO_TEXT_ENCODER_FEATURES;
        float variance = 0.0f;
        for (size_t channel = 0; channel < KOKORO_TEXT_ENCODER_FEATURES; channel++) {
            float centered = values[channel] - mean;
            variance += centered * centered;
        }
        variance /= (float)KOKORO_TEXT_ENCODER_FEATURES;
        float inverse_scale = 1.0f / sqrtf(variance + TEXT_ENCODER_LAYER_NORM_EPSILON);
        for (size_t channel = 0; channel < KOKORO_TEXT_ENCODER_FEATURES; channel++) {
            float value = (values[channel] - mean) * inverse_scale * block->norm_weight[channel] +
                          block->norm_bias[channel];
            if (value < 0.0f) {
                value *= TEXT_ENCODER_LEAKY_RELU_SLOPE;
            }
            values[channel] = value;
        }
    }
}

static bool output_is_empty(const KokoroTextEncoderOutput *output) {
    return output->token_count == 0 && output->frame_count == 0 && output->text_features == NULL &&
           output->aligned_features == NULL;
}

void kokoro_text_encoder_output_free(KokoroTextEncoderOutput *output) {
    if (output == NULL) {
        return;
    }
    free(output->text_features);
    free(output->aligned_features);
    memset(output, 0, sizeof(*output));
}

KokoroStatus kokoro_text_encoder_run(const KokoroTextEncoder *encoder, const int64_t *token_ids,
                                     size_t sequence_length, size_t valid_length,
                                     const KokoroDurationOutput *alignment,
                                     KokoroTextEncoderOutput *output, KokoroError *error) {
    if (encoder == NULL || token_ids == NULL || alignment == NULL || output == NULL) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "encoder, token IDs, alignment, and output are required");
        return KOKORO_INVALID_ARGUMENT;
    }
    if (!output_is_empty(output)) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "text encoder output must be empty before use");
        return KOKORO_INVALID_ARGUMENT;
    }
    if (sequence_length == 0 || sequence_length > KOKORO_TEXT_ENCODER_MAX_TOKENS ||
        valid_length == 0 || valid_length > sequence_length ||
        sequence_length != alignment->token_count || alignment->frame_count == 0 ||
        alignment->frame_count > KOKORO_TEXT_ENCODER_MAX_FRAMES || alignment->alignment == NULL) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "text and alignment lengths are outside the supported range");
        return KOKORO_INVALID_ARGUMENT;
    }
    if (alignment->frame_count > INT_MAX ||
        sequence_length > SIZE_MAX / KOKORO_TEXT_ENCODER_FEATURES ||
        alignment->frame_count > SIZE_MAX / KOKORO_TEXT_ENCODER_FEATURES ||
        sequence_length > SIZE_MAX / alignment->frame_count ||
        sequence_length * alignment->frame_count > SIZE_MAX / sizeof(float)) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "text or alignment dimensions exceed addressable memory");
        return KOKORO_INVALID_ARGUMENT;
    }
    for (size_t token = 0; token < sequence_length; token++) {
        if (token_ids[token] < 0 || token_ids[token] >= TEXT_ENCODER_VOCAB_SIZE) {
            kokoro_error_set(error, KOKORO_INVALID_TOKEN_ID,
                             "token ID at index %zu is outside [0, %u)", token,
                             TEXT_ENCODER_VOCAB_SIZE);
            return KOKORO_INVALID_TOKEN_ID;
        }
    }
    for (size_t frame = 0; frame < alignment->frame_count; frame++) {
        size_t mapped_tokens = 0;
        for (size_t token = 0; token < sequence_length; token++) {
            float value = alignment->alignment[token * alignment->frame_count + frame];
            if (!isfinite(value)) {
                kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                                 "alignment contains a non-finite value at token %zu frame %zu",
                                 token, frame);
                return KOKORO_NUMERIC_ERROR;
            }
            if (value == 1.0f) {
                mapped_tokens++;
            } else if (value != 0.0f) {
                kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                                 "alignment must contain one-hot token columns");
                return KOKORO_INVALID_ARGUMENT;
            }
        }
        if (mapped_tokens != 1) {
            kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                             "each aligned frame must map to exactly one token");
            return KOKORO_INVALID_ARGUMENT;
        }
    }

    const size_t feature_count = sequence_length * KOKORO_TEXT_ENCODER_FEATURES;
    const size_t patch_count = sequence_length * TEXT_ENCODER_CONV_REDUCTION;
    const size_t aligned_count = alignment->frame_count * KOKORO_TEXT_ENCODER_FEATURES;
    float *current = NULL;
    float *next = NULL;
    float *patches = NULL;
    float *recurrent = NULL;
    KokoroTextEncoderOutput result = {.token_count = sequence_length,
                                      .frame_count = alignment->frame_count};
    bool allocated = allocate_floats(feature_count, &current, true) &&
                     allocate_floats(feature_count, &next, false) &&
                     allocate_floats(patch_count, &patches, false) &&
                     allocate_floats(feature_count, &recurrent, true) &&
                     allocate_floats(feature_count, &result.text_features, false) &&
                     allocate_floats(aligned_count, &result.aligned_features, false);
    if (!allocated) {
        free(current);
        free(next);
        free(patches);
        free(recurrent);
        kokoro_text_encoder_output_free(&result);
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY, "cannot allocate text encoder workspace");
        return KOKORO_OUT_OF_MEMORY;
    }

    for (size_t token = 0; token < valid_length; token++) {
        memcpy(current + token * KOKORO_TEXT_ENCODER_FEATURES,
               encoder->embedding + (size_t)token_ids[token] * KOKORO_TEXT_ENCODER_FEATURES,
               KOKORO_TEXT_ENCODER_FEATURES * sizeof(float));
    }
    for (size_t block = 0; block < TEXT_ENCODER_BLOCK_COUNT; block++) {
        run_convolution(&encoder->blocks[block], current, sequence_length, patches, next);
        for (size_t token = valid_length; token < sequence_length; token++) {
            memset(next + token * KOKORO_TEXT_ENCODER_FEATURES, 0,
                   KOKORO_TEXT_ENCODER_FEATURES * sizeof(float));
        }
        float *previous = current;
        current = next;
        next = previous;
    }

    KokoroStatus status =
        kokoro_bilstm_run(&encoder->lstm, current, valid_length, KOKORO_TEXT_ENCODER_FEATURES,
                          TEXT_ENCODER_LSTM_WIDTH, recurrent, error);
    if (status != KOKORO_OK) {
        free(current);
        free(next);
        free(patches);
        free(recurrent);
        kokoro_text_encoder_output_free(&result);
        return status;
    }
    for (size_t token = 0; token < sequence_length; token++) {
        for (size_t channel = 0; channel < KOKORO_TEXT_ENCODER_FEATURES; channel++) {
            result.text_features[channel * sequence_length + token] =
                recurrent[token * KOKORO_TEXT_ENCODER_FEATURES + channel];
        }
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)KOKORO_TEXT_ENCODER_FEATURES,
                (int)alignment->frame_count, (int)sequence_length, 1.0f, result.text_features,
                (int)sequence_length, alignment->alignment, (int)alignment->frame_count, 0.0f,
                result.aligned_features, (int)alignment->frame_count);
    free(current);
    free(next);
    free(patches);
    free(recurrent);

    for (size_t index = 0; index < feature_count; index++) {
        if (!isfinite(result.text_features[index])) {
            kokoro_text_encoder_output_free(&result);
            kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                             "text encoder produced a non-finite feature value");
            return KOKORO_NUMERIC_ERROR;
        }
    }
    for (size_t index = 0; index < aligned_count; index++) {
        if (!isfinite(result.aligned_features[index])) {
            kokoro_text_encoder_output_free(&result);
            kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                             "aligned text encoder produced a non-finite feature value");
            return KOKORO_NUMERIC_ERROR;
        }
    }
    *output = result;
    kokoro_error_set(error, KOKORO_OK, NULL);
    return KOKORO_OK;
}

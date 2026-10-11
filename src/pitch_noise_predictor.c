#include "pitch_noise_predictor.h"

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

#define PITCH_NOISE_STYLE_SIZE 128U
#define PITCH_NOISE_VOICE_SIZE 256U
#define PITCH_NOISE_HIDDEN_SIZE 512U
#define PITCH_NOISE_OUTPUT_CHANNELS 256U
#define PITCH_NOISE_MAX_TOKENS KOKORO_DURATION_MAX_TOKENS
#define PITCH_NOISE_BLOCK_COUNT 3U
#define PITCH_NOISE_ADAIN_EPSILON 1.0e-5f
#define PITCH_NOISE_LEAKY_RELU 0.2f
#define PITCH_NOISE_RESIDUAL_SCALE 0.7071067811865475f

typedef struct {
    size_t input_channels;
    size_t output_channels;
    size_t kernel_size;
    float *weight;
    const float *bias;
} PitchNoiseConv;

typedef struct {
    const float *weight;
    const float *bias;
} PitchNoiseAdaIN;

typedef struct {
    PitchNoiseConv conv1;
    PitchNoiseConv conv2;
    PitchNoiseConv shortcut;
    PitchNoiseConv pool;
    PitchNoiseAdaIN norm1;
    PitchNoiseAdaIN norm2;
    bool upsample;
} PitchNoiseBlock;

struct KokoroPitchNoisePredictor {
    const float *voice_styles;
    size_t voice_style_count;
    KokoroBiLstmWeights shared_lstm;
    PitchNoiseBlock f0_blocks[PITCH_NOISE_BLOCK_COUNT];
    PitchNoiseBlock noise_blocks[PITCH_NOISE_BLOCK_COUNT];
    const float *f0_projection_weight;
    const float *f0_projection_bias;
    const float *noise_projection_weight;
    const float *noise_projection_bias;
};

static bool allocate_floats(size_t count, float **values) {
    if (count == 0 || count > SIZE_MAX / sizeof(float)) {
        return false;
    }
    *values = malloc(count * sizeof(float));
    return *values != NULL;
}

static KokoroStatus bind_weight_norm(PitchNoiseConv *conv, const KokoroPack *pack,
                                     const char *prefix, size_t input_channels,
                                     size_t output_channels, size_t kernel_size, bool has_bias,
                                     KokoroError *error) {
    if (input_channels == 0 || output_channels == 0 || kernel_size == 0 ||
        input_channels > SIZE_MAX / kernel_size || output_channels > SIZE_MAX / input_channels ||
        output_channels * input_channels > SIZE_MAX / kernel_size || output_channels > INT_MAX ||
        input_channels > INT_MAX || kernel_size > INT_MAX) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "invalid convolution dimensions: %s",
                         prefix);
        return KOKORO_INCOMPATIBLE_PACK;
    }
    const uint64_t weight_shape[3] = {output_channels, input_channels, kernel_size};
    const uint64_t norm_shape[3] = {output_channels, 1, 1};
    const uint64_t bias_shape[1] = {output_channels};
    const float *norm = NULL;
    const float *raw_weight = NULL;
    const float *bias = NULL;
    char name[160];
    int length = snprintf(name, sizeof(name), "%s.weight_g", prefix);
    if (length < 0 || (size_t)length >= sizeof(name)) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "predictor tensor name is too long");
        return KOKORO_INCOMPATIBLE_PACK;
    }
    KokoroStatus status = kokoro_tensor_bind_f32(pack, name, 3, norm_shape, &norm, error);
    if (status != KOKORO_OK) {
        return status;
    }
    length = snprintf(name, sizeof(name), "%s.weight_v", prefix);
    if (length < 0 || (size_t)length >= sizeof(name)) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "predictor tensor name is too long");
        return KOKORO_INCOMPATIBLE_PACK;
    }
    status = kokoro_tensor_bind_f32(pack, name, 3, weight_shape, &raw_weight, error);
    if (status != KOKORO_OK) {
        return status;
    }
    if (has_bias) {
        length = snprintf(name, sizeof(name), "%s.bias", prefix);
        if (length < 0 || (size_t)length >= sizeof(name)) {
            kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "predictor tensor name is too long");
            return KOKORO_INCOMPATIBLE_PACK;
        }
        status = kokoro_tensor_bind_f32(pack, name, 1, bias_shape, &bias, error);
        if (status != KOKORO_OK) {
            return status;
        }
    }

    size_t weight_count = output_channels * input_channels * kernel_size;
    float *effective_weight = NULL;
    if (!allocate_floats(weight_count, &effective_weight)) {
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY, "cannot allocate convolution weights");
        return KOKORO_OUT_OF_MEMORY;
    }
    size_t row_size = input_channels * kernel_size;
    for (size_t output = 0; output < output_channels; output++) {
        const float *source = raw_weight + output * row_size;
        double squared_norm = 0.0;
        for (size_t index = 0; index < row_size; index++) {
            squared_norm += (double)source[index] * source[index];
        }
        double magnitude = sqrt(squared_norm);
        if (!(magnitude > 0.0) || !isfinite(magnitude)) {
            free(effective_weight);
            kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK,
                             "weight-normalized convolution has a zero or invalid norm: %s",
                             prefix);
            return KOKORO_INCOMPATIBLE_PACK;
        }
        float scale = (float)((double)norm[output] / magnitude);
        for (size_t input = 0; input < input_channels; input++) {
            for (size_t kernel = 0; kernel < kernel_size; kernel++) {
                size_t source_index = input * kernel_size + kernel;
                size_t target_index = (kernel * output_channels + output) * input_channels + input;
                effective_weight[target_index] = source[source_index] * scale;
            }
        }
    }
    conv->input_channels = input_channels;
    conv->output_channels = output_channels;
    conv->kernel_size = kernel_size;
    conv->weight = effective_weight;
    conv->bias = bias;
    return KOKORO_OK;
}

static KokoroStatus bind_adain(PitchNoiseAdaIN *norm, const KokoroPack *pack, const char *prefix,
                               size_t channels, KokoroError *error) {
    if (channels > SIZE_MAX / 2) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "AdaIN dimensions overflow");
        return KOKORO_INCOMPATIBLE_PACK;
    }
    uint64_t weight_shape[2] = {channels * 2, PITCH_NOISE_STYLE_SIZE};
    uint64_t bias_shape[1] = {channels * 2};
    const float *weight = NULL;
    const float *bias = NULL;
    char name[160];
    int length = snprintf(name, sizeof(name), "%s.fc.weight", prefix);
    if (length < 0 || (size_t)length >= sizeof(name)) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "predictor tensor name is too long");
        return KOKORO_INCOMPATIBLE_PACK;
    }
    KokoroStatus status = kokoro_tensor_bind_f32(pack, name, 2, weight_shape, &weight, error);
    if (status != KOKORO_OK) {
        return status;
    }
    length = snprintf(name, sizeof(name), "%s.fc.bias", prefix);
    if (length < 0 || (size_t)length >= sizeof(name)) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "predictor tensor name is too long");
        return KOKORO_INCOMPATIBLE_PACK;
    }
    status = kokoro_tensor_bind_f32(pack, name, 1, bias_shape, &bias, error);
    if (status != KOKORO_OK) {
        return status;
    }
    norm->weight = weight;
    norm->bias = bias;
    return KOKORO_OK;
}

static KokoroStatus bind_block(PitchNoiseBlock *block, const KokoroPack *pack, const char *prefix,
                               size_t input_channels, size_t output_channels, bool upsample,
                               KokoroError *error) {
    char name[160];
    int length = snprintf(name, sizeof(name), "%s.conv1", prefix);
    if (length < 0 || (size_t)length >= sizeof(name)) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "predictor tensor name is too long");
        return KOKORO_INCOMPATIBLE_PACK;
    }
    KokoroStatus status = bind_weight_norm(&block->conv1, pack, name, input_channels,
                                           output_channels, 3, true, error);
    if (status != KOKORO_OK) {
        return status;
    }
    length = snprintf(name, sizeof(name), "%s.conv2", prefix);
    if (length < 0 || (size_t)length >= sizeof(name)) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "predictor tensor name is too long");
        return KOKORO_INCOMPATIBLE_PACK;
    }
    status = bind_weight_norm(&block->conv2, pack, name, output_channels, output_channels, 3, true,
                              error);
    if (status != KOKORO_OK) {
        return status;
    }
    length = snprintf(name, sizeof(name), "%s.norm1", prefix);
    if (length < 0 || (size_t)length >= sizeof(name)) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "predictor tensor name is too long");
        return KOKORO_INCOMPATIBLE_PACK;
    }
    status = bind_adain(&block->norm1, pack, name, input_channels, error);
    if (status != KOKORO_OK) {
        return status;
    }
    length = snprintf(name, sizeof(name), "%s.norm2", prefix);
    if (length < 0 || (size_t)length >= sizeof(name)) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "predictor tensor name is too long");
        return KOKORO_INCOMPATIBLE_PACK;
    }
    status = bind_adain(&block->norm2, pack, name, output_channels, error);
    if (status != KOKORO_OK) {
        return status;
    }
    if (input_channels != output_channels) {
        length = snprintf(name, sizeof(name), "%s.conv1x1", prefix);
        if (length < 0 || (size_t)length >= sizeof(name)) {
            kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "predictor tensor name is too long");
            return KOKORO_INCOMPATIBLE_PACK;
        }
        status = bind_weight_norm(&block->shortcut, pack, name, input_channels, output_channels, 1,
                                  false, error);
        if (status != KOKORO_OK) {
            return status;
        }
    }
    if (upsample) {
        length = snprintf(name, sizeof(name), "%s.pool", prefix);
        if (length < 0 || (size_t)length >= sizeof(name)) {
            kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "predictor tensor name is too long");
            return KOKORO_INCOMPATIBLE_PACK;
        }
        status = bind_weight_norm(&block->pool, pack, name, 1, input_channels, 3, true, error);
        if (status != KOKORO_OK) {
            return status;
        }
    }
    block->upsample = upsample;
    return KOKORO_OK;
}

KokoroStatus kokoro_pitch_noise_predictor_create(const KokoroPack *pack,
                                                 KokoroPitchNoisePredictor **result,
                                                 KokoroError *error) {
    if (result != NULL) {
        *result = NULL;
    }
    if (pack == NULL || result == NULL) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "model pack and pitch/noise predictor output are required");
        return KOKORO_INVALID_ARGUMENT;
    }
    KokoroPitchNoisePredictor *predictor = calloc(1, sizeof(*predictor));
    if (predictor == NULL) {
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY, "cannot allocate pitch/noise predictor");
        return KOKORO_OUT_OF_MEMORY;
    }
    KokoroTensorView voice_view;
    KokoroStatus status = kokoro_pack_tensor(pack, "voice.af_heart.style", &voice_view, error);
    if (status != KOKORO_OK) {
        free(predictor);
        return status;
    }
    if (voice_view.type != KOKORO_TENSOR_F32 || voice_view.rank != 3 || voice_view.shape[0] == 0 ||
        voice_view.shape[1] != 1 || voice_view.shape[2] != PITCH_NOISE_VOICE_SIZE ||
        voice_view.shape[0] > SIZE_MAX / PITCH_NOISE_VOICE_SIZE) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK,
                         "af_heart style tensor must have shape [rows, 1, 256]");
        free(predictor);
        return KOKORO_INCOMPATIBLE_PACK;
    }
    predictor->voice_styles = voice_view.data;
    predictor->voice_style_count = (size_t)voice_view.shape[0];
    for (size_t index = 0; index < predictor->voice_style_count * PITCH_NOISE_VOICE_SIZE; index++) {
        if (!isfinite(predictor->voice_styles[index])) {
            kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK,
                             "af_heart style tensor contains a non-finite value");
            kokoro_pitch_noise_predictor_free(predictor);
            return KOKORO_INCOMPATIBLE_PACK;
        }
    }

    status = kokoro_bilstm_bind(pack, "predictor.module.shared", KOKORO_DURATION_FEATURES,
                                PITCH_NOISE_HIDDEN_SIZE / 2, &predictor->shared_lstm, error);
    if (status != KOKORO_OK) {
        kokoro_pitch_noise_predictor_free(predictor);
        return status;
    }
    static const size_t input_channels[PITCH_NOISE_BLOCK_COUNT] = {512, 512, 256};
    static const size_t output_channels[PITCH_NOISE_BLOCK_COUNT] = {512, 256, 256};
    for (size_t block = 0; block < PITCH_NOISE_BLOCK_COUNT; block++) {
        bool upsample = block == 1;
        char prefix[128];
        int length = snprintf(prefix, sizeof(prefix), "predictor.module.F0.%zu", block);
        if (length < 0 || (size_t)length >= sizeof(prefix)) {
            kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "predictor tensor name is too long");
            kokoro_pitch_noise_predictor_free(predictor);
            return KOKORO_INCOMPATIBLE_PACK;
        }
        status = bind_block(&predictor->f0_blocks[block], pack, prefix, input_channels[block],
                            output_channels[block], upsample, error);
        if (status != KOKORO_OK) {
            kokoro_pitch_noise_predictor_free(predictor);
            return status;
        }
        length = snprintf(prefix, sizeof(prefix), "predictor.module.N.%zu", block);
        if (length < 0 || (size_t)length >= sizeof(prefix)) {
            kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "predictor tensor name is too long");
            kokoro_pitch_noise_predictor_free(predictor);
            return KOKORO_INCOMPATIBLE_PACK;
        }
        status = bind_block(&predictor->noise_blocks[block], pack, prefix, input_channels[block],
                            output_channels[block], upsample, error);
        if (status != KOKORO_OK) {
            kokoro_pitch_noise_predictor_free(predictor);
            return status;
        }
    }
    const uint64_t projection_shape[3] = {1, PITCH_NOISE_OUTPUT_CHANNELS, 1};
    const uint64_t projection_bias_shape[1] = {1};
    status = kokoro_tensor_bind_f32(pack, "predictor.module.F0_proj.weight", 3, projection_shape,
                                    &predictor->f0_projection_weight, error);
    if (status == KOKORO_OK) {
        status =
            kokoro_tensor_bind_f32(pack, "predictor.module.F0_proj.bias", 1, projection_bias_shape,
                                   &predictor->f0_projection_bias, error);
    }
    if (status == KOKORO_OK) {
        status = kokoro_tensor_bind_f32(pack, "predictor.module.N_proj.weight", 3, projection_shape,
                                        &predictor->noise_projection_weight, error);
    }
    if (status == KOKORO_OK) {
        status =
            kokoro_tensor_bind_f32(pack, "predictor.module.N_proj.bias", 1, projection_bias_shape,
                                   &predictor->noise_projection_bias, error);
    }
    if (status != KOKORO_OK) {
        kokoro_pitch_noise_predictor_free(predictor);
        return status;
    }
    *result = predictor;
    kokoro_error_set(error, KOKORO_OK, NULL);
    return KOKORO_OK;
}

void kokoro_pitch_noise_predictor_free(KokoroPitchNoisePredictor *predictor) {
    if (predictor == NULL) {
        return;
    }
    for (size_t block = 0; block < PITCH_NOISE_BLOCK_COUNT; block++) {
        PitchNoiseBlock *f0 = &predictor->f0_blocks[block];
        PitchNoiseBlock *noise = &predictor->noise_blocks[block];
        free(f0->conv1.weight);
        free(f0->conv2.weight);
        free(f0->shortcut.weight);
        free(f0->pool.weight);
        free(noise->conv1.weight);
        free(noise->conv2.weight);
        free(noise->shortcut.weight);
        free(noise->pool.weight);
    }
    free(predictor);
}

static void apply_leaky_relu(float *values, size_t count) {
    for (size_t index = 0; index < count; index++) {
        if (values[index] < 0.0f) {
            values[index] *= PITCH_NOISE_LEAKY_RELU;
        }
    }
}

static void apply_adain(const PitchNoiseAdaIN *norm, const float *input, const float *style,
                        size_t channels, size_t frame_count, float *output) {
    float affine[PITCH_NOISE_HIDDEN_SIZE * 2];
    cblas_sgemv(CblasRowMajor, CblasNoTrans, (int)(channels * 2), PITCH_NOISE_STYLE_SIZE, 1.0f,
                norm->weight, PITCH_NOISE_STYLE_SIZE, style, 1, 0.0f, affine, 1);
    for (size_t channel = 0; channel < channels * 2; channel++) {
        affine[channel] += norm->bias[channel];
    }
    for (size_t channel = 0; channel < channels; channel++) {
        const float *source = input + channel * frame_count;
        float *target = output + channel * frame_count;
        float mean = 0.0f;
        for (size_t frame = 0; frame < frame_count; frame++) {
            mean += source[frame];
        }
        mean /= (float)frame_count;
        float variance = 0.0f;
        for (size_t frame = 0; frame < frame_count; frame++) {
            float centered = source[frame] - mean;
            variance += centered * centered;
        }
        variance /= (float)frame_count;
        float inverse_scale = 1.0f / sqrtf(variance + PITCH_NOISE_ADAIN_EPSILON);
        float gamma = affine[channel];
        float beta = affine[channels + channel];
        for (size_t frame = 0; frame < frame_count; frame++) {
            float normalized = (source[frame] - mean) * inverse_scale;
            /* The pinned checkpoint omits InstanceNorm affine tensors; PyTorch keeps their
             * initialized identity scale and zero bias. */
            target[frame] = (1.0f + gamma) * normalized + beta;
        }
    }
}

static KokoroStatus run_convolution(const PitchNoiseConv *conv, const float *input,
                                    size_t frame_count, float *output, KokoroError *error) {
    if (frame_count > INT_MAX - 2 || conv->output_channels > INT_MAX ||
        conv->input_channels > INT_MAX || conv->kernel_size == 0 || conv->kernel_size > 3) {
        kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                         "convolution dimensions exceed supported range");
        return KOKORO_NUMERIC_ERROR;
    }
    if (conv->kernel_size == 1) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)conv->output_channels,
                    (int)frame_count, (int)conv->input_channels, 1.0f, conv->weight,
                    (int)conv->input_channels, input, (int)frame_count, 0.0f, output,
                    (int)frame_count);
    } else {
        float *padded = NULL;
        if (frame_count > SIZE_MAX - 2 || conv->input_channels > SIZE_MAX / (frame_count + 2) ||
            conv->input_channels * (frame_count + 2) > SIZE_MAX / sizeof(float)) {
            kokoro_error_set(error, KOKORO_NUMERIC_ERROR, "convolution workspace size overflows");
            return KOKORO_NUMERIC_ERROR;
        }
        padded = calloc(conv->input_channels * (frame_count + 2), sizeof(float));
        if (padded == NULL) {
            free(padded);
            kokoro_error_set(error, KOKORO_OUT_OF_MEMORY, "cannot allocate convolution workspace");
            return KOKORO_OUT_OF_MEMORY;
        }
        for (size_t channel = 0; channel < conv->input_channels; channel++) {
            memcpy(padded + channel * (frame_count + 2) + 1, input + channel * frame_count,
                   frame_count * sizeof(float));
        }
        for (size_t kernel = 0; kernel < conv->kernel_size; kernel++) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)conv->output_channels,
                        (int)frame_count, (int)conv->input_channels, 1.0f,
                        conv->weight + kernel * conv->output_channels * conv->input_channels,
                        (int)conv->input_channels, padded + kernel, (int)(frame_count + 2),
                        kernel == 0 ? 0.0f : 1.0f, output, (int)frame_count);
        }
        free(padded);
    }
    if (conv->bias != NULL) {
        for (size_t channel = 0; channel < conv->output_channels; channel++) {
            float *row = output + channel * frame_count;
            for (size_t frame = 0; frame < frame_count; frame++) {
                row[frame] += conv->bias[channel];
            }
        }
    }
    return KOKORO_OK;
}

static KokoroStatus run_depthwise_upsample(const PitchNoiseConv *conv, const float *input,
                                           size_t input_frames, float *output, KokoroError *error) {
    if (input_frames > SIZE_MAX / 2 || conv->kernel_size != 3 || conv->input_channels != 1 ||
        conv->output_channels > INT_MAX || input_frames > INT_MAX / 2) {
        kokoro_error_set(error, KOKORO_NUMERIC_ERROR, "upsample dimensions exceed supported range");
        return KOKORO_NUMERIC_ERROR;
    }
    size_t output_frames = input_frames * 2;
    if (conv->output_channels > SIZE_MAX / output_frames ||
        conv->output_channels * output_frames > SIZE_MAX / sizeof(float)) {
        kokoro_error_set(error, KOKORO_NUMERIC_ERROR, "upsample output size overflows");
        return KOKORO_NUMERIC_ERROR;
    }
    memset(output, 0, conv->output_channels * output_frames * sizeof(float));
    for (size_t channel = 0; channel < conv->output_channels; channel++) {
        const float *source = input + channel * input_frames;
        float *target = output + channel * output_frames;
        for (size_t frame = 0; frame < input_frames; frame++) {
            for (size_t kernel = 0; kernel < conv->kernel_size; kernel++) {
                int64_t output_frame = (int64_t)(frame * 2) + (int64_t)kernel - 1;
                if (output_frame >= 0 && (size_t)output_frame < output_frames) {
                    target[output_frame] +=
                        source[frame] * conv->weight[kernel * conv->output_channels + channel];
                }
            }
        }
        if (conv->bias != NULL) {
            for (size_t frame = 0; frame < output_frames; frame++) {
                target[frame] += conv->bias[channel];
            }
        }
    }
    return KOKORO_OK;
}

static KokoroStatus run_block(const PitchNoiseBlock *block, const float *input, size_t input_frames,
                              const float *style, float **result, KokoroError *error) {
    if (input_frames == 0 || block->conv1.input_channels > SIZE_MAX / input_frames) {
        kokoro_error_set(error, KOKORO_NUMERIC_ERROR, "predictor block input size overflows");
        return KOKORO_NUMERIC_ERROR;
    }
    size_t output_frames = input_frames;
    if (block->upsample) {
        if (input_frames > SIZE_MAX / 2) {
            kokoro_error_set(error, KOKORO_NUMERIC_ERROR, "predictor frame count overflows");
            return KOKORO_NUMERIC_ERROR;
        }
        output_frames *= 2;
    }
    if (block->conv1.output_channels > SIZE_MAX / output_frames) {
        kokoro_error_set(error, KOKORO_NUMERIC_ERROR, "predictor block output size overflows");
        return KOKORO_NUMERIC_ERROR;
    }
    size_t input_count = block->conv1.input_channels * input_frames;
    size_t output_count = block->conv1.output_channels * output_frames;
    size_t normalized_count = input_count > output_count ? input_count : output_count;
    float *normalized = NULL;
    float *pooled = NULL;
    float *residual = NULL;
    float *shortcut = NULL;
    float *output = NULL;
    if (!allocate_floats(normalized_count, &normalized) ||
        input_count > SIZE_MAX / (block->upsample ? 2U : 1U) ||
        !allocate_floats(input_count * (block->upsample ? 2 : 1), &pooled) ||
        !allocate_floats(output_count, &residual) || !allocate_floats(output_count, &shortcut) ||
        !allocate_floats(output_count, &output)) {
        free(normalized);
        free(pooled);
        free(residual);
        free(shortcut);
        free(output);
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY, "cannot allocate predictor block workspace");
        return KOKORO_OUT_OF_MEMORY;
    }
    apply_adain(&block->norm1, input, style, block->conv1.input_channels, input_frames, normalized);
    apply_leaky_relu(normalized, input_count);
    if (block->upsample) {
        KokoroStatus status =
            run_depthwise_upsample(&block->pool, normalized, input_frames, pooled, error);
        if (status != KOKORO_OK) {
            free(normalized);
            free(pooled);
            free(residual);
            free(shortcut);
            free(output);
            return status;
        }
    } else {
        memcpy(pooled, normalized, input_count * sizeof(float));
    }
    KokoroStatus status = run_convolution(&block->conv1, pooled, output_frames, residual, error);
    if (status == KOKORO_OK) {
        apply_adain(&block->norm2, residual, style, block->conv1.output_channels, output_frames,
                    normalized);
        apply_leaky_relu(normalized, output_count);
        status = run_convolution(&block->conv2, normalized, output_frames, residual, error);
    }
    if (status != KOKORO_OK) {
        free(normalized);
        free(pooled);
        free(residual);
        free(shortcut);
        free(output);
        return status;
    }
    if (block->upsample) {
        for (size_t channel = 0; channel < block->conv1.input_channels; channel++) {
            const float *source = input + channel * input_frames;
            float *target = pooled + channel * output_frames;
            for (size_t frame = 0; frame < input_frames; frame++) {
                target[frame * 2] = source[frame];
                target[frame * 2 + 1] = source[frame];
            }
        }
    } else {
        memcpy(pooled, input, input_count * sizeof(float));
    }
    if (block->shortcut.weight != NULL) {
        status = run_convolution(&block->shortcut, pooled, output_frames, shortcut, error);
    } else {
        memcpy(shortcut, pooled, output_count * sizeof(float));
    }
    if (status == KOKORO_OK) {
        for (size_t index = 0; index < output_count; index++) {
            output[index] = (residual[index] + shortcut[index]) * PITCH_NOISE_RESIDUAL_SCALE;
        }
    }
    free(normalized);
    free(pooled);
    free(residual);
    free(shortcut);
    if (status != KOKORO_OK) {
        free(output);
        return status;
    }
    *result = output;
    return KOKORO_OK;
}

static KokoroStatus run_predictor_blocks(const PitchNoiseBlock *blocks, const float *input,
                                         size_t input_frames, const float *style,
                                         const float *projection_weight,
                                         const float *projection_bias, float **result,
                                         KokoroError *error) {
    float *current = NULL;
    if (!allocate_floats(PITCH_NOISE_HIDDEN_SIZE * input_frames, &current)) {
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY,
                         "cannot allocate pitch/noise recurrent features");
        return KOKORO_OUT_OF_MEMORY;
    }
    for (size_t channel = 0; channel < PITCH_NOISE_HIDDEN_SIZE; channel++) {
        for (size_t frame = 0; frame < input_frames; frame++) {
            current[channel * input_frames + frame] =
                input[frame * PITCH_NOISE_HIDDEN_SIZE + channel];
        }
    }
    size_t current_frames = input_frames;
    for (size_t index = 0; index < PITCH_NOISE_BLOCK_COUNT; index++) {
        float *next = NULL;
        KokoroStatus status =
            run_block(&blocks[index], current, current_frames, style, &next, error);
        free(current);
        if (status != KOKORO_OK) {
            return status;
        }
        current = next;
        current_frames *= blocks[index].upsample ? 2 : 1;
    }
    float *prediction = NULL;
    if (!allocate_floats(current_frames, &prediction)) {
        free(current);
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY, "cannot allocate pitch/noise output");
        return KOKORO_OUT_OF_MEMORY;
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, (int)current_frames,
                PITCH_NOISE_OUTPUT_CHANNELS, 1.0f, projection_weight, PITCH_NOISE_OUTPUT_CHANNELS,
                current, (int)current_frames, 0.0f, prediction, (int)current_frames);
    for (size_t frame = 0; frame < current_frames; frame++) {
        prediction[frame] += projection_bias[0];
    }
    free(current);
    *result = prediction;
    return KOKORO_OK;
}

static KokoroStatus validate_duration(const KokoroDurationOutput *duration, KokoroError *error) {
    if (duration == NULL || duration->token_count == 0 ||
        duration->token_count > PITCH_NOISE_MAX_TOKENS || duration->frame_count == 0 ||
        duration->duration_features == NULL || duration->durations == NULL ||
        duration->alignment == NULL || duration->frame_count > SIZE_MAX / duration->token_count ||
        duration->token_count > SIZE_MAX / KOKORO_DURATION_FEATURES) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "duration output must contain bounded token features and alignment");
        return KOKORO_INVALID_ARGUMENT;
    }
    size_t expected_frames = 0;
    for (size_t token = 0; token < duration->token_count; token++) {
        uint64_t token_duration = duration->durations[token];
        if (token_duration == 0 || token_duration > SIZE_MAX - expected_frames) {
            kokoro_error_set(error, KOKORO_INVALID_ARGUMENT, "duration output has invalid lengths");
            return KOKORO_INVALID_ARGUMENT;
        }
        expected_frames += (size_t)token_duration;
    }
    if (expected_frames != duration->frame_count) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "duration output frame count does not match its token durations");
        return KOKORO_INVALID_ARGUMENT;
    }
    size_t first_frame = 0;
    for (size_t token = 0; token < duration->token_count; token++) {
        size_t last_frame = first_frame + (size_t)duration->durations[token];
        for (size_t frame = 0; frame < duration->frame_count; frame++) {
            float value = duration->alignment[token * duration->frame_count + frame];
            float expected = frame >= first_frame && frame < last_frame ? 1.0f : 0.0f;
            if (value != expected) {
                kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                                 "duration output alignment is not the expected one-hot expansion");
                return KOKORO_INVALID_ARGUMENT;
            }
        }
        first_frame = last_frame;
    }
    size_t feature_count = duration->token_count * KOKORO_DURATION_FEATURES;
    for (size_t index = 0; index < feature_count; index++) {
        if (!isfinite(duration->duration_features[index])) {
            kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                             "duration features contain a non-finite value");
            return KOKORO_NUMERIC_ERROR;
        }
    }
    return KOKORO_OK;
}

void kokoro_pitch_noise_output_free(KokoroPitchNoiseOutput *output) {
    if (output == NULL) {
        return;
    }
    free(output->f0);
    free(output->noise);
    memset(output, 0, sizeof(*output));
}

KokoroStatus kokoro_pitch_noise_predict(const KokoroPitchNoisePredictor *predictor,
                                        const KokoroDurationOutput *duration,
                                        size_t voice_style_index, KokoroPitchNoiseOutput *output,
                                        KokoroError *error) {
    if (predictor == NULL || duration == NULL || output == NULL) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "pitch/noise predictor, duration output, and output are required");
        return KOKORO_INVALID_ARGUMENT;
    }
    if (output->f0 != NULL || output->noise != NULL || output->duration_frame_count != 0 ||
        output->prediction_frame_count != 0) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "pitch/noise output must be empty; free it before reuse");
        return KOKORO_INVALID_ARGUMENT;
    }
    if (voice_style_index >= predictor->voice_style_count) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT, "voice style index is outside the pack");
        return KOKORO_INVALID_ARGUMENT;
    }
    KokoroStatus status = validate_duration(duration, error);
    if (status != KOKORO_OK) {
        return status;
    }
    if (duration->frame_count > SIZE_MAX / 2 || duration->frame_count > INT_MAX / 2 ||
        duration->token_count > INT_MAX ||
        duration->frame_count > SIZE_MAX / PITCH_NOISE_HIDDEN_SIZE) {
        kokoro_error_set(error, KOKORO_NUMERIC_ERROR, "duration output is too long");
        return KOKORO_NUMERIC_ERROR;
    }
    size_t aligned_count = PITCH_NOISE_HIDDEN_SIZE * duration->frame_count;
    if (aligned_count > SIZE_MAX / sizeof(float)) {
        kokoro_error_set(error, KOKORO_NUMERIC_ERROR, "aligned prosody feature size overflows");
        return KOKORO_NUMERIC_ERROR;
    }
    float *aligned = NULL;
    if (!allocate_floats(KOKORO_DURATION_FEATURES * duration->frame_count, &aligned)) {
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY, "cannot allocate aligned prosody features");
        return KOKORO_OUT_OF_MEMORY;
    }
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, (int)duration->frame_count,
                KOKORO_DURATION_FEATURES, (int)duration->token_count, 1.0f, duration->alignment,
                (int)duration->frame_count, duration->duration_features, KOKORO_DURATION_FEATURES,
                0.0f, aligned, KOKORO_DURATION_FEATURES);
    size_t output_frames = duration->frame_count * 2;
    size_t recurrent_count = duration->frame_count * PITCH_NOISE_HIDDEN_SIZE;
    float *recurrent = NULL;
    float *f0 = NULL;
    float *noise = NULL;
    if (!allocate_floats(recurrent_count, &recurrent)) {
        free(aligned);
        free(recurrent);
        free(f0);
        free(noise);
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY, "cannot allocate pitch/noise outputs");
        return KOKORO_OUT_OF_MEMORY;
    }
    status =
        kokoro_bilstm_run(&predictor->shared_lstm, aligned, duration->frame_count,
                          KOKORO_DURATION_FEATURES, PITCH_NOISE_HIDDEN_SIZE / 2, recurrent, error);
    free(aligned);
    if (status != KOKORO_OK) {
        free(recurrent);
        free(f0);
        free(noise);
        return status;
    }
    const float *style = predictor->voice_styles + voice_style_index * PITCH_NOISE_VOICE_SIZE +
                         PITCH_NOISE_STYLE_SIZE;
    status = run_predictor_blocks(predictor->f0_blocks, recurrent, duration->frame_count, style,
                                  predictor->f0_projection_weight, predictor->f0_projection_bias,
                                  &f0, error);
    if (status == KOKORO_OK) {
        status = run_predictor_blocks(predictor->noise_blocks, recurrent, duration->frame_count,
                                      style, predictor->noise_projection_weight,
                                      predictor->noise_projection_bias, &noise, error);
    }
    free(recurrent);
    if (status != KOKORO_OK) {
        free(f0);
        free(noise);
        return status;
    }
    for (size_t frame = 0; frame < output_frames; frame++) {
        if (!isfinite(f0[frame]) || !isfinite(noise[frame])) {
            free(f0);
            free(noise);
            kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                             "pitch/noise predictor produced a non-finite value");
            return KOKORO_NUMERIC_ERROR;
        }
    }
    output->duration_frame_count = duration->frame_count;
    output->prediction_frame_count = output_frames;
    output->f0 = f0;
    output->noise = noise;
    kokoro_error_set(error, KOKORO_OK, NULL);
    return KOKORO_OK;
}

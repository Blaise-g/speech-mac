#include "decoder_conditioner.h"

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

#define DECODER_STYLE_SIZE 128U
#define DECODER_VOICE_SIZE 256U
#define DECODER_MAX_CHANNELS 1090U
#define DECODER_MAX_FRAMES KOKORO_TEXT_ENCODER_MAX_FRAMES
#define DECODER_BLOCK_COUNT 4U
#define DECODER_ADAIN_EPSILON 1.0e-5f
#define DECODER_LEAKY_RELU 0.2f
#define DECODER_RESIDUAL_SCALE 0.7071067811865475f

typedef struct {
    size_t input_channels;
    size_t output_channels;
    size_t kernel_size;
    float *weight;
    const float *bias;
} DecoderConv;

typedef struct {
    const float *weight;
    const float *bias;
} DecoderAdaIN;

typedef struct {
    DecoderConv conv1;
    DecoderConv conv2;
    DecoderConv shortcut;
    DecoderConv pool;
    DecoderAdaIN norm1;
    DecoderAdaIN norm2;
    bool upsample;
} DecoderBlock;

struct KokoroDecoderConditioner {
    const float *voice_styles;
    size_t voice_style_count;
    DecoderConv f0_conv;
    DecoderConv noise_conv;
    DecoderConv asr_res;
    DecoderBlock encode;
    DecoderBlock decode[DECODER_BLOCK_COUNT];
};

static bool allocate_floats(size_t count, float **values) {
    if (count == 0 || count > SIZE_MAX / sizeof(float)) {
        return false;
    }
    *values = malloc(count * sizeof(float));
    return *values != NULL;
}

static KokoroStatus bind_weight_norm(DecoderConv *conv, const KokoroPack *pack, const char *prefix,
                                     size_t input_channels, size_t output_channels,
                                     size_t kernel_size, bool has_bias, KokoroError *error) {
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
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "decoder tensor name is too long");
        return KOKORO_INCOMPATIBLE_PACK;
    }
    KokoroStatus status = kokoro_tensor_bind_f32(pack, name, 3, norm_shape, &norm, error);
    if (status != KOKORO_OK) {
        return status;
    }
    length = snprintf(name, sizeof(name), "%s.weight_v", prefix);
    if (length < 0 || (size_t)length >= sizeof(name)) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "decoder tensor name is too long");
        return KOKORO_INCOMPATIBLE_PACK;
    }
    status = kokoro_tensor_bind_f32(pack, name, 3, weight_shape, &raw_weight, error);
    if (status != KOKORO_OK) {
        return status;
    }
    if (has_bias) {
        length = snprintf(name, sizeof(name), "%s.bias", prefix);
        if (length < 0 || (size_t)length >= sizeof(name)) {
            kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "decoder tensor name is too long");
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
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY,
                         "cannot allocate decoder convolution weights");
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

static KokoroStatus bind_adain(DecoderAdaIN *norm, const KokoroPack *pack, const char *prefix,
                               size_t channels, KokoroError *error) {
    if (channels == 0 || channels > SIZE_MAX / 2) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "AdaIN dimensions overflow");
        return KOKORO_INCOMPATIBLE_PACK;
    }
    const uint64_t weight_shape[2] = {channels * 2, DECODER_STYLE_SIZE};
    const uint64_t bias_shape[1] = {channels * 2};
    const float *weight = NULL;
    const float *bias = NULL;
    char name[160];
    int length = snprintf(name, sizeof(name), "%s.fc.weight", prefix);
    if (length < 0 || (size_t)length >= sizeof(name)) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "decoder tensor name is too long");
        return KOKORO_INCOMPATIBLE_PACK;
    }
    KokoroStatus status = kokoro_tensor_bind_f32(pack, name, 2, weight_shape, &weight, error);
    if (status != KOKORO_OK) {
        return status;
    }
    length = snprintf(name, sizeof(name), "%s.fc.bias", prefix);
    if (length < 0 || (size_t)length >= sizeof(name)) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "decoder tensor name is too long");
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

static KokoroStatus bind_block(DecoderBlock *block, const KokoroPack *pack, const char *prefix,
                               size_t input_channels, size_t output_channels, bool upsample,
                               KokoroError *error) {
    char name[160];
    int length = snprintf(name, sizeof(name), "%s.conv1", prefix);
    if (length < 0 || (size_t)length >= sizeof(name)) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "decoder tensor name is too long");
        return KOKORO_INCOMPATIBLE_PACK;
    }
    KokoroStatus status = bind_weight_norm(&block->conv1, pack, name, input_channels,
                                           output_channels, 3, true, error);
    if (status != KOKORO_OK) {
        return status;
    }
    length = snprintf(name, sizeof(name), "%s.conv2", prefix);
    if (length < 0 || (size_t)length >= sizeof(name)) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "decoder tensor name is too long");
        return KOKORO_INCOMPATIBLE_PACK;
    }
    status = bind_weight_norm(&block->conv2, pack, name, output_channels, output_channels, 3, true,
                              error);
    if (status != KOKORO_OK) {
        return status;
    }
    length = snprintf(name, sizeof(name), "%s.norm1", prefix);
    if (length < 0 || (size_t)length >= sizeof(name)) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "decoder tensor name is too long");
        return KOKORO_INCOMPATIBLE_PACK;
    }
    status = bind_adain(&block->norm1, pack, name, input_channels, error);
    if (status != KOKORO_OK) {
        return status;
    }
    length = snprintf(name, sizeof(name), "%s.norm2", prefix);
    if (length < 0 || (size_t)length >= sizeof(name)) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "decoder tensor name is too long");
        return KOKORO_INCOMPATIBLE_PACK;
    }
    status = bind_adain(&block->norm2, pack, name, output_channels, error);
    if (status != KOKORO_OK) {
        return status;
    }
    if (input_channels != output_channels) {
        length = snprintf(name, sizeof(name), "%s.conv1x1", prefix);
        if (length < 0 || (size_t)length >= sizeof(name)) {
            kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "decoder tensor name is too long");
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
            kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "decoder tensor name is too long");
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

static void free_conv(DecoderConv *conv) {
    free(conv->weight);
    memset(conv, 0, sizeof(*conv));
}

static void free_block(DecoderBlock *block) {
    free_conv(&block->conv1);
    free_conv(&block->conv2);
    free_conv(&block->shortcut);
    free_conv(&block->pool);
}

KokoroStatus kokoro_decoder_conditioner_create(const KokoroPack *pack,
                                               KokoroDecoderConditioner **result,
                                               KokoroError *error) {
    if (result != NULL) {
        *result = NULL;
    }
    if (pack == NULL || result == NULL) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "model pack and decoder conditioner output are required");
        return KOKORO_INVALID_ARGUMENT;
    }
    KokoroDecoderConditioner *conditioner = calloc(1, sizeof(*conditioner));
    if (conditioner == NULL) {
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY, "cannot allocate decoder conditioner");
        return KOKORO_OUT_OF_MEMORY;
    }
    KokoroTensorView voice_view;
    KokoroStatus status = kokoro_pack_tensor(pack, "voice.af_heart.style", &voice_view, error);
    if (status != KOKORO_OK) {
        free(conditioner);
        return status;
    }
    if (voice_view.type != KOKORO_TENSOR_F32 || voice_view.rank != 3 || voice_view.shape[0] == 0 ||
        voice_view.shape[1] != 1 || voice_view.shape[2] != DECODER_VOICE_SIZE ||
        voice_view.shape[0] > SIZE_MAX / DECODER_VOICE_SIZE) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK,
                         "af_heart style tensor must have shape [rows, 1, 256]");
        free(conditioner);
        return KOKORO_INCOMPATIBLE_PACK;
    }
    conditioner->voice_styles = voice_view.data;
    conditioner->voice_style_count = (size_t)voice_view.shape[0];
    for (size_t index = 0; index < conditioner->voice_style_count * DECODER_VOICE_SIZE; index++) {
        if (!isfinite(conditioner->voice_styles[index])) {
            kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK,
                             "af_heart style tensor contains a non-finite value");
            kokoro_decoder_conditioner_free(conditioner);
            return KOKORO_INCOMPATIBLE_PACK;
        }
    }

    status = bind_weight_norm(&conditioner->f0_conv, pack, "decoder.module.F0_conv", 1, 1, 3, true,
                              error);
    if (status == KOKORO_OK) {
        status = bind_weight_norm(&conditioner->noise_conv, pack, "decoder.module.N_conv", 1, 1, 3,
                                  true, error);
    }
    if (status == KOKORO_OK) {
        status = bind_weight_norm(&conditioner->asr_res, pack, "decoder.module.asr_res.0", 512, 64,
                                  1, true, error);
    }
    if (status == KOKORO_OK) {
        status = bind_block(&conditioner->encode, pack, "decoder.module.encode", 514, 1024, false,
                            error);
    }
    static const size_t output_channels[DECODER_BLOCK_COUNT] = {1024, 1024, 1024, 512};
    for (size_t index = 0; status == KOKORO_OK && index < DECODER_BLOCK_COUNT; index++) {
        char prefix[128];
        int length = snprintf(prefix, sizeof(prefix), "decoder.module.decode.%zu", index);
        if (length < 0 || (size_t)length >= sizeof(prefix)) {
            kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "decoder tensor name is too long");
            status = KOKORO_INCOMPATIBLE_PACK;
            break;
        }
        status = bind_block(&conditioner->decode[index], pack, prefix, 1090, output_channels[index],
                            index == DECODER_BLOCK_COUNT - 1, error);
    }
    if (status != KOKORO_OK) {
        kokoro_decoder_conditioner_free(conditioner);
        return status;
    }
    *result = conditioner;
    kokoro_error_set(error, KOKORO_OK, NULL);
    return KOKORO_OK;
}

void kokoro_decoder_conditioner_free(KokoroDecoderConditioner *conditioner) {
    if (conditioner == NULL) {
        return;
    }
    free_conv(&conditioner->f0_conv);
    free_conv(&conditioner->noise_conv);
    free_conv(&conditioner->asr_res);
    free_block(&conditioner->encode);
    for (size_t index = 0; index < DECODER_BLOCK_COUNT; index++) {
        free_block(&conditioner->decode[index]);
    }
    free(conditioner);
}

static void apply_leaky_relu(float *values, size_t count) {
    for (size_t index = 0; index < count; index++) {
        if (values[index] < 0.0f) {
            values[index] *= DECODER_LEAKY_RELU;
        }
    }
}

static void apply_adain(const DecoderAdaIN *norm, const float *input, const float *style,
                        size_t channels, size_t frame_count, float *output) {
    float affine[DECODER_MAX_CHANNELS * 2];
    cblas_sgemv(CblasRowMajor, CblasNoTrans, (int)(channels * 2), DECODER_STYLE_SIZE, 1.0f,
                norm->weight, DECODER_STYLE_SIZE, style, 1, 0.0f, affine, 1);
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
        float inverse_scale = 1.0f / sqrtf(variance + DECODER_ADAIN_EPSILON);
        float gamma = affine[channel];
        float beta = affine[channels + channel];
        for (size_t frame = 0; frame < frame_count; frame++) {
            float normalized = (source[frame] - mean) * inverse_scale;
            /* The checkpoint omits InstanceNorm affine tensors, leaving their identity defaults. */
            target[frame] = (1.0f + gamma) * normalized + beta;
        }
    }
}

static KokoroStatus run_convolution(const DecoderConv *conv, const float *input, size_t frame_count,
                                    float *output, KokoroError *error) {
    if (frame_count == 0 || frame_count > INT_MAX || conv->output_channels > INT_MAX ||
        conv->input_channels > INT_MAX || conv->kernel_size == 0 || conv->kernel_size > 3) {
        kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                         "decoder convolution dimensions exceed supported range");
        return KOKORO_NUMERIC_ERROR;
    }
    if (conv->kernel_size == 1) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)conv->output_channels,
                    (int)frame_count, (int)conv->input_channels, 1.0f, conv->weight,
                    (int)conv->input_channels, input, (int)frame_count, 0.0f, output,
                    (int)frame_count);
    } else {
        if (frame_count > SIZE_MAX - 2 || conv->input_channels > SIZE_MAX / (frame_count + 2) ||
            conv->input_channels * (frame_count + 2) > SIZE_MAX / sizeof(float)) {
            kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                             "decoder convolution workspace size overflows");
            return KOKORO_NUMERIC_ERROR;
        }
        size_t padded_count = conv->input_channels * (frame_count + 2);
        float *padded = calloc(padded_count, sizeof(float));
        if (padded == NULL) {
            kokoro_error_set(error, KOKORO_OUT_OF_MEMORY,
                             "cannot allocate decoder convolution workspace");
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

static KokoroStatus run_downsample(const DecoderConv *conv, const float *input, size_t input_frames,
                                   size_t output_frames, float *output, KokoroError *error) {
    if (conv->input_channels != 1 || conv->output_channels != 1 || conv->kernel_size != 3 ||
        input_frames == 0 || input_frames > INT_MAX || output_frames != (input_frames + 1) / 2) {
        kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                         "decoder F0/noise downsampling dimensions are invalid");
        return KOKORO_NUMERIC_ERROR;
    }
    for (size_t frame = 0; frame < output_frames; frame++) {
        float value = 0.0f;
        for (size_t kernel = 0; kernel < 3; kernel++) {
            int64_t source_frame = (int64_t)(frame * 2) + (int64_t)kernel - 1;
            if (source_frame >= 0 && (size_t)source_frame < input_frames) {
                value += conv->weight[kernel] * input[source_frame];
            }
        }
        output[frame] = value + conv->bias[0];
    }
    return KOKORO_OK;
}

static KokoroStatus run_depthwise_upsample(const DecoderConv *conv, const float *input,
                                           size_t input_frames, float *output, KokoroError *error) {
    if (input_frames > SIZE_MAX / 2 || conv->kernel_size != 3 || conv->input_channels != 1 ||
        conv->output_channels > INT_MAX || input_frames > INT_MAX / 2) {
        kokoro_error_set(error, KOKORO_NUMERIC_ERROR, "decoder upsample dimensions exceed range");
        return KOKORO_NUMERIC_ERROR;
    }
    size_t output_frames = input_frames * 2;
    if (conv->output_channels > SIZE_MAX / output_frames ||
        conv->output_channels * output_frames > SIZE_MAX / sizeof(float)) {
        kokoro_error_set(error, KOKORO_NUMERIC_ERROR, "decoder upsample output size overflows");
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
        for (size_t frame = 0; frame < output_frames; frame++) {
            target[frame] += conv->bias[channel];
        }
    }
    return KOKORO_OK;
}

static KokoroStatus run_block(const DecoderBlock *block, const float *input, size_t input_frames,
                              const float *style, float **result, KokoroError *error) {
    *result = NULL;
    if (input_frames == 0 || block->conv1.input_channels > SIZE_MAX / input_frames) {
        kokoro_error_set(error, KOKORO_NUMERIC_ERROR, "decoder block input size overflows");
        return KOKORO_NUMERIC_ERROR;
    }
    size_t output_frames = input_frames;
    if (block->upsample) {
        if (input_frames > SIZE_MAX / 2) {
            kokoro_error_set(error, KOKORO_NUMERIC_ERROR, "decoder frame count overflows");
            return KOKORO_NUMERIC_ERROR;
        }
        output_frames *= 2;
    }
    if (block->conv1.input_channels > SIZE_MAX / output_frames ||
        block->conv1.output_channels > SIZE_MAX / output_frames) {
        kokoro_error_set(error, KOKORO_NUMERIC_ERROR, "decoder block output size overflows");
        return KOKORO_NUMERIC_ERROR;
    }
    size_t input_count = block->conv1.input_channels * input_frames;
    size_t output_count = block->conv1.output_channels * output_frames;
    size_t normalized_count = input_count > output_count ? input_count : output_count;
    if (normalized_count > SIZE_MAX / sizeof(float) ||
        input_count > SIZE_MAX / (block->upsample ? 2U : 1U)) {
        kokoro_error_set(error, KOKORO_NUMERIC_ERROR, "decoder block workspace size overflows");
        return KOKORO_NUMERIC_ERROR;
    }
    float *normalized = NULL;
    float *pooled = NULL;
    float *residual = NULL;
    float *shortcut = NULL;
    float *output = NULL;
    if (!allocate_floats(normalized_count, &normalized) ||
        !allocate_floats(input_count * (block->upsample ? 2 : 1), &pooled) ||
        !allocate_floats(output_count, &residual) || !allocate_floats(output_count, &shortcut) ||
        !allocate_floats(output_count, &output)) {
        free(normalized);
        free(pooled);
        free(residual);
        free(shortcut);
        free(output);
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY,
                         "cannot allocate decoder residual block workspace");
        return KOKORO_OUT_OF_MEMORY;
    }
    apply_adain(&block->norm1, input, style, block->conv1.input_channels, input_frames, normalized);
    apply_leaky_relu(normalized, input_count);
    KokoroStatus status;
    if (block->upsample) {
        status = run_depthwise_upsample(&block->pool, normalized, input_frames, pooled, error);
    } else {
        memcpy(pooled, normalized, input_count * sizeof(float));
        status = KOKORO_OK;
    }
    if (status == KOKORO_OK) {
        status = run_convolution(&block->conv1, pooled, output_frames, residual, error);
    }
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
        status = KOKORO_OK;
    }
    if (status == KOKORO_OK) {
        for (size_t index = 0; index < output_count; index++) {
            output[index] = (residual[index] + shortcut[index]) * DECODER_RESIDUAL_SCALE;
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

static bool output_is_empty(const KokoroDecoderOutput *output) {
    return output->features == NULL && output->duration_frame_count == 0 &&
           output->feature_frame_count == 0;
}

void kokoro_decoder_output_free(KokoroDecoderOutput *output) {
    if (output == NULL) {
        return;
    }
    free(output->features);
    memset(output, 0, sizeof(*output));
}

KokoroStatus kokoro_decoder_condition(const KokoroDecoderConditioner *conditioner,
                                      const KokoroTextEncoderOutput *text,
                                      const KokoroPitchNoiseOutput *pitch_noise,
                                      size_t voice_style_index, KokoroDecoderOutput *output,
                                      KokoroError *error) {
    if (conditioner == NULL || text == NULL || pitch_noise == NULL || output == NULL) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "decoder conditioner, text, pitch/noise, and output are required");
        return KOKORO_INVALID_ARGUMENT;
    }
    if (!output_is_empty(output)) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "decoder output must be empty; free it before reuse");
        return KOKORO_INVALID_ARGUMENT;
    }
    if (voice_style_index >= conditioner->voice_style_count) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT, "voice style index is outside the pack");
        return KOKORO_INVALID_ARGUMENT;
    }
    if (text->frame_count == 0 || text->frame_count > DECODER_MAX_FRAMES ||
        text->aligned_features == NULL || pitch_noise->f0 == NULL || pitch_noise->noise == NULL ||
        pitch_noise->duration_frame_count == 0) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "decoder conditioning requires aligned text and F0/noise frames");
        return KOKORO_INVALID_ARGUMENT;
    }
    if (text->frame_count != pitch_noise->duration_frame_count) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "decoder text and pitch/noise duration frame counts do not match");
        return KOKORO_INVALID_ARGUMENT;
    }
    if (pitch_noise->duration_frame_count > SIZE_MAX / 2 ||
        pitch_noise->prediction_frame_count != pitch_noise->duration_frame_count * 2) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "decoder F0/noise frame count must be twice the duration frame count");
        return KOKORO_INVALID_ARGUMENT;
    }
    size_t duration_frames = text->frame_count;
    size_t feature_frames = pitch_noise->prediction_frame_count;
    if (feature_frames > INT_MAX || duration_frames > SIZE_MAX / DECODER_MAX_CHANNELS ||
        feature_frames > SIZE_MAX / KOKORO_DECODER_FEATURES ||
        duration_frames * KOKORO_TEXT_ENCODER_FEATURES > SIZE_MAX / sizeof(float) ||
        feature_frames * KOKORO_DECODER_FEATURES > SIZE_MAX / sizeof(float)) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "decoder conditioning dimensions exceed addressable memory");
        return KOKORO_INVALID_ARGUMENT;
    }
    size_t text_count = duration_frames * KOKORO_TEXT_ENCODER_FEATURES;
    for (size_t index = 0; index < text_count; index++) {
        if (!isfinite(text->aligned_features[index])) {
            kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                             "aligned text features contain a non-finite value");
            return KOKORO_NUMERIC_ERROR;
        }
    }
    for (size_t frame = 0; frame < feature_frames; frame++) {
        if (!isfinite(pitch_noise->f0[frame]) || !isfinite(pitch_noise->noise[frame])) {
            kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                             "F0/noise conditioning contains a non-finite value");
            return KOKORO_NUMERIC_ERROR;
        }
    }

    float *f0 = NULL;
    float *noise = NULL;
    float *asr_res = NULL;
    float *combined = NULL;
    if (!allocate_floats(duration_frames, &f0) || !allocate_floats(duration_frames, &noise) ||
        !allocate_floats(64 * duration_frames, &asr_res) ||
        !allocate_floats(514 * duration_frames, &combined)) {
        free(f0);
        free(noise);
        free(asr_res);
        free(combined);
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY,
                         "cannot allocate decoder conditioning workspace");
        return KOKORO_OUT_OF_MEMORY;
    }
    KokoroStatus status = run_downsample(&conditioner->f0_conv, pitch_noise->f0, feature_frames,
                                         duration_frames, f0, error);
    if (status == KOKORO_OK) {
        status = run_downsample(&conditioner->noise_conv, pitch_noise->noise, feature_frames,
                                duration_frames, noise, error);
    }
    if (status == KOKORO_OK) {
        status = run_convolution(&conditioner->asr_res, text->aligned_features, duration_frames,
                                 asr_res, error);
    }
    if (status != KOKORO_OK) {
        free(f0);
        free(noise);
        free(asr_res);
        free(combined);
        return status;
    }
    for (size_t channel = 0; channel < KOKORO_TEXT_ENCODER_FEATURES; channel++) {
        memcpy(combined + channel * duration_frames,
               text->aligned_features + channel * duration_frames, duration_frames * sizeof(float));
    }
    memcpy(combined + 512 * duration_frames, f0, duration_frames * sizeof(float));
    memcpy(combined + 513 * duration_frames, noise, duration_frames * sizeof(float));

    const float *style = conditioner->voice_styles + voice_style_index * DECODER_VOICE_SIZE;
    float *current = NULL;
    status = run_block(&conditioner->encode, combined, duration_frames, style, &current, error);
    free(combined);
    if (status != KOKORO_OK) {
        free(f0);
        free(noise);
        free(asr_res);
        return status;
    }
    size_t current_frames = duration_frames;
    for (size_t index = 0; index < DECODER_BLOCK_COUNT; index++) {
        if (current_frames > SIZE_MAX / 1090 || 1090 * current_frames > SIZE_MAX / sizeof(float)) {
            free(current);
            free(f0);
            free(noise);
            free(asr_res);
            kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                             "decoder concatenated feature size overflows");
            return KOKORO_NUMERIC_ERROR;
        }
        float *input = NULL;
        if (!allocate_floats(1090 * current_frames, &input)) {
            free(current);
            free(f0);
            free(noise);
            free(asr_res);
            kokoro_error_set(error, KOKORO_OUT_OF_MEMORY,
                             "cannot allocate decoder concatenation workspace");
            return KOKORO_OUT_OF_MEMORY;
        }
        const size_t current_channels = 1024;
        memcpy(input, current, current_channels * current_frames * sizeof(float));
        memcpy(input + current_channels * current_frames, asr_res,
               64 * current_frames * sizeof(float));
        memcpy(input + 1088 * current_frames, f0, current_frames * sizeof(float));
        memcpy(input + 1089 * current_frames, noise, current_frames * sizeof(float));
        float *next = NULL;
        status = run_block(&conditioner->decode[index], input, current_frames, style, &next, error);
        free(input);
        free(current);
        if (status != KOKORO_OK) {
            free(f0);
            free(noise);
            free(asr_res);
            return status;
        }
        current = next;
        if (index == DECODER_BLOCK_COUNT - 1) {
            current_frames *= 2;
        }
    }
    free(f0);
    free(noise);
    free(asr_res);
    if (current_frames != feature_frames) {
        free(current);
        kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                         "decoder residual blocks produced an unexpected frame count");
        return KOKORO_NUMERIC_ERROR;
    }
    size_t feature_count = KOKORO_DECODER_FEATURES * feature_frames;
    for (size_t index = 0; index < feature_count; index++) {
        if (!isfinite(current[index])) {
            free(current);
            kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                             "decoder conditioner produced a non-finite feature");
            return KOKORO_NUMERIC_ERROR;
        }
    }
    output->duration_frame_count = duration_frames;
    output->feature_frame_count = feature_frames;
    output->features = current;
    kokoro_error_set(error, KOKORO_OK, NULL);
    return KOKORO_OK;
}

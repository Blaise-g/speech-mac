#include "neural_generator.h"

#include "kokoro_internal.h"
#include "tensor_utils.h"
#include "text_encoder.h"

#define ACCELERATE_NEW_LAPACK 1
#define ACCELERATE_LAPACK_ILP64 1
#include <Accelerate/Accelerate.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GENERATOR_STYLE_SIZE 128U
#define GENERATOR_VOICE_SIZE 256U
#define GENERATOR_INPUT_CHANNELS 512U
#define GENERATOR_SOURCE_HARMONICS 9U
#define GENERATOR_UPSAMPLE_COUNT 2U
#define GENERATOR_RESBLOCKS_PER_STAGE 3U
#define GENERATOR_RESBLOCK_PAIRS 3U
#define GENERATOR_SOURCE_RATE 24000U
#define GENERATOR_F0_UPSAMPLE 300U
#define GENERATOR_FFT_SIZE 20U
#define GENERATOR_HOP_SIZE 5U
#define GENERATOR_SOURCE_CHANNELS (GENERATOR_FFT_SIZE + 2U)
#define GENERATOR_MAX_CHANNELS GENERATOR_INPUT_CHANNELS
#define GENERATOR_MAX_FRAMES KOKORO_TEXT_ENCODER_MAX_FRAMES
#define GENERATOR_ADAIN_EPSILON 1.0e-5f
#define GENERATOR_UPSAMPLE_LEAKY_RELU 0.1f
#define GENERATOR_POST_LEAKY_RELU 0.01f
#define GENERATOR_SINE_AMPLITUDE 0.1f
#define GENERATOR_VOICED_NOISE 0.003f
#define GENERATOR_UNVOICED_NOISE (GENERATOR_SINE_AMPLITUDE / 3.0f)
#define GENERATOR_VOICED_THRESHOLD 10.0f
#define GENERATOR_TWO_PI 6.28318530717958647692

typedef struct {
    size_t input_channels;
    size_t output_channels;
    size_t kernel_size;
    size_t stride;
    size_t padding;
    size_t dilation;
    float *weight; /* [output_channel, kernel, input_channel] */
    const float *bias;
} GeneratorConv;

typedef struct {
    size_t input_channels;
    size_t output_channels;
    size_t kernel_size;
    size_t stride;
    size_t padding;
    float *weight; /* [kernel, output_channel, input_channel] */
    const float *bias;
} GeneratorConvTranspose;

typedef struct {
    const float *weight;
    const float *bias;
} GeneratorAdaIN;

typedef struct {
    size_t channels;
    GeneratorConv conv1[GENERATOR_RESBLOCK_PAIRS];
    GeneratorConv conv2[GENERATOR_RESBLOCK_PAIRS];
    GeneratorAdaIN adain1[GENERATOR_RESBLOCK_PAIRS];
    GeneratorAdaIN adain2[GENERATOR_RESBLOCK_PAIRS];
    const float *alpha1[GENERATOR_RESBLOCK_PAIRS];
    const float *alpha2[GENERATOR_RESBLOCK_PAIRS];
} GeneratorResblock;

struct KokoroNeuralGenerator {
    const float *voice_styles;
    size_t voice_style_count;
    const float *source_weight;
    const float *source_bias;
    GeneratorConvTranspose upsample[GENERATOR_UPSAMPLE_COUNT];
    GeneratorConv noise_conv[GENERATOR_UPSAMPLE_COUNT];
    GeneratorResblock noise_res[GENERATOR_UPSAMPLE_COUNT];
    GeneratorResblock resblocks[GENERATOR_UPSAMPLE_COUNT * GENERATOR_RESBLOCKS_PER_STAGE];
    GeneratorConv conv_post;
};

static bool allocate_floats(size_t count, float **values) {
    if (count == 0 || count > SIZE_MAX / sizeof(float)) {
        return false;
    }
    *values = malloc(count * sizeof(float));
    return *values != NULL;
}

static bool checked_product(size_t left, size_t right, size_t *product) {
    if (right != 0 && left > SIZE_MAX / right) {
        return false;
    }
    *product = left * right;
    return true;
}

static bool make_tensor_name(char *name, size_t capacity, const char *format, const char *prefix,
                             size_t index) {
    int length = snprintf(name, capacity, format, prefix, index);
    return length >= 0 && (size_t)length < capacity;
}

static KokoroStatus bind_conv(GeneratorConv *conv, const KokoroPack *pack, const char *prefix,
                              size_t input_channels, size_t output_channels, size_t kernel_size,
                              size_t stride, size_t padding, size_t dilation, bool weight_norm,
                              bool has_bias, KokoroError *error) {
    if (input_channels == 0 || output_channels == 0 || kernel_size == 0 || stride == 0 ||
        dilation == 0 || input_channels > INT_MAX || output_channels > INT_MAX ||
        kernel_size > INT_MAX || input_channels > SIZE_MAX / kernel_size ||
        output_channels > SIZE_MAX / (input_channels * kernel_size)) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "invalid generator convolution: %s",
                         prefix);
        return KOKORO_INCOMPATIBLE_PACK;
    }

    const uint64_t weight_shape[3] = {output_channels, input_channels, kernel_size};
    const uint64_t norm_shape[3] = {output_channels, 1, 1};
    const uint64_t bias_shape[1] = {output_channels};
    const float *raw_weight = NULL;
    const float *norm = NULL;
    const float *bias = NULL;
    char name[192];
    KokoroStatus status;
    if (weight_norm) {
        int length = snprintf(name, sizeof(name), "%s.weight_g", prefix);
        if (length < 0 || (size_t)length >= sizeof(name)) {
            kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "generator tensor name is too long");
            return KOKORO_INCOMPATIBLE_PACK;
        }
        status = kokoro_tensor_bind_f32(pack, name, 3, norm_shape, &norm, error);
        if (status != KOKORO_OK) {
            return status;
        }
    }
    int length = snprintf(name, sizeof(name), "%s.%s", prefix, weight_norm ? "weight_v" : "weight");
    if (length < 0 || (size_t)length >= sizeof(name)) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "generator tensor name is too long");
        return KOKORO_INCOMPATIBLE_PACK;
    }
    status = kokoro_tensor_bind_f32(pack, name, 3, weight_shape, &raw_weight, error);
    if (status != KOKORO_OK) {
        return status;
    }
    if (has_bias) {
        length = snprintf(name, sizeof(name), "%s.bias", prefix);
        if (length < 0 || (size_t)length >= sizeof(name)) {
            kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "generator tensor name is too long");
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
                         "cannot allocate generator convolution weights");
        return KOKORO_OUT_OF_MEMORY;
    }
    size_t row_size = input_channels * kernel_size;
    for (size_t output = 0; output < output_channels; output++) {
        const float *source = raw_weight + output * row_size;
        float scale = 1.0f;
        if (weight_norm) {
            double squared_norm = 0.0;
            for (size_t index = 0; index < row_size; index++) {
                squared_norm += (double)source[index] * source[index];
            }
            double magnitude = sqrt(squared_norm);
            if (!(magnitude > 0.0) || !isfinite(magnitude)) {
                free(effective_weight);
                kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK,
                                 "weight-normalized convolution has an invalid norm: %s", prefix);
                return KOKORO_INCOMPATIBLE_PACK;
            }
            scale = (float)((double)norm[output] / magnitude);
        }
        for (size_t kernel = 0; kernel < kernel_size; kernel++) {
            for (size_t input = 0; input < input_channels; input++) {
                size_t source_index = input * kernel_size + kernel;
                size_t target_index = (output * kernel_size + kernel) * input_channels + input;
                effective_weight[target_index] = source[source_index] * scale;
            }
        }
    }
    conv->input_channels = input_channels;
    conv->output_channels = output_channels;
    conv->kernel_size = kernel_size;
    conv->stride = stride;
    conv->padding = padding;
    conv->dilation = dilation;
    conv->weight = effective_weight;
    conv->bias = bias;
    return KOKORO_OK;
}

static KokoroStatus bind_conv_transpose(GeneratorConvTranspose *conv, const KokoroPack *pack,
                                        const char *prefix, size_t input_channels,
                                        size_t output_channels, size_t kernel_size, size_t stride,
                                        size_t padding, KokoroError *error) {
    if (input_channels == 0 || output_channels == 0 || kernel_size == 0 || stride == 0 ||
        input_channels > INT_MAX || output_channels > INT_MAX || kernel_size > INT_MAX ||
        input_channels > SIZE_MAX / kernel_size ||
        output_channels > SIZE_MAX / (input_channels * kernel_size)) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK,
                         "invalid generator transposed convolution: %s", prefix);
        return KOKORO_INCOMPATIBLE_PACK;
    }
    const uint64_t weight_shape[3] = {input_channels, output_channels, kernel_size};
    const uint64_t norm_shape[3] = {input_channels, 1, 1};
    const uint64_t bias_shape[1] = {output_channels};
    const float *norm = NULL;
    const float *raw_weight = NULL;
    const float *bias = NULL;
    char name[192];
    int length = snprintf(name, sizeof(name), "%s.weight_g", prefix);
    if (length < 0 || (size_t)length >= sizeof(name)) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "generator tensor name is too long");
        return KOKORO_INCOMPATIBLE_PACK;
    }
    KokoroStatus status = kokoro_tensor_bind_f32(pack, name, 3, norm_shape, &norm, error);
    if (status != KOKORO_OK) {
        return status;
    }
    length = snprintf(name, sizeof(name), "%s.weight_v", prefix);
    if (length < 0 || (size_t)length >= sizeof(name)) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "generator tensor name is too long");
        return KOKORO_INCOMPATIBLE_PACK;
    }
    status = kokoro_tensor_bind_f32(pack, name, 3, weight_shape, &raw_weight, error);
    if (status != KOKORO_OK) {
        return status;
    }
    length = snprintf(name, sizeof(name), "%s.bias", prefix);
    if (length < 0 || (size_t)length >= sizeof(name)) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "generator tensor name is too long");
        return KOKORO_INCOMPATIBLE_PACK;
    }
    status = kokoro_tensor_bind_f32(pack, name, 1, bias_shape, &bias, error);
    if (status != KOKORO_OK) {
        return status;
    }

    size_t weight_count = input_channels * output_channels * kernel_size;
    float *effective_weight = NULL;
    if (!allocate_floats(weight_count, &effective_weight)) {
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY,
                         "cannot allocate generator transposed convolution weights");
        return KOKORO_OUT_OF_MEMORY;
    }
    size_t row_size = output_channels * kernel_size;
    for (size_t input = 0; input < input_channels; input++) {
        const float *source = raw_weight + input * row_size;
        double squared_norm = 0.0;
        for (size_t index = 0; index < row_size; index++) {
            squared_norm += (double)source[index] * source[index];
        }
        double magnitude = sqrt(squared_norm);
        if (!(magnitude > 0.0) || !isfinite(magnitude)) {
            free(effective_weight);
            kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK,
                             "weight-normalized transposed convolution has an invalid norm: %s",
                             prefix);
            return KOKORO_INCOMPATIBLE_PACK;
        }
        float scale = (float)((double)norm[input] / magnitude);
        for (size_t output = 0; output < output_channels; output++) {
            for (size_t kernel = 0; kernel < kernel_size; kernel++) {
                size_t source_index = output * kernel_size + kernel;
                size_t target_index = (kernel * output_channels + output) * input_channels + input;
                effective_weight[target_index] = source[source_index] * scale;
            }
        }
    }
    conv->input_channels = input_channels;
    conv->output_channels = output_channels;
    conv->kernel_size = kernel_size;
    conv->stride = stride;
    conv->padding = padding;
    conv->weight = effective_weight;
    conv->bias = bias;
    return KOKORO_OK;
}

static KokoroStatus bind_adain(GeneratorAdaIN *norm, const KokoroPack *pack, const char *prefix,
                               size_t channels, KokoroError *error) {
    if (channels == 0 || channels > SIZE_MAX / 2) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "generator AdaIN dimensions overflow");
        return KOKORO_INCOMPATIBLE_PACK;
    }
    const uint64_t weight_shape[2] = {channels * 2, GENERATOR_STYLE_SIZE};
    const uint64_t bias_shape[1] = {channels * 2};
    char name[192];
    int length = snprintf(name, sizeof(name), "%s.fc.weight", prefix);
    if (length < 0 || (size_t)length >= sizeof(name)) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "generator tensor name is too long");
        return KOKORO_INCOMPATIBLE_PACK;
    }
    KokoroStatus status = kokoro_tensor_bind_f32(pack, name, 2, weight_shape, &norm->weight, error);
    if (status != KOKORO_OK) {
        return status;
    }
    length = snprintf(name, sizeof(name), "%s.fc.bias", prefix);
    if (length < 0 || (size_t)length >= sizeof(name)) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "generator tensor name is too long");
        return KOKORO_INCOMPATIBLE_PACK;
    }
    return kokoro_tensor_bind_f32(pack, name, 1, bias_shape, &norm->bias, error);
}

static KokoroStatus bind_resblock(GeneratorResblock *block, const KokoroPack *pack,
                                  const char *prefix, size_t channels, size_t kernel_size,
                                  KokoroError *error) {
    static const size_t dilations[GENERATOR_RESBLOCK_PAIRS] = {1, 3, 5};
    block->channels = channels;
    for (size_t pair = 0; pair < GENERATOR_RESBLOCK_PAIRS; pair++) {
        char name[192];
        if (!make_tensor_name(name, sizeof(name), "%s.convs1.%zu", prefix, pair)) {
            kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK,
                             "generator residual tensor name is too long");
            return KOKORO_INCOMPATIBLE_PACK;
        }
        KokoroStatus status =
            bind_conv(&block->conv1[pair], pack, name, channels, channels, kernel_size, 1,
                      dilations[pair] * (kernel_size - 1) / 2, dilations[pair], true, true, error);
        if (status != KOKORO_OK) {
            return status;
        }
        if (!make_tensor_name(name, sizeof(name), "%s.convs2.%zu", prefix, pair)) {
            kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK,
                             "generator residual tensor name is too long");
            return KOKORO_INCOMPATIBLE_PACK;
        }
        status = bind_conv(&block->conv2[pair], pack, name, channels, channels, kernel_size, 1,
                           (kernel_size - 1) / 2, 1, true, true, error);
        if (status != KOKORO_OK) {
            return status;
        }
        if (!make_tensor_name(name, sizeof(name), "%s.adain1.%zu", prefix, pair)) {
            kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK,
                             "generator residual tensor name is too long");
            return KOKORO_INCOMPATIBLE_PACK;
        }
        status = bind_adain(&block->adain1[pair], pack, name, channels, error);
        if (status != KOKORO_OK) {
            return status;
        }
        if (!make_tensor_name(name, sizeof(name), "%s.adain2.%zu", prefix, pair)) {
            kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK,
                             "generator residual tensor name is too long");
            return KOKORO_INCOMPATIBLE_PACK;
        }
        status = bind_adain(&block->adain2[pair], pack, name, channels, error);
        if (status != KOKORO_OK) {
            return status;
        }
        const uint64_t alpha_shape[3] = {1, channels, 1};
        if (!make_tensor_name(name, sizeof(name), "%s.alpha1.%zu", prefix, pair)) {
            kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK,
                             "generator residual tensor name is too long");
            return KOKORO_INCOMPATIBLE_PACK;
        }
        status = kokoro_tensor_bind_f32(pack, name, 3, alpha_shape, &block->alpha1[pair], error);
        if (status != KOKORO_OK) {
            return status;
        }
        if (!make_tensor_name(name, sizeof(name), "%s.alpha2.%zu", prefix, pair)) {
            kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK,
                             "generator residual tensor name is too long");
            return KOKORO_INCOMPATIBLE_PACK;
        }
        status = kokoro_tensor_bind_f32(pack, name, 3, alpha_shape, &block->alpha2[pair], error);
        if (status != KOKORO_OK) {
            return status;
        }
    }
    return KOKORO_OK;
}

static void free_conv(GeneratorConv *conv) {
    free(conv->weight);
    memset(conv, 0, sizeof(*conv));
}

static void free_conv_transpose(GeneratorConvTranspose *conv) {
    free(conv->weight);
    memset(conv, 0, sizeof(*conv));
}

static void free_resblock(GeneratorResblock *block) {
    for (size_t pair = 0; pair < GENERATOR_RESBLOCK_PAIRS; pair++) {
        free_conv(&block->conv1[pair]);
        free_conv(&block->conv2[pair]);
    }
    memset(block, 0, sizeof(*block));
}

void kokoro_neural_generator_free(KokoroNeuralGenerator *generator) {
    if (generator == NULL) {
        return;
    }
    for (size_t stage = 0; stage < GENERATOR_UPSAMPLE_COUNT; stage++) {
        free_conv_transpose(&generator->upsample[stage]);
        free_conv(&generator->noise_conv[stage]);
        free_resblock(&generator->noise_res[stage]);
    }
    for (size_t index = 0; index < GENERATOR_UPSAMPLE_COUNT * GENERATOR_RESBLOCKS_PER_STAGE;
         index++) {
        free_resblock(&generator->resblocks[index]);
    }
    free_conv(&generator->conv_post);
    free(generator);
}

KokoroStatus kokoro_neural_generator_create(const KokoroPack *pack, KokoroNeuralGenerator **result,
                                            KokoroError *error) {
    if (result != NULL) {
        *result = NULL;
    }
    if (pack == NULL || result == NULL) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "model pack and neural generator output are required");
        return KOKORO_INVALID_ARGUMENT;
    }
    KokoroNeuralGenerator *generator = calloc(1, sizeof(*generator));
    if (generator == NULL) {
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY, "cannot allocate neural generator");
        return KOKORO_OUT_OF_MEMORY;
    }

    KokoroTensorView voice_view;
    KokoroStatus status = kokoro_pack_tensor(pack, "voice.af_heart.style", &voice_view, error);
    if (status != KOKORO_OK) {
        kokoro_neural_generator_free(generator);
        return status;
    }
    if (voice_view.type != KOKORO_TENSOR_F32 || voice_view.rank != 3 || voice_view.shape[0] == 0 ||
        voice_view.shape[1] != 1 || voice_view.shape[2] != GENERATOR_VOICE_SIZE ||
        voice_view.shape[0] > SIZE_MAX / GENERATOR_VOICE_SIZE) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK,
                         "af_heart style tensor must have shape [rows, 1, 256]");
        kokoro_neural_generator_free(generator);
        return KOKORO_INCOMPATIBLE_PACK;
    }
    generator->voice_styles = voice_view.data;
    generator->voice_style_count = (size_t)voice_view.shape[0];
    size_t voice_count = generator->voice_style_count * GENERATOR_VOICE_SIZE;
    const float *voice = generator->voice_styles;
    for (size_t index = 0; index < voice_count; index++) {
        if (!isfinite(voice[index])) {
            kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK,
                             "af_heart style tensor contains a non-finite value");
            kokoro_neural_generator_free(generator);
            return KOKORO_INCOMPATIBLE_PACK;
        }
    }

    const uint64_t source_weight_shape[2] = {1, GENERATOR_SOURCE_HARMONICS};
    const uint64_t source_bias_shape[1] = {1};
    status = kokoro_tensor_bind_f32(pack, "decoder.module.generator.m_source.l_linear.weight", 2,
                                    source_weight_shape, &generator->source_weight, error);
    if (status == KOKORO_OK) {
        status = kokoro_tensor_bind_f32(pack, "decoder.module.generator.m_source.l_linear.bias", 1,
                                        source_bias_shape, &generator->source_bias, error);
    }
    static const size_t upsample_rates[GENERATOR_UPSAMPLE_COUNT] = {10, 6};
    static const size_t upsample_kernels[GENERATOR_UPSAMPLE_COUNT] = {20, 12};
    static const size_t stage_channels[GENERATOR_UPSAMPLE_COUNT] = {256, 128};
    static const size_t resblock_kernels[GENERATOR_RESBLOCKS_PER_STAGE] = {3, 7, 11};
    for (size_t stage = 0; status == KOKORO_OK && stage < GENERATOR_UPSAMPLE_COUNT; stage++) {
        char prefix[160];
        size_t input_channels = GENERATOR_INPUT_CHANNELS >> stage;
        size_t output_channels = stage_channels[stage];
        int length = snprintf(prefix, sizeof(prefix), "decoder.module.generator.ups.%zu", stage);
        if (length < 0 || (size_t)length >= sizeof(prefix)) {
            kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "generator tensor name is too long");
            status = KOKORO_INCOMPATIBLE_PACK;
            break;
        }
        size_t padding = (upsample_kernels[stage] - upsample_rates[stage]) / 2;
        status = bind_conv_transpose(&generator->upsample[stage], pack, prefix, input_channels,
                                     output_channels, upsample_kernels[stage],
                                     upsample_rates[stage], padding, error);
        if (status != KOKORO_OK) {
            break;
        }

        length =
            snprintf(prefix, sizeof(prefix), "decoder.module.generator.noise_convs.%zu", stage);
        if (length < 0 || (size_t)length >= sizeof(prefix)) {
            kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "generator tensor name is too long");
            status = KOKORO_INCOMPATIBLE_PACK;
            break;
        }
        size_t source_stride = stage + 1 < GENERATOR_UPSAMPLE_COUNT ? upsample_rates[stage + 1] : 1;
        size_t source_kernel = stage + 1 < GENERATOR_UPSAMPLE_COUNT ? source_stride * 2 : 1;
        size_t source_padding = stage + 1 < GENERATOR_UPSAMPLE_COUNT ? (source_stride + 1) / 2 : 0;
        status = bind_conv(&generator->noise_conv[stage], pack, prefix, GENERATOR_SOURCE_CHANNELS,
                           output_channels, source_kernel, source_stride, source_padding, 1, false,
                           true, error);
        if (status != KOKORO_OK) {
            break;
        }

        length = snprintf(prefix, sizeof(prefix), "decoder.module.generator.noise_res.%zu", stage);
        if (length < 0 || (size_t)length >= sizeof(prefix)) {
            kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "generator tensor name is too long");
            status = KOKORO_INCOMPATIBLE_PACK;
            break;
        }
        status = bind_resblock(&generator->noise_res[stage], pack, prefix, output_channels,
                               stage == 0 ? 7 : 11, error);
        for (size_t block = 0; status == KOKORO_OK && block < GENERATOR_RESBLOCKS_PER_STAGE;
             block++) {
            size_t index = stage * GENERATOR_RESBLOCKS_PER_STAGE + block;
            length =
                snprintf(prefix, sizeof(prefix), "decoder.module.generator.resblocks.%zu", index);
            if (length < 0 || (size_t)length >= sizeof(prefix)) {
                kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK,
                                 "generator tensor name is too long");
                status = KOKORO_INCOMPATIBLE_PACK;
                break;
            }
            status = bind_resblock(&generator->resblocks[index], pack, prefix, output_channels,
                                   resblock_kernels[block], error);
        }
    }
    if (status == KOKORO_OK) {
        status = bind_conv(&generator->conv_post, pack, "decoder.module.generator.conv_post",
                           stage_channels[GENERATOR_UPSAMPLE_COUNT - 1], GENERATOR_SOURCE_CHANNELS,
                           7, 1, 3, 1, true, true, error);
    }
    if (status != KOKORO_OK) {
        kokoro_neural_generator_free(generator);
        return status;
    }
    *result = generator;
    kokoro_error_set(error, KOKORO_OK, NULL);
    return KOKORO_OK;
}

static KokoroStatus convolution_frame_count(const GeneratorConv *conv, size_t input_frames,
                                            size_t *output_frames, KokoroError *error) {
    if (input_frames == 0 || input_frames > INT_MAX || conv->stride == 0 || conv->dilation == 0 ||
        conv->kernel_size == 0 || conv->input_channels > INT_MAX ||
        conv->output_channels > INT_MAX || conv->kernel_size > INT_MAX ||
        conv->kernel_size - 1 > (SIZE_MAX - 1) / conv->dilation) {
        kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                         "generator convolution dimensions exceed supported range");
        return KOKORO_NUMERIC_ERROR;
    }
    size_t effective_kernel = (conv->kernel_size - 1) * conv->dilation + 1;
    if (conv->padding > (SIZE_MAX - input_frames) / 2) {
        kokoro_error_set(error, KOKORO_NUMERIC_ERROR, "generator convolution padding overflows");
        return KOKORO_NUMERIC_ERROR;
    }
    size_t padded_frames = input_frames + 2 * conv->padding;
    if (padded_frames < effective_kernel) {
        kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                         "generator convolution kernel exceeds its input");
        return KOKORO_NUMERIC_ERROR;
    }
    *output_frames = (padded_frames - effective_kernel) / conv->stride + 1;
    if (*output_frames == 0 || *output_frames > INT_MAX) {
        kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                         "generator convolution output frame count is invalid");
        return KOKORO_NUMERIC_ERROR;
    }
    return KOKORO_OK;
}

static void add_channel_bias(float *output, size_t channels, size_t frames, const float *bias) {
    if (bias == NULL) {
        return;
    }
    for (size_t channel = 0; channel < channels; channel++) {
        float *row = output + channel * frames;
        for (size_t frame = 0; frame < frames; frame++) {
            row[frame] += bias[channel];
        }
    }
}

static KokoroStatus run_convolution(const GeneratorConv *conv, const float *input,
                                    size_t input_frames, float *output, size_t *result_frames,
                                    KokoroError *error) {
    if (conv == NULL || input == NULL || output == NULL || result_frames == NULL) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "generator convolution buffers are required");
        return KOKORO_INVALID_ARGUMENT;
    }
    size_t output_frames = 0;
    KokoroStatus status = convolution_frame_count(conv, input_frames, &output_frames, error);
    if (status != KOKORO_OK) {
        return status;
    }
    size_t kernel_channels = 0;
    size_t output_count = 0;
    if (!checked_product(conv->kernel_size, conv->input_channels, &kernel_channels) ||
        kernel_channels > INT_MAX ||
        !checked_product(conv->output_channels, output_frames, &output_count) ||
        output_count > SIZE_MAX / sizeof(float)) {
        kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                         "generator convolution workspace size overflows");
        return KOKORO_NUMERIC_ERROR;
    }

    if (conv->stride == 1 && output_frames == input_frames) {
        size_t padded_frames = input_frames;
        float *padded = NULL;
        const float *convolution_input = input;
        if (conv->padding > 0) {
            if (conv->padding > (SIZE_MAX - input_frames) / 2) {
                kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                                 "generator convolution workspace size overflows");
                return KOKORO_NUMERIC_ERROR;
            }
            padded_frames += 2 * conv->padding;
            size_t padded_count = 0;
            if (!checked_product(conv->input_channels, padded_frames, &padded_count) ||
                !allocate_floats(padded_count, &padded)) {
                kokoro_error_set(error, KOKORO_OUT_OF_MEMORY,
                                 "cannot allocate generator convolution padding");
                return KOKORO_OUT_OF_MEMORY;
            }
            memset(padded, 0, padded_count * sizeof(float));
            for (size_t channel = 0; channel < conv->input_channels; channel++) {
                memcpy(padded + channel * padded_frames + conv->padding,
                       input + channel * input_frames, input_frames * sizeof(float));
            }
            convolution_input = padded;
        }
        for (size_t kernel = 0; kernel < conv->kernel_size; kernel++) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)conv->output_channels,
                        (int)output_frames, (int)conv->input_channels, 1.0f,
                        conv->weight + kernel * conv->input_channels, (int)kernel_channels,
                        convolution_input + kernel * conv->dilation, (int)padded_frames,
                        kernel == 0 ? 0.0f : 1.0f, output, (int)output_frames);
        }
        free(padded);
    } else {
        size_t column_count = 0;
        if (!checked_product(kernel_channels, output_frames, &column_count) ||
            column_count > SIZE_MAX / sizeof(float)) {
            kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                             "generator convolution workspace size overflows");
            return KOKORO_NUMERIC_ERROR;
        }
        float *columns = NULL;
        if (!allocate_floats(column_count, &columns)) {
            kokoro_error_set(error, KOKORO_OUT_OF_MEMORY,
                             "cannot allocate generator convolution columns");
            return KOKORO_OUT_OF_MEMORY;
        }
        for (size_t kernel = 0; kernel < conv->kernel_size; kernel++) {
            for (size_t channel = 0; channel < conv->input_channels; channel++) {
                float *column_row =
                    columns + (kernel * conv->input_channels + channel) * output_frames;
                const float *input_row = input + channel * input_frames;
                for (size_t frame = 0; frame < output_frames; frame++) {
                    int64_t source_frame = (int64_t)(frame * conv->stride) +
                                           (int64_t)(kernel * conv->dilation) -
                                           (int64_t)conv->padding;
                    column_row[frame] = source_frame >= 0 && (uint64_t)source_frame < input_frames
                                            ? input_row[(size_t)source_frame]
                                            : 0.0f;
                }
            }
        }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)conv->output_channels,
                    (int)output_frames, (int)kernel_channels, 1.0f, conv->weight,
                    (int)kernel_channels, columns, (int)output_frames, 0.0f, output,
                    (int)output_frames);
        free(columns);
    }
    add_channel_bias(output, conv->output_channels, output_frames, conv->bias);
    *result_frames = output_frames;
    return KOKORO_OK;
}

static KokoroStatus transposed_frame_count(const GeneratorConvTranspose *conv, size_t input_frames,
                                           size_t *output_frames, KokoroError *error) {
    if (input_frames == 0 || input_frames > INT_MAX || conv->stride == 0 ||
        conv->input_channels > INT_MAX || conv->output_channels > INT_MAX ||
        conv->kernel_size > INT_MAX || input_frames - 1 > SIZE_MAX / conv->stride) {
        kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                         "generator transposed convolution dimensions exceed supported range");
        return KOKORO_NUMERIC_ERROR;
    }
    size_t expanded = (input_frames - 1) * conv->stride;
    if (expanded > SIZE_MAX - conv->kernel_size) {
        kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                         "generator transposed convolution output size overflows");
        return KOKORO_NUMERIC_ERROR;
    }
    size_t padded = 2 * conv->padding;
    if (expanded + conv->kernel_size < padded) {
        kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                         "generator transposed convolution output size is invalid");
        return KOKORO_NUMERIC_ERROR;
    }
    *output_frames = expanded + conv->kernel_size - padded;
    if (*output_frames == 0 || *output_frames > INT_MAX) {
        kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                         "generator transposed convolution frame count is invalid");
        return KOKORO_NUMERIC_ERROR;
    }
    return KOKORO_OK;
}

static KokoroStatus run_transposed_convolution(const GeneratorConvTranspose *conv,
                                               const float *input, size_t input_frames,
                                               float *output, size_t *result_frames,
                                               KokoroError *error) {
    if (conv == NULL || input == NULL || output == NULL || result_frames == NULL) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "generator transposed convolution buffers are required");
        return KOKORO_INVALID_ARGUMENT;
    }
    size_t output_frames = 0;
    KokoroStatus status = transposed_frame_count(conv, input_frames, &output_frames, error);
    if (status != KOKORO_OK) {
        return status;
    }
    size_t input_count = 0;
    size_t output_count = 0;
    size_t product_count = 0;
    if (!checked_product(conv->input_channels, input_frames, &input_count) ||
        !checked_product(conv->output_channels, output_frames, &output_count) ||
        !checked_product(conv->output_channels, input_frames, &product_count) ||
        output_count > SIZE_MAX / sizeof(float) || product_count > SIZE_MAX / sizeof(float)) {
        kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                         "generator transposed convolution workspace size overflows");
        return KOKORO_NUMERIC_ERROR;
    }
    memset(output, 0, output_count * sizeof(float));
    float *product = NULL;
    if (!allocate_floats(product_count, &product)) {
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY,
                         "cannot allocate generator transposed convolution workspace");
        return KOKORO_OUT_OF_MEMORY;
    }
    for (size_t kernel = 0; kernel < conv->kernel_size; kernel++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)conv->output_channels,
                    (int)input_frames, (int)conv->input_channels, 1.0f,
                    conv->weight + kernel * conv->output_channels * conv->input_channels,
                    (int)conv->input_channels, input, (int)input_frames, 0.0f, product,
                    (int)input_frames);
        for (size_t channel = 0; channel < conv->output_channels; channel++) {
            float *target = output + channel * output_frames;
            const float *source = product + channel * input_frames;
            for (size_t frame = 0; frame < input_frames; frame++) {
                int64_t destination_frame =
                    (int64_t)(frame * conv->stride) - (int64_t)conv->padding + (int64_t)kernel;
                if (destination_frame >= 0 && (uint64_t)destination_frame < output_frames) {
                    target[(size_t)destination_frame] += source[frame];
                }
            }
        }
    }
    free(product);
    add_channel_bias(output, conv->output_channels, output_frames, conv->bias);
    *result_frames = output_frames;
    return KOKORO_OK;
}

static void apply_adain(const GeneratorAdaIN *norm, const float *input, const float *style,
                        size_t channels, size_t frame_count, float *output) {
    float affine[GENERATOR_MAX_CHANNELS * 2];
    cblas_sgemv(CblasRowMajor, CblasNoTrans, (int)(channels * 2), GENERATOR_STYLE_SIZE, 1.0f,
                norm->weight, GENERATOR_STYLE_SIZE, style, 1, 0.0f, affine, 1);
    for (size_t channel = 0; channel < channels * 2; channel++) {
        affine[channel] += norm->bias[channel];
    }
    for (size_t channel = 0; channel < channels; channel++) {
        const float *source = input + channel * frame_count;
        float *target = output + channel * frame_count;
        double mean_sum = 0.0;
        for (size_t frame = 0; frame < frame_count; frame++) {
            mean_sum += (double)source[frame];
        }
        float mean = (float)(mean_sum / (double)frame_count);
        double variance_sum = 0.0;
        for (size_t frame = 0; frame < frame_count; frame++) {
            float centered = source[frame] - mean;
            variance_sum += (double)centered * centered;
        }
        double variance = variance_sum / (double)frame_count;
        float inverse_scale = (float)(1.0 / sqrt(variance + GENERATOR_ADAIN_EPSILON));
        float gamma = affine[channel];
        float beta = affine[channels + channel];
        for (size_t frame = 0; frame < frame_count; frame++) {
            float normalized = (source[frame] - mean) * inverse_scale;
            target[frame] = (1.0f + gamma) * normalized + beta;
        }
    }
}

static void apply_snake(float *values, size_t channels, size_t frame_count, const float *alpha) {
    for (size_t channel = 0; channel < channels; channel++) {
        float scale = alpha[channel];
        float *row = values + channel * frame_count;
        if (scale == 0.0f) {
            continue;
        }
        for (size_t frame = 0; frame < frame_count; frame++) {
            float wave = sinf(scale * row[frame]);
            row[frame] += (wave * wave) / scale;
        }
    }
}

static void apply_leaky_relu(float *values, size_t count, float negative_slope) {
    for (size_t index = 0; index < count; index++) {
        if (values[index] < 0.0f) {
            values[index] *= negative_slope;
        }
    }
}

static KokoroStatus run_resblock(const GeneratorResblock *block, const float *input,
                                 size_t frame_count, const float *style, float **result,
                                 KokoroError *error) {
    *result = NULL;
    size_t value_count = 0;
    if (frame_count == 0 || block->channels == 0 ||
        !checked_product(block->channels, frame_count, &value_count) ||
        value_count > SIZE_MAX / sizeof(float)) {
        kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                         "generator residual block dimensions overflow");
        return KOKORO_NUMERIC_ERROR;
    }
    float *running = NULL;
    float *normalized = NULL;
    float *residual = NULL;
    if (!allocate_floats(value_count, &running) || !allocate_floats(value_count, &normalized) ||
        !allocate_floats(value_count, &residual)) {
        free(running);
        free(normalized);
        free(residual);
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY,
                         "cannot allocate generator residual block workspace");
        return KOKORO_OUT_OF_MEMORY;
    }
    memcpy(running, input, value_count * sizeof(float));
    KokoroStatus status = KOKORO_OK;
    for (size_t pair = 0; pair < GENERATOR_RESBLOCK_PAIRS; pair++) {
        apply_adain(&block->adain1[pair], running, style, block->channels, frame_count, normalized);
        apply_snake(normalized, block->channels, frame_count, block->alpha1[pair]);
        size_t convolution_frames = 0;
        status = run_convolution(&block->conv1[pair], normalized, frame_count, residual,
                                 &convolution_frames, error);
        if (status != KOKORO_OK || convolution_frames != frame_count) {
            if (status == KOKORO_OK) {
                kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                                 "generator residual convolution changed frame count");
                status = KOKORO_NUMERIC_ERROR;
            }
            break;
        }
        apply_adain(&block->adain2[pair], residual, style, block->channels, frame_count,
                    normalized);
        apply_snake(normalized, block->channels, frame_count, block->alpha2[pair]);
        status = run_convolution(&block->conv2[pair], normalized, frame_count, residual,
                                 &convolution_frames, error);
        if (status != KOKORO_OK || convolution_frames != frame_count) {
            if (status == KOKORO_OK) {
                kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                                 "generator residual convolution changed frame count");
                status = KOKORO_NUMERIC_ERROR;
            }
            break;
        }
        for (size_t index = 0; index < value_count; index++) {
            running[index] += residual[index];
        }
    }
    free(normalized);
    free(residual);
    if (status != KOKORO_OK) {
        free(running);
        return status;
    }
    *result = running;
    return KOKORO_OK;
}

typedef struct {
    uint64_t state;
    bool has_spare;
    double spare;
} GeneratorRandom;

static uint64_t random_next(GeneratorRandom *random) {
    uint64_t value = random->state;
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    random->state = value;
    return value * UINT64_C(2685821657736338717);
}

static double random_uniform(GeneratorRandom *random) {
    return ((double)(random_next(random) >> 11) + 0.5) / 9007199254740992.0;
}

static float random_normal(GeneratorRandom *random) {
    if (random->has_spare) {
        random->has_spare = false;
        return (float)random->spare;
    }
    double radius = sqrt(-2.0 * log(random_uniform(random)));
    double angle = GENERATOR_TWO_PI * random_uniform(random);
    random->spare = radius * sin(angle);
    random->has_spare = true;
    return (float)(radius * cos(angle));
}

static KokoroStatus generate_harmonic_excitation(const KokoroNeuralGenerator *generator,
                                                 const float *f0, size_t frame_count, uint64_t seed,
                                                 float **result, size_t *sample_count,
                                                 KokoroError *error) {
    *result = NULL;
    if (frame_count == 0 || frame_count > GENERATOR_MAX_FRAMES ||
        frame_count > SIZE_MAX / GENERATOR_F0_UPSAMPLE) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "generator F0 frame count exceeds the supported range");
        return KOKORO_INVALID_ARGUMENT;
    }
    size_t samples = frame_count * GENERATOR_F0_UPSAMPLE;
    size_t phase_count = 0;
    if (!checked_product(frame_count, GENERATOR_SOURCE_HARMONICS, &phase_count) ||
        phase_count > SIZE_MAX / sizeof(float) || samples > SIZE_MAX / sizeof(float)) {
        kokoro_error_set(error, KOKORO_NUMERIC_ERROR, "generator excitation dimensions overflow");
        return KOKORO_NUMERIC_ERROR;
    }
    float *phase_curve = NULL;
    float *harmonic = NULL;
    if (!allocate_floats(phase_count, &phase_curve) || !allocate_floats(samples, &harmonic)) {
        free(phase_curve);
        free(harmonic);
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY,
                         "cannot allocate generator source excitation");
        return KOKORO_OUT_OF_MEMORY;
    }

    const float two_pi_times_scale = (float)(GENERATOR_TWO_PI * GENERATOR_F0_UPSAMPLE);
    for (size_t component = 0; component < GENERATOR_SOURCE_HARMONICS; component++) {
        float cumulative = 0.0f;
        float harmonic_number = (float)(component + 1);
        for (size_t frame = 0; frame < frame_count; frame++) {
            double cycles = ((double)f0[frame] * harmonic_number) / GENERATOR_SOURCE_RATE;
            if (!isfinite(cycles)) {
                free(phase_curve);
                free(harmonic);
                kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                                 "generator F0 cannot produce a finite source phase");
                return KOKORO_NUMERIC_ERROR;
            }
            cycles -= floor(cycles);
            cumulative += (float)cycles;
            phase_curve[component * frame_count + frame] = cumulative * two_pi_times_scale;
        }
    }

    GeneratorRandom random = {.state = seed == 0 ? UINT64_C(0x9e3779b97f4a7c15) : seed};
    for (size_t sample = 0; sample < samples; sample++) {
        size_t f0_frame = sample / GENERATOR_F0_UPSAMPLE;
        bool voiced = f0[f0_frame] > GENERATOR_VOICED_THRESHOLD;
        float noise_amplitude = voiced ? GENERATOR_VOICED_NOISE : GENERATOR_UNVOICED_NOISE;
        double source_position = ((double)sample + 0.5) / GENERATOR_F0_UPSAMPLE - 0.5;
        size_t left_frame = source_position <= 0.0 ? 0 : (size_t)floor(source_position);
        if (left_frame >= frame_count) {
            left_frame = frame_count - 1;
        }
        size_t right_frame = left_frame + 1 < frame_count ? left_frame + 1 : left_frame;
        float fraction = source_position <= 0.0 || right_frame == left_frame
                             ? 0.0f
                             : (float)(source_position - (double)left_frame);
        float merged = generator->source_bias[0];
        for (size_t component = 0; component < GENERATOR_SOURCE_HARMONICS; component++) {
            const float *component_phase = phase_curve + component * frame_count;
            float phase = component_phase[left_frame] +
                          (component_phase[right_frame] - component_phase[left_frame]) * fraction;
            float periodic = voiced ? sinf(phase) * GENERATOR_SINE_AMPLITUDE : 0.0f;
            float noisy = periodic + noise_amplitude * random_normal(&random);
            merged += generator->source_weight[component] * noisy;
        }
        harmonic[sample] = tanhf(merged);
    }
    free(phase_curve);
    *result = harmonic;
    *sample_count = samples;
    return KOKORO_OK;
}

static size_t reflect_stft_index(int64_t index, size_t sample_count) {
    if (index < 0) {
        return (size_t)(-index);
    }
    if ((uint64_t)index >= sample_count) {
        return 2 * sample_count - 2 - (size_t)index;
    }
    return (size_t)index;
}

static KokoroStatus source_spectrogram(const float *harmonic, size_t sample_count, float **result,
                                       size_t *frame_count, KokoroError *error) {
    *result = NULL;
    if (harmonic == NULL || sample_count <= GENERATOR_FFT_SIZE / 2 || sample_count > INT_MAX ||
        sample_count > SIZE_MAX - GENERATOR_HOP_SIZE) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "generator harmonic excitation has an invalid sample count");
        return KOKORO_INVALID_ARGUMENT;
    }
    for (size_t sample = 0; sample < sample_count; sample++) {
        if (!isfinite(harmonic[sample])) {
            kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                             "generator harmonic excitation contains a non-finite value");
            return KOKORO_NUMERIC_ERROR;
        }
    }
    size_t frames = sample_count / GENERATOR_HOP_SIZE + 1;
    size_t feature_count = 0;
    if (frames > INT_MAX || !checked_product(GENERATOR_SOURCE_CHANNELS, frames, &feature_count) ||
        feature_count > SIZE_MAX / sizeof(float)) {
        kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                         "generator source spectrogram dimensions overflow");
        return KOKORO_NUMERIC_ERROR;
    }
    float *features = NULL;
    if (!allocate_floats(feature_count, &features)) {
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY,
                         "cannot allocate generator source spectrogram");
        return KOKORO_OUT_OF_MEMORY;
    }
    double window[GENERATOR_FFT_SIZE];
    double cosine[KOKORO_NEURAL_GENERATOR_SPECTRAL_BINS][GENERATOR_FFT_SIZE];
    double negative_sine[KOKORO_NEURAL_GENERATOR_SPECTRAL_BINS][GENERATOR_FFT_SIZE];
    for (size_t sample = 0; sample < GENERATOR_FFT_SIZE; sample++) {
        double angle = GENERATOR_TWO_PI * (double)sample / GENERATOR_FFT_SIZE;
        window[sample] = 0.5 * (1.0 - cos(angle));
    }
    for (size_t bin = 0; bin < KOKORO_NEURAL_GENERATOR_SPECTRAL_BINS; bin++) {
        for (size_t sample = 0; sample < GENERATOR_FFT_SIZE; sample++) {
            double angle = GENERATOR_TWO_PI * (double)bin * sample / GENERATOR_FFT_SIZE;
            cosine[bin][sample] = cos(angle);
            negative_sine[bin][sample] = -sin(angle);
        }
    }
    for (size_t frame = 0; frame < frames; frame++) {
        int64_t frame_start =
            (int64_t)(frame * GENERATOR_HOP_SIZE) - (int64_t)(GENERATOR_FFT_SIZE / 2);
        for (size_t bin = 0; bin < KOKORO_NEURAL_GENERATOR_SPECTRAL_BINS; bin++) {
            double real = 0.0;
            double imaginary = 0.0;
            for (size_t sample = 0; sample < GENERATOR_FFT_SIZE; sample++) {
                size_t source_index =
                    reflect_stft_index(frame_start + (int64_t)sample, sample_count);
                double weighted = (double)harmonic[source_index] * window[sample];
                real += weighted * cosine[bin][sample];
                imaginary += weighted * negative_sine[bin][sample];
            }
            if (bin == GENERATOR_FFT_SIZE / 2) {
                imaginary = 0.0;
            }
            size_t magnitude_index = bin * frames + frame;
            size_t phase_index = (KOKORO_NEURAL_GENERATOR_SPECTRAL_BINS + bin) * frames + frame;
            double magnitude = sqrt(real * real + imaginary * imaginary);
            double phase =
                imaginary == 0.0 && real < 0.0 ? GENERATOR_TWO_PI / 2.0 : atan2(imaginary, real);
            if (!isfinite(magnitude) || magnitude > FLT_MAX || !isfinite(phase)) {
                free(features);
                kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                                 "generator source STFT exceeds the finite float range");
                return KOKORO_NUMERIC_ERROR;
            }
            features[magnitude_index] = (float)magnitude;
            features[phase_index] = (float)phase;
        }
    }
    *result = features;
    *frame_count = frames;
    return KOKORO_OK;
}

static bool source_output_is_empty(const KokoroNeuralGeneratorSourceOutput *output) {
    return output->frame_count == 0 && output->magnitude == NULL && output->phase == NULL;
}

void kokoro_neural_generator_source_output_free(KokoroNeuralGeneratorSourceOutput *output) {
    if (output == NULL) {
        return;
    }
    free(output->magnitude);
    free(output->phase);
    memset(output, 0, sizeof(*output));
}

KokoroStatus kokoro_neural_generator_source_spectrogram(const float *harmonic, size_t sample_count,
                                                        KokoroNeuralGeneratorSourceOutput *output,
                                                        KokoroError *error) {
    if (output == NULL || !source_output_is_empty(output)) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "source spectrogram output must be empty and non-null");
        return KOKORO_INVALID_ARGUMENT;
    }
    float *features = NULL;
    size_t frames = 0;
    KokoroStatus status = source_spectrogram(harmonic, sample_count, &features, &frames, error);
    if (status != KOKORO_OK) {
        return status;
    }
    size_t count = 0;
    if (!checked_product(KOKORO_NEURAL_GENERATOR_SPECTRAL_BINS, frames, &count)) {
        free(features);
        kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                         "generator source output dimensions overflow");
        return KOKORO_NUMERIC_ERROR;
    }
    float *magnitude = NULL;
    float *phase = NULL;
    if (!allocate_floats(count, &magnitude) || !allocate_floats(count, &phase)) {
        free(features);
        free(magnitude);
        free(phase);
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY,
                         "cannot allocate native source spectrogram output");
        return KOKORO_OUT_OF_MEMORY;
    }
    memcpy(magnitude, features, count * sizeof(float));
    memcpy(phase, features + count, count * sizeof(float));
    free(features);
    output->frame_count = frames;
    output->magnitude = magnitude;
    output->phase = phase;
    kokoro_error_set(error, KOKORO_OK, NULL);
    return KOKORO_OK;
}

static bool output_is_empty(const KokoroSpectralOutput *output) {
    return output->magnitude == NULL && output->phase == NULL && output->input_frame_count == 0 &&
           output->frame_count == 0;
}

void kokoro_spectral_output_free(KokoroSpectralOutput *output) {
    if (output == NULL) {
        return;
    }
    free(output->magnitude);
    free(output->phase);
    memset(output, 0, sizeof(*output));
}

static KokoroStatus validate_excitation(const KokoroNeuralGeneratorExcitation *excitation,
                                        size_t sample_count, KokoroError *error) {
    if (excitation == NULL || excitation->sample_count != sample_count ||
        excitation->harmonic == NULL || excitation->noise == NULL || excitation->voiced == NULL) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "captured generator excitation must contain harmonic, noise, and voiced "
                         "arrays at the expected sample count");
        return KOKORO_INVALID_ARGUMENT;
    }
    for (size_t sample = 0; sample < sample_count; sample++) {
        if (!isfinite(excitation->harmonic[sample]) || !isfinite(excitation->noise[sample]) ||
            !isfinite(excitation->voiced[sample])) {
            kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                             "captured generator excitation contains a non-finite value");
            return KOKORO_NUMERIC_ERROR;
        }
    }
    return KOKORO_OK;
}

static KokoroStatus copy_source_spectrogram(const KokoroNeuralGeneratorSourceSpectrogram *source,
                                            size_t sample_count, float **features,
                                            size_t *frame_count, KokoroError *error) {
    *features = NULL;
    *frame_count = 0;
    if (source == NULL || source->magnitude == NULL || source->phase == NULL) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "captured source spectrogram requires magnitude and phase arrays");
        return KOKORO_INVALID_ARGUMENT;
    }
    size_t expected_frames = sample_count / GENERATOR_HOP_SIZE + 1;
    size_t count = 0;
    size_t feature_count = 0;
    if (source->frame_count != expected_frames || expected_frames > INT_MAX ||
        !checked_product(KOKORO_NEURAL_GENERATOR_SPECTRAL_BINS, expected_frames, &count) ||
        !checked_product(GENERATOR_SOURCE_CHANNELS, expected_frames, &feature_count) ||
        feature_count > SIZE_MAX / sizeof(float)) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "captured source spectrogram has an unexpected frame count");
        return KOKORO_INVALID_ARGUMENT;
    }
    for (size_t index = 0; index < count; index++) {
        if (!isfinite(source->magnitude[index]) || source->magnitude[index] < 0.0f ||
            !isfinite(source->phase[index])) {
            kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                             "captured source spectrogram contains an invalid value");
            return KOKORO_NUMERIC_ERROR;
        }
    }
    float *copy = NULL;
    if (!allocate_floats(feature_count, &copy)) {
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY,
                         "cannot allocate captured source spectrogram");
        return KOKORO_OUT_OF_MEMORY;
    }
    memcpy(copy, source->magnitude, count * sizeof(float));
    memcpy(copy + count, source->phase, count * sizeof(float));
    *features = copy;
    *frame_count = expected_frames;
    return KOKORO_OK;
}

KokoroStatus kokoro_neural_generate(const KokoroNeuralGenerator *generator,
                                    const KokoroDecoderOutput *decoder,
                                    const KokoroPitchNoiseOutput *pitch_noise,
                                    size_t voice_style_index,
                                    const KokoroNeuralGeneratorOptions *options,
                                    KokoroSpectralOutput *output, KokoroError *error) {
    if (generator == NULL || decoder == NULL || pitch_noise == NULL || options == NULL ||
        output == NULL) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "generator, decoder features, pitch/noise, options, and output are "
                         "required");
        return KOKORO_INVALID_ARGUMENT;
    }
    if (!output_is_empty(output)) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "spectral output must be empty; free it before reuse");
        return KOKORO_INVALID_ARGUMENT;
    }
    if (voice_style_index >= generator->voice_style_count) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT, "voice style index is outside the pack");
        return KOKORO_INVALID_ARGUMENT;
    }
    if (decoder->duration_frame_count == 0 ||
        decoder->duration_frame_count > GENERATOR_MAX_FRAMES / 2 ||
        decoder->duration_frame_count > SIZE_MAX / 2 ||
        decoder->feature_frame_count != decoder->duration_frame_count * 2 ||
        pitch_noise->duration_frame_count != decoder->duration_frame_count ||
        pitch_noise->prediction_frame_count != decoder->feature_frame_count ||
        decoder->features == NULL || pitch_noise->f0 == NULL || pitch_noise->noise == NULL) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "generator requires aligned decoder and pitch/noise frame counts");
        return KOKORO_INVALID_ARGUMENT;
    }
    size_t feature_frames = decoder->feature_frame_count;
    size_t feature_count = 0;
    size_t excitation_samples = 0;
    size_t output_frames = 0;
    size_t spectrum_count = 0;
    if (feature_frames > GENERATOR_MAX_FRAMES || feature_frames > INT_MAX ||
        !checked_product(GENERATOR_INPUT_CHANNELS, feature_frames, &feature_count) ||
        feature_count > SIZE_MAX / sizeof(float) ||
        !checked_product(feature_frames, GENERATOR_F0_UPSAMPLE, &excitation_samples) ||
        excitation_samples > INT_MAX || !checked_product(feature_frames, 60, &output_frames) ||
        output_frames == SIZE_MAX ||
        !checked_product(KOKORO_NEURAL_GENERATOR_SPECTRAL_BINS, output_frames + 1,
                         &spectrum_count) ||
        spectrum_count > SIZE_MAX / sizeof(float)) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "generator frame dimensions exceed addressable memory");
        return KOKORO_INVALID_ARGUMENT;
    }
    output_frames += 1;
    for (size_t index = 0; index < feature_count; index++) {
        if (!isfinite(decoder->features[index])) {
            kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                             "decoder features contain a non-finite value");
            return KOKORO_NUMERIC_ERROR;
        }
    }
    for (size_t frame = 0; frame < feature_frames; frame++) {
        if (!isfinite(pitch_noise->f0[frame]) || !isfinite(pitch_noise->noise[frame])) {
            kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                             "generator F0/noise contains a non-finite value");
            return KOKORO_NUMERIC_ERROR;
        }
    }
    if (options->excitation != NULL) {
        KokoroStatus status = validate_excitation(options->excitation, excitation_samples, error);
        if (status != KOKORO_OK) {
            return status;
        }
    }

    KokoroStatus status = KOKORO_OK;
    float *generated_harmonic = NULL;
    const float *harmonic = NULL;
    size_t generated_samples = 0;
    if (options->excitation != NULL) {
        harmonic = options->excitation->harmonic;
    } else {
        status = generate_harmonic_excitation(generator, pitch_noise->f0, feature_frames,
                                              options->random_seed, &generated_harmonic,
                                              &generated_samples, error);
        if (status != KOKORO_OK) {
            return status;
        }
        if (generated_samples != excitation_samples) {
            free(generated_harmonic);
            kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                             "generator source excitation has an unexpected sample count");
            return KOKORO_NUMERIC_ERROR;
        }
        harmonic = generated_harmonic;
    }

    float *source_features = NULL;
    size_t source_frames = 0;
    if (options->source_spectrogram != NULL) {
        status = copy_source_spectrogram(options->source_spectrogram, excitation_samples,
                                         &source_features, &source_frames, error);
    } else {
        status = source_spectrogram(harmonic, excitation_samples, &source_features, &source_frames,
                                    error);
    }
    free(generated_harmonic);
    if (status != KOKORO_OK) {
        return status;
    }
    size_t current_channels = GENERATOR_INPUT_CHANNELS;
    size_t current_frames = feature_frames;
    float *current = NULL;
    if (!allocate_floats(feature_count, &current)) {
        free(source_features);
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY, "cannot allocate generator input features");
        return KOKORO_OUT_OF_MEMORY;
    }
    memcpy(current, decoder->features, feature_count * sizeof(float));
    const float *style = generator->voice_styles + voice_style_index * GENERATOR_VOICE_SIZE;
    static const size_t stage_channels[GENERATOR_UPSAMPLE_COUNT] = {256, 128};
    for (size_t stage = 0; stage < GENERATOR_UPSAMPLE_COUNT; stage++) {
        size_t current_count = 0;
        if (!checked_product(current_channels, current_frames, &current_count)) {
            kokoro_error_set(error, KOKORO_NUMERIC_ERROR, "generator feature dimensions overflow");
            status = KOKORO_NUMERIC_ERROR;
            goto cleanup;
        }
        apply_leaky_relu(current, current_count, GENERATOR_UPSAMPLE_LEAKY_RELU);

        size_t source_output_frames = 0;
        status = convolution_frame_count(&generator->noise_conv[stage], source_frames,
                                         &source_output_frames, error);
        if (status != KOKORO_OK) {
            goto cleanup;
        }
        size_t source_output_count = 0;
        if (!checked_product(stage_channels[stage], source_output_frames, &source_output_count)) {
            kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                             "generator source feature dimensions overflow");
            status = KOKORO_NUMERIC_ERROR;
            goto cleanup;
        }
        float *source = NULL;
        if (!allocate_floats(source_output_count, &source)) {
            kokoro_error_set(error, KOKORO_OUT_OF_MEMORY,
                             "cannot allocate generator source convolution output");
            status = KOKORO_OUT_OF_MEMORY;
            goto cleanup;
        }
        size_t actual_source_frames = 0;
        status = run_convolution(&generator->noise_conv[stage], source_features, source_frames,
                                 source, &actual_source_frames, error);
        if (status != KOKORO_OK || actual_source_frames != source_output_frames) {
            free(source);
            if (status == KOKORO_OK) {
                kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                                 "generator source convolution changed frame count");
                status = KOKORO_NUMERIC_ERROR;
            }
            goto cleanup;
        }
        float *source_residual = NULL;
        status = run_resblock(&generator->noise_res[stage], source, source_output_frames, style,
                              &source_residual, error);
        free(source);
        if (status != KOKORO_OK) {
            free(source_residual);
            goto cleanup;
        }
        source = source_residual;

        size_t upsampled_frames = 0;
        status = transposed_frame_count(&generator->upsample[stage], current_frames,
                                        &upsampled_frames, error);
        if (status != KOKORO_OK) {
            free(source);
            goto cleanup;
        }
        bool reflect_pad = stage + 1 == GENERATOR_UPSAMPLE_COUNT;
        size_t target_frames = upsampled_frames + (reflect_pad ? 1 : 0);
        size_t upsampled_count = 0;
        if ((reflect_pad && upsampled_frames == 0) ||
            !checked_product(stage_channels[stage], upsampled_frames, &upsampled_count) ||
            !checked_product(stage_channels[stage], target_frames, &source_output_count) ||
            target_frames > INT_MAX) {
            free(source);
            kokoro_error_set(error, KOKORO_NUMERIC_ERROR, "generator upsample dimensions overflow");
            status = KOKORO_NUMERIC_ERROR;
            goto cleanup;
        }
        float *upsampled = NULL;
        if (!allocate_floats(upsampled_count, &upsampled)) {
            free(source);
            kokoro_error_set(error, KOKORO_OUT_OF_MEMORY,
                             "cannot allocate generator upsample output");
            status = KOKORO_OUT_OF_MEMORY;
            goto cleanup;
        }
        size_t actual_upsampled_frames = 0;
        status = run_transposed_convolution(&generator->upsample[stage], current, current_frames,
                                            upsampled, &actual_upsampled_frames, error);
        if (status != KOKORO_OK || actual_upsampled_frames != upsampled_frames) {
            free(source);
            free(upsampled);
            if (status == KOKORO_OK) {
                kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                                 "generator transposed convolution changed frame count");
                status = KOKORO_NUMERIC_ERROR;
            }
            goto cleanup;
        }
        float *stage_input = upsampled;
        if (reflect_pad) {
            if (upsampled_frames < 2) {
                free(source);
                free(upsampled);
                kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                                 "generator reflection padding requires two upsampled frames");
                status = KOKORO_NUMERIC_ERROR;
                goto cleanup;
            }
            float *padded = NULL;
            if (!allocate_floats(source_output_count, &padded)) {
                free(source);
                free(upsampled);
                kokoro_error_set(error, KOKORO_OUT_OF_MEMORY,
                                 "cannot allocate generator reflection padding");
                status = KOKORO_OUT_OF_MEMORY;
                goto cleanup;
            }
            for (size_t channel = 0; channel < stage_channels[stage]; channel++) {
                const float *input_row = upsampled + channel * upsampled_frames;
                float *target_row = padded + channel * target_frames;
                target_row[0] = input_row[1];
                memcpy(target_row + 1, input_row, upsampled_frames * sizeof(float));
            }
            free(upsampled);
            stage_input = padded;
        }
        if (target_frames != source_output_frames) {
            free(source);
            free(stage_input);
            kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                             "generator source and learned upsample frame counts do not match");
            status = KOKORO_NUMERIC_ERROR;
            goto cleanup;
        }
        size_t stage_count = 0;
        if (!checked_product(stage_channels[stage], target_frames, &stage_count)) {
            free(source);
            free(stage_input);
            kokoro_error_set(error, KOKORO_NUMERIC_ERROR, "generator stage dimensions overflow");
            status = KOKORO_NUMERIC_ERROR;
            goto cleanup;
        }
        for (size_t index = 0; index < stage_count; index++) {
            stage_input[index] += source[index];
        }
        free(source);

        float *sum = NULL;
        if (!allocate_floats(stage_count, &sum)) {
            free(stage_input);
            kokoro_error_set(error, KOKORO_OUT_OF_MEMORY, "cannot allocate generator residual sum");
            status = KOKORO_OUT_OF_MEMORY;
            goto cleanup;
        }
        memset(sum, 0, stage_count * sizeof(float));
        for (size_t block = 0; block < GENERATOR_RESBLOCKS_PER_STAGE; block++) {
            size_t block_index = stage * GENERATOR_RESBLOCKS_PER_STAGE + block;
            float *block_output = NULL;
            status = run_resblock(&generator->resblocks[block_index], stage_input, target_frames,
                                  style, &block_output, error);
            if (status != KOKORO_OK) {
                free(block_output);
                free(sum);
                free(stage_input);
                goto cleanup;
            }
            for (size_t index = 0; index < stage_count; index++) {
                sum[index] += block_output[index];
            }
            free(block_output);
        }
        for (size_t index = 0; index < stage_count; index++) {
            sum[index] *= 1.0f / GENERATOR_RESBLOCKS_PER_STAGE;
        }
        free(stage_input);
        free(current);
        current = sum;
        current_channels = stage_channels[stage];
        current_frames = target_frames;
    }

    {
        size_t current_count = 0;
        if (!checked_product(current_channels, current_frames, &current_count)) {
            kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                             "generator post-convolution dimensions overflow");
            status = KOKORO_NUMERIC_ERROR;
            goto cleanup;
        }
        apply_leaky_relu(current, current_count, GENERATOR_POST_LEAKY_RELU);
        size_t post_frames = 0;
        status =
            convolution_frame_count(&generator->conv_post, current_frames, &post_frames, error);
        if (status != KOKORO_OK || post_frames != current_frames) {
            if (status == KOKORO_OK) {
                kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                                 "generator post-convolution changed frame count");
                status = KOKORO_NUMERIC_ERROR;
            }
            goto cleanup;
        }
        size_t raw_count = 0;
        if (!checked_product(GENERATOR_SOURCE_CHANNELS, post_frames, &raw_count)) {
            kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                             "generator spectral tensor dimensions overflow");
            status = KOKORO_NUMERIC_ERROR;
            goto cleanup;
        }
        float *raw_spectrum = NULL;
        if (!allocate_floats(raw_count, &raw_spectrum)) {
            kokoro_error_set(error, KOKORO_OUT_OF_MEMORY,
                             "cannot allocate generator spectral tensor");
            status = KOKORO_OUT_OF_MEMORY;
            goto cleanup;
        }
        size_t actual_post_frames = 0;
        status = run_convolution(&generator->conv_post, current, current_frames, raw_spectrum,
                                 &actual_post_frames, error);
        if (status != KOKORO_OK || actual_post_frames != post_frames) {
            free(raw_spectrum);
            if (status == KOKORO_OK) {
                kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                                 "generator post-convolution changed frame count");
                status = KOKORO_NUMERIC_ERROR;
            }
            goto cleanup;
        }
        float *magnitude = NULL;
        float *phase = NULL;
        if (!allocate_floats(spectrum_count, &magnitude) ||
            !allocate_floats(spectrum_count, &phase)) {
            free(raw_spectrum);
            free(magnitude);
            free(phase);
            kokoro_error_set(error, KOKORO_OUT_OF_MEMORY,
                             "cannot allocate generator spectral outputs");
            status = KOKORO_OUT_OF_MEMORY;
            goto cleanup;
        }
        for (size_t bin = 0; bin < KOKORO_NEURAL_GENERATOR_SPECTRAL_BINS; bin++) {
            for (size_t frame = 0; frame < post_frames; frame++) {
                size_t index = bin * post_frames + frame;
                magnitude[index] = expf(raw_spectrum[index]);
                phase[index] =
                    sinf(raw_spectrum[(KOKORO_NEURAL_GENERATOR_SPECTRAL_BINS + bin) * post_frames +
                                      frame]);
                if (!isfinite(magnitude[index]) || !isfinite(phase[index])) {
                    free(raw_spectrum);
                    free(magnitude);
                    free(phase);
                    kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                                     "generator produced a non-finite spectral value");
                    status = KOKORO_NUMERIC_ERROR;
                    goto cleanup;
                }
            }
        }
        free(raw_spectrum);
        output->input_frame_count = feature_frames;
        output->frame_count = post_frames;
        output->magnitude = magnitude;
        output->phase = phase;
    }

cleanup:
    free(source_features);
    free(current);
    if (status != KOKORO_OK) {
        return status;
    }
    kokoro_error_set(error, KOKORO_OK, NULL);
    return KOKORO_OK;
}

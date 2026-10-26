#include "phonon_mimi.h"
#include "phonon_math.h"
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct PhononMimiState {
    const PhononMimiModel *model;
    size_t position;
    int failed;
    float previous[512];
    float keys[2][250][512], values[2][250][512];
    float hidden[16][512], normalized[16][512], qkv[16][1536];
    float attended[16][512], projected[16][512], mlp[16][2048];
    float ordered_keys[250][64], ordered_values[250][64];
};

PhononMimiConfig phonon_mimi_config(void) {
    return (PhononMimiConfig){24000, 12.5f, {6, 5, 4}, 250, 10000.0f};
}
static int config_stride(const PhononMimiConfig *config, size_t *stride) {
    if (!config || config->sample_rate != 24000 || config->frame_rate != 12.5f ||
        config->ratios[0] != 6 || config->ratios[1] != 5 || config->ratios[2] != 4 ||
        config->context != 250 || config->rope_theta != 10000.0f)
        return 0;
    size_t samples = (size_t)((float)config->sample_rate / config->frame_rate);
    size_t ratio = config->ratios[0] * config->ratios[1] * config->ratios[2];
    if (samples % ratio || samples / ratio != 16)
        return 0;
    *stride = samples / ratio;
    return 1;
}
static int tensor_shape(const PhononTensorView *view, size_t rank, size_t a, size_t b, size_t c) {
    if (!view || !view->data || view->dtype != PHONON_PACK_BF16 || view->rank != rank ||
        view->shape[0] != a || (rank > 1 && view->shape[1] != b) ||
        (rank > 2 && view->shape[2] != c))
        return 0;
    size_t count = a * (rank > 1 ? b : 1) * (rank > 2 ? c : 1);
    return view->byte_count == count * sizeof(uint16_t);
}
int phonon_mimi_model_load(const PhononPack *pack, const PhononMimiConfig *config,
                           PhononMimiModel *model) {
    PhononMimiModel next = {0};
    if (!pack || !model || !config_stride(config, &next.stride))
        return 0;
    next.config = *config;
    if (!phonon_pack_tensor(pack, "mimi.quantizer.output_proj.weight", &next.projection) ||
        !tensor_shape(&next.projection, 3, 512, 32, 1) ||
        !phonon_pack_tensor(pack, "mimi.upsample.convtr.convtr.weight", &next.upsample) ||
        !tensor_shape(&next.upsample, 3, 512, 1, 2 * next.stride))
        return 0;
    const char *suffix[] = {"self_attn.in_proj.weight",
                            "self_attn.out_proj.weight",
                            "linear1.weight",
                            "linear2.weight",
                            "norm1.weight",
                            "norm1.bias",
                            "norm2.weight",
                            "norm2.bias",
                            "layer_scale_1.scale",
                            "layer_scale_2.scale"};
    for (size_t layer = 0; layer < 2; layer++)
        for (size_t field = 0; field < 10; field++) {
            char name[160];
            int n =
                snprintf(name, sizeof(name), "mimi.decoder_transformer.transformer.layers.%zu.%s",
                         layer, suffix[field]);
            if (n < 0 || (size_t)n >= sizeof(name) ||
                !phonon_pack_tensor(pack, name, &next.layers[layer][field]))
                return 0;
            size_t rows = field == 0 ? 1536 : (field == 2 ? 2048 : 512);
            size_t cols = field == 3 ? 2048 : 512;
            if (!tensor_shape(&next.layers[layer][field], field < 4 ? 2 : 1, rows, cols, 0))
                return 0;
        }
    *model = next;
    return 1;
}
int phonon_mimi_state_create(const PhononMimiModel *model, PhononMimiState **state) {
    size_t stride;
    if (!state)
        return 0;
    *state = NULL;
    if (!model || !config_stride(&model->config, &stride) || model->stride != stride)
        return 0;
    PhononMimiState *next = calloc(1, sizeof(*next));
    if (!next)
        return 0;
    next->model = model;
    *state = next;
    return 1;
}
void phonon_mimi_state_reset(PhononMimiState *state) {
    if (!state)
        return;
    state->position = 0;
    state->failed = 0;
    memset(state->previous, 0, sizeof(state->previous));
    /* Cache entries become visible only after their absolute position is written. */
}
void phonon_mimi_state_destroy(PhononMimiState *state) { free(state); }
size_t phonon_mimi_positions(const PhononMimiState *state) { return state ? state->position : 0; }

int phonon_mimi_upsample_frame(const PhononTensorView *weight, const float *input,
                               const float *previous, size_t channels, size_t stride,
                               float *output) {
#pragma STDC FP_CONTRACT OFF
    if (!channels || channels > 512 || !stride || stride > 16 || !input || !previous || !output ||
        !tensor_shape(weight, 3, channels, 1, 2 * stride))
        return 0;
    float kernel[512 * 32];
    if (!phonon_bf16_to_float(weight->data, channels * 2 * stride, kernel))
        return 0;
    for (size_t channel = 0; channel < channels; channel++) {
        if (!isfinite(input[channel]) || !isfinite(previous[channel]))
            return 0;
        for (size_t offset = 0; offset < stride; offset++) {
            float current = input[channel] * kernel[channel * 2 * stride + offset];
            float old = previous[channel] * kernel[channel * 2 * stride + stride + offset];
            output[offset * channels + channel] = old + current;
            if (!isfinite(output[offset * channels + channel]))
                return 0;
        }
    }
    return 1;
}
static int vector(const PhononTensorView *view, float *output) {
    return tensor_shape(view, 1, 512, 0, 0) && phonon_bf16_to_float(view->data, 512, output);
}
static void head_order(const float *input, float *output) {
    for (size_t i = 0; i < 32; i++) {
        output[i] = input[2 * i];
        output[32 + i] = input[2 * i + 1];
    }
}
/* One latent frame is a bounded sixteen-query batch. Insert keys query by query
 * so later writes cannot evict history required by the frame's earlier queries.
 * Cache stores original interleaved RoPE; only dot-product scratch is permuted. */
static int attention(PhononMimiState *state, size_t layer) {
#pragma STDC FP_CONTRACT OFF
    for (size_t row = 0; row < 16; row++) {
        size_t position = state->position + row, slot = position % 250;
        float *qkv = state->qkv[row];
        if (!phonon_rope_interleaved(qkv, 512, 8, position, state->model->config.rope_theta) ||
            !phonon_rope_interleaved(qkv + 512, 512, 8, position, state->model->config.rope_theta))
            return 0;
        memcpy(state->keys[layer][slot], qkv + 512, 512 * sizeof(float));
        memcpy(state->values[layer][slot], qkv + 1024, 512 * sizeof(float));
        size_t count = position < 249 ? position + 1 : 250;
        size_t first = position + 1 - count;
        for (size_t head = 0; head < 8; head++) {
            float query[64], scores[250], destination[64];
            head_order(qkv + head * 64, query);
            for (size_t key = 0; key < count; key++) {
                size_t index = (first + key) % 250;
                head_order(state->keys[layer][index] + head * 64, state->ordered_keys[key]);
                memcpy(state->ordered_values[key], state->values[layer][index] + head * 64,
                       64 * sizeof(float));
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, (int)count, 64, 1.0f, query, 64,
                        &state->ordered_keys[0][0], 64, 0.0f, scores, (int)count);
            float maximum = -INFINITY;
            for (size_t key = 0; key < count; key++) {
                scores[key] *= 0.125f;
                if (!isfinite(scores[key]))
                    return 0;
                if (scores[key] > maximum)
                    maximum = scores[key];
            }
            float lanes[4] = {0};
            size_t end = count - count % 4;
            for (size_t key = 0; key < end; key += 4) {
                for (size_t lane = 0; lane < 4; lane++)
                    scores[key + lane] -= maximum;
                phonon_math_exp4(scores + key);
                for (size_t lane = 0; lane < 4; lane++)
                    lanes[lane] += scores[key + lane];
            }
            float total = (lanes[0] + lanes[2]) + (lanes[1] + lanes[3]);
            for (size_t key = end; key < count; key++) {
                scores[key] = expf(scores[key] - maximum);
                total += scores[key];
            }
            if (!(total > 0) || !isfinite(total))
                return 0;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, 64, (int)count, 1.0f, scores,
                        (int)count, &state->ordered_values[0][0], 64, 0.0f, destination, 64);
            float inverse = 1.0f / total;
            for (size_t i = 0; i < 64; i++)
                state->attended[row][head * 64 + i] = destination[i] * inverse;
        }
    }
    return 1;
}
static int layer_forward(PhononMimiState *state, size_t layer) {
#pragma STDC FP_CONTRACT OFF
    const PhononTensorView *w = state->model->layers[layer];
    float norm_w[512], norm_b[512], scale[512];
    if (!vector(w + 4, norm_w) || !vector(w + 5, norm_b))
        return 0;
    for (size_t row = 0; row < 16; row++)
        if (!phonon_layer_norm(state->hidden[row], norm_w, norm_b, 512, 1e-5f,
                               state->normalized[row]))
            return 0;
    for (size_t part = 0; part < 3; part++) {
        PhononTensorView view = w[0];
        view.shape[0] = 512;
        view.byte_count /= 3;
        view.data = (const unsigned char *)view.data + part * view.byte_count;
        if (view.linear_data)
            view.linear_data += part * 512 * 512;
        if (!phonon_linear_batch(&view, &state->normalized[0][0], 512, 512, NULL,
                                 &state->projected[0][0], 16))
            return 0;
        for (size_t row = 0; row < 16; row++)
            memcpy(state->qkv[row] + part * 512, state->projected[row], 512 * sizeof(float));
    }
    if (!attention(state, layer) ||
        !phonon_linear_batch(w + 1, &state->attended[0][0], 512, 512, NULL, &state->projected[0][0],
                             16) ||
        !vector(w + 8, scale))
        return 0;
    for (size_t row = 0; row < 16; row++)
        for (size_t i = 0; i < 512; i++)
            state->hidden[row][i] += scale[i] * state->projected[row][i];
    if (!vector(w + 6, norm_w) || !vector(w + 7, norm_b))
        return 0;
    for (size_t row = 0; row < 16; row++)
        if (!phonon_layer_norm(state->hidden[row], norm_w, norm_b, 512, 1e-5f,
                               state->normalized[row]))
            return 0;
    if (!phonon_linear_batch(w + 2, &state->normalized[0][0], 2048, 512, NULL, &state->mlp[0][0],
                             16) ||
        !phonon_erf_gelu(&state->mlp[0][0], 16 * 2048) ||
        !phonon_linear_batch(w + 3, &state->mlp[0][0], 512, 2048, NULL, &state->projected[0][0],
                             16) ||
        !vector(w + 9, scale))
        return 0;
    for (size_t row = 0; row < 16; row++)
        for (size_t i = 0; i < 512; i++) {
            state->hidden[row][i] += scale[i] * state->projected[row][i];
            if (!isfinite(state->hidden[row][i]))
                return 0;
        }
    return 1;
}
int phonon_mimi_process(PhononMimiState *state, const float *latents, size_t frames,
                        float *features, size_t capacity) {
    if (!state || state->failed || !latents || !features || !frames ||
        frames > PHONON_MIMI_MAX_FRAMES || capacity < frames * 16 ||
        state->position / 16 > PHONON_MIMI_MAX_FRAMES - frames)
        return 0;
    size_t input_bytes = frames * 32 * sizeof(float),
           output_bytes = frames * 16 * 512 * sizeof(float);
    uintptr_t in = (uintptr_t)latents, out = (uintptr_t)features;
    if (in > UINTPTR_MAX - input_bytes || out > UINTPTR_MAX - output_bytes ||
        (in < out + output_bytes && out < in + input_bytes))
        return 0;
    for (size_t i = 0; i < frames * 32; i++)
        if (!isfinite(latents[i]))
            return 0;
    for (size_t frame = 0; frame < frames; frame++) {
        PhononTensorView projection = state->model->projection;
        projection.rank = 2;
        float projected[512];
        if (!phonon_linear(&projection, latents + frame * 32, 512, 32, NULL, projected) ||
            !phonon_mimi_upsample_frame(&state->model->upsample, projected, state->previous, 512,
                                        state->model->stride, &state->hidden[0][0]) ||
            !layer_forward(state, 0) || !layer_forward(state, 1)) {
            state->failed = 1;
            return 0;
        }
        memcpy(state->previous, projected, sizeof(projected));
        memcpy(features + frame * 16 * 512, state->hidden, sizeof(state->hidden));
        state->position += 16;
    }
    return 1;
}

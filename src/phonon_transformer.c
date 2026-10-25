#include "phonon_transformer.h"
#include "phonon_q8.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int phonon_kv_cache_init(PhononKvCache *cache, size_t capacity) {
    if (!cache || !capacity ||
        capacity > SIZE_MAX / (PHONON_FLOW_WIDTH * PHONON_FLOW_LAYERS * sizeof(float)))
        return 0;
    memset(cache, 0, sizeof(*cache));
    size_t count = capacity * PHONON_FLOW_WIDTH * PHONON_FLOW_LAYERS;
    cache->keys = calloc(count, sizeof(float));
    cache->values = calloc(count, sizeof(float));
    if (!cache->keys || !cache->values) {
        phonon_kv_cache_destroy(cache);
        return 0;
    }
    cache->capacity = capacity;
    return 1;
}

void phonon_kv_cache_reset(PhononKvCache *cache) {
    if (cache)
        cache->length = 0;
}

void phonon_kv_cache_destroy(PhononKvCache *cache) {
    if (!cache)
        return;
    free(cache->keys);
    free(cache->values);
    memset(cache, 0, sizeof(*cache));
}

int phonon_kv_cache_append(PhononKvCache *cache, size_t layer, size_t position, const float *key,
                           const float *value, size_t width) {
    if (!cache || !cache->keys || !cache->values || !key || !value || width != PHONON_FLOW_WIDTH ||
        layer >= PHONON_FLOW_LAYERS || position >= cache->capacity)
        return 0;
    for (size_t i = 0; i < width; ++i)
        if (!isfinite(key[i]) || !isfinite(value[i]))
            return 0;
    size_t offset = (layer * cache->capacity + position) * width;
    memcpy(cache->keys + offset, key, width * sizeof(float));
    memcpy(cache->values + offset, value, width * sizeof(float));
    return 1;
}

int phonon_kv_cache_commit_position(PhononKvCache *cache, size_t position) {
    if (!cache || !cache->keys || position != cache->length || position >= cache->capacity)
        return 0;
    cache->length++;
    return 1;
}

int phonon_rms_norm(const float *input, const float *weight, size_t width, float epsilon,
                    float *output) {
    if (!input || !weight || !output || !width || !(epsilon > 0.0f) || !isfinite(epsilon))
        return 0;
    double squares = 0.0;
    for (size_t i = 0; i < width; ++i) {
        if (!isfinite(input[i]) || !isfinite(weight[i]))
            return 0;
        squares += (double)input[i] * input[i];
    }
    float scale = 1.0f / sqrtf((float)(squares / (double)width) + epsilon);
    if (!isfinite(scale))
        return 0;
    for (size_t i = 0; i < width; ++i) {
        output[i] = input[i] * scale * weight[i];
        if (!isfinite(output[i]))
            return 0;
    }
    return 1;
}

int phonon_layer_norm(const float *input, const float *weight, const float *bias, size_t width,
                      float epsilon, float *output) {
    if (!input || !weight || !bias || !output || !width || !(epsilon > 0.0f) || !isfinite(epsilon))
        return 0;
    double sum = 0.0;
    for (size_t i = 0; i < width; ++i) {
        if (!isfinite(input[i]) || !isfinite(weight[i]) || !isfinite(bias[i]))
            return 0;
        sum += input[i];
    }
    double mean = sum / (double)width, squares = 0.0;
    for (size_t i = 0; i < width; ++i) {
        double centered = (double)input[i] - mean;
        squares += centered * centered;
    }
    float scale = 1.0f / sqrtf((float)(squares / (double)width) + epsilon);
    for (size_t i = 0; i < width; ++i) {
        output[i] = ((float)((double)input[i] - mean) * scale) * weight[i] + bias[i];
        if (!isfinite(output[i]))
            return 0;
    }
    return 1;
}

int phonon_erf_gelu(float *values, size_t count) {
    if (!values && count)
        return 0;
    for (size_t i = 0; i < count; ++i) {
        if (!isfinite(values[i]))
            return 0;
        values[i] = 0.5f * values[i] * (1.0f + erff(values[i] * 0.7071067811865475244f));
        if (!isfinite(values[i]))
            return 0;
    }
    return 1;
}

int phonon_rope_interleaved(float *values, size_t width, size_t heads, size_t position,
                            float theta) {
    if (!values || !width || !heads || width % heads || (width / heads) % 2 || !(theta > 0.0f) ||
        !isfinite(theta))
        return 0;
    size_t head_width = width / heads;
    for (size_t h = 0; h < heads; ++h) {
        float *head = values + h * head_width;
        for (size_t i = 0; i < head_width / 2; ++i) {
            float a = head[2 * i], b = head[2 * i + 1];
            if (!isfinite(a) || !isfinite(b))
                return 0;
            double freq = 1.0 / pow((double)theta, (double)(2 * i) / (double)head_width);
            double angle = (double)position * freq;
            float c = (float)cos(angle), s = (float)sin(angle);
            if (!isfinite(c) || !isfinite(s))
                return 0;
            head[2 * i] = a * c - b * s;
            head[2 * i + 1] = a * s + b * c;
            if (!isfinite(head[2 * i]) || !isfinite(head[2 * i + 1]))
                return 0;
        }
    }
    return 1;
}

int phonon_linear(const PhononTensorView *weight, const float *input, size_t rows, size_t columns,
                  const float *bias, float *output) {
    if (!weight || !weight->data || !input || !output || !rows || !columns ||
        rows > SIZE_MAX / columns || weight->rank != 2 || weight->shape[0] != rows ||
        weight->shape[1] != columns || weight->byte_count == 0)
        return 0;
    for (size_t i = 0; i < columns; ++i)
        if (!isfinite(input[i]))
            return 0;
    if (weight->dtype == PHONON_PACK_Q8_0) {
        size_t elements = rows * columns;
        if (columns % 32 || elements / 32 > SIZE_MAX / sizeof(PhononQ8Block) ||
            weight->byte_count != elements / 32 * sizeof(PhononQ8Block) ||
            !phonon_q8_matvec((const PhononQ8Block *)weight->data, input, rows, columns, output))
            return 0;
    } else if (weight->dtype == PHONON_PACK_BF16) {
        size_t elements = rows * columns;
        if (elements > SIZE_MAX / sizeof(uint16_t) ||
            weight->byte_count != elements * sizeof(uint16_t))
            return 0;
        uint16_t *source = (uint16_t *)weight->data;
        for (size_t r = 0; r < rows; ++r) {
            double sum = 0.0;
            for (size_t c = 0; c < columns; ++c) {
                uint32_t bits = (uint32_t)source[r * columns + c] << 16;
                float w;
                memcpy(&w, &bits, sizeof(w));
                sum += (double)w * input[c];
            }
            output[r] = (float)sum;
            if (!isfinite(output[r]))
                return 0;
        }
    } else
        return 0;
    if (bias)
        for (size_t r = 0; r < rows; ++r) {
            if (!isfinite(bias[r]) || !isfinite(output[r] + bias[r]))
                return 0;
            output[r] += bias[r];
        }
    return 1;
}

static int vector_tensor(const PhononTensorView *view, float output[PHONON_FLOW_WIDTH]) {
    if (!view || !view->data || view->rank != 1 || view->shape[0] != PHONON_FLOW_WIDTH ||
        view->dtype != PHONON_PACK_BF16 || view->byte_count != PHONON_FLOW_WIDTH * sizeof(uint16_t))
        return 0;
    return phonon_bf16_to_float((const uint16_t *)view->data, PHONON_FLOW_WIDTH, output);
}

int phonon_flow_layer_step(const PhononFlowLayerWeights *weights, PhononKvCache *cache,
                           size_t layer, size_t position, const float input[PHONON_FLOW_WIDTH],
                           float theta, float output[PHONON_FLOW_WIDTH]) {
    if (!weights || !cache || !input || !output || layer >= PHONON_FLOW_LAYERS ||
        position >= cache->capacity || position >= 8192 || cache->length != position)
        return 0;
    float n1w[PHONON_FLOW_WIDTH], n1b[PHONON_FLOW_WIDTH], n2w[PHONON_FLOW_WIDTH],
        n2b[PHONON_FLOW_WIDTH];
    float normalized[PHONON_FLOW_WIDTH], residual[PHONON_FLOW_WIDTH],
        projected[PHONON_FLOW_WIDTH * 3];
    float attended[PHONON_FLOW_WIDTH], mlp[2048];
    if (!vector_tensor(weights->norm1_weight, n1w) || !vector_tensor(weights->norm1_bias, n1b) ||
        !vector_tensor(weights->norm2_weight, n2w) || !vector_tensor(weights->norm2_bias, n2b) ||
        !phonon_layer_norm(input, n1w, n1b, PHONON_FLOW_WIDTH, 1e-5f, normalized) ||
        !phonon_linear(weights->qkv, normalized, PHONON_FLOW_WIDTH * 3, PHONON_FLOW_WIDTH, NULL,
                       projected))
        return 0;
    float *q = projected, *k = projected + PHONON_FLOW_WIDTH,
          *v = projected + 2 * PHONON_FLOW_WIDTH;
    if (!phonon_rope_interleaved(q, PHONON_FLOW_WIDTH, PHONON_FLOW_HEADS, position, theta) ||
        !phonon_rope_interleaved(k, PHONON_FLOW_WIDTH, PHONON_FLOW_HEADS, position, theta) ||
        !phonon_kv_cache_append(cache, layer, position, k, v, PHONON_FLOW_WIDTH))
        return 0;
    for (size_t h = 0; h < PHONON_FLOW_HEADS; ++h) {
        float scores[8192];
        float maximum = -INFINITY;
        for (size_t t = 0; t <= position; ++t) {
            double dot = 0.0;
            size_t off =
                (layer * cache->capacity + t) * PHONON_FLOW_WIDTH + h * PHONON_FLOW_HEAD_WIDTH;
            for (size_t d = 0; d < PHONON_FLOW_HEAD_WIDTH; ++d)
                dot += (double)q[h * PHONON_FLOW_HEAD_WIDTH + d] * cache->keys[off + d];
            scores[t] = (float)(dot / sqrt((double)PHONON_FLOW_HEAD_WIDTH));
            if (!isfinite(scores[t]))
                return 0;
            if (scores[t] > maximum)
                maximum = scores[t];
        }
        double denom = 0.0;
        for (size_t t = 0; t <= position; ++t) {
            scores[t] = expf(scores[t] - maximum);
            denom += scores[t];
        }
        if (!(denom > 0.0) || !isfinite(denom))
            return 0;
        for (size_t d = 0; d < PHONON_FLOW_HEAD_WIDTH; ++d) {
            double sum = 0.0;
            for (size_t t = 0; t <= position; ++t) {
                size_t off =
                    (layer * cache->capacity + t) * PHONON_FLOW_WIDTH + h * PHONON_FLOW_HEAD_WIDTH;
                sum += (scores[t] / denom) * cache->values[off + d];
            }
            attended[h * PHONON_FLOW_HEAD_WIDTH + d] = (float)sum;
        }
    }
    float attention[PHONON_FLOW_WIDTH];
    if (!phonon_linear(weights->attention_out, attended, PHONON_FLOW_WIDTH, PHONON_FLOW_WIDTH, NULL,
                       attention))
        return 0;
    for (size_t i = 0; i < PHONON_FLOW_WIDTH; ++i)
        residual[i] = input[i] + attention[i];
    if (!phonon_layer_norm(residual, n2w, n2b, PHONON_FLOW_WIDTH, 1e-5f, normalized) ||
        !phonon_linear(weights->mlp_in, normalized, 2048, PHONON_FLOW_WIDTH, NULL, mlp) ||
        !phonon_erf_gelu(mlp, 2048) ||
        !phonon_linear(weights->mlp_out, mlp, PHONON_FLOW_WIDTH, 2048, NULL, output))
        return 0;
    for (size_t i = 0; i < PHONON_FLOW_WIDTH; ++i) {
        output[i] += residual[i];
        if (!isfinite(output[i]))
            return 0;
    }
    return 1;
}

int phonon_flow_stack_step(const PhononFlowLayerWeights weights[PHONON_FLOW_LAYERS],
                           PhononKvCache *cache, size_t position,
                           const float input[PHONON_FLOW_WIDTH], float theta,
                           float output[PHONON_FLOW_WIDTH]) {
    if (!weights || !cache || !input || !output || position != cache->length ||
        position >= cache->capacity)
        return 0;
    float a[PHONON_FLOW_WIDTH], b[PHONON_FLOW_WIDTH];
    memcpy(a, input, sizeof(a));
    for (size_t layer = 0; layer < PHONON_FLOW_LAYERS; ++layer) {
        if (!phonon_flow_layer_step(&weights[layer], cache, layer, position, a, theta, b))
            return 0;
        memcpy(a, b, sizeof(a));
    }
    if (!phonon_kv_cache_commit_position(cache, position))
        return 0;
    memcpy(output, a, sizeof(a));
    return 1;
}

int phonon_voice_project(const float *voice_latents, size_t frames,
                         const PhononTensorView *projection, float *projected) {
    if (!voice_latents || !projection || !projected || frames == 0 || frames > 8192)
        return 0;
    for (size_t i = 0; i < frames; ++i)
        if (!phonon_linear(projection, voice_latents + i * PHONON_FLOW_WIDTH, PHONON_FLOW_WIDTH,
                           PHONON_FLOW_WIDTH, NULL, projected + i * PHONON_FLOW_WIDTH))
            return 0;
    return 1;
}

int phonon_flow_condition(const PhononTensorView *speaker_embeddings,
                          const PhononTensorView *speaker_projection,
                          const PhononTensorView *duration_projection, size_t speaker_count,
                          float duration_delta, float output[PHONON_FLOW_WIDTH]) {
    if (!speaker_embeddings || !speaker_embeddings->data || !speaker_projection ||
        !duration_projection || !output || speaker_embeddings->rank != 2 ||
        speaker_embeddings->shape[1] != 16 || speaker_count >= speaker_embeddings->shape[0] ||
        !isfinite(duration_delta))
        return 0;
    float embedded_speaker[16], speaker[PHONON_FLOW_WIDTH], duration[128],
        duration_out[PHONON_FLOW_WIDTH];
    if (speaker_embeddings->dtype != PHONON_PACK_BF16 ||
        speaker_embeddings->byte_count != speaker_embeddings->shape[0] * 16 * sizeof(uint16_t))
        return 0;
    uint16_t *table = (uint16_t *)speaker_embeddings->data;
    uint32_t bits;
    for (size_t i = 0; i < 16; ++i) {
        bits = (uint32_t)table[speaker_count * 16 + i] << 16;
        memcpy(&embedded_speaker[i], &bits, 4);
    }
    if (!phonon_linear(speaker_projection, embedded_speaker, PHONON_FLOW_WIDTH, 16, NULL, speaker))
        return 0;
    const double scale = 1000.0 * (double)duration_delta;
    for (size_t i = 0; i < 64; ++i) {
        double phase = scale / pow(10000.0, (double)i / 63.0);
        duration[i] = (float)cos(phase);
        duration[64 + i] = (float)sin(phase);
    }
    if (!phonon_linear(duration_projection, duration, PHONON_FLOW_WIDTH, 128, NULL, duration_out))
        return 0;
    for (size_t i = 0; i < PHONON_FLOW_WIDTH; ++i) {
        output[i] = speaker[i] + duration_out[i];
        if (!isfinite(output[i]))
            return 0;
    }
    return 1;
}

int phonon_flow_prepare_prefix(const int32_t *ids, size_t tokens, const float *token_embeddings,
                               size_t prompt_id, size_t bos_id, const float *projected_voice,
                               size_t voice_frames, const float *bos_latent_projection,
                               float *prefix_embeddings) {
    if (!ids || !tokens || !token_embeddings || !projected_voice || !voice_frames ||
        !bos_latent_projection || !prefix_embeddings || tokens > 8192 || voice_frames > tokens ||
        prompt_id > INT32_MAX || bos_id > INT32_MAX)
        return 0;
    size_t prompts = 0, boses = 0;
    for (size_t i = 0; i < tokens; ++i) {
        if (ids[i] == (int32_t)prompt_id)
            prompts++;
        if (ids[i] == (int32_t)bos_id)
            boses++;
    }
    if (prompts != voice_frames || boses != 1)
        return 0;
    size_t voice = 0;
    for (size_t i = 0; i < tokens; ++i) {
        float *dst = prefix_embeddings + i * PHONON_FLOW_WIDTH;
        if (ids[i] == (int32_t)prompt_id)
            memcpy(dst, projected_voice + voice++ * PHONON_FLOW_WIDTH,
                   PHONON_FLOW_WIDTH * sizeof(float));
        else if (ids[i] == (int32_t)bos_id)
            memcpy(dst, bos_latent_projection, PHONON_FLOW_WIDTH * sizeof(float));
        else
            memcpy(dst, token_embeddings + i * PHONON_FLOW_WIDTH,
                   PHONON_FLOW_WIDTH * sizeof(float));
        for (size_t j = 0; j < PHONON_FLOW_WIDTH; ++j)
            if (!isfinite(dst[j]))
                return 0;
    }
    return 1;
}

static int packed_tensor(const PhononPack *pack, const char *name, PhononTensorView *view) {
    return phonon_pack_tensor(pack, name, view);
}

int phonon_flow_load_layers(const PhononPack *pack, PhononTensorView storage[PHONON_FLOW_LAYERS][8],
                            PhononFlowLayerWeights weights[PHONON_FLOW_LAYERS]) {
    if (!pack || !storage || !weights)
        return 0;
    for (size_t i = 0; i < PHONON_FLOW_LAYERS; ++i) {
        char name[160];
#define LOAD_FIELD(field, slot, suffix)                                                            \
    do {                                                                                           \
        (void)snprintf(name, sizeof(name), "flow_lm.transformer.layers.%zu.%s", i, suffix);        \
        if (!packed_tensor(pack, name, &storage[i][slot]))                                         \
            return 0;                                                                              \
        weights[i].field = &storage[i][slot];                                                      \
    } while (0)
        LOAD_FIELD(qkv, 0, "self_attn.in_proj.weight");
        LOAD_FIELD(attention_out, 1, "self_attn.out_proj.weight");
        LOAD_FIELD(mlp_in, 2, "linear1.weight");
        LOAD_FIELD(mlp_out, 3, "linear2.weight");
        LOAD_FIELD(norm1_weight, 4, "norm1.weight");
        LOAD_FIELD(norm1_bias, 5, "norm1.bias");
        LOAD_FIELD(norm2_weight, 6, "norm2.weight");
        LOAD_FIELD(norm2_bias, 7, "norm2.bias");
#undef LOAD_FIELD
    }
    return 1;
}

int phonon_flow_embed_tokens(const PhononPack *pack, const int32_t *ids, size_t count,
                             float *embeddings) {
    PhononTensorView table;
    if (!pack || !ids || !count || !embeddings || count > 8192 ||
        !phonon_pack_tensor(
            pack, "flow_lm.condition_provider.conditioners.transcript_in_segment.embed.weight",
            &table) ||
        table.rank != 2 || table.shape[1] != PHONON_FLOW_WIDTH || table.dtype != PHONON_PACK_BF16 ||
        table.byte_count != table.shape[0] * PHONON_FLOW_WIDTH * sizeof(uint16_t))
        return 0;
    const uint16_t *values = table.data;
    for (size_t i = 0; i < count; ++i) {
        if (ids[i] < 0 || (uint64_t)ids[i] >= table.shape[0] ||
            !phonon_bf16_to_float(values + (size_t)ids[i] * PHONON_FLOW_WIDTH, PHONON_FLOW_WIDTH,
                                  embeddings + i * PHONON_FLOW_WIDTH))
            return 0;
    }
    return 1;
}

int phonon_flow_embed_bos(const PhononPack *pack, const float condition[PHONON_FLOW_WIDTH],
                          float output[PHONON_FLOW_WIDTH]) {
    PhononTensorView bos, projection;
    float latent[32];
    if (!pack || !condition || !output || !phonon_pack_tensor(pack, "flow_lm.bos_emb", &bos) ||
        bos.rank != 1 || bos.shape[0] != 32 || bos.dtype != PHONON_PACK_BF16 ||
        !phonon_pack_tensor(pack, "flow_lm.input_linear.weight", &projection) ||
        !phonon_bf16_to_float((const uint16_t *)bos.data, 32, latent) ||
        !phonon_linear(&projection, latent, PHONON_FLOW_WIDTH, 32, NULL, output))
        return 0;
    for (size_t i = 0; i < PHONON_FLOW_WIDTH; ++i) {
        output[i] += condition[i];
        if (!isfinite(output[i]))
            return 0;
    }
    return 1;
}

int phonon_flow_embed_latent(const PhononPack *pack, const float latent[32],
                             const float condition[PHONON_FLOW_WIDTH],
                             float output[PHONON_FLOW_WIDTH]) {
    PhononTensorView projection;
    if (!pack || !latent || !condition || !output ||
        !phonon_pack_tensor(pack, "flow_lm.input_linear.weight", &projection) ||
        !phonon_linear(&projection, latent, PHONON_FLOW_WIDTH, 32, NULL, output))
        return 0;
    for (size_t i = 0; i < PHONON_FLOW_WIDTH; ++i) {
        output[i] += condition[i];
        if (!isfinite(output[i]))
            return 0;
    }
    return 1;
}

int phonon_flow_normalize_latent(const PhononPack *pack, const float denormalized[32],
                                 float normalized[32]) {
    PhononTensorView mean_view, std_view;
    uint16_t mean_bits[32], std_bits[32];
    float mean[32], std[32];
    if (!pack || !denormalized || !normalized ||
        !phonon_pack_tensor(pack, "flow_lm.emb_mean", &mean_view) ||
        !phonon_pack_tensor(pack, "flow_lm.emb_std", &std_view) ||
        mean_view.dtype != PHONON_PACK_BF16 || std_view.dtype != PHONON_PACK_BF16 ||
        mean_view.rank != 1 || std_view.rank != 1 || mean_view.shape[0] != 32 ||
        std_view.shape[0] != 32 || mean_view.byte_count != sizeof(mean_bits) ||
        std_view.byte_count != sizeof(std_bits) || !mean_view.data || !std_view.data)
        return 0;
    memcpy(mean_bits, mean_view.data, sizeof(mean_bits));
    memcpy(std_bits, std_view.data, sizeof(std_bits));
    if (!phonon_bf16_to_float(mean_bits, 32, mean) || !phonon_bf16_to_float(std_bits, 32, std))
        return 0;
    for (size_t i = 0; i < 32; ++i) {
        if (!isfinite(denormalized[i]) || !isfinite(mean[i]) || !isfinite(std[i]) ||
            !(std[i] > 0.0f))
            return 0;
        normalized[i] = (denormalized[i] - mean[i]) / std[i];
        if (!isfinite(normalized[i]))
            return 0;
    }
    return 1;
}

int phonon_flow_output_norm(const PhononPack *pack, float values[PHONON_FLOW_WIDTH]) {
    PhononTensorView weight, bias;
    float w[PHONON_FLOW_WIDTH], b[PHONON_FLOW_WIDTH], result[PHONON_FLOW_WIDTH];
    if (!pack || !values || !phonon_pack_tensor(pack, "flow_lm.out_norm.weight", &weight) ||
        !phonon_pack_tensor(pack, "flow_lm.out_norm.bias", &bias) || !vector_tensor(&weight, w) ||
        !vector_tensor(&bias, b) ||
        !phonon_layer_norm(values, w, b, PHONON_FLOW_WIDTH, 1e-5f, result))
        return 0;
    memcpy(values, result, sizeof(result));
    return 1;
}

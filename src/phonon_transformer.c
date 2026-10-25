#include "phonon_transformer.h"
#include "phonon_math.h"
#include "phonon_q8.h"

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <limits.h>
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

/* PyTorch 2.7.1 moments_utils.h RowwiseMoments<float>: 16-vector
 * Welford chunks, a three-level cascade, then a scalar lane merge.
 * NEON vector arithmetic is unfused; the scalar mean merge uses FMA. */
static void merge_moments(size_t add, float a1, float a2, size_t *n, float *m1, float *m2) {
#pragma STDC FP_CONTRACT OFF
    size_t total = *n + add;
    float c = total ? (float)add / (float)total : 0;
    float delta = a1 - *m1;
    *m1 += c * delta;
    *m2 += a2 + delta * delta * c * (float)*n;
    *n = total;
}
static void moments_512(const float *x, float *out) {
#pragma STDC FP_CONTRACT OFF
    float s1[3][4] = {{0}}, s2[3][4] = {{0}};
    size_t count[3] = {0};
    for (size_t chunk = 0; chunk < 8; chunk++) {
        float m1[4] = {0}, m2[4] = {0};
        for (size_t j = 0; j < 16; j++)
            for (size_t lane = 0; lane < 4; lane++) {
                float v = x[chunk * 64 + j * 4 + lane], delta = v - m1[lane];
                m1[lane] += delta * (1.0f / (float)(j + 1));
                m2[lane] += delta * (v - m1[lane]);
            }
        size_t n0 = count[0];
        for (size_t lane = 0; lane < 4; lane++) {
            size_t n = n0;
            merge_moments(16, m1[lane], m2[lane], &n, &s1[0][lane], &s2[0][lane]);
        }
        count[0] += 16;
        size_t mask = chunk + 1;
        for (size_t level = 1; level < 3 && (mask & 1) == 0; level++, mask >>= 1) {
            for (size_t lane = 0; lane < 4; lane++) {
                size_t n = count[level];
                merge_moments(count[level - 1], s1[level - 1][lane], s2[level - 1][lane], &n,
                              &s1[level][lane], &s2[level][lane]);
                s1[level - 1][lane] = s2[level - 1][lane] = 0;
            }
            count[level] += count[level - 1];
            count[level - 1] = 0;
        }
    }
    for (size_t level = 1; level < 3; level++) {
        for (size_t lane = 0; lane < 4; lane++) {
            size_t n = count[0];
            merge_moments(count[level], s1[level][lane], s2[level][lane], &n, &s1[0][lane],
                          &s2[0][lane]);
        }
        count[0] += count[level];
    }
    float m1 = 0, m2 = 0;
    size_t n = 0;
    for (size_t lane = 0; lane < 4; lane++) {
        size_t total = n + 128;
        float c = 128.0f / (float)total;
        float delta = s1[0][lane] - m1;
        m1 = fmaf(c, delta, m1);
        m2 += s2[0][lane] + delta * delta * c * (float)n;
        n = total;
    }
    out[0] = m1;
    out[1] = m2 / 512.0f;
}

int phonon_layer_norm(const float *input, const float *weight, const float *bias, size_t width,
                      float epsilon, float *output) {
#pragma STDC FP_CONTRACT OFF
    if (!input || !weight || !bias || !output || !width || !(epsilon > 0.0f) || !isfinite(epsilon))
        return 0;
    double sum = 0.0;
    for (size_t i = 0; i < width; ++i) {
        if (!isfinite(input[i]) || !isfinite(weight[i]) || !isfinite(bias[i]))
            return 0;
        sum += input[i];
    }
    double mean = sum / (double)width, squares = 0.0;
    float variance;
    if (width == 512) {
        float moments[2];
        moments_512(input, moments);
        mean = moments[0];
        variance = moments[1];
    } else {
        for (size_t i = 0; i < width; ++i) {
            double centered = (double)input[i] - mean;
            squares += centered * centered;
        }
        variance = (float)(squares / (double)width);
    }
    float scale = 1.0f / sqrtf(variance + epsilon);
    for (size_t i = 0; i < width; ++i) {
        output[i] = ((float)((double)input[i] - mean) * scale) * weight[i] + bias[i];
        if (!isfinite(output[i]))
            return 0;
    }
    return 1;
}

/* Adapted from PyTorch v2.7.1 vec128_float_neon.h::erf(). Its NEON
 * GELU uses this polynomial with explicit FMA, not libc erff. See
 * LICENSES/pytorch-BSD.txt and reference/phonon/operator-provenance.json. */
static float neon_erf(float x) {
#pragma STDC FP_CONTRACT OFF
    float t = 1.0f / fmaf(0.3275911f, fabsf(x), 1.0f);
    float r = fmaf(1.061405429f, t, -1.453152027f);
    r = fmaf(r, t, 1.421413741f);
    r = fmaf(r, t, -0.284496736f);
    r = fmaf(r, t, 0.254829592f);
    float square = x * x;
    float negative_exp = -expf(-square);
    return copysignf(fmaf(t * negative_exp, r, 1.0f), x);
}

int phonon_erf_gelu(float *values, size_t count) {
#pragma STDC FP_CONTRACT OFF
    if (!values && count)
        return 0;
    for (size_t i = 0; i < count; ++i) {
        if (!isfinite(values[i]))
            return 0;
        float scaled = values[i] * 0.7071067811865475244f;
        float erf_value = i < count - count % 4 ? neon_erf(scaled) : erff(scaled);
        values[i] = 0.5f * values[i] * (1.0f + erf_value);
        if (!isfinite(values[i]))
            return 0;
    }
    return 1;
}

int phonon_rope_interleaved(float *values, size_t width, size_t heads, size_t position,
                            float theta) {
#pragma STDC FP_CONTRACT OFF
    if (!values || !width || !heads || width % heads || (width / heads) % 2 || !(theta > 0.0f) ||
        !isfinite(theta))
        return 0;
    size_t head_width = width / heads;
    for (size_t group = 0; group < head_width / 2; group += 4) {
        size_t count = head_width / 2 - group < 4 ? head_width / 2 - group : 4;
        float phases[4], cosines[4], sines[4];
        for (size_t i = 0; i < count; i++) {
            /* Pinned HF RoPE builds inverse frequencies and phases in float32. */
            float exponent = (float)(2 * (group + i)) / (float)head_width;
            float freq = 1.0f / powf(theta, exponent);
            phases[i] = (float)position * freq;
        }
        phonon_math_sincos(phases, count, cosines, sines);
        for (size_t h = 0; h < heads; h++) {
            float *head = values + h * head_width;
            for (size_t i = 0; i < count; i++) {
                size_t index = 2 * (group + i);
                float a = head[index], b = head[index + 1];
                float c = cosines[i], sine = sines[i];
                if (!isfinite(a) || !isfinite(b) || !isfinite(c) || !isfinite(sine))
                    return 0;
                head[index] = a * c - b * sine;
                head[index + 1] = a * sine + b * c;
                if (!isfinite(head[index]) || !isfinite(head[index + 1]))
                    return 0;
            }
        }
    }
    return 1;
}

static float half_to_float(uint16_t bits) {
    uint32_t sign = (uint32_t)(bits & 0x8000U) << 16;
    uint32_t exponent = (bits >> 10) & 0x1fU;
    uint32_t mantissa = bits & 0x3ffU;
    uint32_t result;
    if (exponent == 0) {
        if (mantissa == 0) {
            result = sign;
        } else {
            int shift = 0;
            while ((mantissa & 0x400U) == 0) {
                mantissa <<= 1;
                shift++;
            }
            mantissa &= 0x3ffU;
            result = sign | ((uint32_t)(127 - 14 - shift) << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1fU) {
        result = sign | 0x7f800000U | (mantissa << 13);
    } else {
        result = sign | ((exponent + 112U) << 23) | (mantissa << 13);
    }
    union {
        uint32_t bits;
        float value;
    } converted = {result};
    return converted.value;
}

int phonon_linear_batch(const PhononTensorView *weight, const float *input, size_t rows,
                        size_t columns, const float *bias, float *output, size_t batch) {
    if (!weight || !weight->data || !input || !output || !rows || !columns || !batch ||
        batch > INT_MAX || batch > SIZE_MAX / rows || batch > SIZE_MAX / columns ||
        rows > SIZE_MAX / columns || weight->rank != 2 || weight->shape[0] != rows ||
        weight->shape[1] != columns || weight->byte_count == 0)
        return 0;
    if (batch * rows > SIZE_MAX / sizeof(float) || batch * columns > SIZE_MAX / sizeof(float))
        return 0;
    for (size_t i = 0; i < batch * columns; ++i)
        if (!isfinite(input[i]))
            return 0;
    size_t elements = rows * columns;
    if (rows > INT_MAX || columns > INT_MAX || elements > SIZE_MAX / sizeof(float))
        return 0;
    if (weight->dtype == PHONON_PACK_Q8_0) {
        if (columns % 32 || elements / 32 > SIZE_MAX / sizeof(PhononQ8Block) ||
            weight->byte_count != elements / 32 * sizeof(PhononQ8Block))
            return 0;
        /* Preserve the packed matvec's non-overlapping output contract. */
        uintptr_t in = (uintptr_t)input, out = (uintptr_t)output, w = (uintptr_t)weight->data;
        size_t in_bytes = batch * columns * sizeof(float), out_bytes = batch * rows * sizeof(float);
        if (in > UINTPTR_MAX - in_bytes || out > UINTPTR_MAX - out_bytes ||
            w > UINTPTR_MAX - weight->byte_count || (in < out + out_bytes && out < in + in_bytes) ||
            (w < out + out_bytes && out < w + weight->byte_count))
            return 0;
    } else if (weight->dtype != PHONON_PACK_BF16 ||
               weight->byte_count != elements * sizeof(uint16_t))
        return 0;
    /* Torch 2.7.1 Linear.cpp flattens contiguous inputs into addmm. Use
     * the same Accelerate SGEMM reduction and bias accumulation here.
     * Expanded weights are call-owned; the borrowed pack stays immutable.
     * Transient expansion is a correctness baseline, not a speed claim. */
    float *expanded = malloc(elements * sizeof(float));
    if (!expanded)
        return 0;
    if (weight->dtype == PHONON_PACK_BF16) {
        if (!phonon_bf16_to_float(weight->data, elements, expanded)) {
            free(expanded);
            return 0;
        }
    } else {
        const PhononQ8Block *blocks = weight->data;
        for (size_t block = 0; block < elements / 32; block++) {
            float scale = half_to_float(blocks[block].scale_bits);
            if (!isfinite(scale)) {
                free(expanded);
                return 0;
            }
            /* Published Q8 expansion multiplies each integer by its scale
             * before projection, rather than scaling a partial dot product. */
            for (size_t i = 0; i < 32; i++)
                expanded[block * 32 + i] = scale * (float)blocks[block].values[i];
        }
    }
    for (size_t r = 0; r < batch * rows; ++r) {
        if (bias && !isfinite(bias[r % rows])) {
            free(expanded);
            return 0;
        }
        output[r] = bias ? bias[r % rows] : 0.0f;
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)batch, (int)rows, (int)columns, 1.0f,
                input, (int)columns, expanded, (int)columns, bias ? 1.0f : 0.0f, output, (int)rows);
    free(expanded);
    for (size_t r = 0; r < batch * rows; ++r)
        if (!isfinite(output[r]))
            return 0;
    return 1;
}

int phonon_linear(const PhononTensorView *weight, const float *input, size_t rows, size_t columns,
                  const float *bias, float *output) {
    return phonon_linear_batch(weight, input, rows, columns, bias, output, 1);
}

static int vector_tensor(const PhononTensorView *view, float output[PHONON_FLOW_WIDTH]) {
    if (!view || !view->data || view->rank != 1 || view->shape[0] != PHONON_FLOW_WIDTH ||
        view->dtype != PHONON_PACK_BF16 || view->byte_count != PHONON_FLOW_WIDTH * sizeof(uint16_t))
        return 0;
    return phonon_bf16_to_float((const uint16_t *)view->data, PHONON_FLOW_WIDTH, output);
}

/* The pack/cache retain raw interleaved RoPE. HF PermuteForRope changes
 * attention reduction order to [even components, odd components]. Reorder only
 * scratch operands so GEMM follows that source order without changing storage. */
static void attention_head_order(const float *input, size_t rows, size_t stride, float *output) {
    for (size_t row = 0; row < rows; row++)
        for (size_t i = 0; i < 32; i++) {
            output[row * 64 + i] = input[row * stride + 2 * i];
            output[row * 64 + 32 + i] = input[row * stride + 2 * i + 1];
        }
}

/* Single-query CPU SDPA, PyTorch 2.7.1 FlashAttentionKernel.cpp.
 * Keep its 512-key online-softmax blocks and normalize after the E@V GEMM. */
static int flow_attention(const float query[64], const float *keys, const float *values,
                          size_t count, size_t stride, float output[64]) {
#pragma STDC FP_CONTRACT OFF
    if (!count || count > 8192 || stride < 64 || stride > INT_MAX)
        return 0;
    float maximum = -INFINITY, total = 0.0f, destination[64] = {0};
    float ordered_query[64], ordered_keys[512 * 64];
    attention_head_order(query, 1, 64, ordered_query);
    for (size_t start = 0; start < count; start += 512) {
        size_t length = count - start < 512 ? count - start : 512;
        float scores[512];
        attention_head_order(keys + start * stride, length, stride, ordered_keys);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, (int)length, 64, 1.0f,
                    ordered_query, 64, ordered_keys, 64, 0.0f, scores, (int)length);
        float next_maximum = maximum;
        for (size_t i = 0; i < length; i++) {
            scores[i] *= 0.125f;
            if (!isfinite(scores[i]))
                return 0;
            if (scores[i] > next_maximum)
                next_maximum = scores[i];
        }
        float lanes[4] = {0}, block_sum = 0.0f;
        size_t vector_end = length - length % 4;
        for (size_t i = 0; i < vector_end; i += 4) {
            for (size_t lane = 0; lane < 4; lane++)
                scores[i + lane] -= next_maximum;
            phonon_math_exp4(scores + i);
            for (size_t lane = 0; lane < 4; lane++)
                lanes[lane] += scores[i + lane];
        }
        /* functional_base.h NEON horizontal reduction: (0+2)+(1+3). */
        block_sum = (lanes[0] + lanes[2]) + (lanes[1] + lanes[3]);
        for (size_t i = vector_end; i < length; i++) {
            scores[i] = expf(scores[i] - next_maximum);
            block_sum += scores[i];
        }
        float rescale = expf(maximum - next_maximum);
        total = fmaf(rescale, total, block_sum);
        maximum = next_maximum;
        if (start)
            for (size_t i = 0; i < 64; i++)
                destination[i] *= rescale;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, 64, (int)length, 1.0f, scores,
                    (int)length, values + start * stride, (int)stride, start ? 1.0f : 0.0f,
                    destination, 64);
    }
    if (!(total > 0.0f) || !isfinite(total))
        return 0;
    float inverse = 1.0f / total;
    for (size_t i = 0; i < 64; i++) {
        output[i] = destination[i] * inverse;
        if (!isfinite(output[i]))
            return 0;
    }
    return 1;
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
        size_t offset = layer * cache->capacity * PHONON_FLOW_WIDTH + h * PHONON_FLOW_HEAD_WIDTH;
        if (!flow_attention(q + h * PHONON_FLOW_HEAD_WIDTH, cache->keys + offset,
                            cache->values + offset, position + 1, PHONON_FLOW_WIDTH,
                            attended + h * PHONON_FLOW_HEAD_WIDTH))
            return 0;
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

/* Full-prefix CPU SDPA uses source query blocking and 512-key blocks. Scores
 * occupy at most 256*512 floats, never a quadratic full-context matrix. */
static int prefill_attention(const float *projected, const float *keys, const float *values,
                             size_t length, float *scores, float *destination, float *output) {
#pragma STDC FP_CONTRACT OFF
    size_t query_block = length >= 768 ? 256 : (length >= 192 ? 64 : 32);
    float ordered_queries[256 * 64], ordered_keys[512 * 64];
    for (size_t head = 0; head < 8; head++) {
        for (size_t start = 0; start < length; start += query_block) {
            size_t rows = length - start < query_block ? length - start : query_block;
            float maxima[256], totals[256];
            attention_head_order(projected + start * 1536 + head * 64, rows, 1536, ordered_queries);
            for (size_t row = 0; row < rows; row++) {
                maxima[row] = -INFINITY;
                totals[row] = 0.0f;
            }
            for (size_t key_start = 0; key_start < start + rows; key_start += 512) {
                size_t cols = length - key_start < 512 ? length - key_start : 512;
                attention_head_order(keys + key_start * 512 + head * 64, cols, 512, ordered_keys);
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)rows, (int)cols, 64, 1.0f,
                            ordered_queries, 64, ordered_keys, 64, 0.0f, scores, (int)cols);
                for (size_t row = 0; row < rows; row++) {
                    float *weights = scores + row * cols;
                    float maximum = maxima[row];
                    for (size_t col = 0; col < cols; col++) {
                        if (key_start + col > start + row)
                            weights[col] = -INFINITY;
                        else {
                            weights[col] *= 0.125f;
                            if (!isfinite(weights[col]))
                                return 0;
                            if (weights[col] > maximum)
                                maximum = weights[col];
                        }
                    }
                    float lanes[4] = {0}, sum = 0.0f;
                    size_t vector_end = cols - cols % 4;
                    for (size_t col = 0; col < vector_end; col += 4) {
                        for (size_t lane = 0; lane < 4; lane++)
                            weights[col + lane] -= maximum;
                        phonon_math_exp4(weights + col);
                        for (size_t lane = 0; lane < 4; lane++)
                            lanes[lane] += weights[col + lane];
                    }
                    sum = (lanes[0] + lanes[2]) + (lanes[1] + lanes[3]);
                    for (size_t col = vector_end; col < cols; col++) {
                        weights[col] = expf(weights[col] - maximum);
                        sum += weights[col];
                    }
                    float rescale = expf(maxima[row] - maximum);
                    totals[row] = fmaf(rescale, totals[row], sum);
                    maxima[row] = maximum;
                    if (key_start)
                        for (size_t d = 0; d < 64; d++)
                            destination[row * 64 + d] *= rescale;
                }
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)rows, 64, (int)cols,
                            1.0f, scores, (int)cols, values + key_start * 512 + head * 64, 512,
                            key_start ? 1.0f : 0.0f, destination, 64);
            }
            for (size_t row = 0; row < rows; row++) {
                if (!(totals[row] > 0.0f) || !isfinite(totals[row]))
                    return 0;
                float inverse = 1.0f / totals[row];
                for (size_t d = 0; d < 64; d++) {
                    float value = destination[row * 64 + d] * inverse;
                    if (!isfinite(value))
                        return 0;
                    output[(start + row) * 512 + head * 64 + d] = value;
                }
            }
        }
    }
    return 1;
}

static int prefill_qkv(const PhononTensorView *qkv, const float *input, size_t length, float *split,
                       float *projected) {
    if (!qkv || !qkv->data || qkv->rank != 2 || qkv->shape[0] != 1536 || qkv->shape[1] != 512 ||
        (qkv->dtype != PHONON_PACK_BF16 && qkv->dtype != PHONON_PACK_Q8_0) || qkv->byte_count % 3)
        return 0;
    /* HF has three separate 512-output projections. Preserve those GEMM shapes;
     * the raw pack stores their rows together, in interleaved RoPE order. */
    for (size_t part = 0; part < 3; part++) {
        PhononTensorView view = *qkv;
        view.shape[0] = 512;
        view.byte_count /= 3;
        view.data = (const unsigned char *)qkv->data + part * view.byte_count;
        if (!phonon_linear_batch(&view, input, 512, 512, NULL, split, length))
            return 0;
        for (size_t row = 0; row < length; row++)
            memcpy(projected + row * 1536 + part * 512, split + row * 512, 512 * sizeof(float));
    }
    return 1;
}

int phonon_flow_stack_prefill(const PhononFlowLayerWeights weights[PHONON_FLOW_LAYERS],
                              PhononKvCache *cache, const float *input, size_t length, float theta,
                              float output[512]) {
    if (!weights || !cache || !cache->keys || !cache->values || !input || !output || !length ||
        length > 8192 || length > cache->capacity || cache->length || !isfinite(theta) ||
        theta <= 0.0f || length > SIZE_MAX / (6144 * sizeof(float)))
        return 0;
    /* All allocations belong to this prefill. Cache is committed only on success;
     * a failed prefill resets it to the empty state and leaves output untouched. */
    float *scratch = malloc(length * 6144 * sizeof(float));
    float *attention_scratch = malloc((256 * 512 + 256 * 64) * sizeof(float));
    if (!scratch || !attention_scratch) {
        free(scratch);
        free(attention_scratch);
        return 0;
    }
    float *state = scratch, *normalized = state + length * 512;
    float *residual = normalized + length * 512, *projected = residual + length * 512;
    float *mlp = projected + length * 1536, *attended = mlp + length * 2048;
    float *split = attended + length * 512;
    float *scores = attention_scratch, *destination = scores + 256 * 512;
    memcpy(state, input, length * 512 * sizeof(float));
    int success = 0;
    for (size_t layer = 0; layer < PHONON_FLOW_LAYERS; layer++) {
        float n1w[512], n1b[512], n2w[512], n2b[512];
        if (!vector_tensor(weights[layer].norm1_weight, n1w) ||
            !vector_tensor(weights[layer].norm1_bias, n1b) ||
            !vector_tensor(weights[layer].norm2_weight, n2w) ||
            !vector_tensor(weights[layer].norm2_bias, n2b))
            goto cleanup;
        for (size_t row = 0; row < length; row++)
            if (!phonon_layer_norm(state + row * 512, n1w, n1b, 512, 1e-5f, normalized + row * 512))
                goto cleanup;
        if (!prefill_qkv(weights[layer].qkv, normalized, length, split, projected))
            goto cleanup;
        size_t offset = layer * cache->capacity * 512;
        for (size_t row = 0; row < length; row++) {
            float *q = projected + row * 1536, *k = q + 512, *v = k + 512;
            if (!phonon_rope_interleaved(q, 512, 8, row, theta) ||
                !phonon_rope_interleaved(k, 512, 8, row, theta))
                goto cleanup;
            memcpy(cache->keys + offset + row * 512, k, 512 * sizeof(float));
            memcpy(cache->values + offset + row * 512, v, 512 * sizeof(float));
        }
        if (!prefill_attention(projected, cache->keys + offset, cache->values + offset, length,
                               scores, destination, attended) ||
            !phonon_linear_batch(weights[layer].attention_out, attended, 512, 512, NULL, normalized,
                                 length))
            goto cleanup;
        for (size_t i = 0; i < length * 512; i++)
            residual[i] = state[i] + normalized[i];
        for (size_t row = 0; row < length; row++)
            if (!phonon_layer_norm(residual + row * 512, n2w, n2b, 512, 1e-5f,
                                   normalized + row * 512))
                goto cleanup;
        if (!phonon_linear_batch(weights[layer].mlp_in, normalized, 2048, 512, NULL, mlp, length) ||
            !phonon_erf_gelu(mlp, length * 2048) ||
            !phonon_linear_batch(weights[layer].mlp_out, mlp, 512, 2048, NULL, state, length))
            goto cleanup;
        for (size_t i = 0; i < length * 512; i++) {
            state[i] += residual[i];
            if (!isfinite(state[i]))
                goto cleanup;
        }
    }
    memcpy(output, state + (length - 1) * 512, 512 * sizeof(float));
    cache->length = length;
    success = 1;
cleanup:
    free(scratch);
    free(attention_scratch);
    if (!success)
        phonon_kv_cache_reset(cache);
    return success;
}

int phonon_voice_project(const float *voice_latents, size_t frames,
                         const PhononTensorView *projection, float *projected) {
    if (!voice_latents || !projection || !projected || frames == 0 || frames > 8192)
        return 0;
    return phonon_linear_batch(projection, voice_latents, PHONON_FLOW_WIDTH, PHONON_FLOW_WIDTH,
                               NULL, projected, frames);
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

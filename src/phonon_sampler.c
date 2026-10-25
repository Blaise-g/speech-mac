#include "phonon_sampler.h"
#include "phonon_math.h"

#include <math.h>
#include <string.h>

#pragma STDC FP_CONTRACT OFF

static uint64_t splitmix64(uint64_t *state) {
    uint64_t value = (*state += UINT64_C(0x9e3779b97f4a7c15));
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

static uint64_t rotl(uint64_t value, int count) {
    return (value << count) | (value >> (64 - count));
}

static uint64_t next_u64(PhononNoiseSource *source) {
    uint64_t *s = source->state;
    uint64_t result = rotl(s[1] * 5, 7) * 9;
    uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return result;
}

int phonon_noise_seed(PhononNoiseSource *source, uint64_t seed) {
    if (!source)
        return 0;
    memset(source, 0, sizeof(*source));
    source->seed = seed;
    uint64_t state = seed;
    for (size_t i = 0; i < 4; i++)
        source->state[i] = splitmix64(&state);
    if (!(source->state[0] | source->state[1] | source->state[2] | source->state[3]))
        source->state[0] = 1;
    return 1;
}

int phonon_noise_replay(PhononNoiseSource *source, const float *draws, size_t count) {
    if (!source || !draws || !count)
        return 0;
    for (size_t i = 0; i < count; i++)
        if (!isfinite(draws[i]))
            return 0;
    memset(source, 0, sizeof(*source));
    source->replay = 1;
    source->replay_draws = draws;
    source->replay_count = count;
    return 1;
}

int phonon_noise_next(PhononNoiseSource *source, float *value) {
    if (!source || !value)
        return 0;
    if (source->replay) {
        if (!source->replay_draws || source->replay_cursor >= source->replay_count)
            return 0;
        *value = source->replay_draws[source->replay_cursor++];
        return isfinite(*value);
    }
    if (source->has_spare) {
        source->has_spare = 0;
        *value = source->spare;
        return isfinite(*value);
    }
    /* Open interval uniforms avoid log(0); Box-Muller values are standard normal. */
    const double unit = 1.0 / 9007199254740992.0;
    double u1 = ((double)(next_u64(source) >> 11) + 0.5) * unit;
    double u2 = ((double)(next_u64(source) >> 11) + 0.5) * unit;
    double radius = sqrt(-2.0 * log(u1));
    double angle = 6.283185307179586476925286766559 * u2;
    double z0 = radius * cos(angle), z1 = radius * sin(angle);
    source->spare = (float)z1;
    source->has_spare = 1;
    *value = (float)z0;
    return isfinite(*value) && isfinite(source->spare);
}

int phonon_noise_frame(PhononNoiseSource *source, float output[PHONON_LATENT_WIDTH]) {
    if (!source || !output)
        return 0;
    for (size_t i = 0; i < PHONON_LATENT_WIDTH; i++)
        if (!phonon_noise_next(source, &output[i]))
            return 0;
    return 1;
}

void phonon_noise_reset(PhononNoiseSource *source) {
    if (!source)
        return;
    if (source->replay) {
        source->replay_cursor = 0;
        return;
    }
    uint64_t seed = source->seed;
    (void)phonon_noise_seed(source, seed);
}

int phonon_stop_init(PhononStopState *state, size_t budget, size_t tail) {
    if (!state || !budget)
        return 0;
    *state = (PhononStopState){.frame_budget = budget, .frames_after_eos = tail};
    return 1;
}

int phonon_stop_after_frame(PhononStopState *state, float logit, float threshold,
                            int *should_stop) {
    if (!state || !should_stop || !isfinite(logit) || !isfinite(threshold) ||
        !state->frame_budget || state->frames_generated >= state->frame_budget ||
        state->reason != PHONON_STOP_NONE)
        return 0;
    *should_stop = 0;
    state->frames_generated++;
    if (!state->eos_seen && logit > threshold) {
        state->eos_seen = 1;
        state->tail_remaining = state->frames_after_eos;
    } else if (state->eos_seen && state->tail_remaining > 0) {
        state->tail_remaining--;
    }
    if (state->eos_seen && state->tail_remaining == 0) {
        state->reason = PHONON_STOP_EOS;
        *should_stop = 1;
    } else if (state->frames_generated == state->frame_budget) {
        state->reason = PHONON_STOP_FRAME_BUDGET;
        *should_stop = 1;
    }
    return 1;
}

#include "phonon_transformer.h"

#include <stdio.h>
#include <stdlib.h>

static int get_tensor(const PhononPack *pack, const char *name, unsigned int rank, uint64_t dim0,
                      uint64_t dim1, PhononTensorView *view) {
    if (!phonon_pack_tensor(pack, name, view) || view->rank != rank || view->shape[0] != dim0 ||
        (rank > 1 && view->shape[1] != dim1) || view->dtype != PHONON_PACK_BF16)
        return 0;
    return 1;
}

static int get_linear(const PhononPack *pack, const char *prefix, size_t rows, size_t cols,
                      PhononTensorView *weight, PhononTensorView *bias) {
    char name[256];
    if (snprintf(name, sizeof(name), "%s.weight", prefix) >= (int)sizeof(name) ||
        !phonon_pack_tensor(pack, name, weight) || weight->rank != 2 || weight->shape[0] != rows ||
        weight->shape[1] != cols)
        return 0;
    if (snprintf(name, sizeof(name), "%s.bias", prefix) >= (int)sizeof(name) ||
        !get_tensor(pack, name, 1, rows, 0, bias))
        return 0;
    return 1;
}

int phonon_flow_head_load(const PhononPack *pack, PhononFlowHead *head) {
    if (!pack || !head)
        return 0;
    memset(head, 0, sizeof(*head));
    if (!get_linear(pack, "flow_lm.flow_net.input_proj", 512, 32, &head->input_weight,
                    &head->input_bias) ||
        !get_linear(pack, "flow_lm.flow_net.cond_embed", 512, 512, &head->cond_weight,
                    &head->cond_bias) ||
        !get_linear(pack, "flow_lm.flow_net.final_layer.adaLN_modulation.1", 1024, 512,
                    &head->final_ada_weight, &head->final_ada_bias) ||
        !get_linear(pack, "flow_lm.flow_net.final_layer.linear", 32, 512, &head->final_weight,
                    &head->final_bias))
        return 0;
    for (size_t i = 0; i < 2; i++) {
        char prefix[256], name[256];
        snprintf(prefix, sizeof(prefix), "flow_lm.flow_net.time_embed.%zu.mlp.0", i);
        if (!get_linear(pack, prefix, 512, 256, &head->time_weight[i][0], &head->time_bias[i][0]))
            return 0;
        snprintf(prefix, sizeof(prefix), "flow_lm.flow_net.time_embed.%zu.mlp.2", i);
        if (!get_linear(pack, prefix, 512, 512, &head->time_weight[i][1], &head->time_bias[i][1]))
            return 0;
        snprintf(name, sizeof(name), "flow_lm.flow_net.time_embed.%zu.freqs", i);
        if (!get_tensor(pack, name, 1, 128, 0, &head->time_freqs[i]))
            return 0;
        snprintf(name, sizeof(name), "flow_lm.flow_net.time_embed.%zu.mlp.3.alpha", i);
        if (!get_tensor(pack, name, 1, 512, 0, &head->time_alpha[i]))
            return 0;
    }
    for (size_t i = 0; i < PHONON_FLOW_HEAD_BLOCKS; i++) {
        char prefix[256], name[256];
        snprintf(prefix, sizeof(prefix), "flow_lm.flow_net.res_blocks.%zu.adaLN_modulation.1", i);
        if (!get_linear(pack, prefix, 1536, 512, &head->block_ada_weight[i],
                        &head->block_ada_bias[i]))
            return 0;
        snprintf(name, sizeof(name), "flow_lm.flow_net.res_blocks.%zu.in_ln.weight", i);
        if (!get_tensor(pack, name, 1, 512, 0, &head->block_norm_weight[i]))
            return 0;
        snprintf(name, sizeof(name), "flow_lm.flow_net.res_blocks.%zu.in_ln.bias", i);
        if (!get_tensor(pack, name, 1, 512, 0, &head->block_norm_bias[i]))
            return 0;
        for (size_t j = 0; j < 2; j++) {
            snprintf(prefix, sizeof(prefix), "flow_lm.flow_net.res_blocks.%zu.mlp.%zu", i,
                     j ? 2UL : 0UL);
            if (!get_linear(pack, prefix, 512, 512, &head->block_mlp_weight[i][j],
                            &head->block_mlp_bias[i][j]))
                return 0;
        }
    }
    return 1;
}

static int vector(const PhononTensorView *view, float *out, size_t count) {
    return view && view->rank == 1 && view->shape[0] == count && view->dtype == PHONON_PACK_BF16 &&
           view->byte_count == count * sizeof(uint16_t) &&
           phonon_bf16_to_float((const uint16_t *)view->data, count, out);
}

static int affine(const PhononTensorView *weight, const PhononTensorView *bias, const float *input,
                  size_t rows, size_t cols, float *output) {
    float b[1536];
    return vector(bias, b, rows) && phonon_linear(weight, input, rows, cols, b, output);
}

/* PyTorch 2.7.1 SumKernel.cpp: four interleaved NEON vectors, two
 * 16-group cascade chunks for this model's 512-wide rows. Keep binary32
 * additions in the source reduction order rather than a serial sum. */
static float mean_512(const float input[512]) {
    float sums[4][4] = {{0}};
    for (size_t chunk = 0; chunk < 2; chunk++) {
        float partial[4][4] = {{0}};
        for (size_t group = 0; group < 16; group++)
            for (size_t vector = 0; vector < 4; vector++)
                for (size_t lane = 0; lane < 4; lane++)
                    partial[vector][lane] += input[chunk * 256 + group * 16 + vector * 4 + lane];
        for (size_t vector = 0; vector < 4; vector++)
            for (size_t lane = 0; lane < 4; lane++)
                sums[vector][lane] += partial[vector][lane];
    }
    for (size_t vector = 1; vector < 4; vector++)
        for (size_t lane = 0; lane < 4; lane++)
            sums[0][lane] += sums[vector][lane];
    float sum = 0.0f;
    for (size_t lane = 0; lane < 4; lane++)
        sum += sums[0][lane];
    return sum / 512.0f;
}

static int flow_layer_norm(const float *input, const float *weight, const float *bias,
                           float output[512]) {
    for (size_t i = 0; i < 512; i++) {
        if (!isfinite(input[i]) || (weight && !isfinite(weight[i])) || (bias && !isfinite(bias[i])))
            return 0;
    }
    float mean = mean_512(input);
    /* ReduceOps.cpp std_var_all_cpu uses double centered squares around
     * the separately computed float mean, then rounds the variance. */
    double variance_sum = 0.0;
    for (size_t i = 0; i < 512; i++) {
        double centered = (double)input[i] - mean;
        variance_sum += centered * centered;
    }
    float variance = (float)(variance_sum / 512.0);
    float inverse = 1.0f / sqrtf(variance + 1e-6f);
    for (size_t i = 0; i < 512; i++) {
        float centered = input[i] - mean;
        float normalized = centered * inverse;
        if (weight)
            normalized = normalized * weight[i];
        if (bias)
            normalized = normalized + bias[i];
        output[i] = normalized;
        if (!isfinite(output[i]))
            return 0;
    }
    return 1;
}

static int time_embedding(const PhononFlowHead *head, size_t which, float value, float out[512]) {
    float freqs[128], phase[128], embedding[256], hidden[512], alpha[512];
    if (!isfinite(value) || !vector(&head->time_freqs[which], freqs, 128))
        return 0;
    for (size_t i = 0; i < 128; i++) {
        phase[i] = value * freqs[i];
    }
    phonon_math_sincos(phase, 128, embedding, embedding + 128);
    if (!affine(&head->time_weight[which][0], &head->time_bias[which][0], embedding, 512, 256,
                hidden))
        return 0;
    phonon_math_silu(hidden, 512);
    if (!affine(&head->time_weight[which][1], &head->time_bias[which][1], hidden, 512, 512, out) ||
        !vector(&head->time_alpha[which], alpha, 512))
        return 0;
    float squares[512];
    for (size_t i = 0; i < 512; i++)
        squares[i] = out[i] * out[i];
    float mean_square = mean_512(squares);
    float inverse = 1.0f / sqrtf(mean_square + 1e-5f);
    for (size_t i = 0; i < 512; i++) {
        float weighted = alpha[i] * out[i];
        out[i] = weighted * inverse;
    }
    return 1;
}

int phonon_flow_head_forward(const PhononFlowHead *head, const float condition[512],
                             const float state[32], float s, float t, float velocity[32]) {
    float x[512], y[512], time_s[512], time_t[512], modulation[1536], norm_w[512], norm_b[512];
    float h[512], first[512], second[512], normalized[512];
    if (!head || !condition || !state || !velocity || !isfinite(s) || !isfinite(t) ||
        !affine(&head->input_weight, &head->input_bias, state, 512, 32, x) ||
        !affine(&head->cond_weight, &head->cond_bias, condition, 512, 512, y) ||
        !time_embedding(head, 0, s, time_s) || !time_embedding(head, 1, t, time_t))
        return 0;
    for (size_t i = 0; i < 512; i++)
        y[i] += 0.5f * (time_s[i] + time_t[i]);
    for (size_t block = 0; block < PHONON_FLOW_HEAD_BLOCKS; block++) {
        memcpy(normalized, y, sizeof(normalized));
        phonon_math_silu(normalized, 512);
        if (!affine(&head->block_ada_weight[block], &head->block_ada_bias[block], normalized, 1536,
                    512, modulation) ||
            !vector(&head->block_norm_weight[block], norm_w, 512) ||
            !vector(&head->block_norm_bias[block], norm_b, 512) ||
            !flow_layer_norm(x, norm_w, norm_b, h))
            return 0;
        for (size_t i = 0; i < 512; i++) {
            float one_plus_scale = 1.0f + modulation[512 + i];
            float scaled = h[i] * one_plus_scale;
            h[i] = scaled + modulation[i];
        }
        if (!affine(&head->block_mlp_weight[block][0], &head->block_mlp_bias[block][0], h, 512, 512,
                    first))
            return 0;
        phonon_math_silu(first, 512);
        if (!affine(&head->block_mlp_weight[block][1], &head->block_mlp_bias[block][1], first, 512,
                    512, second))
            return 0;
        for (size_t i = 0; i < 512; i++) {
            float gated = modulation[1024 + i] * second[i];
            x[i] = x[i] + gated;
        }
    }
    memcpy(normalized, y, sizeof(normalized));
    phonon_math_silu(normalized, 512);
    if (!affine(&head->final_ada_weight, &head->final_ada_bias, normalized, 1024, 512,
                modulation) ||
        !flow_layer_norm(x, NULL, NULL, h))
        return 0;
    for (size_t i = 0; i < 512; i++) {
        float one_plus_scale = 1.0f + modulation[512 + i];
        float scaled = h[i] * one_plus_scale;
        h[i] = scaled + modulation[i];
    }
    return affine(&head->final_weight, &head->final_bias, h, 32, 512, velocity);
}

int phonon_flow_lsd_one_step(const PhononFlowHead *head, const float condition[512],
                             const float raw_noise[32], float temperature, float output[32]) {
    float noise[32], velocity[32];
    if (!head || !condition || !raw_noise || !output || !isfinite(temperature) || temperature <= 0)
        return 0;
    float scale = sqrtf(temperature);
    for (size_t i = 0; i < 32; i++) {
        if (!isfinite(raw_noise[i]))
            return 0;
        noise[i] = raw_noise[i] * scale;
    }
    if (!phonon_flow_head_forward(head, condition, noise, 0.0f, 1.0f, velocity))
        return 0;
    for (size_t i = 0; i < 32; i++) {
        output[i] = noise[i] + velocity[i];
        if (!isfinite(output[i]))
            return 0;
    }
    return 1;
}

int phonon_flow_eos_logit(const PhononPack *pack, const float state[512], float *raw_logit) {
    PhononTensorView weight, bias;
    if (!pack || !state || !raw_logit ||
        !get_linear(pack, "flow_lm.out_eos", 1, 512, &weight, &bias))
        return 0;
    return affine(&weight, &bias, state, 1, 512, raw_logit);
}

int phonon_flow_denormalize_latent(const PhononPack *pack, const float normalized[32],
                                   float denormalized[32]) {
    PhononTensorView mean_view, std_view;
    float mean[32], std[32];
    if (!pack || !normalized || !denormalized ||
        !phonon_pack_tensor(pack, "flow_lm.emb_mean", &mean_view) ||
        !phonon_pack_tensor(pack, "flow_lm.emb_std", &std_view) || !vector(&mean_view, mean, 32) ||
        !vector(&std_view, std, 32))
        return 0;
    for (size_t i = 0; i < 32; i++) {
        denormalized[i] = normalized[i] * std[i] + mean[i];
        if (!isfinite(denormalized[i]))
            return 0;
    }
    return 1;
}

int phonon_flow_generate(const PhononPack *pack,
                         const PhononFlowLayerWeights layers[PHONON_FLOW_LAYERS],
                         const float condition[512], const float *prefix_embeddings,
                         size_t prefix_length, PhononNoiseSource *noise,
                         const PhononGenerationConfig *config, float *output_latents,
                         size_t output_capacity, PhononGenerationTrace *trace,
                         PhononGenerationResult *result) {
    if (!result)
        return 0;
    *result = (PhononGenerationResult){0};
    if (!pack || !layers || !condition || !prefix_embeddings || !prefix_length || !noise ||
        !config || !output_latents || !config->frame_budget ||
        config->frame_budget > output_capacity || config->frame_budget > 8192 ||
        prefix_length > 8192 || config->frame_budget > 8192 - prefix_length ||
        !isfinite(config->eos_threshold) || !isfinite(config->temperature) ||
        config->temperature <= 0.0f || !isfinite(config->rope_theta) ||
        config->rope_theta <= 0.0f ||
        (trace && (trace->capacity < config->frame_budget || !trace->conditions ||
                   !trace->normalized_latents || !trace->eos_logits)))
        return 0;
    PhononKvCache cache = {0};
    PhononFlowHead *head = calloc(1, sizeof(*head));
    float *state = malloc(512 * sizeof(float));
    int success = 0;
    if (!head || !state || !phonon_kv_cache_init(&cache, prefix_length + config->frame_budget) ||
        !phonon_flow_head_load(pack, head))
        goto cleanup;
    if (!phonon_flow_stack_prefill(layers, &cache, prefix_embeddings, prefix_length,
                                   config->rope_theta, state) ||
        !phonon_flow_output_norm(pack, state))
        goto cleanup;
    PhononStopState stop;
    if (!phonon_stop_init(&stop, config->frame_budget, config->frames_after_eos))
        goto cleanup;
    for (size_t frame = 0; frame < config->frame_budget; frame++) {
        float logit, raw[32], normalized[32];
        if (!phonon_flow_eos_logit(pack, state, &logit) || !phonon_noise_frame(noise, raw) ||
            !phonon_flow_lsd_one_step(head, state, raw, config->temperature, normalized) ||
            !phonon_flow_denormalize_latent(pack, normalized,
                                            output_latents + frame * PHONON_LATENT_WIDTH))
            goto cleanup;
        if (trace) {
            memcpy(trace->conditions + frame * 512, state, 512 * sizeof(float));
            memcpy(trace->normalized_latents + frame * 32, normalized, 32 * sizeof(float));
            trace->eos_logits[frame] = logit;
        }
        int should_stop = 0;
        if (!phonon_stop_after_frame(&stop, logit, config->eos_threshold, &should_stop))
            goto cleanup;
        result->frames_generated = stop.frames_generated;
        result->last_eos_logit = logit;
        if (should_stop) {
            result->stop_reason = stop.reason;
            success = 1;
            goto cleanup;
        }
        float input[512];
        if (!phonon_flow_embed_latent(pack, normalized, condition, input) ||
            !phonon_flow_stack_step(layers, &cache, prefix_length + frame, input,
                                    config->rope_theta, state) ||
            !phonon_flow_output_norm(pack, state))
            goto cleanup;
    }
cleanup:
    free(head);
    free(state);
    phonon_kv_cache_destroy(&cache);
    return success;
}

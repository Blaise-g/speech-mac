#ifndef PHONON_SAMPLER_H
#define PHONON_SAMPLER_H

#include <stddef.h>
#include <stdint.h>

#include "phonon_pack.h"
#include "phonon_transformer.h"

#define PHONON_LATENT_WIDTH 32

typedef enum {
    PHONON_STOP_NONE = 0,
    PHONON_STOP_EOS = 1,
    PHONON_STOP_FRAME_BUDGET = 2
} PhononStopReason;

/* Request-owned deterministic standard Gaussian source. Replay values are RAW N(0,1)
 * draws. Temperature scaling belongs to phonon_flow_lsd_one_step and happens once. */
typedef struct {
    uint64_t state[4];
    uint64_t seed;
    const float *replay_draws; /* borrowed until source is reinitialized */
    size_t replay_count;
    size_t replay_cursor;
    int replay;
    int has_spare;
    float spare;
} PhononNoiseSource;

int phonon_noise_seed(PhononNoiseSource *source, uint64_t seed);
int phonon_noise_replay(PhononNoiseSource *source, const float *raw_draws, size_t count);
int phonon_noise_next(PhononNoiseSource *source, float *value);
int phonon_noise_frame(PhononNoiseSource *source, float output[PHONON_LATENT_WIDTH]);
void phonon_noise_reset(PhononNoiseSource *source);

/* EOS frame is included. Tail is counted after its first EOS frame; repeated EOS
 * logits do not restart the countdown. Budget is checked after every produced frame. */
typedef struct {
    size_t frame_budget;
    size_t frames_after_eos;
    size_t frames_generated;
    size_t tail_remaining;
    int eos_seen;
    PhononStopReason reason;
} PhononStopState;

#define PHONON_FLOW_HEAD_BLOCKS 6

/* Tensor views are borrowed from the immutable pack for its lifetime. */
typedef struct {
    PhononTensorView input_weight, input_bias, cond_weight, cond_bias;
    PhononTensorView time_weight[2][2], time_bias[2][2], time_freqs[2], time_alpha[2];
    PhononTensorView block_ada_weight[PHONON_FLOW_HEAD_BLOCKS];
    PhononTensorView block_ada_bias[PHONON_FLOW_HEAD_BLOCKS];
    PhononTensorView block_norm_weight[PHONON_FLOW_HEAD_BLOCKS];
    PhononTensorView block_norm_bias[PHONON_FLOW_HEAD_BLOCKS];
    PhononTensorView block_mlp_weight[PHONON_FLOW_HEAD_BLOCKS][2];
    PhononTensorView block_mlp_bias[PHONON_FLOW_HEAD_BLOCKS][2];
    PhononTensorView final_ada_weight, final_ada_bias, final_weight, final_bias;
} PhononFlowHead;

int phonon_flow_head_load(const PhononPack *pack, PhononFlowHead *head);
/* Current normalized latent state is x; s and t are the flow times. */
int phonon_flow_head_forward(const PhononFlowHead *head, const float condition[512],
                             const float normalized_state[PHONON_LATENT_WIDTH], float s, float t,
                             float velocity[PHONON_LATENT_WIDTH]);
/* The pinned production policy is one LSD Euler step from normalized Gaussian noise. */
int phonon_flow_eos_logit(const PhononPack *pack, const float state[512], float *raw_logit);
int phonon_flow_denormalize_latent(const PhononPack *pack, const float normalized[32],
                                   float denormalized[32]);
int phonon_flow_lsd_one_step(const PhononFlowHead *head, const float condition[512],
                             const float raw_noise[PHONON_LATENT_WIDTH], float temperature,
                             float normalized_latent[PHONON_LATENT_WIDTH]);

/* On cancellation returns 0 without publishing output. */
int phonon_flow_lsd_one_step_cancel(const PhononFlowHead *head, const float condition[512],
                                    const float raw_noise[32], float temperature, float output[32],
                                    PhononCancelPoll cancel, void *context);
typedef struct {
    size_t frame_budget;
    size_t frames_after_eos;
    float eos_threshold;
    float temperature;
    float rope_theta;
} PhononGenerationConfig;

typedef struct {
    size_t frames_generated;
    PhononStopReason stop_reason;
    float last_eos_logit;
} PhononGenerationResult;

/* Caller-owned buffers. If a trace is supplied, all three buffers must hold capacity frames. */
typedef struct {
    size_t capacity;
    float *conditions;         /* [capacity,512] flow-head inputs */
    float *normalized_latents; /* [capacity,32] sampler outputs */
    float *eos_logits;         /* [capacity] raw logits */
} PhononGenerationTrace;

/* prefix_embeddings are caller-prepared 512-wide token/voice/BOS inputs. Output frames are
 * denormalized Mimi-space latents. Cache and stop state are local to this call. The caller owns
 * noise and output buffers; replay exhaustion returns failure and reports any partial frame count.
 */
int phonon_flow_generate(const PhononPack *pack,
                         const PhononFlowLayerWeights layers[PHONON_FLOW_LAYERS],
                         const float condition[512], const float *prefix_embeddings,
                         size_t prefix_length, PhononNoiseSource *noise,
                         const PhononGenerationConfig *config, float *output_latents,
                         size_t output_capacity, PhononGenerationTrace *trace,
                         PhononGenerationResult *result);

int phonon_stop_init(PhononStopState *state, size_t frame_budget, size_t frames_after_eos);
int phonon_stop_after_frame(PhononStopState *state, float raw_eos_logit, float threshold,
                            int *should_stop);

#endif

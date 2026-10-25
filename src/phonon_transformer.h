#ifndef PHONON_TRANSFORMER_H
#define PHONON_TRANSFORMER_H

#include "phonon_pack.h"
#include <stddef.h>

#define PHONON_FLOW_LAYERS 9
#define PHONON_FLOW_WIDTH 512
#define PHONON_FLOW_HEADS 8
#define PHONON_FLOW_HEAD_WIDTH 64

/* Fixed capacity request-owned KV storage. Initialize per request; clear before reuse. */
typedef struct {
    size_t capacity;
    size_t length;
    float *keys;
    float *values;
} PhononKvCache;

typedef struct {
    const PhononTensorView *qkv;           /* [1536,512], checkpoint interleaved Q/K RoPE layout */
    const PhononTensorView *attention_out; /* [512,512] */
    const PhononTensorView *mlp_in;        /* [2048,512] */
    const PhononTensorView *mlp_out;       /* [512,2048] */
    const PhononTensorView *norm1_weight, *norm1_bias;
    const PhononTensorView *norm2_weight, *norm2_bias;
} PhononFlowLayerWeights;

int phonon_kv_cache_init(PhononKvCache *cache, size_t capacity);
void phonon_kv_cache_reset(PhononKvCache *cache);
void phonon_kv_cache_destroy(PhononKvCache *cache);
int phonon_kv_cache_append(PhononKvCache *cache, size_t layer, size_t position, const float *key,
                           const float *value, size_t width);
/* Advance sequence length once after all nine layers have appended a token. */
int phonon_kv_cache_commit_position(PhononKvCache *cache, size_t position);

/* Stable numerical primitives shared by the Flow-LM. Linear projections own
 * and free bounded transient expanded weights; the other primitives do not allocate. */
int phonon_rms_norm(const float *input, const float *weight, size_t width, float epsilon,
                    float *output);
int phonon_layer_norm(const float *input, const float *weight, const float *bias, size_t width,
                      float epsilon, float *output);
int phonon_erf_gelu(float *values, size_t count);
int phonon_rope_interleaved(float *values, size_t width, size_t heads, size_t position,
                            float theta);
/* Applies a row-major [rows, columns] tensor; Q8 stays packed at the call boundary. */
int phonon_linear(const PhononTensorView *weight, const float *input, size_t rows, size_t columns,
                  const float *bias, float *output);
/* Same contract over row-major [batch,columns] and [batch,rows] buffers. */
int phonon_linear_batch(const PhononTensorView *weight, const float *input, size_t rows,
                        size_t columns, const float *bias, float *output, size_t batch);
int phonon_flow_layer_step(const PhononFlowLayerWeights *weights, PhononKvCache *cache,
                           size_t layer, size_t position, const float input[PHONON_FLOW_WIDTH],
                           float theta, float output[PHONON_FLOW_WIDTH]);
int phonon_flow_stack_step(const PhononFlowLayerWeights weights[PHONON_FLOW_LAYERS],
                           PhononKvCache *cache, size_t position,
                           const float input[PHONON_FLOW_WIDTH], float theta,
                           float output[PHONON_FLOW_WIDTH]);
int phonon_flow_condition(const PhononTensorView *speaker_embeddings,
                          const PhononTensorView *speaker_projection,
                          const PhononTensorView *duration_projection, size_t speaker_count,
                          float duration_delta, float output[PHONON_FLOW_WIDTH]);
/* Prefill an empty cache with length<=capacity/8192 prefix rows. Input is borrowed;
 * scratch is call-owned. Success commits the entire prefix and returns its last
 * unnormalized state. Failure leaves output untouched and resets a started cache. */
int phonon_flow_stack_prefill(const PhononFlowLayerWeights weights[PHONON_FLOW_LAYERS],
                              PhononKvCache *cache, const float *input, size_t length, float theta,
                              float output[512]);

int phonon_voice_project(const float *voice_latents, size_t frames,
                         const PhononTensorView *projection, float *projected);
int phonon_flow_prepare_prefix(const int32_t *ids, size_t tokens, const float *token_embeddings,
                               size_t prompt_id, size_t bos_id, const float *projected_voice,
                               size_t voice_frames, const float *bos_latent_projection,
                               float *prefix_embeddings);
int phonon_flow_load_layers(const PhononPack *pack, PhononTensorView storage[PHONON_FLOW_LAYERS][8],
                            PhononFlowLayerWeights weights[PHONON_FLOW_LAYERS]);
int phonon_flow_embed_tokens(const PhononPack *pack, const int32_t *ids, size_t count,
                             float *embeddings);
/* BOS is already in the model's normalized latent space; adds speaker/duration conditioning. */
int phonon_flow_embed_bos(const PhononPack *pack, const float condition[PHONON_FLOW_WIDTH],
                          float output[PHONON_FLOW_WIDTH]);
/* Input is normalized Flow-LM latent space; call normalize_latent for generated Mimi-space data. */
int phonon_flow_embed_latent(const PhononPack *pack, const float latent[32],
                             const float condition[PHONON_FLOW_WIDTH],
                             float output[PHONON_FLOW_WIDTH]);
/* Convert a frozen/generated Mimi-space latent to Flow-LM normalized latent units. */
int phonon_flow_normalize_latent(const PhononPack *pack, const float denormalized[32],
                                 float normalized[32]);
int phonon_flow_output_norm(const PhononPack *pack, float values[PHONON_FLOW_WIDTH]);

#endif

#ifndef PHONON_MIMI_H
#define PHONON_MIMI_H
#include "phonon_transformer.h"

#define PHONON_MIMI_WIDTH 512
#define PHONON_MIMI_CONTEXT 250
#define PHONON_MIMI_STRIDE 16
#define PHONON_MIMI_MAX_FRAMES 8192

typedef struct {
    unsigned sample_rate;
    float frame_rate;
    unsigned ratios[3];
    size_t context;
    float rope_theta;
} PhononMimiConfig;
/* Immutable borrowed pack views; pack must outlive model and its requests.
 * A state borrows the model itself, which must stay immutable and alive. */
typedef struct {
    PhononMimiConfig config;
    size_t stride;
    PhononTensorView projection, upsample;
    PhononTensorView layers[2][10];
} PhononMimiModel;
typedef struct PhononMimiState PhononMimiState;

PhononMimiConfig phonon_mimi_config(void);
int phonon_mimi_model_load(const PhononPack *pack, const PhononMimiConfig *config,
                           PhononMimiModel *model);
int phonon_mimi_state_create(const PhononMimiModel *model, PhononMimiState **state);
void phonon_mimi_state_reset(PhononMimiState *state);
void phonon_mimi_state_destroy(PhononMimiState *state);
size_t phonon_mimi_positions(const PhononMimiState *state);
/* Row-major [frames,32] to [frames*stride,512]. Capacity is in feature rows.
 * Input/output must not overlap. Frames may be partitioned arbitrarily. State
 * owns bounded cache/scratch independent of request length. On any processing
 * failure state becomes unusable until reset; output may contain a partial prefix.
 * Invalid arguments leave state untouched. Maximum request: 8192 latent frames. */
int phonon_mimi_process(PhononMimiState *state, const float *latents, size_t frames,
                        float *features, size_t capacity);
/* Depthwise causal transpose convolution: [channels,2*stride], no bias,
 * trim exactly stride positions from the right. Previous is zero on reset.
 * Per-frame [channels] input to row-major [stride,channels] output. */
int phonon_mimi_upsample_frame(const PhononTensorView *weight, const float *input,
                               const float *previous, size_t channels, size_t stride,
                               float *output);
#endif

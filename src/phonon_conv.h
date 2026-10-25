#ifndef PHONON_CONV_H
#define PHONON_CONV_H
#include "phonon_pack.h"

/* Dense SEANet operators only: constant causal Conv1d, stride/dilation=1,
 * kernel 1,3,7; or causal ConvTranspose1d, stride 4,5,6 and kernel=2*stride.
 * Channels are in [1,512]. BF16 source views are copied/expanded at creation;
 * models own weights and can outlive the source pack. Processing allocates no memory. */
typedef struct PhononConvModel PhononConvModel;
typedef struct PhononConvState PhononConvState;
int phonon_conv_model_create(const PhononTensorView *weight, const PhononTensorView *bias,
                             size_t stride, int transpose, PhononConvModel **model);
void phonon_conv_model_destroy(PhononConvModel *model);
size_t phonon_conv_input_channels(const PhononConvModel *model);
size_t phonon_conv_output_channels(const PhononConvModel *model);
size_t phonon_conv_stride(const PhononConvModel *model);
/* State borrows immutable model; destroy all states before model. History and
 * overlap storage are bounded by the operator shape, independent of duration. */
int phonon_conv_state_create(const PhononConvModel *model, PhononConvState **state);
void phonon_conv_state_destroy(PhononConvState *state);
void phonon_conv_state_reset(PhononConvState *state);
/* Row-major [positions,channels]. Capacity counts output positions (not floats).
 * Each input position emits stride positions immediately; no deferred flush.
 * Max cumulative input positions 15,728,640 (8192 latent frames *16*6*5*4).
 * Zero positions is a no-op. Invalid arguments leave state unchanged. Arithmetic
 * failures poison state until reset; output may contain a partial prefix.
 * Buffers must not overlap; input/output remain caller owned. */
int phonon_conv_process(PhononConvState *state, const float *input, size_t positions, float *output,
                        size_t capacity);

/* Pinned residual seam: identity + Conv1(ELU(Conv3(ELU(input))));
 * hidden width = channels/2. Models are borrowed and must outlive the state. */
typedef struct PhononResidualState PhononResidualState;
int phonon_residual_state_create(const PhononConvModel *conv3, const PhononConvModel *conv1,
                                 PhononResidualState **state);
void phonon_residual_state_destroy(PhononResidualState *state);
void phonon_residual_state_reset(PhononResidualState *state);
int phonon_residual_process(PhononResidualState *state, const float *input, size_t positions,
                            float *output, size_t capacity);
#endif

#ifndef PHONON_PCM_H
#define PHONON_PCM_H
#include "phonon_mimi.h"

#define PHONON_PCM_FRAME_SAMPLES 1920
#define PHONON_PCM_SAMPLE_RATE 24000

typedef struct PhononPCMModel PhononPCMModel;
typedef struct PhononPCMState PhononPCMState;
/* Resident immutable model owns expanded SEANet weights, borrows Mimi views from
 * pack. Pack must outlive model; model must outlive all request states. */
int phonon_pcm_model_create(const PhononPack *pack, PhononPCMModel **model);
void phonon_pcm_model_destroy(PhononPCMModel *model);
int phonon_pcm_state_create(const PhononPCMModel *model, PhononPCMState **state);
void phonon_pcm_state_destroy(PhononPCMState *state);
void phonon_pcm_state_reset(PhononPCMState *state);
size_t phonon_pcm_samples(const PhononPCMState *state);
/* Denormalized row-major [frames,32] -> mono 24k PCM. Every frame immediately
 * emits exactly 1920 samples. Capacities count samples, not bytes. Both output
 * buffers are caller owned; final may be NULL or exactly raw, otherwise buffers
 * must not overlap each other or input. Pinned HF processor has fade=0 and no
 * gain/limiter, so final equals raw. Pass only retained frames (mask prefix): EOS
 * tail is generated upstream. Causal left padding/right crop is inside operators;
 * there is no first-sample delay to remove at this boundary.
 * State owns bounded caches/scratch, never a whole utterance; processing allocates
 * nothing. Limit 8192 frames. Invalid arguments/nonfinite input leave state intact.
 * Arithmetic failure poisons state until reset, outputs may hold a partial prefix.
 * Empty input is a no-op before finish. Process after finish fails, including empty.
 */
int phonon_pcm_process(PhononPCMState *state, const float *latents, size_t frames, float *raw,
                       size_t raw_capacity, float *final, size_t final_capacity);
/* Terminal transition. No extra latent, padding or deferred codec samples are
 * emitted. Repeated finish and process after finish fail until reset. */
int phonon_pcm_finish(PhononPCMState *state);
#endif

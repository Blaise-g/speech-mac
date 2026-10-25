#ifndef PHONON_VOICE_H
#define PHONON_VOICE_H

#include <stddef.h>

typedef struct PhononVoice PhononVoice;

/* Compatible raw safetensors speaker_wavs ([1,512,T] or [512,T]), with T<=4096.
 * Precomputed `emb`/`audio_prompt` forms are outside this reader's contract. */
int phonon_voice_open(const char *path, PhononVoice **voice);
void phonon_voice_close(PhononVoice *voice);
size_t phonon_voice_frames(const PhononVoice *voice);
/* Borrowed until close. Mimi-space latent units, row-major [frames,512]. */
const float *phonon_voice_latents(const PhononVoice *voice);

#endif

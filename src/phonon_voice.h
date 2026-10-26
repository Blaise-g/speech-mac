#ifndef PHONON_VOICE_H
#define PHONON_VOICE_H

#include <stddef.h>

typedef struct PhononVoice PhononVoice;

/* Compatible safetensors voices, 1..4096 frames. */
/* Projected emb/audio_prompt [T,512] or [1,T,512] require matching model_ext.
 * Raw speaker_wavs require sig/epoch. Both return frame-major finite F32. */
int phonon_voice_is_projected(const PhononVoice *voice);
int phonon_voice_open(const char *path, PhononVoice **voice);
void phonon_voice_close(PhononVoice *voice);
size_t phonon_voice_frames(const PhononVoice *voice);
/* Borrowed until close, row-major [frames,512]. Raw Mimi-space units or
 * projected Flow-LM embeddings as identified by phonon_voice_is_projected. */
const float *phonon_voice_latents(const PhononVoice *voice);

#endif

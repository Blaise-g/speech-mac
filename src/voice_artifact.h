#ifndef KOKORO_VOICE_ARTIFACT_H
#define KOKORO_VOICE_ARTIFACT_H

#include "kokoro.h"
#include "model_pack.h"

typedef struct KokoroVoiceArtifact KokoroVoiceArtifact;

/* The mapped style view is borrowed until the artifact is closed. */
KokoroStatus kokoro_voice_artifact_open(const char *path, const char *voice_name,
                                        KokoroVoiceArtifact **artifact, KokoroError *error);
void kokoro_voice_artifact_close(KokoroVoiceArtifact *artifact);
KokoroStatus kokoro_voice_artifact_tensor(const KokoroVoiceArtifact *artifact,
                                          KokoroTensorView *view, KokoroError *error);

#endif

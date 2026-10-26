#ifndef SPEECH_H
#define SPEECH_H
#include "phonon_voice.h"
#include <stddef.h>
#include <stdint.h>

#define SPEECH_SAMPLE_RATE 24000U
#define SPEECH_FRAME_SAMPLES 1920U
#define SPEECH_MAX_INPUT_BYTES (1024U * 1024U)
#define SPEECH_MAX_REQUEST_FRAMES 8192U

typedef struct SpeechEngine SpeechEngine;
typedef enum {
    SPEECH_OK = 0,
    SPEECH_INVALID,
    SPEECH_LIMIT,
    SPEECH_CANCELLED,
    SPEECH_CALLBACK_ERROR,
    SPEECH_FAILED
} SpeechStatus;
/* Return nonzero to cancel. Polled on the synchronous caller thread, including
 * during numerical work; never interrupts a running BLAS call. */
typedef int (*SpeechCancel)(void *context);
typedef struct {
    const char *language; /* exactly "en" */
    uint64_t seed;
    float duration_delta; /* finite native duration conditioning, not the removed speed scale */
    size_t max_frames;    /* total request cap, 1..8192; exceeding it returns SPEECH_LIMIT */
    SpeechCancel cancel;
    void *cancel_context;
} SpeechOptions;
typedef struct {
    size_t chunks_completed;
    size_t frames_generated;
    size_t samples_generated;
} SpeechResult;
/* PCM is borrowed only for this call, exactly 1920 mono samples. Progress counts
 * include this frame; later frames have not been generated. Return 0 to accept,
 * 1 to cancel, any other value for a consumer error. No callback after termination.
 * Callbacks must not destroy the engine or voice. Separate requests have isolated
 * state and may share an immutable engine; callers synchronize their own callbacks. */
typedef int (*SpeechPCMCallback)(void *context, const float *pcm, size_t count,
                                 const SpeechResult *progress);
SpeechOptions speech_options_default(void);
/* Owns mapped pack and resident decoder weights until destroy; no request state is
 * shared. Voice is caller owned and borrowed throughout synthesize/collect. */
SpeechStatus speech_engine_create(const char *pack_path, SpeechEngine **engine);
void speech_engine_destroy(SpeechEngine *engine);
SpeechStatus speech_synthesize(const SpeechEngine *engine, const PhononVoice *voice,
                               const char *text, size_t length, const SpeechOptions *options,
                               SpeechPCMCallback callback, void *context, SpeechResult *result);
/* Bounded collector over the same producer. Caller owns *pcm on success (free).
 * On any error *pcm=NULL, *count=0; result still reports generated partial progress.
 * capacity is a hard sample limit, at most 8192*1920; no hidden unbounded queue. */
SpeechStatus speech_collect(const SpeechEngine *engine, const PhononVoice *voice, const char *text,
                            size_t length, const SpeechOptions *options, size_t capacity,
                            float **pcm, size_t *count, SpeechResult *result);
#endif

#ifndef SPEECH_STREAM_H
#define SPEECH_STREAM_H
#include "speech.h"
#include "stream_playback.h"
#include <signal.h>
typedef struct SpeechStream SpeechStream;
/* CLOCK_MONOTONIC seconds since its unspecified origin; zero means no event.
 * Start means successful running-state observation, not acoustic onset. Reclamation is measured at
 * the AudioQueue callback, not at the speaker. */
typedef struct {
    double synthesis_begin, first_pcm, first_submission, started, synthesis_end, drained;
    size_t submitted_samples, reclaimed_buffers;
    double first_reclaimed;
} SpeechStreamMetrics;
/* Engine, voice, text, options.language/cancel_context and playback context remain
 * borrowed and immutable until wait/free returns. Options itself is copied.
 * Playback submit must copy PCM before returning. No whole-utterance PCM queue.
 * One control owner calls wait/free; cancel may run concurrently with that owner.
 * Cancellation callbacks run only on the producer, including numerical work. */
KokoroStatus speech_stream_start(const SpeechEngine *, const PhononVoice *, const char *, size_t,
                                 const SpeechOptions *, SpeechStream **, KokoroError *);
/* Playback-only injection: inference always runs the actual resident engine. */
KokoroStatus speech_stream_start_with(const SpeechEngine *, const PhononVoice *, const char *,
                                      size_t, const SpeechOptions *,
                                      const KokoroStreamPlaybackOps *, void *, SpeechStream **,
                                      KokoroError *);
KokoroStatus speech_stream_wait(SpeechStream *, const volatile sig_atomic_t *, KokoroError *);
void speech_stream_cancel(SpeechStream *);
void speech_stream_free(SpeechStream *);
/* Read only after wait has returned successfully joining the producer. */
void speech_stream_result(const SpeechStream *, SpeechResult *, SpeechStreamMetrics *);
#endif

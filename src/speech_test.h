#ifndef SPEECH_TEST_H
#define SPEECH_TEST_H
#include "speech.h"
/* Observation only: invoked after each real native head evaluation, before PCM.
 * No model substitution, trace replay or fixture dispatch enters shipped runtime. */
typedef void (*SpeechTestFrameObserver)(void *context, size_t already_generated,
                                        const float condition[512], const float normalized[32],
                                        float raw_eos);
SpeechStatus speech_test_synthesize(const SpeechEngine *engine, const PhononVoice *voice,
                                    const char *text, size_t length, const SpeechOptions *options,
                                    SpeechPCMCallback callback, void *context, SpeechResult *result,
                                    SpeechTestFrameObserver observer, void *observer_context);
#endif

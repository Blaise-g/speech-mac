#ifndef SPEECH_STREAM_TEST_H
#define SPEECH_STREAM_TEST_H
#include "speech_stream.h"
/* Compiled only in test builds; controls PCM production without replacing any
 * numerical operation. Ordinary binaries have no producer-injection entry point. */
typedef SpeechStatus (*SpeechStreamTestProducer)(const SpeechEngine *, const PhononVoice *,
                                                 const char *, size_t, const SpeechOptions *,
                                                 SpeechPCMCallback, void *, SpeechResult *);
KokoroStatus speech_stream_test_start(const SpeechOptions *, const KokoroStreamPlaybackOps *,
                                      void *, SpeechStream **, SpeechStreamTestProducer);
#endif

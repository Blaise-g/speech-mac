#ifndef KOKORO_STREAM_H
#define KOKORO_STREAM_H

#include "ipa_tokenizer.h"
#include "synthesis.h"

#include <signal.h>
#include <stdint.h>

typedef struct KokoroStream KokoroStream;

/* The engine and immutable chunks remain borrowed until wait/free returns. */
KokoroStatus kokoro_stream_start(KokoroEngine *engine, const KokoroIpaChunks *chunks, float speed,
                                 uint64_t random_seed, KokoroStream **stream, KokoroError *error);
/* An optional signal flag is polled in ordinary control flow; the handler only sets that flag. */
KokoroStatus kokoro_stream_wait(KokoroStream *stream, const volatile sig_atomic_t *interrupted,
                                KokoroError *error);
void kokoro_stream_cancel(KokoroStream *stream);
void kokoro_stream_free(KokoroStream *stream);

#endif

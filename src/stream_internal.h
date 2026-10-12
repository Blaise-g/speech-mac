#ifndef KOKORO_STREAM_INTERNAL_H
#define KOKORO_STREAM_INTERNAL_H

#include "stream.h"
#include "stream_playback.h"

typedef KokoroStatus (*KokoroStreamSynthesizeFn)(void *context, KokoroEngine *engine,
                                                 const int64_t *token_ids, size_t token_count,
                                                 const KokoroSynthesisOptions *options,
                                                 KokoroSynthesisOutput *output, KokoroError *error);

/* Stable test seam for deterministic producer and playback tests. */
KokoroStatus kokoro_stream_start_with(KokoroEngine *engine, const KokoroIpaChunks *chunks,
                                      float speed, uint64_t random_seed,
                                      const KokoroStreamPlaybackOps *playback_ops,
                                      void *playback_context, KokoroStreamSynthesizeFn synthesize,
                                      void *synthesize_context, KokoroStream **stream,
                                      KokoroError *error);

#endif

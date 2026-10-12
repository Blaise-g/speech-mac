#ifndef KOKORO_IPA_TOKENIZER_H
#define KOKORO_IPA_TOKENIZER_H

#include "kokoro.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
    int64_t *token_ids;
    size_t token_count;
} KokoroIpaChunk;

typedef struct {
    KokoroIpaChunk *items;
    size_t count;
} KokoroIpaChunks;

/* Encode pinned-v1.0 IPA characters with BOS/EOS IDs; the caller owns token_ids. */
KokoroStatus kokoro_ipa_encode(const char *phonemes, int64_t **token_ids, size_t *token_count,
                               KokoroError *error);
/* Encode complete input into safe chunks. eSpeak output may use line breaks as sentence bounds. */
KokoroStatus kokoro_ipa_encode_chunks(const char *phonemes, bool sentence_lines,
                                      KokoroIpaChunks *chunks, KokoroError *error);
void kokoro_ipa_chunks_free(KokoroIpaChunks *chunks);

#endif

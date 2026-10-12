#ifndef KOKORO_IPA_TOKENIZER_H
#define KOKORO_IPA_TOKENIZER_H

#include "kokoro.h"

#include <stddef.h>
#include <stdint.h>

/* Encode pinned-v1.0 IPA characters with BOS/EOS IDs; the caller owns token_ids. */
KokoroStatus kokoro_ipa_encode(const char *phonemes, int64_t **token_ids, size_t *token_count,
                               KokoroError *error);

#endif

#ifndef CONTEXT_ENCODER_H
#define CONTEXT_ENCODER_H

#include "kokoro.h"
#include "model_pack.h"

#include <stdint.h>

#define KOKORO_CONTEXT_MAX_TOKENS 512U
#define KOKORO_CONTEXT_FEATURES 512U

typedef struct KokoroContextEncoder KokoroContextEncoder;

/* The pack must outlive the encoder; the encoder borrows its read-only weights. */
KokoroStatus kokoro_context_encoder_create(const KokoroPack *pack, KokoroContextEncoder **encoder,
                                           KokoroError *error);
void kokoro_context_encoder_free(KokoroContextEncoder *encoder);

/*
 * Encode one ID sequence, including Kokoro's BOS/EOS IDs. `sequence_length` includes trailing
 * padding, while `valid_length` gives the count of unmasked tokens. Output is channel-major
 * [512, sequence_length], matching the upstream `bert_encoder(...).transpose(-1, -2)` seam.
 * `output_capacity` is measured in float elements. No output is written on failure.
 */
KokoroStatus kokoro_context_encode(const KokoroContextEncoder *encoder, const int64_t *token_ids,
                                   size_t sequence_length, size_t valid_length, float *output,
                                   size_t output_capacity, KokoroError *error);

#endif

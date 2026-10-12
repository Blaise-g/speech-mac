#ifndef KOKORO_TEXT_ENCODER_H
#define KOKORO_TEXT_ENCODER_H

#include "duration_alignment.h"
#include "kokoro.h"
#include "model_pack.h"

#include <stddef.h>
#include <stdint.h>

#define KOKORO_TEXT_ENCODER_FEATURES 512U
#define KOKORO_TEXT_ENCODER_MAX_TOKENS 512U
#define KOKORO_TEXT_ENCODER_MAX_FRAMES 51200U

typedef struct KokoroTextEncoder KokoroTextEncoder;

typedef struct {
    size_t token_count;
    size_t frame_count;
    /* Owned channel-major [512, token] and [512, frame] arrays. */
    float *text_features;
    float *aligned_features;
} KokoroTextEncoderOutput;

/* The pack must outlive the encoder; the encoder borrows its read-only weights. */
KokoroStatus kokoro_text_encoder_create(const KokoroPack *pack, KokoroTextEncoder **encoder,
                                        KokoroError *error);
void kokoro_text_encoder_free(KokoroTextEncoder *encoder);

/*
 * Run the v1.0 text encoder and expand its output with the duration stage's one-hot alignment.
 * `sequence_length` includes trailing padding and must match alignment->token_count; only the
 * first `valid_length` token IDs contribute features. The output must be initialized to {0}; on
 * success it owns both feature arrays and releases them with kokoro_text_encoder_output_free.
 */
KokoroStatus kokoro_text_encoder_run(const KokoroTextEncoder *encoder, const int64_t *token_ids,
                                     size_t sequence_length, size_t valid_length,
                                     const KokoroDurationOutput *alignment,
                                     KokoroTextEncoderOutput *output, KokoroError *error);
void kokoro_text_encoder_output_free(KokoroTextEncoderOutput *output);

#endif

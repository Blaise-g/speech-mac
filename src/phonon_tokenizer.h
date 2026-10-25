#ifndef PHONON_TOKENIZER_H
#define PHONON_TOKENIZER_H
#include <stddef.h>
#include <stdint.h>

#define PHONON_TOKENIZER_MAX_INPUT_BYTES (1024U * 1024U)
#define PHONON_TOKENIZER_MAX_IDS (3U * PHONON_TOKENIZER_MAX_INPUT_BYTES + 4U)

typedef struct {
    uint32_t *ids;
    size_t length;
} PhononTokenIds;
/* Encode ordinary UTF-8 text as HF encode(text, false). Input is borrowed, NUL-free UTF-8. */
int phonon_tokenizer_encode(const char *text, size_t length, PhononTokenIds *out);
/* Decode ids with HF decode(skip_special_tokens=true); output is owned/NUL terminated. */
int phonon_tokenizer_decode(const uint32_t *ids, size_t count, char **out, size_t *length);
void phonon_token_ids_free(PhononTokenIds *ids);
#endif

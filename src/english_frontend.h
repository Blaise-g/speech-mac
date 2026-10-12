#ifndef KOKORO_ENGLISH_FRONTEND_H
#define KOKORO_ENGLISH_FRONTEND_H

#include "kokoro.h"

#include <stddef.h>

/* Run eSpeak NG as an external frontend; the caller owns the returned IPA text. */
KokoroStatus kokoro_english_frontend(const char *program_path, const char *text, size_t text_length,
                                     char **phonemes, size_t *phoneme_length, KokoroError *error);

#endif

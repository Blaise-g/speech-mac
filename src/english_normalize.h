#ifndef ENGLISH_NORMALIZE_H
#define ENGLISH_NORMALIZE_H

#include "kokoro.h"

#include <stdbool.h>
#include <stddef.h>

/* Normalize UTF-8 ordinary English text using the pinned Phonon preprocessing policy.
 * On success, `*normalized` is owned by the caller and NUL terminated; its byte length
 * excludes the terminator. `language` must be exactly "en". */
KokoroStatus phonon_english_normalize(const char *language, const char *text, size_t text_length,
                                      char **normalized, size_t *normalized_length,
                                      KokoroError *error);
void phonon_english_normalize_free(char *normalized);

#ifdef PHONON_ENGLISH_NORMALIZE_TEST_ALLOCATOR
void phonon_english_normalize_test_fail_after(long allocation_index);
bool phonon_english_normalize_test_failure_triggered(void);
#endif

#endif

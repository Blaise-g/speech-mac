#ifndef KOKORO_UTF8_H
#define KOKORO_UTF8_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Decode one Unicode scalar value and advance offset on success. */
bool kokoro_utf8_decode_next(const unsigned char *text, size_t length, size_t *offset,
                             uint32_t *codepoint);

/* Validate a complete UTF-8 byte sequence. Embedded NUL is valid UTF-8. */
bool kokoro_utf8_is_valid(const unsigned char *text, size_t length);

#endif

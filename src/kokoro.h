#ifndef KOKORO_H
#define KOKORO_H

#include <stddef.h>

typedef enum {
    KOKORO_OK = 0,
    KOKORO_INVALID_ARGUMENT,
    KOKORO_IO_ERROR,
    KOKORO_MALFORMED_PACK,
    KOKORO_INCOMPATIBLE_PACK,
    KOKORO_OUT_OF_MEMORY,
    KOKORO_INVALID_TOKEN_ID,
    KOKORO_CONTEXT_TOO_LONG,
    KOKORO_BUFFER_TOO_SMALL,
    KOKORO_NUMERIC_ERROR
} KokoroStatus;

typedef struct {
    KokoroStatus code;
    char message[256];
} KokoroError;

const char *kokoro_status_name(KokoroStatus status);

#endif

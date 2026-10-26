#ifndef SPEECH_ERROR_H
#define SPEECH_ERROR_H

#include <stddef.h>

typedef enum {
    SPEECH_IO_OK = 0,
    SPEECH_IO_INVALID_ARGUMENT,
    SPEECH_IO_ERROR,
    SPEECH_IO_MALFORMED_PACK,
    SPEECH_IO_INCOMPATIBLE_PACK,
    SPEECH_IO_OUT_OF_MEMORY,
    SPEECH_IO_INVALID_TOKEN_ID,
    SPEECH_IO_CONTEXT_TOO_LONG,
    SPEECH_IO_BUFFER_TOO_SMALL,
    SPEECH_IO_NUMERIC_ERROR,
    SPEECH_IO_CANCELLED
} SpeechIOStatus;

typedef struct {
    SpeechIOStatus code;
    char message[256];
} SpeechError;

const char *speech_status_name(SpeechIOStatus status);

#endif

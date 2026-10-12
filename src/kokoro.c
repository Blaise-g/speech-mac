#include "kokoro_internal.h"

#include <stdio.h>

const char *kokoro_status_name(KokoroStatus status) {
    switch (status) {
    case KOKORO_OK:
        return "ok";
    case KOKORO_INVALID_ARGUMENT:
        return "invalid argument";
    case KOKORO_IO_ERROR:
        return "I/O error";
    case KOKORO_MALFORMED_PACK:
        return "malformed model pack";
    case KOKORO_INCOMPATIBLE_PACK:
        return "incompatible model pack";
    case KOKORO_OUT_OF_MEMORY:
        return "out of memory";
    case KOKORO_INVALID_TOKEN_ID:
        return "invalid token ID";
    case KOKORO_CONTEXT_TOO_LONG:
        return "context length error";
    case KOKORO_BUFFER_TOO_SMALL:
        return "output buffer too small";
    case KOKORO_NUMERIC_ERROR:
        return "non-finite model output";
    case KOKORO_CANCELLED:
        return "cancelled";
    }
    return "unknown error";
}

void kokoro_error_set(KokoroError *error, KokoroStatus code, const char *format, ...) {
    if (error == NULL) {
        return;
    }
    error->code = code;
    if (format == NULL) {
        error->message[0] = '\0';
        return;
    }
    va_list arguments;
    va_start(arguments, format);
    (void)vsnprintf(error->message, sizeof(error->message), format, arguments);
    va_end(arguments);
}

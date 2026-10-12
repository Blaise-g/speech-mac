#ifndef KOKORO_INTERNAL_H
#define KOKORO_INTERNAL_H

#include "kokoro.h"

#include <stdarg.h>

void kokoro_error_set(KokoroError *error, KokoroStatus code, const char *format, ...);

#endif

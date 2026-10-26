#ifndef SPEECH_ERROR_INTERNAL_H
#define SPEECH_ERROR_INTERNAL_H

#include "speech_error.h"

#include <stdarg.h>

void speech_error_set(SpeechError *error, SpeechIOStatus code, const char *format, ...);

#endif

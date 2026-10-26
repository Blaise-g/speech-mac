#ifndef SPEECH_SHA256_H
#define SPEECH_SHA256_H

#include <stddef.h>

int speech_sha256_hex(const unsigned char *data, size_t length, char output[65]);

#endif

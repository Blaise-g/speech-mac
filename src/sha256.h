#ifndef KOKORO_SHA256_H
#define KOKORO_SHA256_H

#include <stddef.h>

int kokoro_sha256_hex(const unsigned char *data, size_t length, char output[65]);

#endif

#ifndef WAV_WRITER_H
#define WAV_WRITER_H

#include "kokoro.h"

#include <stddef.h>
#include <stdint.h>

/* Write mono IEEE float samples as little-endian PCM16 WAV. */
KokoroStatus kokoro_wav_write_pcm16(const char *path, const float *samples, size_t sample_count,
                                    uint32_t sample_rate_hz, KokoroError *error);

#endif

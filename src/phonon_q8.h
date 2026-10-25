#ifndef PHONON_Q8_H
#define PHONON_Q8_H

#include <stddef.h>
#include <stdint.h>

/* GGML Q8_0: one little-endian IEEE binary16 scale followed by 32 signed
 * quantized values. The input blocks and vectors are borrowed for the call. */
typedef struct {
    uint16_t scale_bits;
    int8_t values[32];
} PhononQ8Block;

int phonon_q8_dot(const PhononQ8Block *blocks, const float *input, size_t elements, float *result);
int phonon_q8_matvec(const PhononQ8Block *weights, const float *input, size_t rows, size_t columns,
                     float *output);

#endif

#ifndef PHONON_Q8_H
#define PHONON_Q8_H

#include <stddef.h>
#include <stdint.h>

/* GGML Q8_0: one little-endian IEEE binary16 scale followed by 32 signed
 * quantized values. Blocks and input are borrowed. Dot reads elements floats;
 * matvec reads columns floats and writes rows floats. Matvec input/output and
 * weights/output must not overlap. Output contents are unspecified on failure.
 * Both functions reject invalid sizes and non-finite values/results. */
typedef struct {
    uint16_t scale_bits;
    int8_t values[32];
} PhononQ8Block;

_Static_assert(sizeof(PhononQ8Block) == 34, "Q8_0 blocks must retain their packed layout");

int phonon_q8_dot(const PhononQ8Block *blocks, const float *input, size_t elements, float *result);
int phonon_q8_matvec(const PhononQ8Block *weights, const float *input, size_t rows, size_t columns,
                     float *output);

#endif

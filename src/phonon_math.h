#ifndef PHONON_MATH_H
#define PHONON_MATH_H

#include <stddef.h>

/* Private native math boundary matching the pinned Torch CPU vector operators.
 * Buffers are borrowed, finite model inputs are validated by inference callers.
 * Full four-element groups use pinned SLEEF; scalar tails use the system libm.
 * No function allocates. Sincos outputs must not overlap input or each other. */
void phonon_math_silu(float *values, size_t count);
void phonon_math_sincos(const float *input, size_t count, float *cosines, float *sines);
void phonon_math_exp4(float values[4]);

#endif

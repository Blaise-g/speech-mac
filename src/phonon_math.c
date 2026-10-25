#include "phonon_math.h"

#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

/* Unmodified generated upstream code; see vendor/sleef/provenance.json and
 * LICENSES/sleef-BSL-1.0.txt. Keep warning exceptions local to this include. */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-function"
#include "../vendor/sleef/sleefinline_advsimd.h"
#pragma clang diagnostic pop

#pragma STDC FP_CONTRACT OFF

void phonon_math_silu(float *values, size_t count) {
    size_t vector_end = count - count % 4;
    float32x4_t one = vdupq_n_f32(1.0f);
    for (size_t i = 0; i < vector_end; i += 4) {
        float32x4_t x = vld1q_f32(values + i);
        float32x4_t denominator = vaddq_f32(one, Sleef_expf4_u10advsimd(vnegq_f32(x)));
        vst1q_f32(values + i, vdivq_f32(x, denominator));
    }
    for (size_t i = vector_end; i < count; i++)
        values[i] = values[i] / (1.0f + expf(-values[i]));
}

void phonon_math_sincos(const float *input, size_t count, float *cosines, float *sines) {
    size_t vector_end = count - count % 4;
    for (size_t i = 0; i < vector_end; i += 4) {
        float32x4_t x = vld1q_f32(input + i);
        vst1q_f32(cosines + i, Sleef_cosf4_u10advsimd(x));
        vst1q_f32(sines + i, Sleef_sinf4_u10advsimd(x));
    }
    for (size_t i = vector_end; i < count; i++) {
        cosines[i] = cosf(input[i]);
        sines[i] = sinf(input[i]);
    }
}

void phonon_math_exp4(float values[4]) {
    vst1q_f32(values, Sleef_expf4_u10advsimd(vld1q_f32(values)));
}

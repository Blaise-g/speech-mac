#include "phonon_math.h"

#include <stdio.h>
#include <string.h>

#define CHECK(value)                                                                               \
    do {                                                                                           \
        if (!(value)) {                                                                            \
            fprintf(stderr, "failed: %s:%d: %s\n", __FILE__, __LINE__, #value);                    \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)

int main(void) {
    /* Independent locked Torch 2.7.1 CPU results, preserved as exact binary32
     * hex literals. Nine elements exercise two NEON groups and scalar tails.
     * Phase/exp cases include concrete system-libm/SLEEF rounding differences. */
    float silu[] = {-0x1.0000000000000p+3f, -0x1.0000000000000p+0f,
                    -0x1.0000000000000p-2f, 0x0.0p+0f,
                    0x1.0000000000000p-2f,  0x1.0000000000000p+0f,
                    0x1.0000000000000p+3f,  0x1.0000000000000p+4f,
                    0x1.f9a6b60000000p-4f};
    const float expected_silu[] = {-0x1.5fa3de0000000p-9f, -0x1.1365620000000p-2f,
                                   -0x1.c054ce0000000p-4f, 0x0.0p+0f,
                                   0x1.1fd59a0000000p-3f,  0x1.764d500000000p-1f,
                                   0x1.ffd40c0000000p+2f,  0x1.fffffc0000000p+3f,
                                   0x1.0c69560000000p-4f};
    phonon_math_silu(silu, 9);
    CHECK(memcmp(silu, expected_silu, sizeof(silu)) == 0);
    const float phases[] = {-0x1.ff70000000000p+2f, -0x1.ff9c000000000p+2f, 0x1.0000000000000p+0f,
                            -0x1.0000000000000p-1f, 0x1.fff0000000000p+12f, 0x1.0624de0000000p-10f,
                            0x1.4000000000000p+3f,  -0x1.9000000000000p+6f, 0x1.f9a6b60000000p-4f};
    const float expected_cos[] = {
        -0x1.182a120000000p-3f, -0x1.1d9c920000000p-3f, 0x1.14a2820000000p-1f,
        0x1.c152800000000p-1f,  -0x1.4af3b20000000p-1f, 0x1.fffff00000000p-1f,
        -0x1.ad9ac80000000p-1f, 0x1.b981dc0000000p-1f,  0x1.fc1a820000000p-1f};
    const float expected_sin[] = {
        -0x1.fb2fc60000000p-1f, -0x1.faff260000000p-1f, 0x1.aed5480000000p-1f,
        -0x1.eaee880000000p-2f, -0x1.86a8d40000000p-1f, 0x1.0624dc0000000p-10f,
        -0x1.1689f00000000p-1f, 0x1.03425c0000000p-1f,  0x1.f85e2c0000000p-4f};
    float cosines[9], sines[9];
    phonon_math_sincos(phases, 9, cosines, sines);
    CHECK(memcmp(cosines, expected_cos, sizeof(cosines)) == 0);
    CHECK(memcmp(sines, expected_sin, sizeof(sines)) == 0);
    float exponentials[] = {-0x1.fffc000000000p+2f, -0x1.0000000000000p+0f, 0x0.0p+0f,
                            0x1.0000000000000p+2f};
    const float expected_exp[] = {0x1.5fd80c0000000p-12f, 0x1.78b5640000000p-2f,
                                  0x1.0000000000000p+0f, 0x1.b4c9040000000p+5f};
    phonon_math_exp4(exponentials);
    CHECK(memcmp(exponentials, expected_exp, sizeof(exponentials)) == 0);
    phonon_math_silu(NULL, 0);
    phonon_math_sincos(NULL, 0, NULL, NULL);
    puts("Phonon native math matches pinned vector operators and scalar tails");
    return 0;
}

#include "phonon_q8.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

int main(void) {
    PhononQ8Block blocks[2] = {{.scale_bits = 0x3800U}, {.scale_bits = 0x4000U}};
    float input[64];
    for (size_t i = 0; i < 64; i++)
        input[i] = 1.0f;
    for (size_t i = 0; i < 32; i++)
        blocks[0].values[i] = (int8_t)((int)i - 16);
    for (size_t i = 0; i < 32; i++)
        blocks[1].values[i] = 1;
    float value = -99.0f;
    assert(phonon_q8_dot(blocks, input, 64, &value));
    /* GGUF Q8_0 uses d * sum(q[i] * x[i]); expected from the encoded
       half scales 0.5 and 2.0 and the literal integer blocks above. */
    assert(value == 56.0f);

    PhononQ8Block matrix[4] = {blocks[0], blocks[1], blocks[1], blocks[0]};
    float output[2] = {0.0f, 0.0f};
    assert(phonon_q8_matvec(matrix, input, 2, 64, output));
    assert(output[0] == 56.0f && output[1] == 56.0f);
    assert(!phonon_q8_dot(blocks, input, 63, &value));
    assert(!phonon_q8_matvec(blocks, input, 1, 63, output));
    assert(!phonon_q8_dot(NULL, input, 32, &value));
    PhononQ8Block rows[4] = {{.scale_bits = 0x3800U},
                             {.scale_bits = 0x4000U},
                             {.scale_bits = 0x3800U},
                             {.scale_bits = 0x4000U}};
    float asymmetric[64], row_output[2];
    for (size_t i = 0; i < 32; i++) {
        asymmetric[i] = 1.0f;
        asymmetric[i + 32] = 2.0f;
        rows[0].values[i] = 1;
        rows[1].values[i] = 2;
        rows[2].values[i] = -1;
        rows[3].values[i] = 3;
    }
    assert(phonon_q8_matvec(rows, asymmetric, 2, 64, row_output));
    assert(row_output[0] == 272.0f && row_output[1] == 368.0f);
    assert(!phonon_q8_matvec(rows, asymmetric, 2, 64, asymmetric));
    assert(!phonon_q8_matvec(rows, asymmetric, 2, 64, (float *)(void *)rows));
    blocks[0].scale_bits = 0x7c00U;
    assert(!phonon_q8_dot(blocks, input, 32, &value));
    blocks[0].scale_bits = 0x0001U;
    for (size_t i = 0; i < 32; i++)
        blocks[0].values[i] = 1;
    assert(phonon_q8_dot(blocks, input, 32, &value));
    assert(value == 0x1p-19f);
    input[0] = INFINITY;
    assert(!phonon_q8_dot(blocks, input, 32, &value));
    puts("phonon Q8 kernel tests passed");
    return 0;
}

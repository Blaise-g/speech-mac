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
    puts("phonon Q8 kernel tests passed");
    return 0;
}

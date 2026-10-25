#include "phonon_q8.h"

#include <math.h>
#include <stdint.h>

static float half_to_float(uint16_t bits) {
    uint32_t sign = (uint32_t)(bits & 0x8000U) << 16;
    uint32_t exponent = (bits >> 10) & 0x1fU;
    uint32_t mantissa = bits & 0x3ffU;
    uint32_t result;
    if (exponent == 0) {
        if (mantissa == 0) {
            result = sign;
        } else {
            int shift = 0;
            while ((mantissa & 0x400U) == 0) {
                mantissa <<= 1;
                shift++;
            }
            mantissa &= 0x3ffU;
            result = sign | ((uint32_t)(127 - 14 - shift) << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1fU) {
        result = sign | 0x7f800000U | (mantissa << 13);
    } else {
        result = sign | ((exponent + 112U) << 23) | (mantissa << 13);
    }
    union {
        uint32_t bits;
        float value;
    } converted = {result};
    return converted.value;
}

int phonon_q8_dot(const PhononQ8Block *blocks, const float *input, size_t elements, float *result) {
    if (blocks == NULL || input == NULL || result == NULL || elements == 0 || elements % 32 != 0) {
        return 0;
    }
    float sum = 0.0f;
    for (size_t block = 0; block < elements / 32; block++) {
        float scale = half_to_float(blocks[block].scale_bits);
        if (!isfinite(scale))
            return 0;
        float partial = 0.0f;
        for (size_t i = 0; i < 32; i++) {
            if (!isfinite(input[block * 32 + i]))
                return 0;
            partial += (float)blocks[block].values[i] * input[block * 32 + i];
        }
        sum += scale * partial;
    }
    *result = sum;
    return 1;
}

int phonon_q8_matvec(const PhononQ8Block *weights, const float *input, size_t rows, size_t columns,
                     float *output) {
    if (weights == NULL || input == NULL || output == NULL || rows == 0 || columns == 0 ||
        columns % 32 != 0 || rows > SIZE_MAX / (columns / 32))
        return 0;
    size_t blocks_per_row = columns / 32;
    for (size_t row = 0; row < rows; row++) {
        if (!phonon_q8_dot(weights + row * blocks_per_row, input, columns, output + row))
            return 0;
    }
    return 1;
}

#include "phonon_conv.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "failed %s:%d: %s\n", __FILE__, __LINE__, #x);                         \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

static uint16_t bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    CHECK(!(u & 65535));
    return (uint16_t)(u >> 16);
}
static PhononTensorView view(uint16_t *p, size_t a, size_t b, size_t k) {
    PhononTensorView v = {.dtype = PHONON_PACK_BF16,
                          .rank = k ? 3 : 1,
                          .shape = {a, b, k},
                          .data = p,
                          .byte_count = 2 * a * (k ? b * k : 1)};
    return v;
}
static void compare(const float *a, const float *b, size_t n) {
    float worst = 0;
    for (size_t i = 0; i < n; i++) {
        float error = fabsf(a[i] - b[i]);
        if (error > worst)
            worst = error;
        if (!isfinite(a[i]) || !isfinite(b[i]) || error > 1e-5f + 1e-3f * fabsf(b[i])) {
            fprintf(stderr, "parity failure at %zu native %.9g Torch %.9g error %.9g\n", i, a[i],
                    b[i], error);
            abort();
        }
    }
    printf(" max_abs=%.9g", worst);
}
/* Independent scalar cross-correlation / scatter definition; asymmetric exact
 * binary fractions make reversed taps, wrong phase and repeated bias visible. */
static void tiny(int trans, size_t stride) {
    size_t kernel = trans ? 2 * stride : 3, in = 2, out = 2, n = 7;
    uint16_t w[48], bias[2];
    float wf[48], bf[2] = {0.25f, -0.5f}, x[14], expected[84] = {0}, actual[84], part[84],
                  reset[84];
    for (size_t i = 0; i < in * out * kernel; i++) {
        wf[i] = (float)((int)(i % 11) - 5) * 0.125f;
        w[i] = bf16(wf[i]);
    }
    for (size_t i = 0; i < 2; i++)
        bias[i] = bf16(bf[i]);
    for (size_t i = 0; i < n * in; i++)
        x[i] = (float)((int)(i % 9) - 4) * 0.25f;
    x[12] = 0;
    x[13] = 0;
    for (size_t t = 0; t < n * stride; t++)
        for (size_t o = 0; o < out; o++)
            expected[t * out + o] = bf[o];
    if (trans) {
        for (size_t t = 0; t < n; t++)
            for (size_t c = 0; c < in; c++)
                for (size_t o = 0; o < out; o++)
                    for (size_t k = 0; k < kernel && t * stride + k < n * stride; k++)
                        expected[(t * stride + k) * out + o] +=
                            x[t * in + c] * wf[(c * out + o) * kernel + k];
    } else {
        for (size_t t = 0; t < n; t++)
            for (size_t o = 0; o < out; o++)
                for (size_t c = 0; c < in; c++)
                    for (size_t k = 0; k < kernel; k++)
                        if (t + k >= kernel - 1)
                            expected[t * out + o] +=
                                x[(t + k - (kernel - 1)) * in + c] * wf[(o * in + c) * kernel + k];
    }
    PhononTensorView weight = view(w, 2, 2, kernel), bv = view(bias, 2, 0, 0);
    PhononConvModel *m = NULL;
    PhononConvState *s = NULL;
    CHECK(phonon_conv_model_create(&weight, &bv, stride, trans, &m));
    CHECK(phonon_conv_state_create(m, &s));
    CHECK(phonon_conv_input_channels(m) == 2 && phonon_conv_output_channels(m) == 2 &&
          phonon_conv_stride(m) == stride);
    CHECK(phonon_conv_process(s, NULL, 0, NULL, 0));
    CHECK(!phonon_conv_process(s, x, SIZE_MAX, actual, SIZE_MAX));
    CHECK(!phonon_conv_process(s, x, n, actual, n * stride - 1));
    CHECK(!phonon_conv_process(s, x, n, x, n * stride));
    float saved = x[1];
    x[1] = NAN;
    CHECK(!phonon_conv_process(s, x, n, actual, n * stride));
    x[1] = saved;
    CHECK(phonon_conv_process(s, x, n, actual, n * stride));
    CHECK(memcmp(actual, expected, n * stride * out * sizeof(float)) == 0);
    phonon_conv_state_reset(s);
    size_t off = 0, chunks[] = {1, 2, 1, 3};
    for (size_t i = 0; i < 4; i++) {
        size_t count = chunks[i];
        CHECK(
            phonon_conv_process(s, x + off * in, count, part + off * stride * out, count * stride));
        off += count;
    }
    CHECK(memcmp(actual, part, n * stride * out * sizeof(float)) == 0);
    phonon_conv_state_reset(s);
    CHECK(phonon_conv_process(s, x, 1, reset, stride));
    phonon_conv_state_reset(s);
    CHECK(phonon_conv_process(s, x + in, n - 1, reset + stride * out, (n - 1) * stride));
    CHECK(memcmp(actual, reset, n * stride * out * sizeof(float)) != 0);
    phonon_conv_state_destroy(s);
    phonon_conv_model_destroy(m);
    weight.shape[0] = UINT64_MAX;
    CHECK(!phonon_conv_model_create(&weight, &bv, stride, trans, &m));
    weight.shape[0] = 2;
    weight.shape[2] = UINT64_MAX;
    CHECK(!phonon_conv_model_create(&weight, &bv, stride, trans, &m));
    weight.shape[2] = kernel;
    weight.byte_count--;
    CHECK(!phonon_conv_model_create(&weight, &bv, stride, trans, &m));
    weight.byte_count++;
    bv.shape[0] = 1;
    CHECK(!phonon_conv_model_create(&weight, &bv, stride, trans, &m));
}
static void overflow(void) {
    uint16_t w[1] = {0x7f7f}, b[1] = {0};
    PhononTensorView weight = view(w, 1, 1, 1), bias = view(b, 1, 0, 0);
    PhononConvModel *m = NULL;
    PhononConvState *s = NULL;
    float x = FLT_MAX, y = 0;
    CHECK(phonon_conv_model_create(&weight, &bias, 1, 0, &m));
    CHECK(phonon_conv_state_create(m, &s));
    CHECK(!phonon_conv_process(s, &x, 1, &y, 1));
    x = 0;
    CHECK(!phonon_conv_process(s, &x, 1, &y, 1));
    phonon_conv_state_reset(s);
    CHECK(phonon_conv_process(s, &x, 1, &y, 1));
    CHECK(y == 0);
    phonon_conv_state_destroy(s);
    phonon_conv_model_destroy(m);
    w[0] = 0x7f80;
    CHECK(!phonon_conv_model_create(&weight, &bias, 1, 0, &m));
}
static double reference_elu(double x) { return x <= 0 ? exp(x) - 1 : x; }
static void residual_tiny(void) {
    uint16_t w3[6], w1[2], b3[1] = {0x3e80}, b1[2] = {0x3e00, 0xbe80};
    const float weights[6] = {0.5f, -0.25f, 0.125f, -0.5f, 0.75f, 0.25f};
    for (size_t i = 0; i < 6; i++)
        w3[i] = bf16(weights[i]);
    w1[0] = bf16(0.5f);
    w1[1] = bf16(-0.25f);
    PhononTensorView a = view(w3, 1, 2, 3), ab = view(b3, 1, 0, 0), b = view(w1, 2, 1, 1),
                     bb = view(b1, 2, 0, 0);
    PhononConvModel *m3 = NULL, *m1 = NULL;
    PhononResidualState *s = NULL;
    CHECK(phonon_conv_model_create(&a, &ab, 1, 0, &m3));
    CHECK(phonon_conv_model_create(&b, &bb, 1, 0, &m1));
    CHECK(!phonon_residual_state_create(m1, m3, &s));
    CHECK(phonon_residual_state_create(m3, m1, &s));
    float x[10] = {-1, 0.5f, 2, -0.25f, -0.5f, -2, 0, 1, 0, 0}, expected[10], y[10], part[10];
    for (size_t t = 0; t < 5; t++) {
        double hidden = 0.25;
        for (size_t c = 0; c < 2; c++)
            for (size_t k = 0; k < 3; k++)
                if (t + k >= 2)
                    hidden += reference_elu(x[(t + k - 2) * 2 + c]) * weights[c * 3 + k];
        expected[t * 2] = (float)(x[t * 2] + reference_elu(hidden) * 0.5 + 0.125);
        expected[t * 2 + 1] = (float)(x[t * 2 + 1] - reference_elu(hidden) * 0.25 - 0.25);
    }
    CHECK(phonon_residual_process(s, NULL, 0, NULL, 0));
    CHECK(!phonon_residual_process(s, x, 5, y, 4));
    CHECK(!phonon_residual_process(s, x, 5, x, 5));
    float alias[16] = {0};
    CHECK(!phonon_residual_process(s, alias + 6, 5, alias, 5));
    float saved = x[7];
    x[7] = INFINITY;
    CHECK(!phonon_residual_process(s, x, 5, y, 5));
    x[7] = saved;
    CHECK(!phonon_residual_process(s, x, SIZE_MAX, y, SIZE_MAX));
    CHECK(phonon_residual_process(s, x, 5, y, 5));
    compare(y, expected, 10);
    puts(" tiny residual");
    phonon_residual_state_reset(s);
    for (size_t t = 0; t < 5; t++)
        CHECK(phonon_residual_process(s, x + t * 2, 1, part + t * 2, 1));
    CHECK(!memcmp(y, part, sizeof(y)));
    phonon_residual_state_reset(s);
    CHECK(phonon_residual_process(s, x, 5, part, 5));
    CHECK(!memcmp(y, part, sizeof(y)));
    phonon_residual_state_destroy(s);
    phonon_conv_model_destroy(m3);
    /* A permitted finite BF16 model/input can overflow during processing.
     * The residual request must stay poisoned until explicitly reset. */
    for (size_t i = 0; i < 6; i++)
        w3[i] = 0x7f7f;
    CHECK(phonon_conv_model_create(&a, &ab, 1, 0, &m3));
    CHECK(phonon_residual_state_create(m3, m1, &s));
    float extreme[2] = {FLT_MAX, FLT_MAX};
    CHECK(!phonon_residual_process(s, extreme, 1, y, 1));
    extreme[0] = extreme[1] = 0;
    CHECK(!phonon_residual_process(s, extreme, 1, y, 1));
    phonon_residual_state_reset(s);
    CHECK(phonon_residual_process(s, extreme, 1, y, 1));
    phonon_residual_state_destroy(s);
    phonon_conv_model_destroy(m3);
    phonon_conv_model_destroy(m1);
}
static uint32_t read_u32(FILE *f) {
    unsigned char b[4];
    CHECK(fread(b, 1, 4, f) == 4);
    return (uint32_t)b[0] | (uint32_t)b[1] << 8 | (uint32_t)b[2] << 16 | (uint32_t)b[3] << 24;
}
static PhononTensorView read_view(FILE *f) {
    PhononTensorView v = {.dtype = PHONON_PACK_BF16};
    v.rank = read_u32(f);
    size_t count = 1;
    for (size_t i = 0; i < 3; i++) {
        v.shape[i] = read_u32(f);
        if (i < v.rank) {
            CHECK(v.shape[i] && v.shape[i] <= 512);
            count *= v.shape[i];
        }
    }
    CHECK(v.rank == 1 || v.rank == 3);
    v.byte_count = count * 2;
    void *p = malloc(v.byte_count);
    CHECK(p);
    CHECK(fread(p, 1, v.byte_count, f) == v.byte_count);
    v.data = p;
    return v;
}
static void real(void) {
    const char *fixture = getenv("PHONON_CONV_FIXTURE"), *pack_path = getenv("PHONON_NATIVE_PACK");
    if (!fixture && !pack_path) {
        puts("SKIP real q8/BF16 SEANet operator seams: select PHONON_CONV_FIXTURE and "
             "PHONON_NATIVE_PACK");
        return;
    }
    CHECK(fixture && pack_path);
    FILE *f = fopen(fixture, "rb");
    CHECK(f);
    PhononPack *pack = NULL;
    CHECK(phonon_pack_open(pack_path, &pack));
    char magic[8];
    CHECK(fread(magic, 1, 8, f) == 8 && !memcmp(magic, "PHCONV1\0", 8));
    uint32_t records = read_u32(f);
    CHECK(records == 8);
    for (uint32_t r = 0; r < records; r++) {
        size_t index = read_u32(f), kind = read_u32(f), stride = read_u32(f), in = read_u32(f),
               out = read_u32(f), n = read_u32(f), nw = read_u32(f);
        CHECK(kind <= 2 && nw == (kind == 2 ? 4 : 2) && n == 17 && in <= 512 && out <= 512);
        PhononTensorView v[4];
        for (size_t j = 0; j < nw; j++) {
            v[j] = read_view(f);
            char name[128];
            PhononTensorView actual;
            if (kind == 2)
                snprintf(name, sizeof(name), "mimi.decoder.model.%zu.block.%zu.conv.%s", index,
                         j < 2 ? 1UL : 3UL, j % 2 ? "bias" : "weight");
            else
                snprintf(name, sizeof(name), "mimi.decoder.model.%zu.%s.%s", index,
                         kind ? "convtr" : "conv", j % 2 ? "bias" : "weight");
            CHECK(phonon_pack_tensor(pack, name, &actual));
            CHECK(actual.rank == v[j].rank && actual.byte_count == v[j].byte_count);
            CHECK(!memcmp(actual.shape, v[j].shape, v[j].rank * sizeof(uint64_t)));
            CHECK(!memcmp(actual.data, v[j].data, v[j].byte_count));
        }
        size_t input_count = n * in, output_count = n * stride * out;
        float *x = malloc(input_count * 4), *expected = malloc(output_count * 4),
              *y = malloc(output_count * 4), *partition = malloc(output_count * 4),
              *reset = malloc(output_count * 4);
        CHECK(x && expected && y && partition && reset);
        CHECK(fread(x, 4, input_count, f) == input_count);
        CHECK(fread(expected, 4, output_count, f) == output_count);
        PhononConvModel *a = NULL, *b = NULL;
        PhononConvState *s = NULL;
        PhononResidualState *res = NULL;
        CHECK(phonon_conv_model_create(v, v + 1, stride, kind == 1, &a));
        if (kind == 2) {
            CHECK(phonon_conv_model_create(v + 2, v + 3, 1, 0, &b));
            CHECK(phonon_residual_state_create(a, b, &res));
        } else
            CHECK(phonon_conv_state_create(a, &s));
        for (int mode = 0; mode < 3; mode++) {
            if (kind == 2)
                phonon_residual_state_reset(res);
            else
                phonon_conv_state_reset(s);
            float *dest = mode == 0 ? y : mode == 1 ? partition : reset;
            size_t offset = 0;
            while (offset < n) {
                size_t count = mode == 0   ? n
                               : mode == 1 ? (offset == 0   ? 1
                                              : offset == 1 ? 7
                                                            : n - offset)
                                           : 1;
                if (mode == 2) {
                    if (kind == 2)
                        phonon_residual_state_reset(res);
                    else
                        phonon_conv_state_reset(s);
                }
                int ok = kind == 2
                             ? phonon_residual_process(res, x + offset * in, count,
                                                       dest + offset * stride * out, count * stride)
                             : phonon_conv_process(s, x + offset * in, count,
                                                   dest + offset * stride * out, count * stride);
                CHECK(ok);
                offset += count;
            }
        }
        printf("SEANet layer %zu kind %zu", index, kind);
        compare(y, expected, output_count);
        puts("");
        CHECK(!memcmp(y, partition, output_count * 4));
        CHECK(memcmp(y, reset, output_count * 4));
        phonon_conv_state_destroy(s);
        phonon_residual_state_destroy(res);
        phonon_conv_model_destroy(a);
        phonon_conv_model_destroy(b);
        for (size_t j = 0; j < nw; j++)
            free((void *)v[j].data);
        free(x);
        free(expected);
        free(y);
        free(partition);
        free(reset);
    }
    CHECK(fgetc(f) == EOF);
    fclose(f);
    phonon_pack_close(pack);
}
int main(void) {
    tiny(0, 1);
    for (size_t stride = 4; stride <= 6; stride++)
        tiny(1, stride);
    residual_tiny();
    overflow();
    real();
    puts("phonon convolution tests passed");
    return 0;
}

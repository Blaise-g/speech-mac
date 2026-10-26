#include "phonon_pcm_test.h"
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
static PhononTensorView weight(size_t rank, size_t a, size_t b, size_t c) {
    size_t count = a * (rank > 1 ? b : 1) * (rank > 2 ? c : 1);
    PhononTensorView v = {.dtype = PHONON_PACK_BF16,
                          .rank = (unsigned)rank,
                          .shape = {a, b, c},
                          .data = calloc(count, 2),
                          .byte_count = count * 2};
    CHECK(v.data);
    return v;
}
static void put(PhononTensorView *v, size_t at, float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    CHECK(!(u & 65535));
    ((uint16_t *)v->data)[at] = (uint16_t)(u >> 16);
}
static PhononConvModel *conv(size_t in, size_t out, size_t kernel, size_t stride, int route) {
    PhononTensorView w = weight(3, stride > 1 ? in : out, stride > 1 ? out : in, kernel),
                     b = weight(1, out, 0, 0);
    if (route) {
        put(&w, stride > 1 ? 0 : kernel - 1, 1);
        if (stride > 1) {
            for (size_t i = 1; i < stride; i++)
                put(&w, i, 1);
            for (size_t i = stride; i < kernel; i++)
                put(&w, i, .5f);
        } else if (out == 1)
            put(&w, kernel - 2, .25f);
    }
    PhononConvModel *m = NULL;
    CHECK(phonon_conv_model_create(&w, &b, stride, stride > 1, &m));
    free((void *)w.data);
    free((void *)b.data);
    return m;
}
static void parity(const float *a, const float *b, size_t n, const char *label) {
    double square = 0, power = 0;
    float maximum = 0;
    for (size_t i = 0; i < n; i++) {
        CHECK(isfinite(a[i]) && isfinite(b[i]));
        double d = (double)a[i] - b[i];
        square += d * d;
        power += (double)b[i] * b[i];
        if (fabs(d) > maximum)
            maximum = (float)fabs(d);
    }
    double normalized = power ? sqrt(square / power) : sqrt(square / (double)n);
    printf("%s samples=%zu nrmse=%.9g max_abs=%.9g\n", label, n, normalized, maximum);
    CHECK(normalized <= .001 && maximum <= .002f);
}
static float scalar_elu(float x) { return x <= 0 ? expf(x) - 1 : x; }
static void synthetic(void) {
    PhononMimiModel mimi = {.config = phonon_mimi_config(), .stride = 16};
    mimi.projection = weight(3, 512, 32, 1);
    put(&mimi.projection, 0, 1);
    mimi.upsample = weight(3, 512, 1, 32);
    for (size_t i = 0; i < 16; i++)
        put(&mimi.upsample, i, 1);
    for (size_t l = 0; l < 2; l++)
        for (size_t f = 0; f < 10; f++) {
            mimi.layers[l][f] = weight(f < 4 ? 2 : 1,
                                       f == 0   ? 1536
                                       : f == 2 ? 2048
                                                : 512,
                                       f == 3 ? 2048 : 512, 0);
        }
    PhononConvModel *c[11];
    c[0] = conv(512, 512, 7, 1, 1);
    const size_t strides[] = {6, 5, 4};
    for (size_t i = 0, ch = 512; i < 3; i++, ch /= 2) {
        c[1 + i * 3] = conv(ch, ch / 2, 2 * strides[i], strides[i], 1);
        c[2 + i * 3] = conv(ch / 2, ch / 4, 3, 1, 0);
        c[3 + i * 3] = conv(ch / 4, ch / 2, 1, 1, 0);
    }
    c[10] = conv(64, 1, 3, 1, 1);
    PhononPCMModel *m = NULL;
    PhononPCMState *s = NULL, *second = NULL;
    CHECK(phonon_pcm_test_model(&mimi, c, &m));
    CHECK(phonon_pcm_state_create(m, &s));
    CHECK(phonon_pcm_state_create(m, &second));
    enum { FRAMES = 3, COUNT = FRAMES * 1920 };
    float x[FRAMES * 32] = {0}, raw[COUNT], final[COUNT], expected[COUNT], previous[3] = {0},
                     last = 0;
    x[0] = -.5f;
    x[32] = .25f;
    x[64] = 0;
    /* Independent sparse scalar graph: channel zero route; causal transpose
     * scatter with retained half overlap, identity residuals, final 3tap FIR.
     * This tests true negative ELUs, overlap, startup crop and zero latent tail. */
    size_t at = 0;
    for (size_t t = 0; t < FRAMES * 16; t++) {
        float a = scalar_elu(x[t / 16 * 32]);
        for (size_t i = 0; i < 6; i++) {
            float b = scalar_elu(a + previous[0]);
            for (size_t j = 0; j < 5; j++) {
                float d = scalar_elu(b + previous[1]);
                for (size_t k = 0; k < 4; k++) {
                    float e = scalar_elu(d + previous[2]);
                    expected[at++] = e + .25f * last;
                    last = e;
                }
                previous[2] = .5f * d;
            }
            previous[1] = .5f * b;
        }
        previous[0] = .5f * a;
    }
    CHECK(at == COUNT);
    CHECK(phonon_pcm_process(s, NULL, 0, NULL, 0, NULL, 0));
    CHECK(!phonon_pcm_samples(s));
    CHECK(!phonon_pcm_process(s, x, FRAMES, raw, COUNT - 1, final, COUNT));
    CHECK(!phonon_pcm_process(s, x, FRAMES, raw, COUNT, final, COUNT - 1));
    CHECK(!phonon_pcm_process(s, x, SIZE_MAX, raw, SIZE_MAX, NULL, 0));
    CHECK(!phonon_pcm_process(s, x, 1, x, 1920, NULL, 0));
    CHECK(!phonon_pcm_process(s, x, 1, raw, 1920, raw + 1, 1920));
    x[1] = NAN;
    CHECK(!phonon_pcm_process(s, x, 1, raw, 1920, NULL, 0));
    x[1] = 0;
    CHECK(!phonon_pcm_samples(s));
    CHECK(phonon_pcm_process(s, x, FRAMES, raw, COUNT, final, COUNT));
    parity(raw, expected, COUNT, "synthetic scalar complete decoder");
    CHECK(!memcmp(raw, final, sizeof(raw)));
    CHECK(phonon_pcm_samples(s) == COUNT);
    CHECK(!phonon_pcm_samples(second));
    CHECK(phonon_pcm_finish(s));
    CHECK(phonon_pcm_samples(s) == COUNT);
    CHECK(!phonon_pcm_finish(s));
    CHECK(!phonon_pcm_process(s, NULL, 0, NULL, 0, NULL, 0));
    phonon_pcm_state_reset(s);
    for (size_t i = 0; i < FRAMES; i++)
        CHECK(phonon_pcm_process(s, x + i * 32, 1, final + i * 1920, 1920, final + i * 1920, 1920));
    CHECK(!memcmp(raw, final, sizeof(raw)));
    CHECK(!phonon_pcm_process(s, x, PHONON_MIMI_MAX_FRAMES - FRAMES + 1, raw, SIZE_MAX, NULL, 0));
    CHECK(phonon_pcm_samples(s) == COUNT);
    CHECK(phonon_pcm_finish(second));
    CHECK(!phonon_pcm_samples(second));
    phonon_pcm_state_reset(second);
    CHECK(phonon_pcm_process(second, x, 1, final, 1920, NULL, 0));
    CHECK(!memcmp(raw, final, 1920 * 4));
    /* Finite enormous input overflows LayerNorm intermediate in Mimi. */
    phonon_pcm_state_reset(second);
    x[0] = FLT_MAX;
    CHECK(!phonon_pcm_process(second, x, 1, final, 1920, NULL, 0));
    x[0] = -.5f;
    CHECK(!phonon_pcm_process(second, x, 1, final, 1920, NULL, 0));
    CHECK(!phonon_pcm_finish(second));
    phonon_pcm_state_reset(second);
    CHECK(phonon_pcm_process(second, x, 1, final, 1920, NULL, 0));
    phonon_pcm_state_destroy(s);
    phonon_pcm_state_destroy(second);
    phonon_pcm_model_destroy(m);
    free((void *)mimi.projection.data);
    free((void *)mimi.upsample.data);
    for (size_t l = 0; l < 2; l++)
        for (size_t f = 0; f < 10; f++)
            free((void *)mimi.layers[l][f].data);
}
static uint32_t read_u32(FILE *f) {
    unsigned char b[4];
    CHECK(fread(b, 1, 4, f) == 4);
    return (uint32_t)b[0] | (uint32_t)b[1] << 8 | (uint32_t)b[2] << 16 | (uint32_t)b[3] << 24;
}
static void real(void) {
    const char *fixture = getenv("PHONON_PCM_FIXTURE");
    if (!fixture) {
        puts("SKIP actual q8 full PCM: select PHONON_PCM_FIXTURE (requires PHONON_NATIVE_PACK)");
        return;
    }
    const char *pack_path = getenv("PHONON_NATIVE_PACK");
    CHECK(pack_path);
    FILE *f = fopen(fixture, "rb");
    CHECK(f);
    char magic[8];
    CHECK(fread(magic, 1, 8, f) == 8 && !memcmp(magic, "PHPCM1\0\0", 8));
    PhononPack *pack = NULL;
    PhononPCMModel *m = NULL;
    PhononPCMState *s = NULL;
    CHECK(phonon_pack_open(pack_path, &pack));
    CHECK(phonon_pcm_model_create(pack, &m));
    CHECK(phonon_pcm_state_create(m, &s));
    size_t records = read_u32(f);
    CHECK(records == 11);
    for (size_t r = 0; r < records; r++) {
        size_t frames = read_u32(f), count = read_u32(f);
        CHECK(frames && frames <= 263 && count == frames * 1920);
        float *x = malloc(frames * 32 * 4), *raw = malloc(count * 4), *final = malloc(count * 4),
              *actual = malloc(count * 4), *whole = malloc(count * 4),
              *final_actual = malloc(count * 4);
        CHECK(x && raw && final && actual && whole && final_actual);
        CHECK(fread(x, 4, frames * 32, f) == frames * 32);
        CHECK(fread(raw, 4, count, f) == count);
        CHECK(fread(final, 4, count, f) == count);
        for (size_t mode = 0; mode < 3; mode++) {
            phonon_pcm_state_reset(s);
            size_t offset = 0;
            while (offset < frames) {
                size_t n = mode == 0     ? frames
                           : mode == 1   ? 1
                           : offset == 0 ? 1
                           : offset == 1 ? 7
                           : offset == 8 ? 113
                                         : 142;
                if (n > frames - offset)
                    n = frames - offset;
                CHECK(phonon_pcm_process(s, x + offset * 32, n, actual + offset * 1920, n * 1920,
                                         final_actual + offset * 1920, n * 1920));
                offset += n;
                CHECK(phonon_pcm_samples(s) == offset * 1920);
            }
            char label[80];
            snprintf(label, sizeof(label), "actual q8 record=%zu partition=%zu raw", r, mode);
            parity(actual, raw, count, label);
            parity(final_actual, final, count, "actual q8 final");
            CHECK(!memcmp(actual, final_actual, count * 4));
            if (!mode)
                memcpy(whole, actual, count * 4);
            else
                CHECK(!memcmp(whole, actual, count * 4));
            CHECK(phonon_pcm_finish(s));
            CHECK(phonon_pcm_samples(s) == count);
            CHECK(!phonon_pcm_finish(s));
        }
        free(x);
        free(raw);
        free(final);
        free(actual);
        free(whole);
        free(final_actual);
    }
    CHECK(fgetc(f) == EOF);
    fclose(f);
    phonon_pcm_state_destroy(s);
    phonon_pcm_model_destroy(m);
    phonon_pack_close(pack);
}
int main(void) {
    synthetic();
    real();
    return 0;
}

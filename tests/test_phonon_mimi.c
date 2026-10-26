#include "phonon_mimi.h"
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
static uint16_t bf(float value) {
    uint32_t bits;
    memcpy(&bits, &value, 4);
    return (uint16_t)(bits >> 16);
}
static PhononTensorView weight(size_t rank, size_t a, size_t b, size_t c) {
    size_t count = a * (rank > 1 ? b : 1) * (rank > 2 ? c : 1);
    PhononTensorView w = {.dtype = PHONON_PACK_BF16,
                          .rank = (unsigned)rank,
                          .shape = {a, b, c},
                          .byte_count = count * 2};
    w.data = calloc(count, 2);
    CHECK(w.data);
    return w;
}
static void put(PhononTensorView *w, size_t i, float x) { ((uint16_t *)w->data)[i] = bf(x); }
static void compare(const float *actual, const float *expected, size_t count, float rtol,
                    float atol, const char *label) {
    double maximum = 0;
    for (size_t i = 0; i < count; i++) {
        double error = fabs((double)actual[i] - expected[i]);
        if (error > maximum)
            maximum = error;
        if (!isfinite(actual[i]) || error > atol + rtol * fabs((double)expected[i])) {
            fprintf(stderr, "%s mismatch i=%zu actual=%.9g ref=%.9g error=%.9g\n", label, i,
                    actual[i], expected[i], error);
            exit(1);
        }
    }
    fprintf(stderr, "%s passed max_abs=%.9g\n", label, maximum);
}
static void upsample_test(void) {
    PhononTensorView w = weight(3, 2, 1, 4);
    for (size_t i = 0; i < 8; i++)
        put(&w, i, (float)(i + 1));
    float input[] = {2, 3}, previous[] = {5, 7}, output[4], expected[] = {17, 64, 24, 74};
    CHECK(phonon_mimi_upsample_frame(&w, input, previous, 2, 2, output));
    compare(output, expected, 4, 0, 0, "depthwise overlap");
    float zero[2] = {0}, startup[] = {2, 15, 4, 18};
    CHECK(phonon_mimi_upsample_frame(&w, input, zero, 2, 2, output));
    compare(output, startup, 4, 0, 0, "depthwise startup/right trim");
    input[0] = NAN;
    CHECK(!phonon_mimi_upsample_frame(&w, input, zero, 2, 2, output));
    CHECK(!phonon_mimi_upsample_frame(&w, zero, zero, 2, 0, output));
    w.byte_count--;
    CHECK(!phonon_mimi_upsample_frame(&w, zero, zero, 2, 2, output));
    free((void *)w.data);
}
static void normalized(const float *input, float *output) {
    double mean = 0, variance = 0;
    for (size_t i = 0; i < 512; i++)
        mean += input[i];
    mean /= 512;
    for (size_t i = 0; i < 512; i++)
        variance += ((double)input[i] - mean) * ((double)input[i] - mean);
    float scale = 1.0f / sqrtf((float)(variance / 512) + 1e-5f);
    for (size_t i = 0; i < 512; i++)
        output[i] = (float)((double)input[i] - mean) * scale;
}
static void synthetic_test(void) {
    enum { FRAMES = 19, POSITIONS = FRAMES * 16 };
    PhononMimiModel model = {.config = phonon_mimi_config(), .stride = 16};
    model.projection = weight(3, 512, 32, 1);
    model.upsample = weight(3, 512, 1, 32);
    for (size_t i = 0; i < 512; i++) {
        put(&model.projection, i * 32 + i % 32, 1);
        for (size_t k = 0; k < 32; k++)
            put(&model.upsample, i * 32 + k, k < 16 ? 1.0f : 0.5f);
    }
    for (size_t l = 0; l < 2; l++)
        for (size_t f = 0; f < 10; f++) {
            size_t rows = f == 0 ? 1536 : (f == 2 ? 2048 : 512), cols = f == 3 ? 2048 : 512;
            model.layers[l][f] = weight(f < 4 ? 2 : 1, rows, cols, 0);
            for (size_t i = 0; i < 512; i++) {
                if (f == 0)
                    put(&model.layers[l][f], (1024 + i) * 512 + i, 1);
                if (f == 1)
                    put(&model.layers[l][f], i * 512 + i, 1);
                if (f == 4 || f == 6)
                    put(&model.layers[l][f], i, 1);
                if (f == 8)
                    put(&model.layers[l][f], i, l ? 0.25f : 0.5f);
            }
        }
    float *input = calloc(FRAMES * 32, sizeof(float)),
          *actual = calloc(POSITIONS * 512, sizeof(float));
    float *expected = calloc(POSITIONS * 512, sizeof(float)),
          *norm = calloc(POSITIONS * 512, sizeof(float));
    CHECK(input && actual && expected && norm);
    for (size_t t = 0; t < FRAMES; t++)
        for (size_t i = 0; i < 32; i++)
            input[t * 32 + i] = sinf((float)(t * 7 + i) * 0.17f);
    for (size_t t = 0; t < POSITIONS; t++)
        for (size_t i = 0; i < 512; i++)
            expected[t * 512 + i] = input[t / 16 * 32 + i % 32] +
                                    (t >= 16 ? 0.5f * input[(t / 16 - 1) * 32 + i % 32] : 0);
    /* Independent scalar graph: Q=K=0 gives uniform causal mean over exactly
     * min(position+1,250) V rows; learned scales are .5 and .25, MLP=0. */
    for (size_t l = 0; l < 2; l++) {
        for (size_t t = 0; t < POSITIONS; t++)
            normalized(expected + t * 512, norm + t * 512);
        for (size_t t = 0; t < POSITIONS; t++) {
            size_t first = t >= 249 ? t - 249 : 0;
            for (size_t i = 0; i < 512; i++) {
                double sum = 0;
                for (size_t k = first; k <= t; k++)
                    sum += norm[k * 512 + i];
                expected[t * 512 + i] +=
                    (l ? 0.25f : 0.5f) * (float)(sum / (double)(t + 1 - first));
            }
        }
    }
    PhononMimiState *a = NULL, *b = NULL;
    CHECK(phonon_mimi_state_create(&model, &a));
    CHECK(phonon_mimi_state_create(&model, &b));
    CHECK(!phonon_mimi_process(a, input, FRAMES, actual, POSITIONS - 1));
    CHECK(phonon_mimi_positions(a) == 0);
    input[0] = NAN;
    CHECK(!phonon_mimi_process(a, input, FRAMES, actual, POSITIONS));
    input[0] = 0;
    CHECK(!phonon_mimi_process(a, input, PHONON_MIMI_MAX_FRAMES + 1, actual, POSITIONS));
    CHECK(!phonon_mimi_process(a, input, 1, input, 16));
    CHECK(phonon_mimi_process(a, input, FRAMES, actual, POSITIONS));
    compare(actual, expected, POSITIONS * 512, 0, 4e-6f,
            "synthetic 250-key ring and learned scales");
    CHECK(phonon_mimi_positions(a) == POSITIONS);
    /* Valid per-call count that exceeds this request's remaining budget must
     * fail before accessing the borrowed buffers and leave progress intact. */
    size_t over_remaining = PHONON_MIMI_MAX_FRAMES - FRAMES + 1;
    CHECK(!phonon_mimi_process(a, input, over_remaining, actual, over_remaining * 16));
    CHECK(phonon_mimi_positions(a) == POSITIONS);
    phonon_mimi_state_reset(a);
    for (size_t t = 0; t < FRAMES; t++)
        CHECK(phonon_mimi_process(a, input + t * 32, 1, actual + t * 16 * 512, 16));
    compare(actual, expected, POSITIONS * 512, 0, 4e-6f, "synthetic reset/framewise");
    CHECK(phonon_mimi_positions(b) == 0);
    CHECK(phonon_mimi_process(b, input, 1, actual, 16));
    compare(actual, expected, 16 * 512, 0, 4e-6f, "independent request");
    /* A corrupt model fails after processing starts, then requires explicit reset. */
    model.layers[0][0].rank = 1;
    CHECK(!phonon_mimi_process(b, input, 1, actual, 16));
    model.layers[0][0].rank = 2;
    CHECK(!phonon_mimi_process(b, input, 1, actual, 16));
    phonon_mimi_state_reset(b);
    CHECK(phonon_mimi_process(b, input, 1, actual, 16));
    PhononMimiConfig bad = phonon_mimi_config();
    bad.context = 251;
    PhononMimiModel invalid = {.config = bad, .stride = 16};
    PhononMimiState *none = NULL;
    CHECK(!phonon_mimi_state_create(&invalid, &none));
    CHECK(!none);
    CHECK(!phonon_mimi_model_load(NULL, &bad, &invalid));
    phonon_mimi_state_destroy(a);
    phonon_mimi_state_destroy(b);
    free(input);
    free(actual);
    free(expected);
    free(norm);
    free((void *)model.projection.data);
    free((void *)model.upsample.data);
    for (size_t l = 0; l < 2; l++)
        for (size_t f = 0; f < 10; f++)
            free((void *)model.layers[l][f].data);
}
static void model_test(void) {
    const char *fixture = getenv("PHONON_MIMI_FIXTURE");
    if (!fixture) {
        fprintf(stderr, "real Mimi feature parity skipped (set PHONON_MIMI_FIXTURE)\n");
        return;
    }
    const char *path = getenv("PHONON_NATIVE_PACK");
    CHECK(path);
    FILE *file = fopen(fixture, "rb");
    CHECK(file);
    PhononPack *pack = NULL;
    CHECK(phonon_pack_open(path, &pack));
    PhononMimiModel model;
    PhononMimiConfig config = phonon_mimi_config();
    CHECK(phonon_mimi_model_load(pack, &config, &model));
    config.frame_rate = 0;
    CHECK(!phonon_mimi_model_load(pack, &config, &model));
    PhononMimiState *state = NULL;
    CHECK(phonon_mimi_state_create(&model, &state));
    unsigned char magic[8];
    uint32_t runs;
    CHECK(fread(magic, 1, 8, file) == 8 && !memcmp(magic, "PHMIMI1\0", 8));
    CHECK(fread(&runs, 4, 1, file) == 1 && runs == 5);
    for (size_t run = 0; run < runs; run++) {
        uint32_t frames, expectations;
        CHECK(fread(&frames, 4, 1, file) == 1 && fread(&expectations, 4, 1, file) == 1);
        CHECK(frames == (run == 0 ? 40U : run == 1 ? 32U : run == 4 ? 257U : 263U));
        CHECK(expectations == (run == 2 ? 3U : run >= 3 ? 2U : 1U));
        size_t positions = frames * 16, count = positions * 512;
        float *latent = malloc(frames * 32 * sizeof(float)),
              *hidden = malloc(count * sizeof(float));
        float *actual = malloc(count * sizeof(float)), *expected = malloc(count * sizeof(float));
        CHECK(latent && hidden && actual && expected);
        CHECK(fread(latent, 4, frames * 32, file) == frames * 32 &&
              fread(hidden, 4, count, file) == count);
        /* Verify projection + learned overlap separately from the transformer. */
        float previous[512] = {0}, projected[512], up[16 * 512];
        PhononTensorView projection = model.projection;
        projection.rank = 2;
        for (size_t t = 0; t < frames; t++) {
            CHECK(phonon_linear(&projection, latent + t * 32, 512, 32, NULL, projected));
            CHECK(phonon_mimi_upsample_frame(&model.upsample, projected, previous, 512, 16, up));
            for (size_t i = 0; i < 16 * 512; i++)
                CHECK(fabsf(up[i] - hidden[t * 16 * 512 + i]) <=
                      1e-5f + 1e-3f * fabsf(hidden[t * 16 * 512 + i]));
            memcpy(previous, projected, sizeof(previous));
        }
        for (size_t partition = 0; partition < expectations; partition++) {
            CHECK(fread(expected, 4, count, file) == count);
            phonon_mimi_state_reset(state);
            size_t offset = 0;
            const size_t chunks[] = {1, 7, 113, 142};
            while (offset < frames) {
                size_t chunk = partition == 0   ? frames
                               : partition == 1 ? 1
                                                : chunks[offset == 0   ? 0
                                                         : offset == 1 ? 1
                                                         : offset == 8 ? 2
                                                                       : 3];
                CHECK(phonon_mimi_process(state, latent + offset * 32, chunk,
                                          actual + offset * 16 * 512, chunk * 16));
                offset += chunk;
            }
            char label[80];
            snprintf(label, sizeof(label), "real Mimi run=%zu partition=%zu", run, partition);
            compare(actual, expected, count, 1e-3f, 1e-5f, label);
            if (run == 2) {
                if (partition == 0)
                    memcpy(hidden, actual, count * sizeof(float));
                else
                    compare(actual, hidden, count, 0, 1e-5f, "native whole/partition features");
            }
            CHECK(phonon_mimi_positions(state) == positions);
        }
        free(latent);
        free(hidden);
        free(actual);
        free(expected);
    }
    CHECK(fgetc(file) == EOF);
    fclose(file);
    phonon_mimi_state_destroy(state);
    phonon_pack_close(pack);
}
int main(void) {
    upsample_test();
    synthetic_test();
    model_test();
    return 0;
}

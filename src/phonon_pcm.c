#include "phonon_pcm.h"
#include "phonon_conv.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct PhononPCMModel {
    PhononMimiModel mimi;
    PhononConvModel *conv[11];
};
struct PhononPCMState {
    const PhononPCMModel *model;
    PhononMimiState *mimi;
    PhononConvState *conv[5];
    PhononResidualState *residual[3];
    size_t frames;
    int failed, finished;
    float features[16 * 512], a[120 * 64], b[120 * 64];
};
static int load_conv(const PhononPack *pack, size_t layer, size_t block, size_t stride, size_t in,
                     size_t out, size_t kernel, PhononConvModel **model) {
    char prefix[128], name[160];
    if (block)
        snprintf(prefix, sizeof(prefix), "mimi.decoder.model.%zu.block.%zu.conv", layer, block);
    else
        snprintf(prefix, sizeof(prefix), "mimi.decoder.model.%zu.%s", layer,
                 stride > 1 ? "convtr" : "conv");
    PhononTensorView w, b;
    snprintf(name, sizeof(name), "%s.weight", prefix);
    if (!phonon_pack_tensor(pack, name, &w))
        return 0;
    snprintf(name, sizeof(name), "%s.bias", prefix);
    if (!phonon_pack_tensor(pack, name, &b) || w.rank != 3 ||
        w.shape[0] != (stride > 1 ? in : out) || w.shape[1] != (stride > 1 ? out : in) ||
        w.shape[2] != kernel)
        return 0;
    return phonon_conv_model_create(&w, &b, stride, stride > 1, model);
}
int phonon_pcm_model_create(const PhononPack *pack, PhononPCMModel **out) {
    if (!out)
        return 0;
    *out = NULL;
    if (!pack)
        return 0;
    PhononPCMModel *m = calloc(1, sizeof(*m));
    if (!m)
        return 0;
    PhononMimiConfig config = phonon_mimi_config();
    if (!phonon_mimi_model_load(pack, &config, &m->mimi) ||
        !load_conv(pack, 0, 0, 1, 512, 512, 7, &m->conv[0]))
        goto fail;
    const size_t strides[] = {6, 5, 4};
    for (size_t i = 0, channels = 512; i < 3; i++, channels /= 2) {
        if (!load_conv(pack, 2 + i * 3, 0, strides[i], channels, channels / 2, 2 * strides[i],
                       &m->conv[1 + i * 3]) ||
            !load_conv(pack, 3 + i * 3, 1, 1, channels / 2, channels / 4, 3, &m->conv[2 + i * 3]) ||
            !load_conv(pack, 3 + i * 3, 3, 1, channels / 4, channels / 2, 1, &m->conv[3 + i * 3]))
            goto fail;
    }
    if (!load_conv(pack, 11, 0, 1, 64, 1, 3, &m->conv[10]))
        goto fail;
    *out = m;
    return 1;
fail:
    phonon_pcm_model_destroy(m);
    return 0;
}
void phonon_pcm_model_destroy(PhononPCMModel *m) {
    if (!m)
        return;
    for (size_t i = 0; i < 11; i++)
        phonon_conv_model_destroy(m->conv[i]);
    free(m);
}
int phonon_pcm_state_create(const PhononPCMModel *m, PhononPCMState **out) {
    if (!out)
        return 0;
    *out = NULL;
    if (!m)
        return 0;
    PhononPCMState *s = calloc(1, sizeof(*s));
    if (!s)
        return 0;
    s->model = m;
    if (!phonon_mimi_state_create(&m->mimi, &s->mimi) ||
        !phonon_conv_state_create(m->conv[0], &s->conv[0]) ||
        !phonon_conv_state_create(m->conv[10], &s->conv[4]))
        goto fail;
    for (size_t i = 0; i < 3; i++)
        if (!phonon_conv_state_create(m->conv[1 + i * 3], &s->conv[1 + i]) ||
            !phonon_residual_state_create(m->conv[2 + i * 3], m->conv[3 + i * 3], &s->residual[i]))
            goto fail;
    *out = s;
    return 1;
fail:
    phonon_pcm_state_destroy(s);
    return 0;
}
void phonon_pcm_state_destroy(PhononPCMState *s) {
    if (!s)
        return;
    phonon_mimi_state_destroy(s->mimi);
    for (size_t i = 0; i < 5; i++)
        phonon_conv_state_destroy(s->conv[i]);
    for (size_t i = 0; i < 3; i++)
        phonon_residual_state_destroy(s->residual[i]);
    free(s);
}
void phonon_pcm_state_reset(PhononPCMState *s) {
    if (!s)
        return;
    phonon_mimi_state_reset(s->mimi);
    for (size_t i = 0; i < 5; i++)
        phonon_conv_state_reset(s->conv[i]);
    for (size_t i = 0; i < 3; i++)
        phonon_residual_state_reset(s->residual[i]);
    s->frames = 0;
    s->failed = 0;
    s->finished = 0;
}
size_t phonon_pcm_samples(const PhononPCMState *s) {
    return s ? s->frames * PHONON_PCM_FRAME_SAMPLES : 0;
}
static int overlap(const void *a, size_t an, const void *b, size_t bn) {
    uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return x > UINTPTR_MAX - an || y > UINTPTR_MAX - bn || (x < y + bn && y < x + an);
}
int phonon_pcm_process(PhononPCMState *s, const float *x, size_t n, float *raw, size_t cap,
                       float *final, size_t final_cap) {
    if (!s || s->failed || s->finished || n > PHONON_MIMI_MAX_FRAMES - s->frames)
        return 0;
    if (!n)
        return 1;
    size_t count = n * PHONON_PCM_FRAME_SAMPLES, bytes = count * sizeof(float);
    if (!x || !raw || cap < count || (final && final_cap < count) ||
        overlap(x, n * 32 * sizeof(float), raw, bytes) ||
        (final && (overlap(x, n * 32 * sizeof(float), final, bytes) ||
                   (final != raw && overlap(raw, bytes, final, bytes)))))
        return 0;
    for (size_t i = 0; i < n * 32; i++)
        if (!isfinite(x[i]))
            return 0;
    const size_t strides[] = {6, 5, 4};
    for (size_t frame = 0; frame < n; frame++) {
        if (!phonon_mimi_process(s->mimi, x + frame * 32, 1, s->features, 16))
            goto fail;
        for (size_t row = 0; row < 16; row++) {
            if (!phonon_conv_process(s->conv[0], s->features + row * 512, 1, s->a, 1))
                goto fail;
            size_t positions = 1, channels = 512;
            for (size_t stage = 0; stage < 3; stage++) {
                phonon_conv_elu(s->a, positions * channels);
                if (!phonon_conv_process(s->conv[1 + stage], s->a, positions, s->b,
                                         positions * strides[stage]))
                    goto fail;
                positions *= strides[stage];
                channels /= 2;
                if (!phonon_residual_process(s->residual[stage], s->b, positions, s->a, positions))
                    goto fail;
            }
            phonon_conv_elu(s->a, positions * channels);
            if (!phonon_conv_process(s->conv[4], s->a, 120, raw + frame * 1920 + row * 120, 120))
                goto fail;
        }
        s->frames++;
    }
    if (final && final != raw)
        memcpy(final, raw, bytes);
    return 1;
fail:
    s->failed = 1;
    return 0;
}
int phonon_pcm_finish(PhononPCMState *s) {
    if (!s || s->failed || s->finished)
        return 0;
    s->finished = 1;
    return 1;
}
#ifdef PHONON_PACK_TEST_API
#include "phonon_pcm_test.h"
int phonon_pcm_test_model(const PhononMimiModel *mimi, PhononConvModel *conv[11],
                          PhononPCMModel **out) {
    if (!out)
        return 0;
    *out = NULL;
    if (!mimi || !conv)
        return 0;
    for (size_t i = 0; i < 11; i++)
        if (!conv[i])
            return 0;
    PhononPCMModel *m = calloc(1, sizeof(*m));
    if (!m)
        return 0;
    m->mimi = *mimi;
    memcpy(m->conv, conv, sizeof(m->conv));
    *out = m;
    return 1;
}
#endif

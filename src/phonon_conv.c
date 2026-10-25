#include "phonon_conv.h"
#include "phonon_math.h"
#define ACCELERATE_NEW_LAPACK

#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MAX_POSITIONS ((size_t)15728640)
struct PhononConvModel {
    size_t in, out, kernel, stride;
    int transpose;
    float *weight, *bias;
};
struct PhononConvState {
    const PhononConvModel *model;
    size_t positions;
    int failed;
    float *history, *scratch;
};
struct PhononResidualState {
    PhononConvState *first, *second;
    int failed;
    float activated[512], hidden[256], result[512];
};
static int finite_values(const float *p, size_t n) {
    for (size_t i = 0; i < n; i++)
        if (!isfinite(p[i]))
            return 0;
    return 1;
}
static int overlap(const void *a, size_t an, const void *b, size_t bn) {
    uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return x > UINTPTR_MAX - an || y > UINTPTR_MAX - bn || (x < y + bn && y < x + an);
}
static int bf16_shape(const PhononTensorView *v, size_t count) {
    return v && v->dtype == PHONON_PACK_BF16 && v->data &&
           v->byte_count == count * sizeof(uint16_t);
}
int phonon_conv_model_create(const PhononTensorView *w, const PhononTensorView *b, size_t stride,
                             int transpose, PhononConvModel **out) {
    if (!out)
        return 0;
    *out = NULL;
    if (!w || w->rank != 3 || (transpose != 0 && transpose != 1) || !w->shape[0] ||
        w->shape[0] > 512 || !w->shape[1] || w->shape[1] > 512)
        return 0;
    size_t in = (size_t)w->shape[transpose ? 0 : 1];
    size_t channels = (size_t)w->shape[transpose ? 1 : 0];
    size_t kernel = (size_t)w->shape[2];
    if (transpose ? (stride < 4 || stride > 6 || kernel != 2 * stride)
                  : (stride != 1 || (kernel != 1 && kernel != 3 && kernel != 7)))
        return 0;
    size_t count = in * channels * kernel;
    if (!bf16_shape(w, count) || !b || b->rank != 1 || b->shape[0] != channels ||
        !bf16_shape(b, channels))
        return 0;
    PhononConvModel *m = calloc(1, sizeof(*m));
    if (!m)
        return 0;
    m->in = in;
    m->out = channels;
    m->kernel = kernel;
    m->stride = stride;
    m->transpose = transpose;
    m->weight = malloc(count * sizeof(float));
    m->bias = malloc(channels * sizeof(float));
    if (!m->weight || !m->bias || !phonon_bf16_to_float(w->data, count, m->weight) ||
        !phonon_bf16_to_float(b->data, channels, m->bias) || !finite_values(m->weight, count) ||
        !finite_values(m->bias, channels)) {
        phonon_conv_model_destroy(m);
        return 0;
    }
    *out = m;
    return 1;
}
void phonon_conv_model_destroy(PhononConvModel *m) {
    if (m) {
        free(m->weight);
        free(m->bias);
        free(m);
    }
}
size_t phonon_conv_input_channels(const PhononConvModel *m) { return m ? m->in : 0; }
size_t phonon_conv_output_channels(const PhononConvModel *m) { return m ? m->out : 0; }
size_t phonon_conv_stride(const PhononConvModel *m) { return m ? m->stride : 0; }
int phonon_conv_state_create(const PhononConvModel *m, PhononConvState **out) {
    if (!out)
        return 0;
    *out = NULL;
    if (!m)
        return 0;
    PhononConvState *s = calloc(1, sizeof(*s));
    if (!s)
        return 0;
    s->model = m;
    size_t h = m->transpose ? m->out * m->stride : m->in * m->kernel;
    size_t scratch = m->transpose ? m->out * m->kernel : m->in * m->kernel;
    s->history = calloc(h, sizeof(float));
    s->scratch = calloc(scratch, sizeof(float));
    if (!s->history || !s->scratch) {
        phonon_conv_state_destroy(s);
        return 0;
    }
    *out = s;
    return 1;
}
void phonon_conv_state_destroy(PhononConvState *s) {
    if (s) {
        free(s->history);
        free(s->scratch);
        free(s);
    }
}
void phonon_conv_state_reset(PhononConvState *s) {
    if (!s)
        return;
    const PhononConvModel *m = s->model;
    memset(s->history, 0, sizeof(float) * (m->transpose ? m->out * m->stride : m->in * m->kernel));
    s->positions = 0;
    s->failed = 0;
}
static int arguments(const PhononConvState *s, const float *x, size_t n, float *y, size_t cap) {
    if (!s || s->failed || n > MAX_POSITIONS - s->positions)
        return 0;
    if (!n)
        return 1;
    const PhononConvModel *m = s->model;
    return x && y && cap >= n * m->stride &&
           !overlap(x, n * m->in * sizeof(float), y, n * m->stride * m->out * sizeof(float)) &&
           finite_values(x, n * m->in);
}
int phonon_conv_process(PhononConvState *s, const float *x, size_t n, float *y, size_t cap) {
    if (!arguments(s, x, n, y, cap))
        return 0;
    const PhononConvModel *m = s->model;
    for (size_t t = 0; t < n; t++) {
        if (!m->transpose) {
            /* Torch contiguous im2col order is [input channel,kernel tap]. */
            for (size_t c = 0; c < m->in; c++) {
                float *h = s->history + c * m->kernel;
                memmove(h, h + 1, (m->kernel - 1) * sizeof(float));
                h[m->kernel - 1] = x[t * m->in + c];
            }
            memcpy(y + t * m->out, m->bias, m->out * sizeof(float));
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)m->out, 1,
                        (int)(m->in * m->kernel), 1, m->weight, (int)(m->in * m->kernel),
                        s->history, 1, 1, y + t * m->out, 1);
        } else {
            /* Torch computes [output channel,kernel tap] columns, col2im adds
             * previous-frame right half to this frame's left half, then bias. */
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, (int)(m->out * m->kernel), 1,
                        (int)m->in, 1, m->weight, (int)(m->out * m->kernel), x + t * m->in, 1, 0,
                        s->scratch, 1);
            for (size_t k = 0; k < m->stride; k++)
                for (size_t c = 0; c < m->out; c++) {
                    size_t h = k * m->out + c;
                    y[(t * m->stride + k) * m->out + c] =
                        (s->scratch[c * m->kernel + k] + s->history[h]) + m->bias[c];
                    s->history[h] = s->scratch[c * m->kernel + m->stride + k];
                }
        }
        if (!finite_values(y + t * m->stride * m->out, m->stride * m->out)) {
            s->failed = 1;
            return 0;
        }
        s->positions++;
    }
    return 1;
}
void phonon_conv_elu(float *x, size_t n) {
    size_t end = n - n % 4;
    for (size_t i = 0; i < end; i += 4) {
        float e[4];
        memcpy(e, x + i, sizeof(e));
        phonon_math_exp4(e);
        for (size_t j = 0; j < 4; j++)
            if (x[i + j] <= 0)
                x[i + j] = e[j] - 1.0f;
    }
    for (size_t i = end; i < n; i++)
        if (x[i] <= 0)
            x[i] = expf(x[i]) - 1.0f;
}
int phonon_residual_state_create(const PhononConvModel *a, const PhononConvModel *b,
                                 PhononResidualState **out) {
    if (!out)
        return 0;
    *out = NULL;
    if (!a || !b || a->transpose || b->transpose || a->kernel != 3 || b->kernel != 1 ||
        a->in != b->out || a->in != 2 * a->out || a->out != b->in)
        return 0;
    PhononResidualState *s = calloc(1, sizeof(*s));
    if (!s)
        return 0;
    if (!phonon_conv_state_create(a, &s->first) || !phonon_conv_state_create(b, &s->second)) {
        phonon_residual_state_destroy(s);
        return 0;
    }
    *out = s;
    return 1;
}
void phonon_residual_state_destroy(PhononResidualState *s) {
    if (s) {
        phonon_conv_state_destroy(s->first);
        phonon_conv_state_destroy(s->second);
        free(s);
    }
}
void phonon_residual_state_reset(PhononResidualState *s) {
    if (s) {
        phonon_conv_state_reset(s->first);
        phonon_conv_state_reset(s->second);
        s->failed = 0;
    }
}
int phonon_residual_process(PhononResidualState *s, const float *x, size_t n, float *y,
                            size_t cap) {
    if (!s || s->failed || !arguments(s->first, x, n, y, cap))
        return 0;
    size_t channels = s->first->model->in, hidden = s->first->model->out;
    if (n && overlap(x, n * channels * sizeof(float), y, n * channels * sizeof(float)))
        return 0;
    for (size_t t = 0; t < n; t++) {
        memcpy(s->activated, x + t * channels, channels * sizeof(float));
        phonon_conv_elu(s->activated, channels);
        if (!phonon_conv_process(s->first, s->activated, 1, s->hidden, 1)) {
            s->failed = 1;
            return 0;
        }
        phonon_conv_elu(s->hidden, hidden);
        if (!phonon_conv_process(s->second, s->hidden, 1, s->result, 1)) {
            s->failed = 1;
            return 0;
        }
        for (size_t c = 0; c < channels; c++)
            y[t * channels + c] = x[t * channels + c] + s->result[c];
        if (!finite_values(y + t * channels, channels)) {
            s->failed = 1;
            return 0;
        }
    }
    return 1;
}

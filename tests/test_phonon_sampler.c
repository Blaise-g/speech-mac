#include "phonon_sampler.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "check failed: %s:%d: %s\n", __FILE__, __LINE__, #x);                  \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

static void test_seed_repeatability_and_interleaving(void) {
    PhononNoiseSource a, b, c;
    CHECK(phonon_noise_seed(&a, 42));
    CHECK(phonon_noise_seed(&b, 42));
    CHECK(phonon_noise_seed(&c, 43));
    for (int i = 0; i < 96; i++) {
        float x, y, z;
        CHECK(phonon_noise_next(&a, &x));
        CHECK(phonon_noise_next(&b, &y));
        CHECK(phonon_noise_next(&c, &z));
        CHECK(x == y);
        if (i == 0)
            CHECK(x != z);
    }
    phonon_noise_reset(&a);
    float first, again;
    CHECK(phonon_noise_next(&a, &first));
    phonon_noise_reset(&a);
    CHECK(phonon_noise_next(&a, &again) && first == again);
}

static void test_raw_replay_scaling_and_exhaustion(void) {
    const float raw[] = {-1.0f, 0.25f, 2.0f};
    PhononNoiseSource source;
    CHECK(phonon_noise_replay(&source, raw, 3));
    float value;
    CHECK(phonon_noise_next(&source, &value));
    CHECK(value == -1.0f);
    CHECK(phonon_noise_next(&source, &value));
    CHECK(value == 0.25f);
    CHECK(phonon_noise_next(&source, &value));
    CHECK(value == 2.0f);
    CHECK(!phonon_noise_next(&source, &value));
    phonon_noise_reset(&source);
    CHECK(phonon_noise_next(&source, &value) && value == -1.0f);
    CHECK(!phonon_noise_replay(&source, raw, 0));
    const float bad[] = {NAN};
    CHECK(!phonon_noise_replay(&source, bad, 1));
    float frame[32];
    CHECK(phonon_noise_replay(&source, raw, 3));
    CHECK(!phonon_noise_frame(&source, frame)); /* finite cursor: no wrap or silent reuse */
}

static void test_eos_tail_and_budget(void) {
    PhononStopState state;
    int stop;
    CHECK(phonon_stop_init(&state, 8, 3));
    CHECK(phonon_stop_after_frame(&state, 4.01f, 4.0f, &stop) && !stop);
    CHECK(state.eos_seen && state.frames_generated == 1 && state.tail_remaining == 3);
    CHECK(phonon_stop_after_frame(&state, 5.0f, 4.0f, &stop) && !stop && state.tail_remaining == 2);
    CHECK(phonon_stop_after_frame(&state, 0.0f, 4.0f, &stop) && !stop && state.tail_remaining == 1);
    CHECK(phonon_stop_after_frame(&state, 9.0f, 4.0f, &stop) && stop &&
          state.reason == PHONON_STOP_EOS && state.frames_generated == 4);

    CHECK(phonon_stop_init(&state, 2, 4));
    CHECK(phonon_stop_after_frame(&state, 4.1f, 4.0f, &stop) && !stop);
    CHECK(phonon_stop_after_frame(&state, 0.0f, 4.0f, &stop) && stop &&
          state.reason == PHONON_STOP_FRAME_BUDGET && state.eos_seen);

    CHECK(phonon_stop_init(&state, 1, 0));
    CHECK(phonon_stop_after_frame(&state, 4.0f, 4.0f, &stop) && stop &&
          state.reason == PHONON_STOP_FRAME_BUDGET && !state.eos_seen);
    CHECK(!phonon_stop_after_frame(&state, 0.0f, 4.0f, &stop));

    CHECK(phonon_stop_init(&state, 4, 3));
    CHECK(phonon_stop_after_frame(&state, 4.1f, 4.0f, &stop) && !stop);
    CHECK(phonon_stop_after_frame(&state, 0.0f, 4.0f, &stop) && !stop);
    CHECK(phonon_stop_after_frame(&state, 4.2f, 4.0f, &stop) && !stop);
    CHECK(phonon_stop_after_frame(&state, 0.0f, 4.0f, &stop) && stop &&
          state.frames_generated == state.frame_budget && state.reason == PHONON_STOP_EOS);

    CHECK(phonon_stop_init(&state, 1, 0));
    CHECK(phonon_stop_after_frame(&state, 4.01f, 4.0f, &stop) && stop &&
          state.reason == PHONON_STOP_EOS && state.frames_generated == 1);
}

static void test_nonzero_flow_head_against_independent_reference(void);

int main(void) {
    test_seed_repeatability_and_interleaving();
    test_raw_replay_scaling_and_exhaustion();
    test_eos_tail_and_budget();
    test_nonzero_flow_head_against_independent_reference();
    puts("Phonon sampler tests passed");
    return 0;
}

static uint16_t synthetic_matrix[1536 * 512 + 32];
static uint16_t synthetic_vectors[60][1536];

static uint16_t float_to_bf16_exact(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

static void synthetic_tensor(PhononTensorView *view, size_t rows, size_t cols, size_t offset) {
    view->dtype = PHONON_PACK_BF16;
    view->rank = 2;
    view->shape[0] = rows;
    view->shape[1] = cols;
    view->data = synthetic_matrix + offset;
    view->byte_count = rows * cols * sizeof(uint16_t);
}

static void synthetic_vector(PhononTensorView *view, size_t count, size_t storage_slot,
                             size_t pattern_slot, int kind) {
    view->dtype = PHONON_PACK_BF16;
    view->rank = 1;
    view->shape[0] = count;
    view->data = synthetic_vectors[storage_slot];
    view->byte_count = count * sizeof(uint16_t);
    for (size_t i = 0; i < count; i++) {
        float value;
        if (kind == 1 || kind == 2)
            value = 1.0f + (float)((int)((i * 3 + pattern_slot * 5) % 7) - 3) / 128.0f;
        else if (kind == 3)
            value = (float)(i % 8 + 1) / 128.0f;
        else
            value = (float)((int)((i * 3 + pattern_slot * 5) % 11) - 5) / 128.0f;
        synthetic_vectors[storage_slot][i] = float_to_bf16_exact(value);
    }
}

static void synthetic_linear(PhononTensorView *weight, PhononTensorView *bias, size_t rows,
                             size_t cols, size_t matrix_offset, size_t vector_storage,
                             size_t vector_pattern) {
    synthetic_tensor(weight, rows, cols, matrix_offset);
    synthetic_vector(bias, rows, vector_storage, vector_pattern, 0);
}

static void test_nonzero_flow_head_against_independent_reference(void) {
    for (size_t i = 0; i < sizeof(synthetic_matrix) / sizeof(synthetic_matrix[0]); i++)
        synthetic_matrix[i] = float_to_bf16_exact((float)((int)((i * 5) % 7) - 3) / 128.0f);
    PhononFlowHead head = {0};
    synthetic_linear(&head.input_weight, &head.input_bias, 512, 32, 1, 0, 1);
    synthetic_linear(&head.cond_weight, &head.cond_bias, 512, 512, 2, 1, 2);
    for (size_t i = 0; i < 2; i++) {
        synthetic_linear(&head.time_weight[i][0], &head.time_bias[i][0], 512, 256, i + 1, 2 + i * 2,
                         i + 3);
        synthetic_linear(&head.time_weight[i][1], &head.time_bias[i][1], 512, 512, i + 4, 3 + i * 2,
                         i + 5);
        synthetic_vector(&head.time_freqs[i], 128, 6 + i, 30 + i, 3);
        synthetic_vector(&head.time_alpha[i], 512, 8 + i, 32 + i, 2);
    }
    for (size_t i = 0; i < PHONON_FLOW_HEAD_BLOCKS; i++) {
        synthetic_linear(&head.block_ada_weight[i], &head.block_ada_bias[i], 1536, 512, i + 3,
                         10 + i * 3, i + 4);
        synthetic_vector(&head.block_norm_weight[i], 512, 28 + i, 10 + i, 1);
        synthetic_vector(&head.block_norm_bias[i], 512, 34 + i, 16 + i, 0);
        synthetic_linear(&head.block_mlp_weight[i][0], &head.block_mlp_bias[i][0], 512, 512, i + 9,
                         40 + i * 2, i + 10);
        synthetic_linear(&head.block_mlp_weight[i][1], &head.block_mlp_bias[i][1], 512, 512, i + 11,
                         41 + i * 2, i + 12);
    }
    synthetic_linear(&head.final_ada_weight, &head.final_ada_bias, 1024, 512, 22, 52, 23);
    synthetic_linear(&head.final_weight, &head.final_bias, 32, 512, 24, 53, 25);

    float condition[512], state[32], actual[32];
    for (size_t i = 0; i < 512; i++)
        condition[i] = (float)((int)((i * 7) % 13) - 6) / 8.0f;
    for (size_t i = 0; i < 32; i++)
        state[i] = (float)((int)((i * 11 + 3) % 17) - 8) / 16.0f;
    /* Independent CPU PyTorch 2.7.1 reference: six eager AdaLN blocks using F.linear,
     * F.layer_norm(eps=1e-6), F.silu, and the stored BF16 tensors constructed above.
     * The condition/state formulas above and matrix/vector patterns in the setup fully
     * specify the oracle input; time frequencies use the same deterministic BF16 pattern. */
    const float expected[32] = {
        6.749069214f,  -2.760953903f, -6.280110359f, 1.966097355f,  3.173623323f,  -1.909553528f,
        -0.930359602f, 6.741256714f,  -2.768766403f, -6.287922859f, 1.958284855f,  3.165810823f,
        -1.917366028f, -0.852234602f, 6.733444214f,  -2.776578903f, -6.295735359f, 1.950472355f,
        3.157998323f,  -1.925178528f, -0.860047102f, 6.725631714f,  -2.784391403f, -6.303547859f,
        2.028597355f,  3.150185823f,  -1.932991028f, -0.867859602f, 6.717819214f,  -2.792203903f,
        -6.311360359f, 2.020784855f};
    CHECK(phonon_flow_head_forward(&head, condition, state, 0.0f, 1.0f, actual));
    for (size_t i = 0; i < 32; i++) {
        if (fabsf(actual[i] - expected[i]) >= 2e-4f)
            fprintf(stderr, "synthetic head mismatch %zu got %.9g expected %.9g\n", i, actual[i],
                    expected[i]);
        CHECK(fabsf(actual[i] - expected[i]) < 2e-4f);
    }
}

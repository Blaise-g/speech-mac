#include "phonon_sampler.h"
#include "phonon_transformer_test.h"
#include "speech_test.h"
#include <math.h>
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
typedef struct {
    float *pcm;
    size_t capacity, samples, calls, head_calls;
    const float *expected_trace;
    int response;
} Consumer;
static int receive(void *context, const float *pcm, size_t count, const SpeechResult *r) {
    Consumer *c = context;
    CHECK(count == 1920 && r->frames_generated == c->calls + 1);
    CHECK(r->samples_generated == (c->calls + 1) * 1920);
    /* Actual head observer proves no future latent exists at this callback. A
     * bulk-generate-then-emit mutation fails on the first callback. */
    CHECK(c->head_calls == c->calls + 1);
    CHECK(count <= c->capacity - c->samples);
    for (size_t i = 0; i < count; i++)
        CHECK(isfinite(pcm[i]));
    memcpy(c->pcm + c->samples, pcm, count * sizeof(float));
    c->samples += count;
    c->calls++;
    return c->response;
}
static void observe(void *context, size_t generated, const float *condition, const float *latent,
                    float eos) {
    Consumer *c = context;
    CHECK(generated == c->head_calls);
    if (c->expected_trace) {
        const float *want = c->expected_trace + generated * 545;
        for (size_t j = 0; j < 545; j++) {
            float actual = j < 512 ? condition[j] : j < 544 ? latent[j - 512] : eos;
            if (fabsf(actual - want[j]) > 1e-5f + 1e-3f * fabsf(want[j])) {
                fprintf(stderr,
                        "engine seam mismatch frame=%zu component=%zu actual=%.9g "
                        "expected=%.9g\n",
                        generated, j, actual, want[j]);
                CHECK(0);
            }
        }
    }
    c->head_calls++;
}
static int no_output(void *context, const float *pcm, size_t count, const SpeechResult *r) {
    (void)context;
    (void)pcm;
    (void)count;
    (void)r;
    CHECK(0);
    return 0;
}
typedef struct {
    size_t calls, stop;
} Cancel;
static int cancel(void *context) {
    Cancel *c = context;
    return ++c->calls == c->stop;
}
static void parity(const float *a, const float *b, size_t count, unsigned record) {
    double square = 0, power = 0;
    float maximum = 0;
    for (size_t i = 0; i < count; i++) {
        CHECK(isfinite(a[i]) && isfinite(b[i]));
        double d = (double)a[i] - b[i];
        square += d * d;
        power += (double)b[i] * b[i];
        if (fabs(d) > maximum)
            maximum = (float)fabs(d);
    }
    double nrmse = power ? sqrt(square / power) : sqrt(square / count);
    printf("engine record=%u samples=%zu nrmse=%.9g max_abs=%.9g\n", record, count, nrmse, maximum);
    CHECK(nrmse <= .001 && maximum <= .002);
}
#include "phonon_layer_norm_regression.inc"
static void layer_norm_regression(void) {
    float input[512], weight[512], bias[512] = {0}, output[512];
    for (size_t i = 0; i < 512; i++) {
        memcpy(input + i, layer_norm_regression_bits + i, sizeof(float));
        weight[i] = 1.f;
    }
    CHECK(phonon_layer_norm(input, weight, bias, 512, 1e-5f, output));
    uint32_t first;
    memcpy(&first, output, sizeof(first));
    CHECK(first == UINT32_C(0x3f31cd52));
}
static void attention_regression(void) {
    /* Independent pinned Torch2.7.1 SDPA outputs. K=1 after a full block
     * accumulates a rounded product, rather than contracting it with prior C.
     * The1025 case also rescales prior output when the middle block raises max. */
    const struct {
        size_t count;
        float middle, tail;
        uint32_t expected;
    } cases[] = {{513, 0, 1, UINT32_C(0x394b6175)},
                 {513, 0, 3, UINT32_C(0x3a05c0fa)},
                 {1025, 3, 1, UINT32_C(0x38c973b4)},
                 {1025, 1, 1, UINT32_C(0x38ccd53a)},
                 {1537, 3, 1, UINT32_C(0x388782fa)}};
    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        size_t n = cases[c].count;
        float query[64] = {8}, out[64];
        float *k = calloc(n * 64, sizeof(float)), *v = calloc(n * 64, sizeof(float));
        CHECK(k && v);
        k[0] = 2;
        k[(n - 1) * 64] = cases[c].tail;
        v[0] = v[(n - 1) * 64] = .01f;
        if (cases[c].middle)
            k[512 * 64] = cases[c].middle;
        CHECK(phonon_attention_test(query, k, v, n, 64, out));
        uint32_t bits;
        memcpy(&bits, out, 4);
        CHECK(bits == cases[c].expected);
        for (size_t d = 1; d < 64; d++)
            CHECK(out[d] == 0);
        if (n <= 1025) {
            float *q = calloc(n * 1536, sizeof(float)), *pk = calloc(n * 512, sizeof(float)),
                  *pv = calloc(n * 512, sizeof(float)), *po = malloc(n * 512 * sizeof(float));
            CHECK(q && pk && pv && po);
            for (size_t row = 0; row < n; row++) {
                q[row * 1536] = 8;
                pk[row * 512] = k[row * 64];
                pv[row * 512] = v[row * 64];
            }
            CHECK(phonon_prefill_attention_test(q, pk, pv, n, po));
            memcpy(&bits, po + (n - 1) * 512, 4);
            CHECK(bits == cases[c].expected);
            free(q);
            free(pk);
            free(pv);
            free(po);
        }
        free(k);
        free(v);
    }
}
static void contracts(void) {
    SpeechEngine *e = (void *)1;
    CHECK(speech_engine_create(NULL, &e) == SPEECH_INVALID && !e);
    CHECK(speech_engine_create("/no/such/pack", &e) == SPEECH_FAILED && !e);
    speech_engine_destroy(NULL);
    SpeechResult r = {1, 2, 3};
    SpeechOptions o = speech_options_default();
    CHECK(o.max_frames == 8192 && !strcmp(o.language, "en"));
    CHECK(speech_synthesize(NULL, NULL, "x", 1, &o, no_output, NULL, &r) == SPEECH_INVALID);
    CHECK(!r.frames_generated && !r.samples_generated && !r.chunks_completed);
    float *pcm = (void *)1;
    size_t n = 99;
    CHECK(speech_collect(NULL, NULL, "x", 1, &o, 0, &pcm, &n, &r) == SPEECH_INVALID);
    CHECK(!pcm && !n);
    CHECK(speech_collect(NULL, NULL, "x", 1, &o, (size_t)8192 * 1920 + 1, &pcm, &n, &r) ==
          SPEECH_LIMIT);
}
static void full_model(void) {
    const char *pack = getenv("PHONON_NATIVE_PACK"), *fixture = getenv("PHONON_ENGINE_FIXTURE");
    if (!pack && !fixture) {
        puts("SKIP real engine: set PHONON_NATIVE_PACK and PHONON_ENGINE_FIXTURE");
        return;
    }
    CHECK(pack && fixture);
    FILE *f = fopen(fixture, "rb");
    CHECK(f);
    char magic[8];
    uint32_t records;
    CHECK(fread(magic, 1, 8, f) == 8 && !memcmp(magic, "PHENG2\0\0", 8));
    CHECK(fread(&records, 4, 1, f) == 1 && records == 5);
    SpeechEngine *e = NULL;
    CHECK(speech_engine_create(pack, &e) == SPEECH_OK);
    PhononVoice *voice = NULL, *short_voice = NULL;
    const char *voice_path = getenv("PHONON_DEFAULT_VOICE");
    if (!voice_path) {
        fprintf(stderr, "Real engine requires PHONON_DEFAULT_VOICE when model gate selected\n");
        CHECK(voice_path);
    }
    CHECK(phonon_voice_open(voice_path, &voice));
    CHECK(phonon_voice_open(".build/phonon-engine-short-voice.safetensors", &short_voice));
    for (uint32_t record = 0; record < records; record++) {
        uint32_t bytes, vframes, chunks, samples, draw_count;
        uint64_t seed;
        float delta;
        CHECK(fread(&bytes, 4, 1, f) == 1 && fread(&seed, 8, 1, f) == 1 &&
              fread(&delta, 4, 1, f) == 1);
        CHECK(fread(&vframes, 4, 1, f) == 1 && fread(&chunks, 4, 1, f) == 1 &&
              fread(&samples, 4, 1, f) == 1 && fread(&draw_count, 4, 1, f) == 1);
        CHECK(bytes < 4096 && samples && samples <= 8192 * 1920 &&
              draw_count == samples / 1920 * 32);
        char *text = calloc(bytes + 1, 1);
        float *draws = malloc(draw_count * sizeof(float)),
              *expected = malloc(samples * sizeof(float));
        float *trace = malloc((samples / 1920) * 545 * sizeof(float));
        CHECK(text && draws && expected && trace);
        CHECK(fread(text, 1, bytes, f) == bytes && fread(draws, 4, draw_count, f) == draw_count &&
              fread(trace, sizeof(float), (samples / 1920) * 545, f) == (samples / 1920) * 545 &&
              fread(expected, 4, samples, f) == samples);
        const char *selection = getenv("PHONON_ENGINE_RECORD");
        if (selection && record != (uint32_t)atoi(selection)) {
            free(text);
            free(draws);
            free(expected);
            free(trace);
            continue;
        }
        PhononNoiseSource rng;
        CHECK(phonon_noise_seed(&rng, seed));
        for (size_t i = 0; i < draw_count; i++) {
            float draw;
            CHECK(phonon_noise_next(&rng, &draw));
            CHECK(draw == draws[i]);
        }
        SpeechOptions o = speech_options_default();
        o.seed = seed;
        o.duration_delta = delta;
        PhononVoice *selected = vframes == 32 ? short_voice : voice;
        Consumer c = {.capacity = 8192U * 1920, .pcm = malloc(8192U * 1920 * sizeof(float))};
        CHECK(c.pcm);
        c.expected_trace = trace;
        SpeechResult r;
        CHECK(speech_test_synthesize(e, selected, text, bytes, &o, receive, &c, &r, observe, &c) ==
              SPEECH_OK);
        printf("engine actual record=%u samples=%zu chunks=%zu expected=%u/%u\n", record, c.samples,
               r.chunks_completed, samples, chunks);
        fflush(stdout);
        CHECK(c.calls > 1 && c.samples == samples && r.chunks_completed == chunks &&
              r.samples_generated == samples);
        parity(c.pcm, expected, samples, record);
        float *batch = NULL;
        size_t count = 0;
        SpeechResult br;
        CHECK(speech_collect(e, selected, text, bytes, &o, samples, &batch, &count, &br) ==
              SPEECH_OK);
        CHECK(count == samples && !memcmp(batch, c.pcm, samples * sizeof(float)));
        CHECK(br.frames_generated == r.frames_generated &&
              br.chunks_completed == r.chunks_completed);
        free(batch);
        if (record == 0) {
            /* Cancellation directly inside head/stack numerical work must not
             * publish partial output or commit a partially evaluated cache. */
            PhononPack *native_pack = NULL;
            PhononFlowHead head;
            PhononTensorView views[9][8];
            PhononFlowLayerWeights layers[9];
            CHECK(phonon_pack_open(pack, &native_pack));
            CHECK(phonon_flow_head_load(native_pack, &head));
            CHECK(phonon_flow_load_layers(native_pack, views, layers));
            float condition[512] = {0}, raw[32] = {0}, output[512];
            for (size_t j = 0; j < 512; j++)
                output[j] = 123.f;
            Cancel block_cancel = {.stop = 3};
            CHECK(!phonon_flow_lsd_one_step_cancel(&head, condition, raw, .8f, output, cancel,
                                                   &block_cancel));
            CHECK(block_cancel.calls == 3);
            for (size_t j = 0; j < 512; j++)
                CHECK(output[j] == 123.f);
            PhononKvCache cache = {0};
            CHECK(phonon_kv_cache_init(&cache, 2));
            Cancel layer_cancel = {.stop = 3};
            CHECK(!phonon_flow_stack_step_cancel(layers, &cache, 0, condition, 10000.f, output,
                                                 cancel, &layer_cancel));
            CHECK(layer_cancel.calls == 3 && !cache.length);
            for (size_t j = 0; j < 512; j++)
                CHECK(output[j] == 123.f);
            Cancel prefill_cancel = {.stop = 4};
            CHECK(!phonon_flow_stack_prefill_cancel(layers, &cache, condition, 1, 10000.f, output,
                                                    cancel, &prefill_cancel));
            CHECK(prefill_cancel.calls == 4 && !cache.length);
            for (size_t j = 0; j < 512; j++)
                CHECK(output[j] == 123.f);
            phonon_kv_cache_destroy(&cache);
            phonon_pack_close(native_pack);
            /* Cancel during prefill, edge-triggered token, before first PCM. */
            Cancel cancellation = {.stop = 8};
            o.cancel = cancel;
            o.cancel_context = &cancellation;
            CHECK(speech_synthesize(e, selected, text, bytes, &o, no_output, NULL, &r) ==
                  SPEECH_CANCELLED);
            CHECK(cancellation.calls == 8 && !r.frames_generated);
            o.cancel = NULL;
            for (int response = 1; response <= 2; response++) {
                c.calls = c.samples = c.head_calls = 0;
                c.response = response;
                CHECK(speech_test_synthesize(e, selected, text, bytes, &o, receive, &c, &r, observe,
                                             &c) ==
                      (response == 1 ? SPEECH_CANCELLED : SPEECH_CALLBACK_ERROR));
                CHECK(c.calls == 1 && c.head_calls == 1 && r.frames_generated == 1 &&
                      !r.chunks_completed);
            }
            c.response = 0;
            CHECK(speech_collect(e, selected, text, bytes, &o, samples, &batch, &count, &r) ==
                  SPEECH_OK);
            parity(batch, expected, count, record);
            free(batch);
            CHECK(speech_collect(e, selected, text, bytes, &o, 1919, &batch, &count, &r) ==
                      SPEECH_LIMIT &&
                  !batch && !count);
            o.max_frames = 1;
            CHECK(speech_collect(e, selected, text, bytes, &o, samples, &batch, &count, &r) ==
                      SPEECH_LIMIT &&
                  !batch && !count && r.frames_generated == 1);
            o.max_frames = 8192;
            o.language = "fr";
            CHECK(speech_synthesize(e, selected, text, bytes, &o, no_output, NULL, &r) ==
                  SPEECH_INVALID);
            o.language = "en";
            o.duration_delta = NAN;
            CHECK(speech_synthesize(e, selected, text, bytes, &o, no_output, NULL, &r) ==
                  SPEECH_INVALID);
            o.duration_delta = 0;
            CHECK(speech_synthesize(e, selected, "\xff", 1, &o, no_output, NULL, &r) ==
                  SPEECH_INVALID);
            CHECK(speech_synthesize(e, selected, "x", SPEECH_MAX_INPUT_BYTES + 1, &o, no_output,
                                    NULL, &r) == SPEECH_LIMIT);
            CHECK(speech_synthesize(e, selected, " \t ", 3, &o, no_output, NULL, &r) ==
                      SPEECH_INVALID &&
                  !r.frames_generated);
        }
        free(text);
        free(draws);
        free(expected);
        free(trace);
        free(c.pcm);
    }
    CHECK(fgetc(f) == EOF);
    fclose(f);
    phonon_voice_close(short_voice);
    phonon_voice_close(voice);
    speech_engine_destroy(e);
}
int main(void) {
    layer_norm_regression();
    attention_regression();
    contracts();
    full_model();
    puts("speech tests passed");
    return 0;
}

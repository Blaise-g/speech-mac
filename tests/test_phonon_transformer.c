#include "phonon_sampler.h"
#include "phonon_transformer.h"
#include "phonon_voice.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "failed: %s:%d: %s\n", __FILE__, __LINE__, #x);                        \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)

#undef CHECK
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "failed: %s:%d: %s\n", __FILE__, __LINE__, #x);                        \
            result = 1;                                                                            \
            goto cleanup;                                                                          \
        }                                                                                          \
    } while (0)

static int real_sampler_fixture_check(void) {
    const char *pack_path = getenv("PHONON_NATIVE_PACK");
    const char *fixture_path = getenv("PHONON_FLOW_SAMPLER_FIXTURE");
    if (!pack_path && !fixture_path)
        return 0;
    if (!pack_path || !fixture_path)
        return 1;
    FILE *file = fopen(fixture_path, "rb");
    if (!file)
        return 1;
    int result = 1;
    PhononPack *pack = NULL;
    PhononFlowHead *head = calloc(1, sizeof(*head));
    if (!head || !phonon_pack_open(pack_path, &pack) || !phonon_flow_head_load(pack, head)) {
        fprintf(stderr, "flow head load failed\n");
        goto cleanup;
    }
    unsigned char magic[8];
    uint32_t runs;
    if (fread(magic, 1, 8, file) != 8 || memcmp(magic, "PHSAMP1\0", 8) ||
        fread(&runs, sizeof(runs), 1, file) != 1 || runs != 2)
        goto cleanup;
    for (size_t run = 0; run < 2; run++) {
        uint32_t frames;
        if (fread(&frames, sizeof(frames), 1, file) != 1 || frames != (run == 0 ? 40U : 32U))
            goto cleanup;
        PhononStopState stop_state;
        if (!phonon_stop_init(&stop_state, frames, run == 1 ? 3 : frames))
            goto cleanup;
        for (size_t frame = 0; frame < frames; frame++) {
            float condition[512], raw[32], expected_flow[32], expected_latent[32], expected_eos;
            float actual[32], actual_eos, normalized_expected[32];
            int should_stop = 0;
            if (fread(condition, sizeof(float), 512, file) != 512 ||
                fread(raw, sizeof(float), 32, file) != 32 ||
                fread(expected_flow, sizeof(float), 32, file) != 32 ||
                fread(expected_latent, sizeof(float), 32, file) != 32 ||
                fread(&expected_eos, sizeof(float), 1, file) != 1 ||
                !phonon_flow_normalize_latent(pack, expected_latent, normalized_expected) ||
                !phonon_flow_lsd_one_step(head, condition, raw, 0.8f, actual) ||
                !phonon_flow_eos_logit(pack, condition, &actual_eos)) {
                fprintf(stderr, "flow sample or EOS projection failed run=%zu frame=%zu\n", run,
                        frame);
                goto cleanup;
            }
            for (size_t i = 0; i < 32; i++) {
                if (fabsf(expected_flow[i] - normalized_expected[i]) >
                    1e-5f + 1e-3f * fabsf(normalized_expected[i])) {
                    fprintf(stderr,
                            "normalization seam mismatch run=%zu frame=%zu channel=%zu "
                            "normalized=%.9g flow=%.9g denorm=%.9g\n",
                            run, frame, i, normalized_expected[i], expected_flow[i],
                            expected_latent[i]);
                    goto cleanup;
                }
                double err = fabs((double)actual[i] - expected_flow[i]);
                if (err > 1e-5 + 1e-3 * fabs((double)expected_flow[i])) {
                    fprintf(stderr,
                            "sampler mismatch run=%zu frame=%zu channel=%zu got=%.9g ref=%.9g\n",
                            run, frame, i, actual[i], expected_flow[i]);
                    goto cleanup;
                }
            }
            if (fabsf(actual_eos - expected_eos) > 1e-5f + 1e-3f * fabsf(expected_eos)) {
                fprintf(stderr, "EOS mismatch run=%zu frame=%zu got=%.9g ref=%.9g\n", run, frame,
                        actual_eos, expected_eos);
                goto cleanup;
            }
            if (!phonon_stop_after_frame(&stop_state, actual_eos, run == 0 ? FLT_MAX : 4.0f,
                                         &should_stop))
                goto cleanup;
            if (run == 1 && should_stop != (frame == 31))
                goto cleanup;
        }
        if (run == 1 && (stop_state.reason != PHONON_STOP_EOS || stop_state.frames_generated != 32))
            goto cleanup;
        if (run == 0 && (stop_state.reason != PHONON_STOP_FRAME_BUDGET || stop_state.eos_seen))
            goto cleanup;
    }
    if (fgetc(file) != EOF)
        goto cleanup;
    fprintf(stderr, "real Q8 one-step flow head parity: 40 fixed + 32 natural frames passed\n");
    result = 0;
cleanup:
    free(head);
    phonon_pack_close(pack);
    fclose(file);
    return result;
}

static int real_prefix_check(void) {
    int result = 1;
    FILE *file = NULL, *condition_file = NULL;
    int32_t *ids = NULL;
    float *voice = NULL, *expected_voice = NULL, *expected = NULL, *latents = NULL;
    float *embedded = NULL, *projected = NULL, *prefix = NULL, *condition = NULL;
    PhononVoice *voice_artifact = NULL;
    PhononPack *pack = NULL;
    PhononKvCache cache = {0};
    const char *pack_path = getenv("PHONON_NATIVE_PACK"),
               *fixture_path = getenv("PHONON_FLOW_FIXTURE"),
               *voice_path = getenv("PHONON_DEFAULT_VOICE"),
               *condition_path = getenv("PHONON_CONDITION_FIXTURE");
    if (!fixture_path)
        return 0;
    CHECK(pack_path && voice_path && condition_path);
    file = fopen(fixture_path, "rb");
    CHECK(file);
    unsigned char header[24];
    CHECK(fread(header, 1, sizeof(header), file) == sizeof(header));
    CHECK(memcmp(header, "PHFLOW1\0", 8) == 0);
    uint32_t tokens, frames, width, states;
    memcpy(&tokens, header + 8, 4);
    memcpy(&frames, header + 12, 4);
    memcpy(&width, header + 16, 4);
    memcpy(&states, header + 20, 4);
    CHECK(tokens > 0 && tokens <= 8192 && frames > 0 && frames <= tokens && width == 512 &&
          states >= tokens && states <= tokens + 8192);
    ids = malloc((size_t)tokens * sizeof(*ids));
    voice = malloc((size_t)frames * 512 * sizeof(float));
    expected_voice = malloc((size_t)frames * 512 * sizeof(float));
    expected = malloc((size_t)states * 512 * sizeof(float));
    latents = malloc((size_t)(states - tokens + 1) * 32 * sizeof(float));
    embedded = malloc((size_t)tokens * 512 * sizeof(float));
    projected = malloc((size_t)frames * 512 * sizeof(float));
    prefix = malloc((size_t)tokens * 512 * sizeof(float));
    condition = malloc(512 * sizeof(float));
    float expected_bos[512];
    CHECK(ids && voice && expected_voice && expected && latents && embedded && projected &&
          prefix && condition);
    int fixture_ok =
        fread(ids, sizeof(*ids), tokens, file) == tokens &&
        fread(voice, sizeof(float), (size_t)frames * 512, file) == (size_t)frames * 512 &&
        fread(expected_voice, sizeof(float), (size_t)frames * 512, file) == (size_t)frames * 512 &&
        fread(expected_bos, sizeof(float), 512, file) == 512 &&
        fread(latents, sizeof(float), (size_t)(states - tokens + 1) * 32, file) ==
            (size_t)(states - tokens + 1) * 32 &&
        fread(expected, sizeof(float), (size_t)states * 512, file) == (size_t)states * 512 &&
        fgetc(file) == EOF;
    fclose(file);
    file = NULL;
    CHECK(fixture_ok);
    CHECK(phonon_voice_open(voice_path, &voice_artifact));
    CHECK(phonon_voice_frames(voice_artifact) == frames);
    const float *parsed_voice = phonon_voice_latents(voice_artifact);
    for (size_t i = 0; i < (size_t)frames * 512; ++i)
        CHECK(fabsf(parsed_voice[i] - voice[i]) < 1e-7f);
    phonon_voice_close(voice_artifact);
    voice_artifact = NULL;
    CHECK(phonon_pack_open(pack_path, &pack));
    PhononTensorView storage[PHONON_FLOW_LAYERS][8];
    PhononFlowLayerWeights layers[PHONON_FLOW_LAYERS];
    PhononTensorView speakers, spkproj, durproj, voiceproj;
    CHECK(phonon_flow_load_layers(pack, storage, layers));
    CHECK(phonon_pack_tensor(
        pack, "flow_lm.condition_provider.conditioners.num_speakers.embed.weight", &speakers));
    CHECK(phonon_pack_tensor(
        pack, "flow_lm.condition_provider.conditioners.num_speakers.output_proj.weight", &spkproj));
    CHECK(phonon_pack_tensor(
        pack, "flow_lm.condition_provider.conditioners.duration_delta.output_proj.weight",
        &durproj));
    CHECK(phonon_pack_tensor(
        pack, "flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight",
        &voiceproj));
    condition_file = fopen(condition_path, "rb");
    CHECK(condition_file);
    unsigned char condition_header[16];
    CHECK(fread(condition_header, 1, sizeof(condition_header), condition_file) ==
          sizeof(condition_header));
    CHECK(memcmp(condition_header, "PHCOND1\0", 8) == 0);
    uint32_t condition_count, condition_width;
    memcpy(&condition_count, condition_header + 8, 4);
    memcpy(&condition_width, condition_header + 12, 4);
    float hf_conditions[2][512];
    int condition_ok = condition_count == 2 && condition_width == 512 &&
                       fread(hf_conditions, sizeof(float), 1024, condition_file) == 1024 &&
                       fgetc(condition_file) == EOF;
    fclose(condition_file);
    condition_file = NULL;
    CHECK(condition_ok);
    float native_condition[512];
    CHECK(phonon_flow_condition(&speakers, &spkproj, &durproj, 1, 0.003f, native_condition));
    for (size_t i = 0; i < 512; ++i)
        CHECK(fabsf(native_condition[i] - hf_conditions[0][i]) <=
              1e-5f + 1e-3f * fabsf(hf_conditions[0][i]));
    CHECK(phonon_flow_condition(&speakers, &spkproj, &durproj, 1, -0.003f, native_condition));
    for (size_t i = 0; i < 512; ++i)
        CHECK(fabsf(native_condition[i] - hf_conditions[1][i]) <=
              1e-5f + 1e-3f * fabsf(hf_conditions[1][i]));
    CHECK(phonon_flow_condition(&speakers, &spkproj, &durproj, 1, 0.0f, condition));
    CHECK(phonon_flow_embed_tokens(pack, ids, tokens, embedded));
    CHECK(phonon_voice_project(voice, frames, &voiceproj, projected));
    for (size_t i = 0; i < (size_t)frames * 512; ++i)
        CHECK(fabsf(projected[i] - expected_voice[i]) <= 1e-5f + 1e-3f * fabsf(expected_voice[i]));
    float bos[512];
    CHECK(phonon_flow_embed_bos(pack, condition, bos));
    for (size_t i = 0; i < 512; ++i)
        CHECK(fabsf(bos[i] - expected_bos[i]) <= 1e-5f + 1e-3f * fabsf(expected_bos[i]));
    CHECK(phonon_flow_prepare_prefix(ids, tokens, embedded, 5, 4, projected, frames, bos, prefix));
    CHECK(phonon_kv_cache_init(&cache, states));
    float state[512];
    double max_error = 0.0;
    for (size_t t = 0; t < tokens; ++t) {
        CHECK(phonon_flow_stack_step(layers, &cache, t, prefix + t * 512, 10000.0f, state));
        CHECK(phonon_flow_output_norm(pack, state));
        for (size_t j = 0; j < 512; ++j) {
            double err = fabs((double)state[j] - expected[t * 512 + j]);
            if (err > max_error)
                max_error = err;
            if (err > 1e-5 + 1e-3 * fabs((double)expected[t * 512 + j])) {
                fprintf(
                    stderr,
                    "real prefix mismatch token=%zu channel=%zu native=%.9g ref=%.9g max=%.9g\n", t,
                    j, state[j], expected[t * 512 + j], max_error);
                CHECK(0);
            }
        }
    }
    size_t incremental = states - tokens;
    for (size_t step = 0; step < incremental; ++step) {
        float latent_input[512], normalized_latent[32];
        CHECK(phonon_flow_normalize_latent(pack, latents + step * 32, normalized_latent));
        CHECK(phonon_flow_embed_latent(pack, normalized_latent, condition, latent_input));
        CHECK(phonon_flow_stack_step(layers, &cache, tokens + step, latent_input, 10000.0f, state));
        CHECK(phonon_flow_output_norm(pack, state));
        for (size_t j = 0; j < 512; ++j) {
            double err = fabs((double)state[j] - expected[(tokens + step) * 512 + j]);
            if (err > max_error)
                max_error = err;
            if (err > 1e-5 + 1e-3 * fabs((double)expected[(tokens + step) * 512 + j])) {
                fprintf(stderr,
                        "real incremental mismatch step=%zu channel=%zu native=%.9g ref=%.9g\n",
                        step, j, state[j], expected[(tokens + step) * 512 + j]);
                CHECK(0);
            }
        }
    }
    fprintf(stderr, "real pinned Q8 cached parity: prefill=%u incremental=%zu max_abs=%.9g\n",
            tokens, incremental, max_error);
    result = 0;
cleanup:
    if (file)
        fclose(file);
    if (condition_file)
        fclose(condition_file);
    phonon_voice_close(voice_artifact);
    phonon_pack_close(pack);
    phonon_kv_cache_destroy(&cache);
    free(ids);
    free(voice);
    free(expected_voice);
    free(expected);
    free(latents);
    free(embedded);
    free(projected);
    free(prefix);
    free(condition);
    return result;
}

#undef CHECK
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "failed: %s:%d: %s\n", __FILE__, __LINE__, #x);                        \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)

static int real_generation_fixture_check(void) {
    const char *pack_path = getenv("PHONON_NATIVE_PACK");
    const char *sampler_path = getenv("PHONON_FLOW_SAMPLER_FIXTURE");
    if (!sampler_path)
        return 0;
    const char *prefix_path = ".build/phonon-flow-prefix-v1.bin";
    if (!pack_path)
        return 1;
    int result = 1;
    FILE *prefix_file = NULL, *sampler_file = NULL;
    PhononPack *pack = NULL;
    int32_t *ids = NULL;
    float *voice = NULL, *expected_voice = NULL, *latents = NULL, *states = NULL;
    float *embedded = NULL, *projected = NULL, *prefix = NULL;
    prefix_file = fopen(prefix_path, "rb");
    sampler_file = fopen(sampler_path, "rb");
    if (!prefix_file || !sampler_file || !phonon_pack_open(pack_path, &pack))
        goto cleanup;
    unsigned char header[24];
    uint32_t tokens, voice_frames, width, expected_states;
    if (fread(header, 1, sizeof(header), prefix_file) != sizeof(header) ||
        memcmp(header, "PHFLOW1\0", 8))
        goto cleanup;
    memcpy(&tokens, header + 8, 4);
    memcpy(&voice_frames, header + 12, 4);
    memcpy(&width, header + 16, 4);
    memcpy(&expected_states, header + 20, 4);
    if (!tokens || tokens > 8192 || !voice_frames || voice_frames > tokens || width != 512 ||
        expected_states < tokens)
        goto cleanup;
    ids = malloc((size_t)tokens * sizeof(*ids));
    voice = malloc((size_t)voice_frames * 512 * sizeof(float));
    expected_voice = malloc((size_t)voice_frames * 512 * sizeof(float));
    latents = malloc((size_t)40 * 32 * sizeof(float));
    states = malloc((size_t)expected_states * 512 * sizeof(float));
    embedded = malloc((size_t)tokens * 512 * sizeof(float));
    projected = malloc((size_t)voice_frames * 512 * sizeof(float));
    prefix = malloc((size_t)tokens * 512 * sizeof(float));
    float bos_expected[512], condition[512];
    if (!ids || !voice || !expected_voice || !latents || !states || !embedded || !projected ||
        !prefix)
        goto cleanup;
    if (fread(ids, sizeof(*ids), tokens, prefix_file) != tokens ||
        fread(voice, sizeof(float), (size_t)voice_frames * 512, prefix_file) !=
            (size_t)voice_frames * 512 ||
        fread(expected_voice, sizeof(float), (size_t)voice_frames * 512, prefix_file) !=
            (size_t)voice_frames * 512 ||
        fread(bos_expected, sizeof(float), 512, prefix_file) != 512 ||
        fread(latents, sizeof(float), (size_t)(expected_states - tokens + 1) * 32, prefix_file) !=
            (size_t)(expected_states - tokens + 1) * 32 ||
        fread(states, sizeof(float), (size_t)expected_states * 512, prefix_file) !=
            (size_t)expected_states * 512)
        goto cleanup;
    PhononTensorView storage[PHONON_FLOW_LAYERS][8], speakers, speaker_projection,
        duration_projection, voice_projection;
    PhononFlowLayerWeights layers[PHONON_FLOW_LAYERS];
    if (!phonon_flow_load_layers(pack, storage, layers) ||
        !phonon_pack_tensor(
            pack, "flow_lm.condition_provider.conditioners.num_speakers.embed.weight", &speakers) ||
        !phonon_pack_tensor(
            pack, "flow_lm.condition_provider.conditioners.num_speakers.output_proj.weight",
            &speaker_projection) ||
        !phonon_pack_tensor(
            pack, "flow_lm.condition_provider.conditioners.duration_delta.output_proj.weight",
            &duration_projection) ||
        !phonon_pack_tensor(
            pack, "flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight",
            &voice_projection) ||
        !phonon_flow_condition(&speakers, &speaker_projection, &duration_projection, 1, 0.0f,
                               condition) ||
        !phonon_flow_embed_tokens(pack, ids, tokens, embedded) ||
        !phonon_voice_project(voice, voice_frames, &voice_projection, projected))
        goto cleanup;
    float bos[512];
    if (!phonon_flow_embed_bos(pack, condition, bos) ||
        !phonon_flow_prepare_prefix(ids, tokens, embedded, 5, 4, projected, voice_frames, bos,
                                    prefix)) {
        fprintf(stderr, "closed-loop prefix construction failed\n");
        goto cleanup;
    }
    for (size_t i = 0; i < 512; i++)
        if (fabsf(bos[i] - bos_expected[i]) > 1e-5f + 1e-3f * fabsf(bos_expected[i])) {
            fprintf(stderr, "closed-loop BOS mismatch channel=%zu got=%.9g expected=%.9g\n", i,
                    bos[i], bos_expected[i]);
            goto cleanup;
        }
    unsigned char sampler_magic[8];
    uint32_t runs;
    if (fread(sampler_magic, 1, 8, sampler_file) != 8 || memcmp(sampler_magic, "PHSAMP1\0", 8) ||
        fread(&runs, sizeof(runs), 1, sampler_file) != 1 || runs != 2)
        goto cleanup;
    float replay_draws[32 * 32], expected_latents[32 * 32], expected_eos[32];
    for (size_t run = 0; run < 2; run++) {
        uint32_t frames;
        if (fread(&frames, sizeof(frames), 1, sampler_file) != 1 ||
            frames != (run == 0 ? 40U : 32U))
            goto cleanup;
        for (size_t frame = 0; frame < frames; frame++) {
            float saved_condition[512], raw[32], integrated[32], output[32], eos;
            if (fread(saved_condition, sizeof(float), 512, sampler_file) != 512 ||
                fread(raw, sizeof(float), 32, sampler_file) != 32 ||
                fread(integrated, sizeof(float), 32, sampler_file) != 32 ||
                fread(output, sizeof(float), 32, sampler_file) != 32 ||
                fread(&eos, sizeof(float), 1, sampler_file) != 1)
                goto cleanup;
            if (run == 1) {
                memcpy(replay_draws + frame * 32, raw, sizeof(raw));
                memcpy(expected_latents + frame * 32, output, sizeof(output));
                expected_eos[frame] = eos;
            }
        }
    }
    PhononNoiseSource replay;
    PhononGenerationConfig config = {.frame_budget = 32,
                                     .frames_after_eos = 3,
                                     .eos_threshold = 4.0f,
                                     .temperature = 0.8f,
                                     .rope_theta = 10000.0f};
    PhononGenerationResult generated;
    float actual[32 * 32], trace_conditions[32 * 512], trace_normalized[32 * 32], trace_eos[32];
    PhononGenerationTrace trace = {.capacity = 32,
                                   .conditions = trace_conditions,
                                   .normalized_latents = trace_normalized,
                                   .eos_logits = trace_eos};
    if (!phonon_noise_replay(&replay, replay_draws, 32 * 32)) {
        fprintf(stderr, "closed-loop replay setup failed\n");
        goto cleanup;
    }
    if (!phonon_flow_generate(pack, layers, condition, prefix, tokens, &replay, &config, actual, 32,
                              &trace, &generated)) {
        fprintf(stderr, "closed-loop generator failed after %zu frames\n",
                generated.frames_generated);
        goto cleanup;
    }
    if (generated.frames_generated != 32 || generated.stop_reason != PHONON_STOP_EOS ||
        replay.replay_cursor != 32 * 32) {
        fprintf(stderr, "closed-loop stop/replay mismatch frames=%zu reason=%d draws=%zu\n",
                generated.frames_generated, generated.stop_reason, replay.replay_cursor);
        goto cleanup;
    }
    double max_abs = 0.0;
    size_t budget_exceedances = 0;
    for (size_t i = 0; i < 32 * 32; i++) {
        double error = fabs((double)actual[i] - expected_latents[i]);
        if (error > max_abs)
            max_abs = error;
        if (error > 1e-5 + 1e-3 * fabs((double)expected_latents[i]))
            budget_exceedances++;
    }
    fprintf(stderr,
            "closed-loop Q8 generation observed max_abs=%.9g budget_exceedances=%zu/1024; stop=32 "
            "EOS+3\n",
            max_abs, budget_exceedances);
    FILE *trace_output = fopen(".build/phonon-flow-generated-trace-v1.bin", "wb");
    if (!trace_output)
        goto cleanup;
    const unsigned char trace_magic[8] = {'P', 'H', 'G', 'E', 'N', '1', 0, 0};
    uint32_t trace_frames = 32;
    int trace_ok =
        fwrite(trace_magic, 1, sizeof(trace_magic), trace_output) == sizeof(trace_magic) &&
        fwrite(&trace_frames, sizeof(trace_frames), 1, trace_output) == 1;
    for (size_t frame = 0; frame < 32 && trace_ok; frame++) {
        trace_ok =
            fwrite(trace.conditions + frame * 512, sizeof(float), 512, trace_output) == 512 &&
            fwrite(replay_draws + frame * 32, sizeof(float), 32, trace_output) == 32 &&
            fwrite(trace.normalized_latents + frame * 32, sizeof(float), 32, trace_output) == 32 &&
            fwrite(trace.eos_logits + frame, sizeof(float), 1, trace_output) == 1 &&
            fwrite(actual + frame * 32, sizeof(float), 32, trace_output) == 32;
    }
    fclose(trace_output);
    if (!trace_ok || fabsf(generated.last_eos_logit - expected_eos[31]) >
                         1e-5f + 1e-3f * fabsf(expected_eos[31]))
        goto cleanup;
    config.frame_budget = 1;
    config.eos_threshold = FLT_MAX;
    float seeded_first[32], seeded_repeat[32], seeded_other[32];
    PhononNoiseSource seeded_a, seeded_b, seeded_c;
    CHECK(phonon_noise_seed(&seeded_a, UINT64_C(987654321)) &&
          phonon_noise_seed(&seeded_b, UINT64_C(987654321)) &&
          phonon_noise_seed(&seeded_c, UINT64_C(123456789)));
    CHECK(phonon_flow_generate(pack, layers, condition, prefix, tokens, &seeded_a, &config,
                               seeded_first, 1, NULL, &generated));
    CHECK(generated.frames_generated == 1 && generated.stop_reason == PHONON_STOP_FRAME_BUDGET);
    CHECK(phonon_flow_generate(pack, layers, condition, prefix, tokens, &seeded_c, &config,
                               seeded_other, 1, NULL, &generated));
    CHECK(phonon_flow_generate(pack, layers, condition, prefix, tokens, &seeded_b, &config,
                               seeded_repeat, 1, NULL, &generated));
    CHECK(memcmp(seeded_first, seeded_repeat, sizeof(seeded_first)) == 0);
    CHECK(memcmp(seeded_first, seeded_other, sizeof(seeded_first)) != 0);

    config.frame_budget = 33;
    config.eos_threshold = FLT_MAX;
    phonon_noise_reset(&replay);
    float partial_output[33 * 32];
    if (phonon_flow_generate(pack, layers, condition, prefix, tokens, &replay, &config,
                             partial_output, 33, NULL, &generated) ||
        generated.frames_generated != 32 || generated.stop_reason != PHONON_STOP_NONE)
        goto cleanup;
    fprintf(
        stderr,
        "closed-loop Q8 generation completed: 32 frames, EOS+3 tail; replay exhaustion rejected\n");
    result = 0;
cleanup:
    free(ids);
    free(voice);
    free(expected_voice);
    free(latents);
    free(states);
    free(embedded);
    free(projected);
    free(prefix);
    phonon_pack_close(pack);
    if (prefix_file)
        fclose(prefix_file);
    if (sampler_file)
        fclose(sampler_file);
    return result;
}

int main(void) {
    CHECK(real_sampler_fixture_check() == 0);
    CHECK(real_generation_fixture_check() == 0);
    float x[] = {1.0f, -2.0f, 3.0f}, w[] = {2.0f, 0.5f, -1.0f}, y[3];
    CHECK(phonon_rms_norm(x, w, 3, 1e-5f, y));
    float inv = 1.0f / sqrtf((14.0f / 3.0f) + 1e-5f);
    CHECK(fabsf(y[0] - 2.0f * inv) < 1e-6f);
    CHECK(fabsf(y[1] + inv) < 1e-6f);
    CHECK(fabsf(y[2] + 3.0f * inv) < 1e-6f);
    float lnw[] = {1.0f, 2.0f, 0.5f}, lnb[] = {0.0f, 0.1f, -0.1f};
    CHECK(phonon_layer_norm(x, lnw, lnb, 3, 1e-5f, y));
    CHECK(fabsf(y[0] - (1.0f / 3.0f) / sqrtf(38.0f / 9.0f + 1e-5f)) < 1e-5f);
    CHECK(fabsf(y[1] - ((-8.0f / 3.0f) / sqrtf(38.0f / 9.0f + 1e-5f) * 2.0f + 0.1f)) < 1e-5f);
    CHECK(fabsf(y[2] - ((7.0f / 3.0f) / sqrtf(38.0f / 9.0f + 1e-5f) * 0.5f - 0.1f)) < 1e-5f);
    float g[] = {-1.0f, 0.0f, 1.0f};
    CHECK(phonon_erf_gelu(g, 3));
    CHECK(fabsf(g[0] - (-0.15865526f)) < 1e-6f);
    CHECK(g[1] == 0.0f && fabsf(g[2] - 0.8413447f) < 1e-6f);
    float q[] = {1.0f, 0.0f, 0.0f, 1.0f};
    CHECK(phonon_rope_interleaved(q, 4, 1, 1, 10000.0f));
    CHECK(fabsf(q[0] - cosf(1.0f)) < 1e-6f);
    CHECK(fabsf(q[1] - sinf(1.0f)) < 1e-6f);
    CHECK(fabsf(q[2] + sinf(0.01f)) < 1e-6f);
    CHECK(fabsf(q[3] - cosf(0.01f)) < 1e-6f);
    CHECK(!phonon_rope_interleaved(q, 4, 3, 0, 10000.0f));
    CHECK(real_prefix_check() == 0);
    PhononKvCache cache;
    CHECK(phonon_kv_cache_init(&cache, 2));
    PhononKvCache other;
    CHECK(phonon_kv_cache_init(&other, 2));
    float k[PHONON_FLOW_WIDTH], v[PHONON_FLOW_WIDTH];
    for (size_t i = 0; i < PHONON_FLOW_WIDTH; ++i) {
        k[i] = (float)i;
        v[i] = -(float)i;
    }
    CHECK(phonon_kv_cache_append(&cache, 8, 1, k, v, PHONON_FLOW_WIDTH));
    CHECK(cache.keys[(8 * 2 + 1) * PHONON_FLOW_WIDTH + 7] == 7.0f);
    CHECK(cache.length == 0);
    CHECK(phonon_kv_cache_commit_position(&cache, 0));
    CHECK(cache.length == 1);
    CHECK(!phonon_kv_cache_commit_position(&cache, 0));
    CHECK(other.length == 0 && other.keys != cache.keys);
    CHECK(!phonon_kv_cache_append(&cache, 9, 0, k, v, PHONON_FLOW_WIDTH));
    phonon_kv_cache_reset(&cache);
    CHECK(cache.length == 0);
    phonon_kv_cache_destroy(&cache);
    phonon_kv_cache_destroy(&other);
    CHECK(cache.keys == NULL && cache.values == NULL);
    uint16_t bf16[] = {0x3f80, 0xc000, 0x4000, 0x3f00};
    PhononTensorView matrix = {.dtype = PHONON_PACK_BF16,
                               .rank = 2,
                               .shape = {2, 2},
                               .data = bf16,
                               .byte_count = sizeof(bf16)};
    float input[] = {3.0f, 2.0f}, output[2];
    CHECK(phonon_linear(&matrix, input, 2, 2, NULL, output));
    CHECK(output[0] == -1.0f && output[1] == 7.0f);
    uint16_t speakers[32 * 16] = {0}, speaker_proj[512 * 16] = {0}, duration_proj[512 * 128] = {0};
    speakers[16] = 0x3f80;
    speaker_proj[0] = 0x4000;
    duration_proj[0] = 0x3f80;
    duration_proj[64] = 0x3f80;
    duration_proj[32] = 0x3f80;
    duration_proj[96] = 0x3f80;
    duration_proj[127] = 0x3f80;
    PhononTensorView sv = {.dtype = PHONON_PACK_BF16,
                           .rank = 2,
                           .shape = {32, 16},
                           .data = speakers,
                           .byte_count = sizeof(speakers)};
    PhononTensorView sp = {.dtype = PHONON_PACK_BF16,
                           .rank = 2,
                           .shape = {512, 16},
                           .data = speaker_proj,
                           .byte_count = sizeof(speaker_proj)};
    PhononTensorView dp = {.dtype = PHONON_PACK_BF16,
                           .rank = 2,
                           .shape = {512, 128},
                           .data = duration_proj,
                           .byte_count = sizeof(duration_proj)};
    float condition[512];
    CHECK(phonon_flow_condition(&sv, &sp, &dp, 1, 0.0f, condition));
    CHECK(fabsf(condition[0] - 4.0f) < 1e-6f);
    CHECK(phonon_flow_condition(&sv, &sp, &dp, 1, 0.003f, condition));
    float phase32 = 3.0f / powf(10000.0f, 32.0f / 63.0f), phase63 = 3.0f / 10000.0f;
    CHECK(fabsf(condition[0] - (2.0f + cosf(3.0f) + sinf(3.0f) + cosf(phase32) + sinf(phase32) +
                                sinf(phase63))) < 1e-6f);
    CHECK(phonon_flow_condition(&sv, &sp, &dp, 1, -0.003f, condition));
    CHECK(fabsf(condition[0] - (2.0f + cosf(-3.0f) + sinf(-3.0f) + cosf(-phase32) + sinf(-phase32) +
                                sinf(-phase63))) < 1e-6f);
    CHECK(!phonon_flow_condition(&sv, &sp, &dp, 32, 0.0f, condition));
    int32_t ids[] = {3, 5, 5, 4};
    float embeds[4 * 512] = {0}, voices[2 * 512], bos[512], prefix[4 * 512];
    for (size_t i = 0; i < 2 * 512; ++i)
        voices[i] = 10.0f + (float)i;
    for (size_t i = 0; i < 512; ++i)
        bos[i] = 20.0f;
    CHECK(phonon_flow_prepare_prefix(ids, 4, embeds, 5, 4, voices, 2, bos, prefix));
    CHECK(prefix[0] == 0.0f && prefix[512] == 10.0f && prefix[2 * 512] == 522.0f &&
          prefix[3 * 512] == 20.0f);
    CHECK(!phonon_flow_prepare_prefix(ids, 4, embeds, 5, 4, voices, 1, bos, prefix));
    size_t counts[] = {1536U * 512U, 512U * 512U, 2048U * 512U, 512U * 2048U};
    uint16_t *zero[4];
    for (size_t i = 0; i < 4; ++i) {
        zero[i] = calloc(counts[i], sizeof(uint16_t));
        CHECK(zero[i]);
    }
    uint16_t ones[512], biases[512] = {0};
    for (size_t i = 0; i < 512; ++i)
        ones[i] = 0x3f80;
    PhononTensorView views[4] = {{.dtype = PHONON_PACK_BF16,
                                  .rank = 2,
                                  .shape = {1536, 512},
                                  .data = zero[0],
                                  .byte_count = counts[0] * 2},
                                 {.dtype = PHONON_PACK_BF16,
                                  .rank = 2,
                                  .shape = {512, 512},
                                  .data = zero[1],
                                  .byte_count = counts[1] * 2},
                                 {.dtype = PHONON_PACK_BF16,
                                  .rank = 2,
                                  .shape = {2048, 512},
                                  .data = zero[2],
                                  .byte_count = counts[2] * 2},
                                 {.dtype = PHONON_PACK_BF16,
                                  .rank = 2,
                                  .shape = {512, 2048},
                                  .data = zero[3],
                                  .byte_count = counts[3] * 2}};
    PhononTensorView nw = {
        .dtype = PHONON_PACK_BF16, .rank = 1, .shape = {512}, .data = ones, .byte_count = 1024};
    PhononTensorView nb = {
        .dtype = PHONON_PACK_BF16, .rank = 1, .shape = {512}, .data = biases, .byte_count = 1024};
    PhononFlowLayerWeights lw = {.qkv = &views[0],
                                 .attention_out = &views[1],
                                 .mlp_in = &views[2],
                                 .mlp_out = &views[3],
                                 .norm1_weight = &nw,
                                 .norm1_bias = &nb,
                                 .norm2_weight = &nw,
                                 .norm2_bias = &nb};
    float state[512], next[512];
    for (size_t i = 0; i < 512; ++i)
        state[i] = (float)i / 512.0f;
    PhononKvCache layer_cache;
    CHECK(phonon_kv_cache_init(&layer_cache, 2));
    CHECK(phonon_flow_layer_step(&lw, &layer_cache, 0, 0, state, 10000.0f, next));
    for (size_t i = 0; i < 512; ++i)
        CHECK(next[i] == state[i]);
    PhononFlowLayerWeights stack[PHONON_FLOW_LAYERS];
    for (size_t i = 0; i < PHONON_FLOW_LAYERS; ++i)
        stack[i] = lw;
    CHECK(phonon_flow_stack_step(stack, &layer_cache, 0, state, 10000.0f, next));
    CHECK(layer_cache.length == 1);
    for (size_t i = 0; i < 512; ++i)
        CHECK(next[i] == state[i]);
    CHECK(phonon_flow_stack_step(stack, &layer_cache, 1, state, 10000.0f, next));
    CHECK(layer_cache.length == 2);
    phonon_kv_cache_destroy(&layer_cache);
    /* Nonzero value and output paths make causal attention observably depend on the cache. */
    zero[0][1024U * 512U] = 0x3f80; /* V[0,0] */
    zero[1][0] = 0x3f80;            /* output[0,0] */
    PhononKvCache causal;
    CHECK(phonon_kv_cache_init(&causal, 2));
    float first[512] = {0}, second[512] = {0}, out0[512], out1[512];
    first[0] = 1.0f;
    second[0] = 3.0f;
    CHECK(phonon_flow_stack_step(stack, &causal, 0, first, 10000.0f, out0));
    CHECK(phonon_flow_stack_step(stack, &causal, 1, second, 10000.0f, out1));
    CHECK(out0[0] > 1.0f && out1[0] > 3.0f);
    phonon_kv_cache_destroy(&causal);
    PhononKvCache req_a, req_b, req_reference;
    CHECK(phonon_kv_cache_init(&req_a, 2));
    CHECK(phonon_kv_cache_init(&req_b, 2));
    CHECK(phonon_kv_cache_init(&req_reference, 2));
    float divergent[512] = {0}, shared[512] = {0}, aout[512], bout[512], refout[512];
    divergent[0] = 1.0f;
    shared[0] = 3.0f;
    CHECK(phonon_flow_stack_step(stack, &req_a, 0, divergent, 10000.0f, aout));
    divergent[0] = -7.0f;
    CHECK(phonon_flow_stack_step(stack, &req_b, 0, divergent, 10000.0f, bout));
    CHECK(phonon_flow_stack_step(stack, &req_a, 1, shared, 10000.0f, aout));
    CHECK(phonon_flow_stack_step(stack, &req_reference, 0, (float[512]){[0] = 1.0f}, 10000.0f,
                                 refout));
    CHECK(phonon_flow_stack_step(stack, &req_reference, 1, shared, 10000.0f, refout));
    for (size_t i = 0; i < 512; ++i)
        CHECK(fabsf(aout[i] - refout[i]) < 1e-6f);
    phonon_kv_cache_reset(&req_a);
    CHECK(phonon_flow_stack_step(stack, &req_a, 0, (float[512]){[0] = 1.0f}, 10000.0f, aout));
    CHECK(req_a.length == 1 && fabsf(aout[0] - bout[0]) > 1e-4f);
    phonon_kv_cache_destroy(&req_a);
    phonon_kv_cache_destroy(&req_b);
    phonon_kv_cache_destroy(&req_reference);
    for (size_t i = 0; i < 4; ++i)
        free(zero[i]);
    puts("phonon transformer primitives: ok");
    return 0;
}

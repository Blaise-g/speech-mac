#include "speech.h"
#include "english_normalize.h"
#include "phonon_pcm.h"
#include "phonon_prompt_plan.h"
#include "phonon_sampler.h"
#include "speech_profile.h"
#include "speech_test.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

struct SpeechEngine {
    PhononPack *pack;
    PhononTensorView storage[PHONON_FLOW_LAYERS][8];
    PhononFlowLayerWeights layers[PHONON_FLOW_LAYERS];
    PhononFlowHead head;
    PhononTensorView speakers, speaker_projection, duration_projection, voice_projection;
    PhononPCMModel *pcm;
};
SpeechOptions speech_options_default(void) {
    return (SpeechOptions){.language = "en", .seed = 0, .max_frames = SPEECH_MAX_REQUEST_FRAMES};
}
void speech_engine_destroy(SpeechEngine *e) {
    if (!e)
        return;
    phonon_pcm_model_destroy(e->pcm);
    phonon_pack_close(e->pack);
    free(e);
}
SpeechStatus speech_engine_create(const char *path, SpeechEngine **out) {
    if (!out)
        return SPEECH_INVALID;
    *out = NULL;
    if (!path)
        return SPEECH_INVALID;
    /* Allocate in its final location: layer pointers target storage inside e. */
    SpeechEngine *e = calloc(1, sizeof(*e));
    if (!e)
        return SPEECH_FAILED;
    if (!phonon_pack_open(path, &e->pack) || !phonon_pack_prepare_linears(e->pack) ||
        !phonon_flow_load_layers(e->pack, e->storage, e->layers) ||
        !phonon_flow_head_load(e->pack, &e->head) ||
        !phonon_pack_tensor(e->pack,
                            "flow_lm.condition_provider.conditioners.num_speakers.embed.weight",
                            &e->speakers) ||
        !phonon_pack_tensor(
            e->pack, "flow_lm.condition_provider.conditioners.num_speakers.output_proj.weight",
            &e->speaker_projection) ||
        !phonon_pack_tensor(
            e->pack, "flow_lm.condition_provider.conditioners.duration_delta.output_proj.weight",
            &e->duration_projection) ||
        !phonon_pack_tensor(
            e->pack, "flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight",
            &e->voice_projection) ||
        !phonon_pcm_model_create(e->pack, &e->pcm)) {
        speech_engine_destroy(e);
        return SPEECH_FAILED;
    }
    *out = e;
    return SPEECH_OK;
}
typedef void (*FrameObserver)(void *, size_t, const float *, const float *, float);
typedef struct {
    const SpeechOptions *options;
    int stopped;
    FrameObserver observer;
    void *observer_context;
} Cancellation;
static int poll(void *context) {
    Cancellation *c = context;
    if (!c->stopped && c->options->cancel)
        c->stopped = c->options->cancel(c->options->cancel_context) != 0;
    return c->stopped;
}
/* One chunk owns all numerical mutable state. The noise source belongs to the
 * request so chunk resets never reseed it. Full-prefix batching is intentional. */
static SpeechStatus chunk(const SpeechEngine *e, const PhononPromptChunk *c, const float *projected,
                          size_t voice_frames, const float condition[512], const SpeechOptions *o,
                          PhononNoiseSource *noise, PhononPCMState *pcm, SpeechPCMCallback callback,
                          void *context, SpeechResult *result, Cancellation *cancel) {
    size_t prefix_length = voice_frames + c->tokens.length + 1;
    if (prefix_length > 8192 || c->frame_budget > 8192 - prefix_length)
        return SPEECH_LIMIT;
    float *prefix = malloc(prefix_length * 512 * sizeof(float));
    int32_t *ids = malloc(prefix_length * sizeof(*ids));
    PhononKvCache cache = {0};
    SpeechStatus status = SPEECH_FAILED;
    float state[512], bos[512];
    if (!prefix || !ids || !phonon_kv_cache_init(&cache, prefix_length + c->frame_budget))
        goto done;
    for (size_t i = 0; i < voice_frames; i++)
        ids[i] = 5;
    for (size_t i = 0; i < c->tokens.length; i++)
        ids[voice_frames + i] = (int32_t)c->tokens.ids[i];
    ids[prefix_length - 1] = 4;
    if (poll(cancel)) {
        status = SPEECH_CANCELLED;
        goto done;
    }
    if (!phonon_flow_embed_tokens(e->pack, ids, prefix_length, prefix) ||
        !phonon_flow_embed_bos(e->pack, condition, bos))
        goto done;
    memcpy(prefix, projected, voice_frames * 512 * sizeof(float));
    memcpy(prefix + (prefix_length - 1) * 512, bos, sizeof(bos));
    if (poll(cancel)) {
        status = SPEECH_CANCELLED;
        goto done;
    }
    SPEECH_PROFILE_BEGIN("prefill");
    if (!phonon_flow_stack_prefill_cancel(e->layers, &cache, prefix, prefix_length, 10000.f, state,
                                          poll, cancel) ||
        !phonon_flow_output_norm(e->pack, state))
        goto done;
    SPEECH_PROFILE_END("prefill");
    PhononStopState stop;
    if (!phonon_stop_init(&stop, c->frame_budget, c->frames_after_eos))
        goto done;
    for (size_t frame = 0; frame < c->frame_budget; frame++) {
        float logit, raw[32], normalized[32], latent[32], samples[1920];
        if (poll(cancel)) {
            status = SPEECH_CANCELLED;
            goto done;
        }
        if (result->frames_generated == o->max_frames) {
            status = SPEECH_LIMIT;
            goto done;
        }
        SPEECH_PROFILE_BEGIN("flow_head");
        if (!phonon_flow_eos_logit(e->pack, state, &logit) || !phonon_noise_frame(noise, raw) ||
            !phonon_flow_lsd_one_step_cancel(&e->head, state, raw, .8f, normalized, poll, cancel) ||
            !phonon_flow_denormalize_latent(e->pack, normalized, latent))
            goto done;
        SPEECH_PROFILE_END("flow_head");
        if (cancel->observer)
            cancel->observer(cancel->observer_context, result->frames_generated, state, normalized,
                             logit);
        if (poll(cancel)) {
            status = SPEECH_CANCELLED;
            goto done;
        }
        SPEECH_PROFILE_BEGIN("mimi_pcm");
        if (!phonon_pcm_process(pcm, latent, 1, samples, 1920, NULL, 0))
            goto done;
        SPEECH_PROFILE_END("mimi_pcm");
        if (poll(cancel)) {
            status = SPEECH_CANCELLED;
            goto done;
        }
        int stop_now;
        if (!phonon_stop_after_frame(&stop, logit, 4.f, &stop_now))
            goto done;
        result->frames_generated++;
        result->samples_generated += 1920;
        int accepted = callback(context, samples, 1920, result);
        if (accepted) {
            status = accepted == 1 ? SPEECH_CANCELLED : SPEECH_CALLBACK_ERROR;
            goto done;
        }
        if (poll(cancel)) {
            status = SPEECH_CANCELLED;
            goto done;
        }
        if (stop_now) {
            status = phonon_pcm_finish(pcm) ? SPEECH_OK : SPEECH_FAILED;
            goto done;
        }
        float input[512];
        SPEECH_PROFILE_BEGIN("incremental_backbone");
        if (!phonon_flow_embed_latent(e->pack, normalized, condition, input) ||
            !phonon_flow_stack_step_cancel(e->layers, &cache, prefix_length + frame, input, 10000.f,
                                           state, poll, cancel) ||
            !phonon_flow_output_norm(e->pack, state))
            goto done;
        SPEECH_PROFILE_END("incremental_backbone");
    }
done:
    if (cancel->stopped)
        status = SPEECH_CANCELLED;
    free(prefix);
    free(ids);
    phonon_kv_cache_destroy(&cache);
    return status;
}
static SpeechStatus synthesize(const SpeechEngine *e, const PhononVoice *voice, const char *text,
                               size_t length, const SpeechOptions *options,
                               SpeechPCMCallback callback, void *context, SpeechResult *result,
                               FrameObserver observer, void *observer_context) {
    if (!result)
        return SPEECH_INVALID;
    *result = (SpeechResult){0};
    SpeechOptions defaults = speech_options_default();
    const SpeechOptions *o = options ? options : &defaults;
    if (!e || !voice || !text || !callback || !o->language || strcmp(o->language, "en") ||
        !isfinite(o->duration_delta) || !o->max_frames || o->max_frames > SPEECH_MAX_REQUEST_FRAMES)
        return SPEECH_INVALID;
    Cancellation cancellation = {
        .options = o, .observer = observer, .observer_context = observer_context};
    Cancellation *cancel = &cancellation;
    if (length > SPEECH_MAX_INPUT_BYTES)
        return SPEECH_LIMIT;
    if (poll(cancel))
        return SPEECH_CANCELLED;
    char *normalized = NULL;
    size_t normalized_length = 0;
    SpeechError error;
    SpeechIOStatus normalization =
        phonon_english_normalize("en", text, length, &normalized, &normalized_length, &error);
    if (normalization != SPEECH_IO_OK)
        return normalization == SPEECH_IO_OUT_OF_MEMORY ? SPEECH_FAILED : SPEECH_INVALID;
    PhononPromptPlan plan = {0};
    PhononPCMState *pcm = NULL;
    float *projected = NULL;
    SpeechStatus status = SPEECH_FAILED;
    if (poll(cancel)) {
        status = SPEECH_CANCELLED;
        goto done;
    }
    if (!normalized_length) {
        status = SPEECH_INVALID;
        goto done;
    }
    if (normalized_length > SPEECH_MAX_INPUT_BYTES) {
        status = SPEECH_LIMIT;
        goto done;
    }
    if (!phonon_prompt_plan(normalized, normalized_length, 50, 12.5, &plan))
        goto done;
    if (poll(cancel)) {
        status = SPEECH_CANCELLED;
        goto done;
    }
    size_t frames = phonon_voice_frames(voice);
    if (!frames || frames > 4096) {
        status = SPEECH_INVALID;
        goto done;
    }
    /* Validate every chunk before producing a partial request. Planner reserves
     * only 512 voice positions; actual voice length is authoritative here. */
    for (size_t i = 0; i < plan.count; i++) {
        size_t prefix = frames + plan.chunks[i].tokens.length + 1;
        if (prefix > 8192 || plan.chunks[i].frame_budget > 8192 - prefix) {
            status = SPEECH_LIMIT;
            goto done;
        }
    }
    if (!plan.count) {
        status = SPEECH_OK;
        goto done;
    }
    projected = malloc(frames * 512 * sizeof(float));
    float condition[512];
    PhononNoiseSource noise;
    if (!projected || !phonon_noise_seed(&noise, o->seed) ||
        !phonon_flow_condition(&e->speakers, &e->speaker_projection, &e->duration_projection, 1,
                               o->duration_delta, condition) ||
        !phonon_pcm_state_create(e->pcm, &pcm))
        goto done;
    if (phonon_voice_is_projected(voice))
        memcpy(projected, phonon_voice_latents(voice), frames * 512 * sizeof(float));
    else if (!phonon_voice_project(phonon_voice_latents(voice), frames, &e->voice_projection,
                                   projected))
        goto done;
    for (size_t i = 0; i < plan.count; i++) {
        phonon_pcm_state_reset(pcm);
        status = chunk(e, &plan.chunks[i], projected, frames, condition, o, &noise, pcm, callback,
                       context, result, cancel);
        if (status != SPEECH_OK)
            goto done;
        result->chunks_completed++;
    }
    status = SPEECH_OK;
done:
    phonon_pcm_state_destroy(pcm);
    free(projected);
    phonon_prompt_plan_free(&plan);
    phonon_english_normalize_free(normalized);
    return status;
}
SpeechStatus speech_synthesize(const SpeechEngine *e, const PhononVoice *voice, const char *text,
                               size_t length, const SpeechOptions *options,
                               SpeechPCMCallback callback, void *context, SpeechResult *result) {
    return synthesize(e, voice, text, length, options, callback, context, result, NULL, NULL);
}
#ifdef PHONON_PACK_TEST_API
SpeechStatus speech_test_synthesize(const SpeechEngine *e, const PhononVoice *voice,
                                    const char *text, size_t length, const SpeechOptions *options,
                                    SpeechPCMCallback callback, void *context, SpeechResult *result,
                                    SpeechTestFrameObserver observer, void *observer_context) {
    return synthesize(e, voice, text, length, options, callback, context, result, observer,
                      observer_context);
}
#endif
typedef struct {
    float *pcm;
    size_t length, capacity;
    int full;
} Collector;
static int collect_frame(void *context, const float *pcm, size_t count,
                         const SpeechResult *result) {
    (void)result;
    Collector *c = context;
    if (count > c->capacity - c->length) {
        c->full = 1;
        return 2;
    }
    memcpy(c->pcm + c->length, pcm, count * sizeof(float));
    c->length += count;
    return 0;
}
SpeechStatus speech_collect(const SpeechEngine *e, const PhononVoice *voice, const char *text,
                            size_t length, const SpeechOptions *o, size_t capacity, float **pcm,
                            size_t *count, SpeechResult *result) {
    if (pcm)
        *pcm = NULL;
    if (count)
        *count = 0;
    if (result)
        *result = (SpeechResult){0};
    if (!pcm || !count || !result)
        return SPEECH_INVALID;
    if (capacity > SPEECH_MAX_REQUEST_FRAMES * SPEECH_FRAME_SAMPLES)
        return SPEECH_LIMIT;
    Collector c = {.capacity = capacity};
    c.pcm = malloc((capacity ? capacity : 1) * sizeof(float));
    if (!c.pcm)
        return SPEECH_FAILED;
    SpeechStatus status = speech_synthesize(e, voice, text, length, o, collect_frame, &c, result);
    if (c.full)
        status = SPEECH_LIMIT;
    if (status == SPEECH_OK) {
        *pcm = c.pcm;
        *count = c.length;
    } else
        free(c.pcm);
    return status;
}

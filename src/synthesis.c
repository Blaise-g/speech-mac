#include "synthesis.h"

#include "context_encoder.h"
#include "decoder_conditioner.h"
#include "duration_alignment.h"
#include "kokoro_internal.h"
#include "pitch_noise_predictor.h"
#include "text_encoder.h"
#include "voice_artifact.h"

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    KokoroContextEncoder *context_encoder;
    KokoroDurationPredictor *duration_predictor;
    KokoroPitchNoisePredictor *pitch_noise_predictor;
    KokoroTextEncoder *text_encoder;
    KokoroDecoderConditioner *decoder_conditioner;
    KokoroNeuralGenerator *neural_generator;
} KokoroStages;

struct KokoroEngine {
    KokoroPack *pack;
    KokoroVoiceArtifact *voice_artifact;
    KokoroTensorView voice_style;
    KokoroStages stages;
};

static void stages_free(KokoroStages *stages) {
    if (stages == NULL) {
        return;
    }
    kokoro_neural_generator_free(stages->neural_generator);
    kokoro_decoder_conditioner_free(stages->decoder_conditioner);
    kokoro_text_encoder_free(stages->text_encoder);
    kokoro_pitch_noise_predictor_free(stages->pitch_noise_predictor);
    kokoro_duration_predictor_free(stages->duration_predictor);
    kokoro_context_encoder_free(stages->context_encoder);
    *stages = (KokoroStages){0};
}

static KokoroStatus stages_create(const KokoroPack *pack, const KokoroTensorView *voice_style,
                                  KokoroStages *stages, KokoroError *error) {
    if (pack == NULL || stages == NULL) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "model pack and stage output are required");
        return KOKORO_INVALID_ARGUMENT;
    }
    *stages = (KokoroStages){0};
    KokoroStatus status = kokoro_context_encoder_create(pack, &stages->context_encoder, error);
    if (status == KOKORO_OK) {
        status =
            kokoro_duration_predictor_create(pack, voice_style, &stages->duration_predictor, error);
    }
    if (status == KOKORO_OK) {
        status = kokoro_pitch_noise_predictor_create(pack, voice_style,
                                                     &stages->pitch_noise_predictor, error);
    }
    if (status == KOKORO_OK) {
        status = kokoro_text_encoder_create(pack, &stages->text_encoder, error);
    }
    if (status == KOKORO_OK) {
        status = kokoro_decoder_conditioner_create(pack, voice_style, &stages->decoder_conditioner,
                                                   error);
    }
    if (status == KOKORO_OK) {
        status =
            kokoro_neural_generator_create(pack, voice_style, &stages->neural_generator, error);
    }
    if (status != KOKORO_OK) {
        stages_free(stages);
    }
    return status;
}

static KokoroStatus validate_request(const int64_t *token_ids, size_t token_count,
                                     const KokoroSynthesisOptions *options,
                                     const KokoroSynthesisOutput *output, KokoroError *error) {
    if (token_ids == NULL || token_count < 2 || options == NULL || output == NULL ||
        output->durations != NULL || output->samples != NULL || output->token_count != 0 ||
        output->frame_count != 0 || output->sample_count != 0 || !isfinite(options->speed) ||
        options->speed < 0.5f || options->speed > 2.0f) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT, "invalid synthesis input or output");
        return KOKORO_INVALID_ARGUMENT;
    }
    if (token_count > KOKORO_CONTEXT_MAX_TOKENS) {
        kokoro_error_set(error, KOKORO_CONTEXT_TOO_LONG, "synthesis input exceeds %u tokens",
                         KOKORO_CONTEXT_MAX_TOKENS);
        return KOKORO_CONTEXT_TOO_LONG;
    }
    return KOKORO_OK;
}

static KokoroStatus synthesis_run(const KokoroStages *stages, const int64_t *token_ids,
                                  size_t token_count, const KokoroSynthesisOptions *options,
                                  KokoroSynthesisOutput *output, KokoroError *error) {
    KokoroStatus status = KOKORO_OK;
    float *context_features = NULL;
    KokoroDurationOutput duration = {0};
    KokoroPitchNoiseOutput pitch_noise = {0};
    KokoroTextEncoderOutput text = {0};
    KokoroDecoderOutput decoder = {0};
    KokoroSpectralOutput spectrum = {0};
    KokoroWaveformOutput waveform = {0};

    context_features = malloc(token_count * KOKORO_CONTEXT_FEATURES * sizeof(float));
    if (context_features == NULL) {
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY, "cannot allocate contextual features");
        status = KOKORO_OUT_OF_MEMORY;
        goto cleanup;
    }
    status = kokoro_context_encode(stages->context_encoder, token_ids, token_count, token_count,
                                   context_features, token_count * KOKORO_CONTEXT_FEATURES, error);
    if (status != KOKORO_OK) {
        goto cleanup;
    }
    status = kokoro_duration_predict(stages->duration_predictor, context_features, token_count,
                                     options->voice_style_index, options->speed, &duration, error);
    if (status != KOKORO_OK) {
        goto cleanup;
    }
    status = kokoro_pitch_noise_predict(stages->pitch_noise_predictor, &duration,
                                        options->voice_style_index, &pitch_noise, error);
    if (status != KOKORO_OK) {
        goto cleanup;
    }
    status = kokoro_text_encoder_run(stages->text_encoder, token_ids, token_count, token_count,
                                     &duration, &text, error);
    if (status != KOKORO_OK) {
        goto cleanup;
    }
    status = kokoro_decoder_condition(stages->decoder_conditioner, &text, &pitch_noise,
                                      options->voice_style_index, &decoder, error);
    if (status != KOKORO_OK) {
        goto cleanup;
    }

    KokoroNeuralGeneratorOptions generator_options = {
        .random_seed = options->random_seed,
        .excitation = options->reference_excitation,
        .source_spectrogram = options->reference_source_spectrogram,
    };
    status =
        kokoro_neural_generate(stages->neural_generator, &decoder, &pitch_noise,
                               options->voice_style_index, &generator_options, &spectrum, error);
    if (status != KOKORO_OK) {
        goto cleanup;
    }
    status = kokoro_istft_reconstruct(spectrum.magnitude, spectrum.phase, spectrum.frame_count,
                                      &waveform, error);
    if (status != KOKORO_OK) {
        goto cleanup;
    }

    if (duration.token_count != token_count || duration.frame_count == 0 ||
        waveform.sample_count == 0 || duration.durations == NULL) {
        kokoro_error_set(error, KOKORO_NUMERIC_ERROR,
                         "assembled synthesis output has invalid shape");
        status = KOKORO_NUMERIC_ERROR;
        goto cleanup;
    }
    output->token_count = token_count;
    output->frame_count = duration.frame_count;
    output->sample_count = waveform.sample_count;
    output->durations = duration.durations;
    output->samples = waveform.samples;
    duration.durations = NULL;
    waveform.samples = NULL;

cleanup:
    free(context_features);
    kokoro_waveform_output_free(&waveform);
    kokoro_spectral_output_free(&spectrum);
    kokoro_decoder_output_free(&decoder);
    kokoro_text_encoder_output_free(&text);
    kokoro_pitch_noise_output_free(&pitch_noise);
    kokoro_duration_output_free(&duration);
    return status;
}

KokoroStatus kokoro_synthesize(const KokoroPack *pack, const int64_t *token_ids, size_t token_count,
                               const KokoroSynthesisOptions *options, KokoroSynthesisOutput *output,
                               KokoroError *error) {
    if (pack == NULL) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT, "model pack is required");
        return KOKORO_INVALID_ARGUMENT;
    }
    KokoroStatus status = validate_request(token_ids, token_count, options, output, error);
    if (status != KOKORO_OK) {
        return status;
    }
    KokoroStages stages = {0};
    status = stages_create(pack, NULL, &stages, error);
    if (status == KOKORO_OK) {
        status = synthesis_run(&stages, token_ids, token_count, options, output, error);
    }
    stages_free(&stages);
    if (status == KOKORO_OK) {
        kokoro_error_set(error, KOKORO_OK, NULL);
    }
    return status;
}

KokoroStatus kokoro_engine_create(const char *model_pack_path, const char *voice_name,
                                  const char *voice_path, KokoroEngine **result,
                                  KokoroError *error) {
    if (result != NULL) {
        *result = NULL;
    }
    if (model_pack_path == NULL || voice_name == NULL || result == NULL) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "model pack, voice name, and engine output are required");
        return KOKORO_INVALID_ARGUMENT;
    }
    bool is_af_heart = strcmp(voice_name, "af_heart") == 0;
    bool is_am_adam = strcmp(voice_name, "am_adam") == 0;
    if (!is_af_heart && !is_am_adam) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT, "unsupported voice: %s", voice_name);
        return KOKORO_INVALID_ARGUMENT;
    }
    if ((is_af_heart && voice_path != NULL) || (is_am_adam && voice_path == NULL)) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "voice artifact path does not match the selected voice");
        return KOKORO_INVALID_ARGUMENT;
    }

    KokoroEngine *engine = calloc(1, sizeof(*engine));
    if (engine == NULL) {
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY, "cannot allocate resident engine");
        return KOKORO_OUT_OF_MEMORY;
    }
    KokoroStatus status = kokoro_pack_open(model_pack_path, &engine->pack, error);
    if (status != KOKORO_OK) {
        goto fail;
    }
    if (is_am_adam) {
        status = kokoro_voice_artifact_open(voice_path, voice_name, &engine->voice_artifact, error);
        if (status != KOKORO_OK) {
            goto fail;
        }
        status = kokoro_voice_artifact_tensor(engine->voice_artifact, &engine->voice_style, error);
        if (status != KOKORO_OK) {
            goto fail;
        }
        KokoroTensorView checked_style;
        status = kokoro_pack_voice_style(engine->pack, &engine->voice_style, &checked_style, error);
        if (status != KOKORO_OK) {
            goto fail;
        }
        engine->voice_style = checked_style;
    } else {
        status = kokoro_pack_voice_style(engine->pack, NULL, &engine->voice_style, error);
        if (status != KOKORO_OK) {
            goto fail;
        }
    }
    status = stages_create(engine->pack, &engine->voice_style, &engine->stages, error);
    if (status != KOKORO_OK) {
        goto fail;
    }
    *result = engine;
    kokoro_error_set(error, KOKORO_OK, NULL);
    return KOKORO_OK;

fail:
    kokoro_engine_free(engine);
    return status;
}

void kokoro_engine_free(KokoroEngine *engine) {
    if (engine == NULL) {
        return;
    }
    stages_free(&engine->stages);
    kokoro_voice_artifact_close(engine->voice_artifact);
    kokoro_pack_close(engine->pack);
    free(engine);
}

KokoroStatus kokoro_engine_synthesize(KokoroEngine *engine, const int64_t *token_ids,
                                      size_t token_count, const KokoroSynthesisOptions *options,
                                      KokoroSynthesisOutput *output, KokoroError *error) {
    if (engine == NULL) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT, "resident engine is required");
        return KOKORO_INVALID_ARGUMENT;
    }
    KokoroStatus status = validate_request(token_ids, token_count, options, output, error);
    if (status != KOKORO_OK) {
        return status;
    }
    status = synthesis_run(&engine->stages, token_ids, token_count, options, output, error);
    if (status == KOKORO_OK) {
        kokoro_error_set(error, KOKORO_OK, NULL);
    }
    return status;
}

void kokoro_synthesis_output_free(KokoroSynthesisOutput *output) {
    if (output == NULL) {
        return;
    }
    free(output->durations);
    free(output->samples);
    *output = (KokoroSynthesisOutput){0};
}

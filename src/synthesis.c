#include "synthesis.h"

#include "context_encoder.h"
#include "decoder_conditioner.h"
#include "duration_alignment.h"
#include "kokoro_internal.h"
#include "pitch_noise_predictor.h"
#include "text_encoder.h"

#include <math.h>
#include <stdlib.h>

KokoroStatus kokoro_synthesize(const KokoroPack *pack, const int64_t *token_ids, size_t token_count,
                               const KokoroSynthesisOptions *options, KokoroSynthesisOutput *output,
                               KokoroError *error) {
    if (pack == NULL || token_ids == NULL || token_count < 2 || options == NULL || output == NULL ||
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

    KokoroStatus status = KOKORO_OK;
    KokoroContextEncoder *context_encoder = NULL;
    KokoroDurationPredictor *duration_predictor = NULL;
    KokoroPitchNoisePredictor *pitch_noise_predictor = NULL;
    KokoroTextEncoder *text_encoder = NULL;
    KokoroDecoderConditioner *decoder_conditioner = NULL;
    KokoroNeuralGenerator *neural_generator = NULL;
    float *context_features = NULL;
    KokoroDurationOutput duration = {0};
    KokoroPitchNoiseOutput pitch_noise = {0};
    KokoroTextEncoderOutput text = {0};
    KokoroDecoderOutput decoder = {0};
    KokoroSpectralOutput spectrum = {0};
    KokoroWaveformOutput waveform = {0};

    status = kokoro_context_encoder_create(pack, &context_encoder, error);
    if (status != KOKORO_OK) {
        goto cleanup;
    }
    status = kokoro_duration_predictor_create(pack, &duration_predictor, error);
    if (status != KOKORO_OK) {
        goto cleanup;
    }
    status = kokoro_pitch_noise_predictor_create(pack, &pitch_noise_predictor, error);
    if (status != KOKORO_OK) {
        goto cleanup;
    }
    status = kokoro_text_encoder_create(pack, &text_encoder, error);
    if (status != KOKORO_OK) {
        goto cleanup;
    }
    status = kokoro_decoder_conditioner_create(pack, &decoder_conditioner, error);
    if (status != KOKORO_OK) {
        goto cleanup;
    }
    status = kokoro_neural_generator_create(pack, &neural_generator, error);
    if (status != KOKORO_OK) {
        goto cleanup;
    }

    context_features = malloc(token_count * KOKORO_CONTEXT_FEATURES * sizeof(float));
    if (context_features == NULL) {
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY, "cannot allocate contextual features");
        status = KOKORO_OUT_OF_MEMORY;
        goto cleanup;
    }
    status = kokoro_context_encode(context_encoder, token_ids, token_count, token_count,
                                   context_features, token_count * KOKORO_CONTEXT_FEATURES, error);
    if (status != KOKORO_OK) {
        goto cleanup;
    }
    status = kokoro_duration_predict(duration_predictor, context_features, token_count,
                                     options->voice_style_index, options->speed, &duration, error);
    if (status != KOKORO_OK) {
        goto cleanup;
    }
    status = kokoro_pitch_noise_predict(pitch_noise_predictor, &duration,
                                        options->voice_style_index, &pitch_noise, error);
    if (status != KOKORO_OK) {
        goto cleanup;
    }
    status = kokoro_text_encoder_run(text_encoder, token_ids, token_count, token_count, &duration,
                                     &text, error);
    if (status != KOKORO_OK) {
        goto cleanup;
    }
    status = kokoro_decoder_condition(decoder_conditioner, &text, &pitch_noise,
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
        kokoro_neural_generate(neural_generator, &decoder, &pitch_noise, options->voice_style_index,
                               &generator_options, &spectrum, error);
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
    kokoro_neural_generator_free(neural_generator);
    kokoro_decoder_conditioner_free(decoder_conditioner);
    kokoro_text_encoder_free(text_encoder);
    kokoro_pitch_noise_predictor_free(pitch_noise_predictor);
    kokoro_duration_predictor_free(duration_predictor);
    kokoro_context_encoder_free(context_encoder);
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

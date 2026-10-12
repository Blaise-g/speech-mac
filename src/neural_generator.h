#ifndef KOKORO_NEURAL_GENERATOR_H
#define KOKORO_NEURAL_GENERATOR_H

#include "decoder_conditioner.h"
#include "kokoro.h"
#include "model_pack.h"
#include "pitch_noise_predictor.h"

#include <stddef.h>
#include <stdint.h>

#define KOKORO_NEURAL_GENERATOR_SPECTRAL_BINS 11U

typedef struct KokoroNeuralGenerator KokoroNeuralGenerator;

typedef struct {
    size_t sample_count;
    /* Captured SourceModuleHnNSF outputs, each a borrowed [sample_count] array. */
    const float *harmonic;
    const float *noise;
    const float *voiced;
} KokoroNeuralGeneratorExcitation;

typedef struct {
    size_t frame_count;
    /* Borrowed channel-major [11, frame_count] arrays. */
    const float *magnitude;
    const float *phase;
} KokoroNeuralGeneratorSourceSpectrogram;

typedef struct {
    uint64_t random_seed;
    /* Optional captured excitation for deterministic reference replay; otherwise generated from F0.
     */
    const KokoroNeuralGeneratorExcitation *excitation;
    /* Optional source-STFT conditioning for deterministic learned-network replay. */
    const KokoroNeuralGeneratorSourceSpectrogram *source_spectrogram;
} KokoroNeuralGeneratorOptions;

typedef struct {
    size_t frame_count;
    /* Owned channel-major [11, frame_count] arrays, released with the source output free call. */
    float *magnitude;
    float *phase;
} KokoroNeuralGeneratorSourceOutput;

typedef struct {
    size_t input_frame_count;
    size_t frame_count;
    /* Owned channel-major [11, frame_count] outputs, released with kokoro_spectral_output_free. */
    float *magnitude;
    float *phase;
} KokoroSpectralOutput;

/* The pack must outlive the generator; it borrows voice, normalization, and convolution biases. */
KokoroStatus kokoro_neural_generator_create(const KokoroPack *pack,
                                            KokoroNeuralGenerator **generator, KokoroError *error);
void kokoro_neural_generator_free(KokoroNeuralGenerator *generator);

/* Compute the native centered 20-point Hann source STFT used to condition ISTFTNet. */
KokoroStatus kokoro_neural_generator_source_spectrogram(const float *harmonic, size_t sample_count,
                                                        KokoroNeuralGeneratorSourceOutput *output,
                                                        KokoroError *error);
void kokoro_neural_generator_source_output_free(KokoroNeuralGeneratorSourceOutput *output);

/*
 * Generate ISTFTNet's magnitude and phase from decoder features and aligned F0. The optional
 * excitation replaces the three captured SourceModuleHnNSF outputs at their exact boundary;
 * when absent, a deterministic C source is generated from F0 using random_seed. The optional
 * source spectrogram replaces only harmonic-to-STFT conditioning and supports exact learned-network
 * fixture replay. Native inference leaves it NULL. Initialize output to {0}; on success its
 * buffers are owned by the caller and released with kokoro_spectral_output_free. On failure output
 * is unchanged.
 */
KokoroStatus kokoro_neural_generate(const KokoroNeuralGenerator *generator,
                                    const KokoroDecoderOutput *decoder,
                                    const KokoroPitchNoiseOutput *pitch_noise,
                                    size_t voice_style_index,
                                    const KokoroNeuralGeneratorOptions *options,
                                    KokoroSpectralOutput *output, KokoroError *error);
void kokoro_spectral_output_free(KokoroSpectralOutput *output);

#endif

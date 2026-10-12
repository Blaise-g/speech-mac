#ifndef SYNTHESIS_H
#define SYNTHESIS_H

#include "istft.h"
#include "model_pack.h"
#include "neural_generator.h"

#include <stddef.h>
#include <stdint.h>

typedef struct {
    size_t voice_style_index;
    float speed;
    uint64_t random_seed;
    /* Optional reference inputs; leave NULL for fully native inference. */
    const KokoroNeuralGeneratorExcitation *reference_excitation;
    const KokoroNeuralGeneratorSourceSpectrogram *reference_source_spectrogram;
} KokoroSynthesisOptions;

typedef struct {
    size_t token_count;
    size_t frame_count;
    size_t sample_count;
    /* Owned per-token durations and mono waveform samples. */
    uint64_t *durations;
    float *samples;
} KokoroSynthesisOutput;

typedef struct KokoroEngine KokoroEngine;

/*
 * Run the complete v1.0 phoneme-ID-to-waveform path. IDs include Kokoro BOS/EOS and have no
 * padding. The pack is borrowed for the duration of the call. Initialize output to {0}; on
 * success its buffers are owned by the caller and released with kokoro_synthesis_output_free.
 */
KokoroStatus kokoro_synthesize(const KokoroPack *pack, const int64_t *token_ids, size_t token_count,
                               const KokoroSynthesisOptions *options, KokoroSynthesisOutput *output,
                               KokoroError *error);
/*
 * Create an engine with resident model weights and one supported voice. `voice_path` is NULL
 * for the embedded af_heart style and required for the pinned am_adam artifact.
 */
KokoroStatus kokoro_engine_create(const char *model_pack_path, const char *voice_name,
                                  const char *voice_path, KokoroEngine **engine,
                                  KokoroError *error);
void kokoro_engine_free(KokoroEngine *engine);
KokoroStatus kokoro_engine_synthesize(KokoroEngine *engine, const int64_t *token_ids,
                                      size_t token_count, const KokoroSynthesisOptions *options,
                                      KokoroSynthesisOutput *output, KokoroError *error);
void kokoro_synthesis_output_free(KokoroSynthesisOutput *output);

#endif

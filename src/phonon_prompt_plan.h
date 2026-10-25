#ifndef PHONON_PROMPT_PLAN_H
#define PHONON_PROMPT_PLAN_H
#include "phonon_tokenizer.h"
#include <stddef.h>
#include <stdint.h>
#define PHONON_PLAN_MAX_FIT_TOKENS 200U
#define PHONON_PLAN_PROMPT_HEADROOM 512U
#define PHONON_PLAN_MAX_FRAMES 4096U
#define PHONON_PLAN_MAX_CONTEXT 8192U
typedef struct {
    char *text;
    PhononTokenIds tokens;
    size_t frames_after_eos, frame_budget, seq_budget;
} PhononPromptChunk;
typedef struct {
    PhononPromptChunk *chunks;
    size_t count;
} PhononPromptPlan;
/* Prepare text, group whole sentences toward max_tokens, then fit every piece to the fixed
 * 200-token model cap. Input is ordinary normalized English UTF-8, never IPA or token IDs. */
int phonon_prepare_text_prompt(const char *text, size_t length, char **prepared,
                               size_t *prepared_length, size_t *frames_after_eos);
int phonon_prompt_plan(const char *text, size_t length, size_t max_tokens, double frame_rate,
                       PhononPromptPlan *out);
/* Refit an existing plan to a smaller hard token limit; failure leaves the original plan intact. */
int phonon_prompt_fit(PhononPromptPlan *plan, size_t max_tokens, double frame_rate);
void phonon_prompt_plan_free(PhononPromptPlan *plan);
#endif

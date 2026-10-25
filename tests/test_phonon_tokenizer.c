#include "../src/phonon_prompt_plan.h"
#include "../src/phonon_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(condition)                                                                           \
    do {                                                                                           \
        if (!(condition)) {                                                                        \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);                   \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)
struct TokenizerVector {
    const unsigned char *text;
    size_t text_length;
    const uint32_t *ids;
    size_t id_count;
    const unsigned char *decoded;
    size_t decoded_length;
};
#include "phonon_tokenizer_vectors.inc"
struct PromptExpected {
    size_t index;
    const unsigned char *text;
    size_t text_length;
    const uint32_t *ids;
    size_t id_count, tail, frames, seq;
};
struct PromptVector {
    const char *text;
    size_t text_length, grouping_max, fit_max;
    const struct PromptExpected *expected;
    size_t expected_count;
};
#include "phonon_prompt_plan_vectors.inc"
int main(void) {
    for (size_t i = 0; i < sizeof(tokenizer_vectors) / sizeof(tokenizer_vectors[0]); i++) {
        const struct TokenizerVector *v = &tokenizer_vectors[i];
        PhononTokenIds actual = {0};
        CHECK(phonon_tokenizer_encode((const char *)v->text, v->text_length, &actual));
        CHECK(actual.length == v->id_count &&
              (!actual.length ||
               memcmp(actual.ids, v->ids, actual.length * sizeof(*actual.ids)) == 0));
        char *decoded = NULL;
        size_t decoded_length = 0;
        CHECK(phonon_tokenizer_decode(actual.ids, actual.length, &decoded, &decoded_length));
        CHECK(decoded_length == v->decoded_length &&
              memcmp(decoded, v->decoded, decoded_length) == 0);
        free(decoded);
        phonon_token_ids_free(&actual);
    }
    for (size_t i = 0; i < sizeof(prompt_vectors) / sizeof(prompt_vectors[0]); i++) {
        const struct PromptVector *v = &prompt_vectors[i];
        PhononPromptPlan actual = {0};
        CHECK(phonon_prompt_plan((const char *)v->text, v->text_length, v->grouping_max, 12.5,
                                 &actual));
        if (v->fit_max)
            CHECK(phonon_prompt_fit(&actual, v->fit_max, 12.5));
        CHECK(actual.count == v->expected_count);
        for (size_t j = 0; j < actual.count; j++) {
            const struct PromptExpected *expected = &v->expected[j];
            const PhononPromptChunk *got = &actual.chunks[j];
            CHECK(expected->index == j);
            if (strlen(got->text) != expected->text_length ||
                memcmp(got->text, expected->text, expected->text_length) != 0)
                fprintf(stderr, "planner case=%zu chunk=%zu got=%s expected=%.*s\n", i, j,
                        got->text, (int)expected->text_length, expected->text);
            CHECK(strlen(got->text) == expected->text_length &&
                  memcmp(got->text, expected->text, expected->text_length) == 0);
            CHECK(got->tokens.length == expected->id_count &&
                  memcmp(got->tokens.ids, expected->ids,
                         expected->id_count * sizeof(*expected->ids)) == 0);
            CHECK(got->frames_after_eos == expected->tail &&
                  got->frame_budget == expected->frames && got->seq_budget == expected->seq);
        }
        phonon_prompt_plan_free(&actual);
    }
    PhononTokenIds ids = {0};
    CHECK(phonon_tokenizer_encode("hello", 5, &ids));
    CHECK(ids.length == 2 && ids.ids[0] == 2229 && ids.ids[1] == 805);
    char *decoded = NULL;
    size_t n = 0;
    CHECK(phonon_tokenizer_decode(ids.ids, ids.length, &decoded, &n));
    CHECK(n == 5 && memcmp(decoded, "hello", 5) == 0);
    free(decoded);
    phonon_token_ids_free(&ids);
    CHECK(phonon_tokenizer_encode("two  spaces", 11, &ids));
    const uint32_t repeated_space[] = {845, 660, 1531, 662};
    CHECK(ids.length == 4 && memcmp(ids.ids, repeated_space, sizeof(repeated_space)) == 0);
    phonon_token_ids_free(&ids);
    CHECK(phonon_tokenizer_encode("Hello, world!", 13, &ids));
    const uint32_t punctuation[] = {3248, 661, 956, 1367};
    CHECK(ids.length == 4 && memcmp(ids.ids, punctuation, sizeof(punctuation)) == 0);
    phonon_token_ids_free(&ids);
    CHECK(phonon_tokenizer_encode("hello world", 11, &ids));
    const uint32_t hello_world[] = {2229, 805, 956};
    CHECK(ids.length == 3 && memcmp(ids.ids, hello_world, sizeof(hello_world)) == 0);
    phonon_token_ids_free(&ids);
    CHECK(phonon_tokenizer_encode("café", 5, &ids));
    const uint32_t cafe[] = {723, 2758, 1053};
    CHECK(ids.length == 3 && memcmp(ids.ids, cafe, sizeof(cafe)) == 0);
    phonon_token_ids_free(&ids);
    CHECK(phonon_tokenizer_encode("🙂", strlen("🙂"), &ids));
    const uint32_t emoji[] = {660, 644, 563, 557, 534};
    CHECK(ids.length == 5 && memcmp(ids.ids, emoji, sizeof(emoji)) == 0);
    decoded = NULL;
    CHECK(phonon_tokenizer_decode(ids.ids, ids.length, &decoded, &n));
    CHECK(n == strlen("🙂") && memcmp(decoded, "🙂", n) == 0);
    free(decoded);
    phonon_token_ids_free(&ids);
    const uint32_t malformed_byte[] = {659};
    CHECK(phonon_tokenizer_decode(malformed_byte, 1, &decoded, &n));
    CHECK(n == 3 && memcmp(decoded, "\xef\xbf\xbd", 3) == 0);
    free(decoded);
    CHECK(phonon_tokenizer_encode("", 0, &ids) && ids.length == 0);
    CHECK(!phonon_tokenizer_encode("\xff", 1, &ids));
    CHECK(!phonon_tokenizer_encode("\xc0\xaf", 2, &ids));
    CHECK(!phonon_tokenizer_encode("\xf0\x9f", 2, &ids));
    const char embedded_nul[] = {'a', '\0', 'b'};
    CHECK(!phonon_tokenizer_encode(embedded_nul, sizeof(embedded_nul), &ids));
    CHECK(!phonon_tokenizer_encode("x", PHONON_TOKENIZER_MAX_INPUT_BYTES + 1, &ids));
    const uint32_t late_invalid[] = {659, 4000};
    decoded = NULL;
    CHECK(!phonon_tokenizer_decode(late_invalid, 2, &decoded, &n));
    CHECK(decoded == NULL && n == 0);
    decoded = (char *)1;
    n = 99;
    CHECK(!phonon_tokenizer_decode(malformed_byte, PHONON_TOKENIZER_MAX_IDS + 1, &decoded, &n));
    CHECK(decoded == NULL && n == 0);
    char *prepared = NULL;
    size_t prepared_length = 0, tail = 0;
    CHECK(phonon_prepare_text_prompt("  hello\nworld  ", 15, &prepared, &prepared_length, &tail));

    CHECK(prepared_length == strlen("Hello world.") &&
          memcmp(prepared, "Hello world.", prepared_length) == 0);
    CHECK(tail == 3);
    free(prepared);
    CHECK(phonon_prepare_text_prompt("ß", strlen("ß"), &prepared, &prepared_length, &tail));
    CHECK(strcmp(prepared, "SS.") == 0 && tail == 3);
    free(prepared);
    CHECK(phonon_prepare_text_prompt("🙂", strlen("🙂"), &prepared, &prepared_length, &tail));
    CHECK(strcmp(prepared, "🙂") == 0);
    free(prepared);
    CHECK(phonon_prepare_text_prompt("one\xe2\x80\x83two", 9, &prepared, &prepared_length, &tail));
    CHECK(strcmp(prepared, "One two.") == 0);
    free(prepared);
    CHECK(!phonon_prepare_text_prompt("\xff", 1, &prepared, &prepared_length, &tail));
    CHECK(prepared == NULL && prepared_length == 0 && tail == 0);
    CHECK(!phonon_prepare_text_prompt(embedded_nul, sizeof(embedded_nul), &prepared,
                                      &prepared_length, &tail));
    CHECK(prepared == NULL && prepared_length == 0 && tail == 0);
    prepared = (char *)1;
    prepared_length = 99;
    tail = 99;
    CHECK(phonon_prepare_text_prompt("  ", 2, &prepared, &prepared_length, &tail));
    CHECK(prepared != NULL && prepared_length == 0 && tail == 3);
    free(prepared);
    PhononPromptPlan plan = {0};
    CHECK(phonon_prompt_plan("Hello. Hi.", 10, 3, 12.5, &plan));
    CHECK(plan.count == 2);
    CHECK(strcmp(plan.chunks[0].text, "Hello.") == 0);
    CHECK(strcmp(plan.chunks[1].text, "Hi.") == 0);
    CHECK(plan.chunks[0].frame_budget > 0 && plan.chunks[0].seq_budget <= PHONON_PLAN_MAX_CONTEXT);
    phonon_prompt_plan_free(&plan);
    const char *fit_text = "one two three four five, six seven eight nine ten eleven twelve";
    CHECK(phonon_prompt_plan(fit_text, strlen(fit_text), 50, 12.5, &plan));
    CHECK(plan.count == 1);
    CHECK(phonon_prompt_fit(&plan, 8, 12.5));
    CHECK(plan.count == 4);
    CHECK(strcmp(plan.chunks[0].text, "One two three four five,") == 0);
    CHECK(plan.chunks[0].tokens.length == 6 && plan.chunks[0].frame_budget == 50 &&
          plan.chunks[0].seq_budget == 568 && plan.chunks[0].frames_after_eos == 1);
    CHECK(strcmp(plan.chunks[1].text, "six seven eight nine ten") == 0);
    CHECK(plan.chunks[1].tokens.length == 7 && plan.chunks[1].frame_budget == 55 &&
          plan.chunks[1].seq_budget == 574 && plan.chunks[1].frames_after_eos == 1);
    CHECK(strcmp(plan.chunks[2].text, "eleven") == 0 && plan.chunks[2].tokens.length == 5 &&
          plan.chunks[2].frame_budget == 46 && plan.chunks[2].seq_budget == 563 &&
          plan.chunks[2].frames_after_eos == 3);
    CHECK(strcmp(plan.chunks[3].text, "twelve.") == 0 && plan.chunks[3].tokens.length == 5 &&
          plan.chunks[3].frame_budget == 46 && plan.chunks[3].seq_budget == 563 &&
          plan.chunks[3].frames_after_eos == 3);
    phonon_prompt_plan_free(&plan);
    const char *ceil_fit = "red, one two three four five six seven eight nine ten eleven twelve";
    CHECK(phonon_prompt_plan(ceil_fit, strlen(ceil_fit), 50, 12.5, &plan));
    CHECK(phonon_prompt_fit(&plan, 9, 12.5));
    CHECK(plan.count == 4);
    CHECK(strcmp(plan.chunks[0].text, "Red, one two three") == 0 &&
          plan.chunks[0].tokens.length == 6);
    CHECK(strcmp(plan.chunks[1].text, "four five six seven eight") == 0 &&
          plan.chunks[1].tokens.length == 7);
    CHECK(strcmp(plan.chunks[2].text, "nine ten eleven") == 0 && plan.chunks[2].tokens.length == 7);
    CHECK(strcmp(plan.chunks[3].text, "twelve.") == 0 && plan.chunks[3].tokens.length == 5);
    phonon_prompt_plan_free(&plan);
    CHECK(phonon_prompt_plan("🙂", strlen("🙂"), 50, 12.5, &plan));
    CHECK(plan.count == 1 && plan.chunks[0].tokens.length == 5);
    CHECK(!phonon_prompt_fit(&plan, 1, 12.5));
    CHECK(plan.count == 1 && strcmp(plan.chunks[0].text, "🙂") == 0);
    phonon_prompt_plan_free(&plan);
    const char *punctuated = "One. Two! Three? Four...";
    CHECK(phonon_prompt_plan(punctuated, strlen(punctuated), 1, 12.5, &plan));
    CHECK(plan.count == 4);
    CHECK(strcmp(plan.chunks[0].text, "One.") == 0);
    CHECK(strcmp(plan.chunks[1].text, "Two!") == 0);
    CHECK(strcmp(plan.chunks[2].text, "Three?") == 0);
    CHECK(strcmp(plan.chunks[3].text, "Four...") == 0);
    phonon_prompt_plan_free(&plan);
    CHECK(phonon_prompt_plan("a.a.a", 5, 50, 12.5, &plan));
    CHECK(plan.count == 1 && strcmp(plan.chunks[0].text, "A. a. a.") == 0);
    phonon_prompt_plan_free(&plan);
    CHECK(!phonon_prompt_plan("   ", 3, 50, 12.5, &plan));
    size_t long_len = 0;
    char *long_text = malloc(3000);
    CHECK(long_text != NULL);
    for (size_t i = 0; i < 400; i++) {
        if (i)
            long_text[long_len++] = ' ';
        memcpy(long_text + long_len, "word", 4);
        long_len += 4;
    }
    CHECK(phonon_prompt_plan(long_text, long_len, 50, 12.5, &plan));
    size_t word_count = 0;
    for (size_t i = 0; i < plan.count; i++) {
        CHECK(plan.chunks[i].tokens.length <= PHONON_PLAN_MAX_FIT_TOKENS);
        for (size_t j = 0; j + 4 <= strlen(plan.chunks[i].text); j++)
            if (memcmp(plan.chunks[i].text + j, "word", 4) == 0 ||
                memcmp(plan.chunks[i].text + j, "Word", 4) == 0)
                word_count++;
    }
    CHECK(word_count == 400);
    phonon_prompt_plan_free(&plan);
    free(long_text);
    char digits[3001];
    memset(digits, '9', 3000);
    digits[3000] = 0;
    CHECK(phonon_prompt_plan(digits, 3000, 50, 12.5, &plan));
    size_t digit_count = 0;
    for (size_t i = 0; i < plan.count; i++) {
        CHECK(plan.chunks[i].tokens.length <= PHONON_PLAN_MAX_FIT_TOKENS);
        for (size_t j = 0; j < strlen(plan.chunks[i].text); j++)
            if (plan.chunks[i].text[j] == '9')
                digit_count++;
    }
    CHECK(digit_count == 3000);
    phonon_prompt_plan_free(&plan);
    CHECK(!phonon_prompt_plan(digits, 3000, 50, 100.0, &plan));
    CHECK(phonon_tokenizer_encode("<0xFF>", 6, &ids));
    const uint32_t literal_byte_token[] = {660, 659};
    CHECK(ids.length == 2 && memcmp(ids.ids, literal_byte_token, sizeof(literal_byte_token)) == 0);
    phonon_token_ids_free(&ids);
    CHECK(phonon_tokenizer_encode("<s>", 3, &ids));
    const uint32_t literal_special[] = {660, 464, 662, 466};
    CHECK(ids.length == 4 && memcmp(ids.ids, literal_special, sizeof(literal_special)) == 0);
    phonon_token_ids_free(&ids);
    const uint32_t filtered_special_in_byte_run[] = {630, 0, 554, 533};
    CHECK(phonon_tokenizer_decode(filtered_special_in_byte_run, 4, &decoded, &n));
    CHECK(n == 3 && memcmp(decoded, "\xe2\x96\x81", 3) == 0);
    free(decoded);
    puts("phonon tokenizer checks passed");
    return 0;
}

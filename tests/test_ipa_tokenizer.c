#include "ipa_tokenizer.h"

#include "context_encoder.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int check(int condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "test_ipa_tokenizer: %s\n", message);
        return 0;
    }
    return 1;
}

int main(void) {
    const char *phonemes = "həˈloʊ";
    const int64_t expected[] = {0, 50, 83, 156, 54, 57, 135, 0};
    int64_t *token_ids = NULL;
    size_t token_count = 0;
    KokoroError error = {0};
    KokoroStatus status = kokoro_ipa_encode(phonemes, &token_ids, &token_count, &error);
    int passed = check(status == KOKORO_OK, error.message);
    if (passed) {
        passed = check(token_count == sizeof(expected) / sizeof(expected[0]) &&
                           memcmp(token_ids, expected, sizeof(expected)) == 0,
                       "pinned hello phonemes did not map to the oracle IDs");
    }
    free(token_ids);

    token_ids = NULL;
    token_count = 0;
    status = kokoro_ipa_encode("", &token_ids, &token_count, &error);
    passed &= check(status == KOKORO_INVALID_ARGUMENT && token_ids == NULL && token_count == 0,
                    "empty IPA input was not rejected without output");

    const char unsupported[] = "☃";
    status = kokoro_ipa_encode(unsupported, &token_ids, &token_count, &error);
    passed &= check(status == KOKORO_INVALID_TOKEN_ID && token_ids == NULL && token_count == 0,
                    "unknown phoneme was not rejected without output");

    const char invalid_utf8[] = {(char)0xc3, (char)0x28, '\0'};
    status = kokoro_ipa_encode(invalid_utf8, &token_ids, &token_count, &error);
    passed &= check(status == KOKORO_INVALID_ARGUMENT && token_ids == NULL && token_count == 0,
                    "malformed UTF-8 was not rejected");

    char oversized[1024];
    memset(oversized, 'a', sizeof(oversized) - 1);
    oversized[sizeof(oversized) - 1] = '\0';
    status = kokoro_ipa_encode(oversized, &token_ids, &token_count, &error);
    passed &= check(status == KOKORO_CONTEXT_TOO_LONG && token_ids == NULL && token_count == 0,
                    "overlong phoneme input was not rejected");

    KokoroIpaChunks chunks = {0};
    status = kokoro_ipa_encode_chunks("həlˈoʊ\nwɜːld", true, &chunks, &error);
    const int64_t first_sentence[] = {0, 50, 83, 54, 156, 57, 135, 0};
    const int64_t second_sentence[] = {0, 16, 65, 87, 158, 54, 46, 0};
    passed &= check(
        status == KOKORO_OK && chunks.count == 2 &&
            chunks.items[0].token_count == sizeof(first_sentence) / sizeof(first_sentence[0]) &&
            memcmp(chunks.items[0].token_ids, first_sentence, sizeof(first_sentence)) == 0 &&
            chunks.items[1].token_count == sizeof(second_sentence) / sizeof(second_sentence[0]) &&
            memcmp(chunks.items[1].token_ids, second_sentence, sizeof(second_sentence)) == 0,
        "eSpeak sentence lines did not map to the expected Kokoro token IDs");
    kokoro_ipa_chunks_free(&chunks);

    char safe_split[512];
    memset(safe_split, 'a', 509);
    safe_split[509] = ' ';
    safe_split[510] = 'a';
    safe_split[511] = '\0';
    status = kokoro_ipa_encode_chunks(safe_split, false, &chunks, &error);
    passed &= check(status == KOKORO_OK && chunks.count == 2 &&
                        chunks.items[0].token_count == KOKORO_CONTEXT_MAX_TOKENS &&
                        chunks.items[0].token_ids[KOKORO_CONTEXT_MAX_TOKENS - 2] == 16 &&
                        chunks.items[1].token_count == 3 && chunks.items[1].token_ids[1] == 43,
                    "long IPA input was not split at a safe boundary without losing symbols");
    kokoro_ipa_chunks_free(&chunks);

    char exact_limit_split[514];
    memset(exact_limit_split, 'a', 510);
    exact_limit_split[510] = ' ';
    exact_limit_split[511] = 'a';
    exact_limit_split[512] = '\0';
    status = kokoro_ipa_encode_chunks(exact_limit_split, false, &chunks, &error);
    passed &= check(status == KOKORO_OK && chunks.count == 2 &&
                        chunks.items[0].token_count == KOKORO_CONTEXT_MAX_TOKENS &&
                        chunks.items[0].token_ids[KOKORO_CONTEXT_MAX_TOKENS - 2] == 43 &&
                        chunks.items[1].token_count == 3 && chunks.items[1].token_ids[1] == 43,
                    "the exact context limit did not split at the following whitespace boundary");
    kokoro_ipa_chunks_free(&chunks);

    char multibyte_sentence[523];
    multibyte_sentence[0] = 'a';
    multibyte_sentence[1] = '\n';
    for (size_t index = 0; index < 260; index++) {
        multibyte_sentence[2 + index * 2] = (char)0xc9;
        multibyte_sentence[3 + index * 2] = (char)0x99;
    }
    multibyte_sentence[522] = '\0';
    status = kokoro_ipa_encode_chunks(multibyte_sentence, true, &chunks, &error);
    passed &= check(status == KOKORO_OK && chunks.count == 2 &&
                        chunks.items[1].token_count == 263 && chunks.items[1].token_ids[1] == 16 &&
                        chunks.items[1].token_ids[2] == 83 && chunks.items[1].token_ids[261] == 83,
                    "a multi-byte sentence lost its separator under the phoneme-symbol limit");
    kokoro_ipa_chunks_free(&chunks);

    status = kokoro_ipa_encode_chunks("həlˈoʊ\nwɜːld", false, &chunks, &error);
    passed &= check(status == KOKORO_INVALID_TOKEN_ID && chunks.count == 0,
                    "prepared IPA accepted an unsupported newline symbol");

    char unsafe_split[512];
    memset(unsafe_split, 'a', sizeof(unsafe_split) - 1);
    unsafe_split[sizeof(unsafe_split) - 1] = '\0';
    status = kokoro_ipa_encode_chunks(unsafe_split, false, &chunks, &error);
    passed &= check(status == KOKORO_CONTEXT_TOO_LONG && chunks.count == 0 &&
                        strstr(error.message, "whitespace boundary") != NULL,
                    "an overlong word without a safe split was not reported clearly");
    return passed ? 0 : 1;
}

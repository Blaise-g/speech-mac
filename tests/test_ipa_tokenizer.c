#include "ipa_tokenizer.h"

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
    return passed ? 0 : 1;
}

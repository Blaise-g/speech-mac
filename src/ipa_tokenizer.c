#include "ipa_tokenizer.h"

#include "context_encoder.h"
#include "kokoro_internal.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t codepoint;
    int64_t token_id;
} PhonemeToken;

/* Character IDs come from the pinned Kokoro v1.0 config.json vocabulary. */
static const PhonemeToken vocabulary[] = {
    {UINT32_C(0x20), INT64_C(16)},    {UINT32_C(0x21), INT64_C(5)},
    {UINT32_C(0x22), INT64_C(11)},    {UINT32_C(0x28), INT64_C(12)},
    {UINT32_C(0x29), INT64_C(13)},    {UINT32_C(0x2C), INT64_C(3)},
    {UINT32_C(0x2E), INT64_C(4)},     {UINT32_C(0x3A), INT64_C(2)},
    {UINT32_C(0x3B), INT64_C(1)},     {UINT32_C(0x3F), INT64_C(6)},
    {UINT32_C(0x41), INT64_C(24)},    {UINT32_C(0x49), INT64_C(25)},
    {UINT32_C(0x4F), INT64_C(31)},    {UINT32_C(0x51), INT64_C(33)},
    {UINT32_C(0x53), INT64_C(35)},    {UINT32_C(0x54), INT64_C(36)},
    {UINT32_C(0x57), INT64_C(39)},    {UINT32_C(0x59), INT64_C(41)},
    {UINT32_C(0x61), INT64_C(43)},    {UINT32_C(0x62), INT64_C(44)},
    {UINT32_C(0x63), INT64_C(45)},    {UINT32_C(0x64), INT64_C(46)},
    {UINT32_C(0x65), INT64_C(47)},    {UINT32_C(0x66), INT64_C(48)},
    {UINT32_C(0x68), INT64_C(50)},    {UINT32_C(0x69), INT64_C(51)},
    {UINT32_C(0x6A), INT64_C(52)},    {UINT32_C(0x6B), INT64_C(53)},
    {UINT32_C(0x6C), INT64_C(54)},    {UINT32_C(0x6D), INT64_C(55)},
    {UINT32_C(0x6E), INT64_C(56)},    {UINT32_C(0x6F), INT64_C(57)},
    {UINT32_C(0x70), INT64_C(58)},    {UINT32_C(0x71), INT64_C(59)},
    {UINT32_C(0x72), INT64_C(60)},    {UINT32_C(0x73), INT64_C(61)},
    {UINT32_C(0x74), INT64_C(62)},    {UINT32_C(0x75), INT64_C(63)},
    {UINT32_C(0x76), INT64_C(64)},    {UINT32_C(0x77), INT64_C(65)},
    {UINT32_C(0x78), INT64_C(66)},    {UINT32_C(0x79), INT64_C(67)},
    {UINT32_C(0x7A), INT64_C(68)},    {UINT32_C(0xE6), INT64_C(72)},
    {UINT32_C(0xE7), INT64_C(78)},    {UINT32_C(0xF0), INT64_C(81)},
    {UINT32_C(0xF8), INT64_C(116)},   {UINT32_C(0x14B), INT64_C(112)},
    {UINT32_C(0x153), INT64_C(120)},  {UINT32_C(0x250), INT64_C(70)},
    {UINT32_C(0x251), INT64_C(69)},   {UINT32_C(0x252), INT64_C(71)},
    {UINT32_C(0x254), INT64_C(76)},   {UINT32_C(0x255), INT64_C(77)},
    {UINT32_C(0x256), INT64_C(80)},   {UINT32_C(0x259), INT64_C(83)},
    {UINT32_C(0x25A), INT64_C(85)},   {UINT32_C(0x25B), INT64_C(86)},
    {UINT32_C(0x25C), INT64_C(87)},   {UINT32_C(0x25F), INT64_C(90)},
    {UINT32_C(0x261), INT64_C(92)},   {UINT32_C(0x263), INT64_C(139)},
    {UINT32_C(0x264), INT64_C(140)},  {UINT32_C(0x265), INT64_C(99)},
    {UINT32_C(0x268), INT64_C(101)},  {UINT32_C(0x26A), INT64_C(102)},
    {UINT32_C(0x26F), INT64_C(110)},  {UINT32_C(0x270), INT64_C(111)},
    {UINT32_C(0x272), INT64_C(114)},  {UINT32_C(0x273), INT64_C(113)},
    {UINT32_C(0x274), INT64_C(115)},  {UINT32_C(0x278), INT64_C(118)},
    {UINT32_C(0x279), INT64_C(123)},  {UINT32_C(0x27B), INT64_C(126)},
    {UINT32_C(0x27D), INT64_C(129)},  {UINT32_C(0x27E), INT64_C(125)},
    {UINT32_C(0x281), INT64_C(128)},  {UINT32_C(0x282), INT64_C(130)},
    {UINT32_C(0x283), INT64_C(131)},  {UINT32_C(0x288), INT64_C(132)},
    {UINT32_C(0x28A), INT64_C(135)},  {UINT32_C(0x28B), INT64_C(136)},
    {UINT32_C(0x28C), INT64_C(138)},  {UINT32_C(0x28E), INT64_C(143)},
    {UINT32_C(0x292), INT64_C(147)},  {UINT32_C(0x294), INT64_C(148)},
    {UINT32_C(0x29D), INT64_C(103)},  {UINT32_C(0x2A3), INT64_C(18)},
    {UINT32_C(0x2A4), INT64_C(82)},   {UINT32_C(0x2A5), INT64_C(19)},
    {UINT32_C(0x2A6), INT64_C(20)},   {UINT32_C(0x2A7), INT64_C(133)},
    {UINT32_C(0x2A8), INT64_C(21)},   {UINT32_C(0x2B0), INT64_C(162)},
    {UINT32_C(0x2B2), INT64_C(164)},  {UINT32_C(0x2C8), INT64_C(156)},
    {UINT32_C(0x2CC), INT64_C(157)},  {UINT32_C(0x2D0), INT64_C(158)},
    {UINT32_C(0x303), INT64_C(17)},   {UINT32_C(0x3B2), INT64_C(75)},
    {UINT32_C(0x3B8), INT64_C(119)},  {UINT32_C(0x3C7), INT64_C(142)},
    {UINT32_C(0x1D4A), INT64_C(42)},  {UINT32_C(0x1D5D), INT64_C(22)},
    {UINT32_C(0x1D7B), INT64_C(177)}, {UINT32_C(0x2014), INT64_C(9)},
    {UINT32_C(0x201C), INT64_C(14)},  {UINT32_C(0x201D), INT64_C(15)},
    {UINT32_C(0x2026), INT64_C(10)},  {UINT32_C(0x2192), INT64_C(171)},
    {UINT32_C(0x2193), INT64_C(169)}, {UINT32_C(0x2197), INT64_C(172)},
    {UINT32_C(0x2198), INT64_C(173)}, {UINT32_C(0xAB67), INT64_C(23)},
};

static bool decode_utf8(const unsigned char *text, size_t length, size_t *offset,
                        uint32_t *codepoint) {
    unsigned char first = text[*offset];
    if (first < 0x80) {
        *codepoint = first;
        (*offset)++;
        return true;
    }
    size_t continuation_count;
    uint32_t value;
    if (first >= 0xc2 && first <= 0xdf) {
        continuation_count = 1;
        value = (uint32_t)(first & 0x1fU);
    } else if (first >= 0xe0 && first <= 0xef) {
        continuation_count = 2;
        value = (uint32_t)(first & 0x0fU);
    } else if (first >= 0xf0 && first <= 0xf4) {
        continuation_count = 3;
        value = (uint32_t)(first & 0x07U);
    } else {
        return false;
    }
    if (continuation_count > length - *offset - 1) {
        return false;
    }
    for (size_t index = 1; index <= continuation_count; index++) {
        unsigned char continuation = text[*offset + index];
        if ((continuation & 0xc0U) != 0x80U) {
            return false;
        }
        value = (value << 6) | (uint32_t)(continuation & 0x3fU);
    }
    if ((continuation_count == 2 && value < 0x800) ||
        (continuation_count == 3 && value < 0x10000) || (value >= 0xd800 && value <= 0xdfff) ||
        value > 0x10ffff) {
        return false;
    }
    *codepoint = value;
    *offset += continuation_count + 1;
    return true;
}

static bool token_for_codepoint(uint32_t codepoint, int64_t *token_id) {
    size_t lower = 0;
    size_t upper = sizeof(vocabulary) / sizeof(vocabulary[0]);
    while (lower < upper) {
        size_t middle = lower + (upper - lower) / 2;
        if (vocabulary[middle].codepoint < codepoint) {
            lower = middle + 1;
        } else {
            upper = middle;
        }
    }
    if (lower >= sizeof(vocabulary) / sizeof(vocabulary[0]) ||
        vocabulary[lower].codepoint != codepoint) {
        return false;
    }
    *token_id = vocabulary[lower].token_id;
    return true;
}

KokoroStatus kokoro_ipa_encode(const char *phonemes, int64_t **result, size_t *result_count,
                               KokoroError *error) {
    if (result != NULL) {
        *result = NULL;
    }
    if (result_count != NULL) {
        *result_count = 0;
    }
    if (phonemes == NULL || result == NULL || result_count == NULL || phonemes[0] == '\0') {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "nonempty IPA input and output pointers are required");
        return KOKORO_INVALID_ARGUMENT;
    }
    size_t byte_count = strlen(phonemes);
    if (byte_count > (SIZE_MAX / sizeof(int64_t)) - 2) {
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY, "IPA input is too large");
        return KOKORO_OUT_OF_MEMORY;
    }
    int64_t *token_ids = malloc((byte_count + 2) * sizeof(*token_ids));
    if (token_ids == NULL) {
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY, "cannot allocate IPA token IDs");
        return KOKORO_OUT_OF_MEMORY;
    }
    token_ids[0] = 0;
    size_t offset = 0;
    size_t token_count = 1;
    while (offset < byte_count) {
        uint32_t codepoint;
        if (!decode_utf8((const unsigned char *)phonemes, byte_count, &offset, &codepoint)) {
            free(token_ids);
            kokoro_error_set(error, KOKORO_INVALID_ARGUMENT, "IPA input is not valid UTF-8");
            return KOKORO_INVALID_ARGUMENT;
        }
        if (!token_for_codepoint(codepoint, &token_ids[token_count])) {
            free(token_ids);
            kokoro_error_set(error, KOKORO_INVALID_TOKEN_ID,
                             "IPA character U+%04X is not in the pinned vocabulary", codepoint);
            return KOKORO_INVALID_TOKEN_ID;
        }
        token_count++;
    }
    if (token_count + 1 > KOKORO_CONTEXT_MAX_TOKENS) {
        free(token_ids);
        kokoro_error_set(error, KOKORO_CONTEXT_TOO_LONG, "IPA input exceeds %u phoneme tokens",
                         KOKORO_CONTEXT_MAX_TOKENS - 2);
        return KOKORO_CONTEXT_TOO_LONG;
    }
    token_ids[token_count++] = 0;
    *result = token_ids;
    *result_count = token_count;
    kokoro_error_set(error, KOKORO_OK, NULL);
    return KOKORO_OK;
}

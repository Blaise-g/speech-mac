#include "ipa_tokenizer.h"

#include "context_encoder.h"
#include "kokoro_internal.h"
#include "utf8.h"

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
        if (!kokoro_utf8_decode_next((const unsigned char *)phonemes, byte_count, &offset,
                                     &codepoint)) {
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

void kokoro_ipa_chunks_free(KokoroIpaChunks *chunks) {
    if (chunks == NULL) {
        return;
    }
    for (size_t index = 0; index < chunks->count; index++) {
        free(chunks->items[index].token_ids);
    }
    free(chunks->items);
    *chunks = (KokoroIpaChunks){0};
}

static KokoroStatus append_chunk(KokoroIpaChunks *chunks, const char *phonemes, size_t length,
                                 KokoroError *error) {
    if (chunks->count >= SIZE_MAX / sizeof(*chunks->items)) {
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY, "too many IPA chunks");
        return KOKORO_OUT_OF_MEMORY;
    }
    char *chunk_text = malloc(length + 1);
    if (chunk_text == NULL) {
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY, "cannot allocate IPA chunk");
        return KOKORO_OUT_OF_MEMORY;
    }
    memcpy(chunk_text, phonemes, length);
    chunk_text[length] = '\0';
    int64_t *token_ids = NULL;
    size_t token_count = 0;
    KokoroStatus status = kokoro_ipa_encode(chunk_text, &token_ids, &token_count, error);
    free(chunk_text);
    if (status != KOKORO_OK) {
        return status;
    }
    KokoroIpaChunk *items = realloc(chunks->items, (chunks->count + 1) * sizeof(*items));
    if (items == NULL) {
        free(token_ids);
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY, "cannot allocate IPA chunk list");
        return KOKORO_OUT_OF_MEMORY;
    }
    chunks->items = items;
    chunks->items[chunks->count++] = (KokoroIpaChunk){
        .token_ids = token_ids,
        .token_count = token_count,
    };
    return KOKORO_OK;
}

static KokoroStatus encode_line(const char *line, size_t line_length, bool prefix_space,
                                KokoroIpaChunks *chunks, KokoroError *error) {
    if (line_length == 0) {
        return KOKORO_OK;
    }
    if (prefix_space && line_length == SIZE_MAX) {
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY, "IPA line is too large");
        return KOKORO_OUT_OF_MEMORY;
    }
    const size_t max_phonemes = KOKORO_CONTEXT_MAX_TOKENS - 2U;
    /* A chunk boundary already separates overlong sentences from the previous line. */
    if (prefix_space && line_length >= max_phonemes) {
        size_t offset = 0;
        size_t symbols = 0;
        while (offset < line_length && symbols < max_phonemes) {
            uint32_t codepoint;
            if (!kokoro_utf8_decode_next((const unsigned char *)line, line_length, &offset,
                                         &codepoint)) {
                kokoro_error_set(error, KOKORO_INVALID_ARGUMENT, "IPA input is not valid UTF-8");
                return KOKORO_INVALID_ARGUMENT;
            }
            symbols++;
        }
        if (symbols == max_phonemes) {
            prefix_space = false;
        }
    }
    size_t buffer_length = line_length + (prefix_space ? 1U : 0U);
    char *buffer = malloc(buffer_length);
    if (buffer == NULL) {
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY, "cannot allocate IPA line");
        return KOKORO_OUT_OF_MEMORY;
    }
    size_t line_offset = 0;
    if (prefix_space) {
        buffer[0] = ' ';
        line_offset = 1;
    }
    memcpy(buffer + line_offset, line, line_length);

    size_t start = 0;
    KokoroStatus status = KOKORO_OK;
    while (start < buffer_length && status == KOKORO_OK) {
        size_t offset = start;
        size_t codepoints = 0;
        size_t last_space_end = 0;
        while (offset < buffer_length && codepoints < max_phonemes) {
            size_t codepoint_start = offset;
            uint32_t codepoint;
            if (!kokoro_utf8_decode_next((const unsigned char *)buffer, buffer_length, &offset,
                                         &codepoint)) {
                kokoro_error_set(error, KOKORO_INVALID_ARGUMENT, "IPA input is not valid UTF-8");
                status = KOKORO_INVALID_ARGUMENT;
                break;
            }
            codepoints++;
            if (codepoint == UINT32_C(0x20) && codepoints > 1) {
                last_space_end = offset;
            }
            if (offset <= codepoint_start) {
                kokoro_error_set(error, KOKORO_INVALID_ARGUMENT, "invalid IPA character boundary");
                status = KOKORO_INVALID_ARGUMENT;
                break;
            }
        }
        if (status != KOKORO_OK) {
            break;
        }
        size_t chunk_end = offset;
        size_t next_start = offset;
        if (offset < buffer_length) {
            if (last_space_end <= start) {
                if (buffer[offset] == ' ') {
                    /* The space itself falls past the context limit; the chunk boundary separates
                     * these words without exceeding the model's phoneme limit. */
                    while (next_start < buffer_length && buffer[next_start] == ' ') {
                        next_start++;
                    }
                } else {
                    kokoro_error_set(
                        error, KOKORO_CONTEXT_TOO_LONG,
                        "cannot split an IPA word longer than %zu phoneme symbols at a "
                        "whitespace boundary",
                        max_phonemes);
                    status = KOKORO_CONTEXT_TOO_LONG;
                    break;
                }
            } else {
                chunk_end = last_space_end;
                next_start = chunk_end;
            }
        }
        status = append_chunk(chunks, buffer + start, chunk_end - start, error);
        start = next_start;
    }
    free(buffer);
    return status;
}

static KokoroStatus validate_chunk_input(const char *phonemes, size_t length, bool sentence_lines,
                                         KokoroError *error) {
    size_t offset = 0;
    while (offset < length) {
        if (sentence_lines && (phonemes[offset] == '\n' || phonemes[offset] == '\r')) {
            offset++;
            continue;
        }
        size_t start = offset;
        uint32_t codepoint;
        if (!kokoro_utf8_decode_next((const unsigned char *)phonemes, length, &offset,
                                     &codepoint)) {
            kokoro_error_set(error, KOKORO_INVALID_ARGUMENT, "IPA input is not valid UTF-8");
            return KOKORO_INVALID_ARGUMENT;
        }
        int64_t ignored_token;
        if (!token_for_codepoint(codepoint, &ignored_token)) {
            kokoro_error_set(error, KOKORO_INVALID_TOKEN_ID,
                             "IPA character U+%04X is not in the pinned vocabulary", codepoint);
            return KOKORO_INVALID_TOKEN_ID;
        }
        if (offset <= start) {
            kokoro_error_set(error, KOKORO_INVALID_ARGUMENT, "invalid IPA character boundary");
            return KOKORO_INVALID_ARGUMENT;
        }
    }
    return KOKORO_OK;
}

KokoroStatus kokoro_ipa_encode_chunks(const char *phonemes, bool sentence_lines,
                                      KokoroIpaChunks *chunks, KokoroError *error) {
    if (chunks != NULL) {
        *chunks = (KokoroIpaChunks){0};
    }
    if (phonemes == NULL || chunks == NULL || phonemes[0] == '\0') {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "nonempty IPA input and chunk output are required");
        return KOKORO_INVALID_ARGUMENT;
    }
    size_t length = strlen(phonemes);
    KokoroStatus status = validate_chunk_input(phonemes, length, sentence_lines, error);
    if (!sentence_lines) {
        if (status == KOKORO_OK) {
            status = encode_line(phonemes, length, false, chunks, error);
        }
    } else {
        size_t start = 0;
        bool have_line = false;
        while (start <= length && status == KOKORO_OK) {
            size_t end = start;
            while (end < length && phonemes[end] != '\n' && phonemes[end] != '\r') {
                end++;
            }
            if (end > start) {
                status = encode_line(phonemes + start, end - start, have_line, chunks, error);
                have_line = true;
            }
            if (end == length) {
                break;
            }
            start = end + 1;
            if (phonemes[end] == '\r' && start < length && phonemes[start] == '\n') {
                start++;
            }
        }
    }
    if (status == KOKORO_OK && chunks->count == 0) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT, "IPA input contains no phonemes");
        status = KOKORO_INVALID_ARGUMENT;
    }
    if (status != KOKORO_OK) {
        kokoro_ipa_chunks_free(chunks);
        return status;
    }
    kokoro_error_set(error, KOKORO_OK, NULL);
    return KOKORO_OK;
}

#include "utf8.h"

bool speech_utf8_decode_next(const unsigned char *text, size_t length, size_t *offset,
                             uint32_t *codepoint) {
    if (text == NULL || offset == NULL || codepoint == NULL || *offset >= length) {
        return false;
    }
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

bool speech_utf8_is_valid(const unsigned char *text, size_t length) {
    if (text == NULL && length != 0) {
        return false;
    }
    size_t offset = 0;
    while (offset < length) {
        uint32_t ignored_codepoint;
        if (!speech_utf8_decode_next(text, length, &offset, &ignored_codepoint)) {
            return false;
        }
    }
    return true;
}

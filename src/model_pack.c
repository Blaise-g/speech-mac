#include "model_pack.h"

#include "kokoro_internal.h"

#include <CommonCrypto/CommonDigest.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define PACK_HEADER_SIZE 40U
#define PACK_VERSION 1U
#define PACK_MAX_TENSORS 4096U
#define PACK_MAX_NAME 1024U
#define PACK_SHA256_SIZE 32U

static const unsigned char pack_magic[8] = {'K', 'O', 'K', 'O', 'R', 'O', '1', '\0'};

static const char *const expected_metadata[] = {
    "kokoro-native-tensor-pack",
    "hexgrad/Kokoro-82M",
    "f3ff3571791e39611d31c381e3a41a3af07b4987",
    "dfb907a02bba8152ca444717ca5d78747ccb4bec",
    "af_heart",
    "496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4",
    "5abb01e2403b072bf03d04fde160443e209d7a0dad49a423be15196b9b43c17f",
    "0ab5709b8ffab19bfd849cd11d98f75b60af7733253ad0d67b12382a102cb4ff",
};

typedef struct {
    const unsigned char *name;
    size_t name_length;
    KokoroTensorType type;
    unsigned int rank;
    uint64_t shape[KOKORO_PACK_MAX_RANK];
    uint64_t payload_offset;
    uint64_t byte_count;
} PackTensor;

typedef struct {
    uint64_t start;
    uint64_t end;
} PackRange;

struct KokoroPack {
    int descriptor;
    const unsigned char *mapping;
    size_t mapping_size;
    const unsigned char *payload;
    size_t payload_size;
    PackTensor *tensors;
    size_t tensor_count;
};

typedef struct {
    const unsigned char *data;
    size_t length;
    size_t position;
} JsonParser;

enum {
    META_FORMAT = 1U << 0,
    META_FORMAT_VERSION = 1U << 1,
    META_MODEL = 1U << 2,
    META_MODEL_REVISION = 1U << 3,
    META_SOURCE_REVISION = 1U << 4,
    META_VOICE = 1U << 5,
    META_CHECKPOINT_HASH = 1U << 6,
    META_CONFIG_HASH = 1U << 7,
    META_VOICE_HASH = 1U << 8,
    META_PAYLOAD_HASH = 1U << 9,
    META_TENSOR_COUNT = 1U << 10,
};

static uint32_t read_u32_le(const unsigned char *bytes) {
    return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) | ((uint32_t)bytes[2] << 16) |
           ((uint32_t)bytes[3] << 24);
}

static uint64_t read_u64_le(const unsigned char *bytes) {
    return (uint64_t)read_u32_le(bytes) | ((uint64_t)read_u32_le(bytes + 4) << 32);
}

static void skip_json_space(JsonParser *parser) {
    while (parser->position < parser->length) {
        unsigned char byte = parser->data[parser->position];
        if (byte != ' ' && byte != '\t' && byte != '\r' && byte != '\n') {
            break;
        }
        parser->position++;
    }
}

static bool append_utf8(char *output, size_t capacity, size_t *length, uint32_t codepoint) {
    unsigned char encoded[4];
    size_t count;
    if (codepoint <= 0x7f) {
        encoded[0] = (unsigned char)codepoint;
        count = 1;
    } else if (codepoint <= 0x7ff) {
        encoded[0] = (unsigned char)(0xc0U | (codepoint >> 6));
        encoded[1] = (unsigned char)(0x80U | (codepoint & 0x3fU));
        count = 2;
    } else if (codepoint <= 0xffff) {
        encoded[0] = (unsigned char)(0xe0U | (codepoint >> 12));
        encoded[1] = (unsigned char)(0x80U | ((codepoint >> 6) & 0x3fU));
        encoded[2] = (unsigned char)(0x80U | (codepoint & 0x3fU));
        count = 3;
    } else if (codepoint <= 0x10ffff) {
        encoded[0] = (unsigned char)(0xf0U | (codepoint >> 18));
        encoded[1] = (unsigned char)(0x80U | ((codepoint >> 12) & 0x3fU));
        encoded[2] = (unsigned char)(0x80U | ((codepoint >> 6) & 0x3fU));
        encoded[3] = (unsigned char)(0x80U | (codepoint & 0x3fU));
        count = 4;
    } else {
        return false;
    }
    if (output != NULL) {
        if (*length > capacity || count > capacity - *length) {
            return false;
        }
        memcpy(output + *length, encoded, count);
    }
    *length += count;
    return true;
}

static bool read_hex_quad(JsonParser *parser, uint32_t *value) {
    if (parser->length - parser->position < 4) {
        return false;
    }
    uint32_t result = 0;
    for (unsigned int index = 0; index < 4; index++) {
        unsigned char byte = parser->data[parser->position++];
        result <<= 4;
        if (byte >= '0' && byte <= '9') {
            result |= (uint32_t)(byte - '0');
        } else if (byte >= 'a' && byte <= 'f') {
            result |= (uint32_t)(byte - 'a' + 10);
        } else if (byte >= 'A' && byte <= 'F') {
            result |= (uint32_t)(byte - 'A' + 10);
        } else {
            return false;
        }
    }
    *value = result;
    return true;
}

static bool parse_json_string(JsonParser *parser, char *output, size_t capacity, size_t *length) {
    if (parser->position >= parser->length || parser->data[parser->position++] != '"') {
        return false;
    }
    size_t decoded_length = 0;
    while (parser->position < parser->length) {
        unsigned char byte = parser->data[parser->position++];
        if (byte == '"') {
            if (output != NULL) {
                if (decoded_length >= capacity) {
                    return false;
                }
                output[decoded_length] = '\0';
            }
            *length = decoded_length;
            return true;
        }
        if (byte < 0x20) {
            return false;
        }
        if (byte >= 0x80) {
            size_t continuation_count;
            uint32_t codepoint;
            if (byte >= 0xc2 && byte <= 0xdf) {
                continuation_count = 1;
                codepoint = (uint32_t)(byte & 0x1fU);
            } else if (byte >= 0xe0 && byte <= 0xef) {
                continuation_count = 2;
                codepoint = (uint32_t)(byte & 0x0fU);
            } else if (byte >= 0xf0 && byte <= 0xf4) {
                continuation_count = 3;
                codepoint = (uint32_t)(byte & 0x07U);
            } else {
                return false;
            }
            if (parser->length - parser->position < continuation_count) {
                return false;
            }
            for (size_t index = 0; index < continuation_count; index++) {
                unsigned char continuation = parser->data[parser->position++];
                if ((continuation & 0xc0U) != 0x80U) {
                    return false;
                }
                codepoint = (codepoint << 6) | (uint32_t)(continuation & 0x3fU);
            }
            if ((continuation_count == 2 && codepoint < 0x800) ||
                (continuation_count == 3 && codepoint < 0x10000) ||
                (codepoint >= 0xd800 && codepoint <= 0xdfff) || codepoint > 0x10ffff) {
                return false;
            }
            size_t encoded_length = continuation_count + 1;
            if (output != NULL) {
                if (decoded_length > capacity || encoded_length > capacity - decoded_length) {
                    return false;
                }
                memcpy(output + decoded_length, parser->data + parser->position - encoded_length,
                       encoded_length);
            }
            decoded_length += encoded_length;
            continue;
        }
        uint32_t codepoint = byte;
        if (byte == '\\') {
            if (parser->position >= parser->length) {
                return false;
            }
            byte = parser->data[parser->position++];
            switch (byte) {
            case '"':
            case '\\':
            case '/':
                codepoint = byte;
                break;
            case 'b':
                codepoint = '\b';
                break;
            case 'f':
                codepoint = '\f';
                break;
            case 'n':
                codepoint = '\n';
                break;
            case 'r':
                codepoint = '\r';
                break;
            case 't':
                codepoint = '\t';
                break;
            case 'u': {
                if (!read_hex_quad(parser, &codepoint)) {
                    return false;
                }
                if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
                    if (parser->length - parser->position < 6 ||
                        parser->data[parser->position] != '\\' ||
                        parser->data[parser->position + 1] != 'u') {
                        return false;
                    }
                    parser->position += 2;
                    uint32_t low;
                    if (!read_hex_quad(parser, &low) || low < 0xdc00 || low > 0xdfff) {
                        return false;
                    }
                    codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + (low - 0xdc00);
                } else if (codepoint >= 0xdc00 && codepoint <= 0xdfff) {
                    return false;
                }
                break;
            }
            default:
                return false;
            }
        }
        if (!append_utf8(output, capacity, &decoded_length, codepoint)) {
            return false;
        }
    }
    return false;
}

static bool parse_json_number(JsonParser *parser, uint64_t *integer, bool *is_unsigned_integer) {
    size_t start = parser->position;
    bool negative = false;
    if (parser->position < parser->length && parser->data[parser->position] == '-') {
        negative = true;
        parser->position++;
    }
    if (parser->position >= parser->length) {
        return false;
    }
    if (parser->data[parser->position] == '0') {
        parser->position++;
        if (parser->position < parser->length && parser->data[parser->position] >= '0' &&
            parser->data[parser->position] <= '9') {
            return false;
        }
    } else {
        if (parser->data[parser->position] < '1' || parser->data[parser->position] > '9') {
            return false;
        }
        while (parser->position < parser->length && parser->data[parser->position] >= '0' &&
               parser->data[parser->position] <= '9') {
            parser->position++;
        }
    }
    bool integral = !negative;
    if (parser->position < parser->length && parser->data[parser->position] == '.') {
        integral = false;
        parser->position++;
        size_t fraction_start = parser->position;
        while (parser->position < parser->length && parser->data[parser->position] >= '0' &&
               parser->data[parser->position] <= '9') {
            parser->position++;
        }
        if (fraction_start == parser->position) {
            return false;
        }
    }
    if (parser->position < parser->length &&
        (parser->data[parser->position] == 'e' || parser->data[parser->position] == 'E')) {
        integral = false;
        parser->position++;
        if (parser->position < parser->length &&
            (parser->data[parser->position] == '+' || parser->data[parser->position] == '-')) {
            parser->position++;
        }
        size_t exponent_start = parser->position;
        while (parser->position < parser->length && parser->data[parser->position] >= '0' &&
               parser->data[parser->position] <= '9') {
            parser->position++;
        }
        if (exponent_start == parser->position) {
            return false;
        }
    }
    if (integer != NULL) {
        uint64_t parsed = 0;
        for (size_t index = start; index < parser->position; index++) {
            unsigned char byte = parser->data[index];
            if (byte < '0' || byte > '9') {
                continue;
            }
            unsigned int digit = (unsigned int)(byte - '0');
            if (parsed > (UINT64_MAX - digit) / 10) {
                return false;
            }
            parsed = parsed * 10 + digit;
        }
        *integer = parsed;
    }
    if (is_unsigned_integer != NULL) {
        *is_unsigned_integer = integral;
    }
    return true;
}

static bool skip_json_value(JsonParser *parser, unsigned int depth);

static bool skip_json_object(JsonParser *parser, unsigned int depth) {
    parser->position++;
    skip_json_space(parser);
    if (parser->position < parser->length && parser->data[parser->position] == '}') {
        parser->position++;
        return true;
    }
    while (parser->position < parser->length) {
        size_t ignored_length;
        if (!parse_json_string(parser, NULL, 0, &ignored_length)) {
            return false;
        }
        skip_json_space(parser);
        if (parser->position >= parser->length || parser->data[parser->position++] != ':') {
            return false;
        }
        skip_json_space(parser);
        if (!skip_json_value(parser, depth + 1)) {
            return false;
        }
        skip_json_space(parser);
        if (parser->position < parser->length && parser->data[parser->position] == '}') {
            parser->position++;
            return true;
        }
        if (parser->position >= parser->length || parser->data[parser->position++] != ',') {
            return false;
        }
        skip_json_space(parser);
    }
    return false;
}

static bool skip_json_array(JsonParser *parser, unsigned int depth) {
    parser->position++;
    skip_json_space(parser);
    if (parser->position < parser->length && parser->data[parser->position] == ']') {
        parser->position++;
        return true;
    }
    while (parser->position < parser->length) {
        if (!skip_json_value(parser, depth + 1)) {
            return false;
        }
        skip_json_space(parser);
        if (parser->position < parser->length && parser->data[parser->position] == ']') {
            parser->position++;
            return true;
        }
        if (parser->position >= parser->length || parser->data[parser->position++] != ',') {
            return false;
        }
        skip_json_space(parser);
    }
    return false;
}

static bool skip_json_value(JsonParser *parser, unsigned int depth) {
    if (depth > 64) {
        return false;
    }
    skip_json_space(parser);
    if (parser->position >= parser->length) {
        return false;
    }
    unsigned char byte = parser->data[parser->position];
    if (byte == '"') {
        size_t ignored_length;
        return parse_json_string(parser, NULL, 0, &ignored_length);
    }
    if (byte == '{') {
        return skip_json_object(parser, depth);
    }
    if (byte == '[') {
        return skip_json_array(parser, depth);
    }
    if (byte == '-' || (byte >= '0' && byte <= '9')) {
        return parse_json_number(parser, NULL, NULL);
    }
    static const char *const literals[] = {"true", "false", "null"};
    for (size_t index = 0; index < sizeof(literals) / sizeof(literals[0]); index++) {
        size_t length = strlen(literals[index]);
        if (parser->length - parser->position >= length &&
            memcmp(parser->data + parser->position, literals[index], length) == 0) {
            parser->position += length;
            return true;
        }
    }
    return false;
}

static bool string_matches(const char *value, size_t length, const char *expected) {
    return length == strlen(expected) && memcmp(value, expected, length) == 0;
}

static bool metadata_key(const char *key, size_t length, const char *expected) {
    return string_matches(key, length, expected);
}

static bool parse_metadata(const unsigned char *data, size_t length, uint32_t tensor_count,
                           const char *payload_sha256) {
    JsonParser parser = {.data = data, .length = length, .position = 0};
    unsigned int seen = 0;
    bool compatible = true;
    unsigned int required = META_FORMAT | META_FORMAT_VERSION | META_MODEL | META_MODEL_REVISION |
                            META_SOURCE_REVISION | META_VOICE | META_CHECKPOINT_HASH |
                            META_CONFIG_HASH | META_VOICE_HASH | META_PAYLOAD_HASH |
                            META_TENSOR_COUNT;
    skip_json_space(&parser);
    if (parser.position >= parser.length || parser.data[parser.position++] != '{') {
        return false;
    }
    skip_json_space(&parser);
    if (parser.position < parser.length && parser.data[parser.position] == '}') {
        return false;
    }
    while (parser.position < parser.length) {
        char key[64];
        size_t key_length;
        if (!parse_json_string(&parser, key, sizeof(key) - 1, &key_length)) {
            return false;
        }
        skip_json_space(&parser);
        if (parser.position >= parser.length || parser.data[parser.position++] != ':') {
            return false;
        }
        skip_json_space(&parser);
        unsigned int bit = 0;
        const char *expected = NULL;
        if (metadata_key(key, key_length, "format")) {
            bit = META_FORMAT;
            expected = expected_metadata[0];
        } else if (metadata_key(key, key_length, "format_version")) {
            bit = META_FORMAT_VERSION;
        } else if (metadata_key(key, key_length, "model")) {
            bit = META_MODEL;
            expected = expected_metadata[1];
        } else if (metadata_key(key, key_length, "model_revision")) {
            bit = META_MODEL_REVISION;
            expected = expected_metadata[2];
        } else if (metadata_key(key, key_length, "source_revision")) {
            bit = META_SOURCE_REVISION;
            expected = expected_metadata[3];
        } else if (metadata_key(key, key_length, "voice")) {
            bit = META_VOICE;
            expected = expected_metadata[4];
        } else if (metadata_key(key, key_length, "checkpoint_sha256")) {
            bit = META_CHECKPOINT_HASH;
            expected = expected_metadata[5];
        } else if (metadata_key(key, key_length, "config_sha256")) {
            bit = META_CONFIG_HASH;
            expected = expected_metadata[6];
        } else if (metadata_key(key, key_length, "voice_sha256")) {
            bit = META_VOICE_HASH;
            expected = expected_metadata[7];
        } else if (metadata_key(key, key_length, "payload_sha256")) {
            bit = META_PAYLOAD_HASH;
            expected = payload_sha256;
        } else if (metadata_key(key, key_length, "tensor_count")) {
            bit = META_TENSOR_COUNT;
        }

        if (bit == 0) {
            if (!skip_json_value(&parser, 0)) {
                return false;
            }
        } else {
            if ((seen & bit) != 0) {
                return false;
            }
            seen |= bit;
            if (expected != NULL) {
                char value[160];
                size_t value_length;
                if (!parse_json_string(&parser, value, sizeof(value) - 1, &value_length)) {
                    return false;
                }
                if (!string_matches(value, value_length, expected)) {
                    compatible = false;
                }
            } else {
                uint64_t value;
                bool unsigned_integer;
                if (!parse_json_number(&parser, &value, &unsigned_integer)) {
                    return false;
                }
                uint64_t expected_value = bit == META_FORMAT_VERSION ? PACK_VERSION : tensor_count;
                if (!unsigned_integer || value != expected_value) {
                    compatible = false;
                }
            }
        }
        skip_json_space(&parser);
        if (parser.position < parser.length && parser.data[parser.position] == '}') {
            parser.position++;
            break;
        }
        if (parser.position >= parser.length || parser.data[parser.position++] != ',') {
            return false;
        }
        skip_json_space(&parser);
    }
    skip_json_space(&parser);
    return parser.position == parser.length && seen == required && compatible;
}

static bool valid_tensor_name(const unsigned char *name, size_t length) {
    static const char *const groups[] = {"bert", "bert_encoder", "predictor", "text_encoder",
                                         "decoder"};
    static const char voice_name[] = "voice.af_heart.style";
    if (length == sizeof(voice_name) - 1 && memcmp(name, voice_name, length) == 0) {
        return true;
    }
    size_t prefix_length = 0;
    for (size_t group = 0; group < sizeof(groups) / sizeof(groups[0]); group++) {
        size_t group_length = strlen(groups[group]);
        if (length > group_length && memcmp(name, groups[group], group_length) == 0 &&
            name[group_length] == '.') {
            prefix_length = group_length + 1;
            break;
        }
    }
    if (prefix_length == 0 || prefix_length == length) {
        return false;
    }
    bool expect_character = true;
    for (size_t index = prefix_length; index < length; index++) {
        unsigned char byte = name[index];
        bool valid = (byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') ||
                     (byte >= '0' && byte <= '9') || byte == '_';
        if (byte == '.') {
            if (expect_character) {
                return false;
            }
            expect_character = true;
        } else if (valid) {
            expect_character = false;
        } else {
            return false;
        }
    }
    return !expect_character;
}

static uint64_t name_hash(const unsigned char *name, size_t length) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t index = 0; index < length; index++) {
        hash ^= name[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static bool tensor_name_equals(const PackTensor *tensor, const unsigned char *name, size_t length) {
    return tensor->name_length == length && memcmp(tensor->name, name, length) == 0;
}

static int compare_ranges(const void *left, const void *right) {
    const PackRange *a = left;
    const PackRange *b = right;
    if (a->start < b->start) {
        return -1;
    }
    if (a->start > b->start) {
        return 1;
    }
    if (a->end < b->end) {
        return -1;
    }
    return a->end > b->end ? 1 : 0;
}

static bool sha256_hex(const unsigned char *data, size_t length, char output[65]) {
    CC_SHA256_CTX context;
    unsigned char digest[PACK_SHA256_SIZE];
    if (CC_SHA256_Init(&context) != 1) {
        return false;
    }
    size_t offset = 0;
    while (offset < length) {
        size_t remaining = length - offset;
        CC_LONG chunk = remaining > UINT32_MAX ? UINT32_MAX : (CC_LONG)remaining;
        if (CC_SHA256_Update(&context, data + offset, chunk) != 1) {
            return false;
        }
        offset += chunk;
    }
    if (CC_SHA256_Final(digest, &context) != 1) {
        return false;
    }
    static const char digits[] = "0123456789abcdef";
    for (size_t index = 0; index < sizeof(digest); index++) {
        output[index * 2] = digits[digest[index] >> 4];
        output[index * 2 + 1] = digits[digest[index] & 0x0f];
    }
    output[64] = '\0';
    return true;
}

static void pack_release(KokoroPack *pack) {
    if (pack == NULL) {
        return;
    }
    free(pack->tensors);
    if (pack->mapping != NULL) {
        (void)munmap((void *)pack->mapping, pack->mapping_size);
    }
    if (pack->descriptor >= 0) {
        (void)close(pack->descriptor);
    }
    free(pack);
}

void kokoro_pack_close(KokoroPack *pack) { pack_release(pack); }

KokoroStatus kokoro_pack_open(const char *path, KokoroPack **result, KokoroError *error) {
    if (result != NULL) {
        *result = NULL;
    }
    if (path == NULL || result == NULL) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT, "model pack path and output are required");
        return KOKORO_INVALID_ARGUMENT;
    }
    int descriptor = open(path, O_RDONLY);
    if (descriptor < 0) {
        kokoro_error_set(error, KOKORO_IO_ERROR, "cannot open model pack: %s", strerror(errno));
        return KOKORO_IO_ERROR;
    }
    struct stat info;
    if (fstat(descriptor, &info) != 0 || !S_ISREG(info.st_mode) || info.st_size < 0 ||
        (uintmax_t)info.st_size > SIZE_MAX) {
        kokoro_error_set(error, KOKORO_IO_ERROR, "cannot inspect model pack file");
        (void)close(descriptor);
        return KOKORO_IO_ERROR;
    }
    size_t mapping_size = (size_t)info.st_size;
    if (mapping_size < PACK_HEADER_SIZE) {
        kokoro_error_set(error, KOKORO_MALFORMED_PACK, "truncated pack header");
        (void)close(descriptor);
        return KOKORO_MALFORMED_PACK;
    }
    const unsigned char *mapping = mmap(NULL, mapping_size, PROT_READ, MAP_PRIVATE, descriptor, 0);
    if (mapping == MAP_FAILED) {
        kokoro_error_set(error, KOKORO_IO_ERROR, "cannot map model pack: %s", strerror(errno));
        (void)close(descriptor);
        return KOKORO_IO_ERROR;
    }
    if (memcmp(mapping, pack_magic, sizeof(pack_magic)) != 0) {
        kokoro_error_set(error, KOKORO_MALFORMED_PACK, "invalid pack magic");
        (void)munmap((void *)mapping, mapping_size);
        (void)close(descriptor);
        return KOKORO_MALFORMED_PACK;
    }
    uint32_t version = read_u32_le(mapping + 8);
    uint32_t tensor_count = read_u32_le(mapping + 12);
    uint32_t index_size = read_u32_le(mapping + 16);
    uint32_t metadata_size = read_u32_le(mapping + 20);
    uint64_t payload_size = read_u64_le(mapping + 24);
    uint64_t reserved = read_u64_le(mapping + 32);
    if (version != PACK_VERSION) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "unsupported pack version: %u", version);
        (void)munmap((void *)mapping, mapping_size);
        (void)close(descriptor);
        return KOKORO_INCOMPATIBLE_PACK;
    }
    if (reserved != 0 || tensor_count == 0 || tensor_count > PACK_MAX_TENSORS) {
        kokoro_error_set(error, KOKORO_MALFORMED_PACK, "invalid pack header fields");
        (void)munmap((void *)mapping, mapping_size);
        (void)close(descriptor);
        return KOKORO_MALFORMED_PACK;
    }
    uint64_t remaining = (uint64_t)(mapping_size - PACK_HEADER_SIZE);
    if (index_size > remaining) {
        kokoro_error_set(error, KOKORO_MALFORMED_PACK, "truncated pack index");
        (void)munmap((void *)mapping, mapping_size);
        (void)close(descriptor);
        return KOKORO_MALFORMED_PACK;
    }
    remaining -= index_size;
    if (metadata_size > remaining) {
        kokoro_error_set(error, KOKORO_MALFORMED_PACK, "truncated pack metadata");
        (void)munmap((void *)mapping, mapping_size);
        (void)close(descriptor);
        return KOKORO_MALFORMED_PACK;
    }
    remaining -= metadata_size;
    if (payload_size != remaining) {
        kokoro_error_set(error, KOKORO_MALFORMED_PACK,
                         payload_size > remaining ? "truncated pack payload"
                                                  : "trailing bytes after pack");
        (void)munmap((void *)mapping, mapping_size);
        (void)close(descriptor);
        return KOKORO_MALFORMED_PACK;
    }

    const unsigned char *metadata = mapping + PACK_HEADER_SIZE + index_size;
    const unsigned char *payload = metadata + metadata_size;
    if ((uintptr_t)payload % _Alignof(int64_t) != 0) {
        kokoro_error_set(error, KOKORO_MALFORMED_PACK,
                         "tensor payload is not aligned for native tensor access");
        (void)munmap((void *)mapping, mapping_size);
        (void)close(descriptor);
        return KOKORO_MALFORMED_PACK;
    }
    char payload_sha256[65];
    if (!sha256_hex(payload, (size_t)payload_size, payload_sha256)) {
        kokoro_error_set(error, KOKORO_IO_ERROR, "cannot calculate pack payload SHA-256");
        (void)munmap((void *)mapping, mapping_size);
        (void)close(descriptor);
        return KOKORO_IO_ERROR;
    }
    if (!parse_metadata(metadata, metadata_size, tensor_count, payload_sha256)) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK,
                         "malformed or incompatible model pack metadata");
        (void)munmap((void *)mapping, mapping_size);
        (void)close(descriptor);
        return KOKORO_INCOMPATIBLE_PACK;
    }

    KokoroPack *pack = calloc(1, sizeof(*pack));
    PackRange *ranges = calloc(tensor_count, sizeof(*ranges));
    size_t table_capacity = 1;
    while (table_capacity < (size_t)tensor_count * 2) {
        table_capacity *= 2;
    }
    size_t *name_table = calloc(table_capacity, sizeof(*name_table));
    if (pack == NULL || ranges == NULL || name_table == NULL) {
        free(pack);
        free(ranges);
        free(name_table);
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY, "cannot allocate pack index");
        (void)munmap((void *)mapping, mapping_size);
        (void)close(descriptor);
        return KOKORO_OUT_OF_MEMORY;
    }
    pack->descriptor = descriptor;
    pack->mapping = mapping;
    pack->mapping_size = mapping_size;
    pack->payload = payload;
    pack->payload_size = (size_t)payload_size;
    pack->tensor_count = tensor_count;
    pack->tensors = calloc(tensor_count, sizeof(*pack->tensors));
    if (pack->tensors == NULL) {
        free(ranges);
        free(name_table);
        pack_release(pack);
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY, "cannot allocate pack tensor index");
        return KOKORO_OUT_OF_MEMORY;
    }

    size_t cursor = PACK_HEADER_SIZE;
    size_t index_end = PACK_HEADER_SIZE + index_size;
    unsigned int groups = 0;
    bool saw_voice = false;
    KokoroStatus status = KOKORO_OK;
    for (uint32_t index = 0; index < tensor_count; index++) {
        if (cursor > index_end || index_end - cursor < 8) {
            status = KOKORO_MALFORMED_PACK;
            kokoro_error_set(error, status, "truncated tensor index entry");
            break;
        }
        const unsigned char *entry = mapping + cursor;
        uint16_t name_length = (uint16_t)entry[0] | ((uint16_t)entry[1] << 8);
        unsigned int type_code = entry[2];
        unsigned int rank = entry[3];
        uint32_t entry_reserved = read_u32_le(entry + 4);
        cursor += 8;
        if (entry_reserved != 0 || name_length == 0 || name_length > PACK_MAX_NAME ||
            rank > KOKORO_PACK_MAX_RANK) {
            status = KOKORO_MALFORMED_PACK;
            kokoro_error_set(error, status, "invalid tensor index entry");
            break;
        }
        size_t tail_size = (size_t)rank * 8 + 16 + name_length;
        if (cursor > index_end || tail_size > index_end - cursor) {
            status = KOKORO_MALFORMED_PACK;
            kokoro_error_set(error, status, "truncated tensor index entry");
            break;
        }
        PackTensor *tensor = &pack->tensors[index];
        tensor->rank = rank;
        uint64_t elements = 1;
        for (unsigned int dimension = 0; dimension < rank; dimension++) {
            uint64_t size = read_u64_le(mapping + cursor);
            cursor += 8;
            if (size == 0 || elements > UINT64_MAX / size) {
                status = KOKORO_MALFORMED_PACK;
                kokoro_error_set(error, status, "invalid tensor shape");
                break;
            }
            elements *= size;
            tensor->shape[dimension] = size;
        }
        if (status != KOKORO_OK) {
            break;
        }
        tensor->payload_offset = read_u64_le(mapping + cursor);
        tensor->byte_count = read_u64_le(mapping + cursor + 8);
        cursor += 16;
        tensor->name = mapping + cursor;
        tensor->name_length = name_length;
        cursor += name_length;
        size_t element_size;
        if (type_code == KOKORO_TENSOR_F32) {
            tensor->type = KOKORO_TENSOR_F32;
            element_size = sizeof(float);
        } else if (type_code == KOKORO_TENSOR_I64) {
            tensor->type = KOKORO_TENSOR_I64;
            element_size = sizeof(int64_t);
        } else {
            status = KOKORO_MALFORMED_PACK;
            kokoro_error_set(error, status, "unsupported tensor data type");
            break;
        }
        if (elements > UINT64_MAX / element_size || tensor->byte_count != elements * element_size) {
            status = KOKORO_MALFORMED_PACK;
            kokoro_error_set(error, status, "tensor byte size does not match shape");
            break;
        }
        if (tensor->payload_offset > payload_size ||
            tensor->byte_count > payload_size - tensor->payload_offset) {
            status = KOKORO_MALFORMED_PACK;
            kokoro_error_set(error, status, "tensor data lies outside the payload");
            break;
        }
        if ((tensor->payload_offset % element_size) != 0) {
            status = KOKORO_MALFORMED_PACK;
            kokoro_error_set(error, status, "tensor payload is not naturally aligned");
            break;
        }
        if (!valid_tensor_name(tensor->name, tensor->name_length)) {
            status = KOKORO_MALFORMED_PACK;
            kokoro_error_set(error, status, "invalid tensor name");
            break;
        }
        size_t slot = (size_t)name_hash(tensor->name, tensor->name_length) & (table_capacity - 1);
        while (name_table[slot] != 0) {
            PackTensor *previous = &pack->tensors[name_table[slot] - 1];
            if (tensor_name_equals(previous, tensor->name, tensor->name_length)) {
                status = KOKORO_MALFORMED_PACK;
                kokoro_error_set(error, status, "duplicate tensor name");
                break;
            }
            slot = (slot + 1) & (table_capacity - 1);
        }
        if (status != KOKORO_OK) {
            break;
        }
        name_table[slot] = (size_t)index + 1;
        ranges[index].start = tensor->payload_offset;
        ranges[index].end = tensor->payload_offset + tensor->byte_count;
        const unsigned char *dot = memchr(tensor->name, '.', tensor->name_length);
        size_t group_length = (size_t)(dot - tensor->name);
        if (group_length == 4 && memcmp(tensor->name, "bert", 4) == 0) {
            groups |= 1U << 0;
        } else if (group_length == 12 && memcmp(tensor->name, "bert_encoder", 12) == 0) {
            groups |= 1U << 1;
        } else if (group_length == 9 && memcmp(tensor->name, "predictor", 9) == 0) {
            groups |= 1U << 2;
        } else if (group_length == 12 && memcmp(tensor->name, "text_encoder", 12) == 0) {
            groups |= 1U << 3;
        } else if (group_length == 7 && memcmp(tensor->name, "decoder", 7) == 0) {
            groups |= 1U << 4;
        } else if (tensor->name_length == strlen("voice.af_heart.style") &&
                   memcmp(tensor->name, "voice.af_heart.style", tensor->name_length) == 0) {
            saw_voice = true;
            if (tensor->type != KOKORO_TENSOR_F32 || tensor->rank != 3 || tensor->shape[0] == 0 ||
                tensor->shape[1] != 1 || tensor->shape[2] != 256) {
                status = KOKORO_INCOMPATIBLE_PACK;
                kokoro_error_set(error, status, "voice tensor shape must be [styles, 1, 256]");
                break;
            }
        }
    }
    if (status == KOKORO_OK && cursor != index_end) {
        status = KOKORO_MALFORMED_PACK;
        kokoro_error_set(error, status, "tensor index size mismatch");
    }
    if (status == KOKORO_OK) {
        qsort(ranges, tensor_count, sizeof(*ranges), compare_ranges);
        for (size_t index = 1; index < tensor_count; index++) {
            if (ranges[index].start < ranges[index - 1].end) {
                status = KOKORO_MALFORMED_PACK;
                kokoro_error_set(error, status, "overlapping tensor data");
                break;
            }
        }
    }
    free(ranges);
    free(name_table);
    if (status == KOKORO_OK && groups != 0x1fU) {
        status = KOKORO_INCOMPATIBLE_PACK;
        kokoro_error_set(error, status, "pack is missing one or more model tensor groups");
    }
    if (status == KOKORO_OK && !saw_voice) {
        status = KOKORO_INCOMPATIBLE_PACK;
        kokoro_error_set(error, status, "pack is missing voice.af_heart.style");
    }
    if (status != KOKORO_OK) {
        pack_release(pack);
        return status;
    }
    *result = pack;
    kokoro_error_set(error, KOKORO_OK, NULL);
    return KOKORO_OK;
}

KokoroStatus kokoro_pack_tensor(const KokoroPack *pack, const char *name, KokoroTensorView *view,
                                KokoroError *error) {
    if (pack == NULL || name == NULL || view == NULL) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "pack, tensor name, and view are required");
        return KOKORO_INVALID_ARGUMENT;
    }
    size_t name_length = strlen(name);
    for (size_t index = 0; index < pack->tensor_count; index++) {
        const PackTensor *tensor = &pack->tensors[index];
        if (!tensor_name_equals(tensor, (const unsigned char *)name, name_length)) {
            continue;
        }
        view->type = tensor->type;
        view->rank = tensor->rank;
        memcpy(view->shape, tensor->shape, sizeof(view->shape));
        view->data = pack->payload + tensor->payload_offset;
        view->byte_count = (size_t)tensor->byte_count;
        kokoro_error_set(error, KOKORO_OK, NULL);
        return KOKORO_OK;
    }
    kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK, "required tensor is missing: %s", name);
    return KOKORO_INCOMPATIBLE_PACK;
}

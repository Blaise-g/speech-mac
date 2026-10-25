#include "phonon_pack.h"

#include "sha256.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HEADER_SIZE 192U
#define MAX_TENSORS 512U
#define MAX_PACK_BYTES (512U * 1024U * 1024U)
static const unsigned char magic[8] = {'P', 'H', 'N', 'Q', '8', 'P', '1', 0};
static const char model_digest[] =
    "f6acc8f43cb296be31341b4428a6f01b6402659cd0039b624c4aea224936cf95";
static const char config_digest[] =
    "3e89e90b9e7ae5a74c6b4e694960c37fd2ec85906d84adfcc1440c64307aa4f1";
static const char tensor_index_digest[] =
    "97512628c18e7532f1c3fd5fa53bd45e58cdd46175b304d3125bea6833cff254";

typedef struct {
    const char *name;
    size_t name_length;
    PhononTensorView view;
    uint64_t offset;
} PackEntry;

struct PhononPack {
    unsigned char *bytes;
    size_t size;
    const unsigned char *payload;
    size_t payload_size;
    PackEntry *entries;
    size_t count;
};

static uint32_t u32(const unsigned char *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static uint64_t u64(const unsigned char *p) {
    return (uint64_t)u32(p) | (uint64_t)u32(p + 4) << 32;
}
static size_t product(const uint64_t *shape, unsigned int rank) {
    size_t result = 1;
    for (unsigned int i = 0; i < rank; i++) {
        if (!shape[i] || shape[i] > SIZE_MAX || result > SIZE_MAX / (size_t)shape[i])
            return 0;
        result *= (size_t)shape[i];
    }
    return result;
}
static int has_text(const unsigned char *p, size_t n, const char *needle) {
    size_t m = strlen(needle);
    if (m > n)
        return 0;
    for (size_t i = 0; i <= n - m; i++)
        if (memcmp(p + i, needle, m) == 0)
            return 1;
    return 0;
}
static int check_hash(const unsigned char *p, size_t n, const unsigned char *expected) {
    char got[65], wanted[65];
    if (!kokoro_sha256_hex(p, n, got))
        return 0;
    for (size_t i = 0; i < 32; i++)
        sprintf(wanted + 2 * i, "%02x", expected[i]);
    wanted[64] = 0;
    return strcmp(got, wanted) == 0;
}
static void free_pack(PhononPack *pack) {
    if (pack) {
        free(pack->entries);
        free(pack->bytes);
        free(pack);
    }
}

static int phonon_pack_open_impl(const char *path, PhononPack **result, int require_pinned_index) {
    if (!path || !result)
        return 0;
    *result = NULL;
    FILE *file = fopen(path, "rb");
    if (!file)
        return 0;
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return 0;
    }
    long length = ftell(file);
    if (length < (long)(HEADER_SIZE + 4) || (unsigned long)length > MAX_PACK_BYTES ||
        fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return 0;
    }
    PhononPack *pack = calloc(1, sizeof(*pack));
    if (!pack) {
        fclose(file);
        return 0;
    }
    pack->size = (size_t)length;
    pack->bytes = malloc(pack->size);
    if (!pack->bytes || fread(pack->bytes, 1, pack->size, file) != pack->size) {
        fclose(file);
        free_pack(pack);
        return 0;
    }
    fclose(file);
    const unsigned char *b = pack->bytes;
    uint32_t version = u32(b + 8), count = u32(b + 12);
    uint64_t index_size = u64(b + 16), payload_size = u64(b + 24);
    size_t metadata_size_pos;
    if (memcmp(b, magic, 8) || version != 1 || count != 209 || count > MAX_TENSORS) {
        free_pack(pack);
        return 0;
    }
    char source_hex[65], config_hex[65];
    for (size_t i = 0; i < 32; i++) {
        sprintf(source_hex + i * 2, "%02x", b[32 + i]);
        sprintf(config_hex + i * 2, "%02x", b[64 + i]);
    }
    source_hex[64] = 0;
    config_hex[64] = 0;
    if (strcmp(source_hex, model_digest) || strcmp(config_hex, config_digest)) {
        free_pack(pack);
        return 0;
    }
    if (u32(b + HEADER_SIZE) > pack->size - HEADER_SIZE - 4) {
        free_pack(pack);
        return 0;
    }
    metadata_size_pos = HEADER_SIZE + 4;
    uint32_t metadata_size = u32(b + HEADER_SIZE);
    size_t entry_start = metadata_size_pos + metadata_size;
    if (entry_start > pack->size || index_size > pack->size - entry_start) {
        free_pack(pack);
        return 0;
    }
    size_t payload_start = entry_start + (size_t)index_size;
    if (payload_start > pack->size || payload_size != pack->size - payload_start ||
        (payload_start & 7U)) {
        free_pack(pack);
        return 0;
    }
    if (!check_hash(b + entry_start, (size_t)index_size, b + 96)) {
        free_pack(pack);
        return 0;
    }
    if (require_pinned_index) {
        char observed_index[65];
        if (!kokoro_sha256_hex(b + entry_start, (size_t)index_size, observed_index) ||
            strcmp(observed_index, tensor_index_digest)) {
            free_pack(pack);
            return 0;
        }
    }
    const unsigned char *metadata = b + metadata_size_pos;
    if (!check_hash(metadata, metadata_size, b + 160)) {
        free_pack(pack);
        return 0;
    }
    char expected_metadata[512];
    int expected_size =
        snprintf(expected_metadata, sizeof(expected_metadata),
                 "{\"config_sha256\":\"%s\",\"model\":\"gradium/"
                 "phonon-40m-en\",\"revision\":\"cc738b7e4d182c2ba1bafb8062aa4846af000637\","
                 "\"source\":\"model.q8.gguf\",\"source_sha256\":\"%s\",\"tensors\":209}",
                 config_digest, model_digest);
    if (expected_size < 0 || (size_t)expected_size > metadata_size ||
        memcmp(metadata, expected_metadata, (size_t)expected_size) != 0) {
        free_pack(pack);
        return 0;
    }
    for (size_t i = (size_t)expected_size; i < metadata_size; i++) {
        if (metadata[i] != ' ') {
            free_pack(pack);
            return 0;
        }
    }
    pack->payload = b + payload_start;
    pack->payload_size = (size_t)payload_size;
    pack->count = count;
    if (!check_hash(pack->payload, pack->payload_size, b + 128)) {
        free_pack(pack);
        return 0;
    }
    pack->entries = calloc(count, sizeof(*pack->entries));
    if (!pack->entries) {
        free_pack(pack);
        return 0;
    }
    size_t cursor = entry_start, end = entry_start + (size_t)index_size;
    size_t q8_count = 0, bf16_count = 0;
    for (size_t i = 0; i < count; i++) {
        if (end - cursor < 4) {
            free_pack(pack);
            return 0;
        }
        uint16_t name_size = (uint16_t)(b[cursor] | (uint16_t)b[cursor + 1] << 8);
        uint8_t dtype = b[cursor + 2], rank = b[cursor + 3];
        cursor += 4;
        if (!name_size || name_size > 1024 || rank == 0 || rank > PHONON_PACK_MAX_RANK ||
            end - cursor < (size_t)rank * 8 + 16 + name_size) {
            free_pack(pack);
            return 0;
        }
        PackEntry *entry = &pack->entries[i];
        entry->view.dtype = (PhononPackDType)dtype;
        entry->view.rank = rank;
        for (unsigned int d = 0; d < rank; d++)
            entry->view.shape[d] = u64(b + cursor + d * 8);
        cursor += (size_t)rank * 8;
        entry->offset = u64(b + cursor);
        uint64_t bytes = u64(b + cursor + 8);
        cursor += 16;
        entry->name = (const char *)(b + cursor);
        entry->name_length = name_size;
        cursor += name_size;
        for (size_t j = 0; j < name_size; j++) {
            unsigned char ch = (unsigned char)entry->name[j];
            if (!((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                  (ch >= '0' && ch <= '9') || ch == '_' || ch == '.')) {
                free_pack(pack);
                return 0;
            }
        }
        size_t elements = product(entry->view.shape, rank), expected;
        if (!elements || (dtype != PHONON_PACK_Q8_0 && dtype != PHONON_PACK_BF16)) {
            free_pack(pack);
            return 0;
        }
        if (dtype == PHONON_PACK_Q8_0) {
            q8_count++;
            if (rank != 2 || entry->view.shape[1] % 32 || elements / 32 > SIZE_MAX / 34) {
                free_pack(pack);
                return 0;
            }
            expected = elements / 32 * 34;
        } else {
            bf16_count++;
            if (elements > SIZE_MAX / 2) {
                free_pack(pack);
                return 0;
            }
            expected = elements * 2;
        }
        if (bytes != expected || entry->offset % 8 != 0 || entry->offset > payload_size ||
            bytes > payload_size - entry->offset) {
            free_pack(pack);
            return 0;
        }
        entry->view.data = pack->payload + entry->offset;
        entry->view.byte_count = (size_t)bytes;
        if (memchr(entry->name, 0, name_size) ||
            !has_text((const unsigned char *)entry->name, name_size, ".")) {
            free_pack(pack);
            return 0;
        }
        for (size_t j = 0; j < i; j++) {
            PackEntry *prev = &pack->entries[j];
            if (prev->name_length == name_size && !memcmp(prev->name, entry->name, name_size)) {
                free_pack(pack);
                return 0;
            }
            if (entry->offset < prev->offset + prev->view.byte_count &&
                prev->offset < entry->offset + bytes) {
                free_pack(pack);
                return 0;
            }
        }
    }
    if (cursor != end) {
        free_pack(pack);
        return 0;
    }
    if (q8_count != 36 || bf16_count != 173) {
        free_pack(pack);
        return 0;
    }
    *result = pack;
    return 1;
}

int phonon_pack_open(const char *path, PhononPack **result) {
    return phonon_pack_open_impl(path, result, 1);
}
#ifdef PHONON_PACK_TEST_API
int phonon_pack_open_for_tests(const char *path, PhononPack **result) {
    return phonon_pack_open_impl(path, result, 0);
}
#endif

void phonon_pack_close(PhononPack *pack) { free_pack(pack); }
size_t phonon_pack_tensor_count(const PhononPack *pack) { return pack ? pack->count : 0; }
int phonon_pack_tensor_at(const PhononPack *pack, size_t index, const char **name,
                          size_t *name_length, PhononTensorView *view) {
    if (!pack || index >= pack->count || !name || !name_length || !view)
        return 0;
    *name = pack->entries[index].name;
    *name_length = pack->entries[index].name_length;
    *view = pack->entries[index].view;
    return 1;
}
int phonon_pack_tensor(const PhononPack *pack, const char *name, PhononTensorView *view) {
    if (!pack || !name || !view)
        return 0;
    size_t length = strlen(name);
    for (size_t i = 0; i < pack->count; i++) {
        const PackEntry *entry = &pack->entries[i];
        if (entry->name_length == length && !memcmp(entry->name, name, length)) {
            *view = entry->view;
            return 1;
        }
    }
    return 0;
}
int phonon_bf16_to_float(const uint16_t *values, size_t count, float *output) {
    if (!values || !output || !count)
        return 0;
    uintptr_t input_start = (uintptr_t)values, output_start = (uintptr_t)output;
    if (count > SIZE_MAX / sizeof(*values) || count > SIZE_MAX / sizeof(*output))
        return 0;
    size_t input_bytes = count * sizeof(*values), output_bytes = count * sizeof(*output);
    if (input_start > UINTPTR_MAX - input_bytes || output_start > UINTPTR_MAX - output_bytes ||
        (input_start < output_start + output_bytes && output_start < input_start + input_bytes))
        return 0;
    for (size_t i = 0; i < count; i++) {
        union {
            uint32_t bits;
            float value;
        } converted = {(uint32_t)values[i] << 16};
        if (!isfinite(converted.value))
            return 0;
        output[i] = converted.value;
    }
    return 1;
}

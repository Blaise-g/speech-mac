#include "phonon_pack.h"
#include "phonon_pack_test.h"
#include "phonon_q8.h"
#include "phonon_transformer.h"
#include "sha256.h"
#include <pthread.h>

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static uint32_t read_u32(FILE *file) {
    unsigned char b[4];
    assert(fread(b, 1, sizeof(b), file) == sizeof(b));
    return (uint32_t)b[0] | (uint32_t)b[1] << 8 | (uint32_t)b[2] << 16 | (uint32_t)b[3] << 24;
}

static void put_u32(unsigned char *b, uint32_t value) {
    for (unsigned int i = 0; i < 4; i++)
        b[i] = (unsigned char)(value >> (8U * i));
}
static void put_u64(unsigned char *b, uint64_t value) {
    for (unsigned int i = 0; i < 8; i++)
        b[i] = (unsigned char)(value >> (8U * i));
}
static void hash_bytes(const unsigned char *data, size_t size, unsigned char output[32]) {
    char hex[65];
    assert(speech_sha256_hex(data, size, hex));
    for (size_t i = 0; i < 32; i++) {
        unsigned char high =
            (unsigned char)(hex[2 * i] <= '9' ? hex[2 * i] - '0' : hex[2 * i] - 'a' + 10);
        unsigned char low = (unsigned char)(hex[2 * i + 1] <= '9' ? hex[2 * i + 1] - '0'
                                                                  : hex[2 * i + 1] - 'a' + 10);
        output[i] = (unsigned char)((high << 4) | low);
    }
}
static void append(unsigned char *buffer, size_t *length, const void *data, size_t size) {
    memcpy(buffer + *length, data, size);
    *length += size;
}
static size_t hex_bytes(const char *hex, unsigned char *output, size_t count) {
    for (size_t i = 0; i < count; i++) {
        unsigned char high =
            (unsigned char)(hex[2 * i] <= '9' ? hex[2 * i] - '0' : hex[2 * i] - 'a' + 10);
        unsigned char low = (unsigned char)(hex[2 * i + 1] <= '9' ? hex[2 * i + 1] - '0'
                                                                  : hex[2 * i + 1] - 'a' + 10);
        output[i] = (unsigned char)((high << 4) | low);
    }
    return count;
}

static const char *make_synthetic_pack(void) {
    static unsigned char entries[50000], payload[20000], metadata[512], header[192];
    size_t index_size = 0, payload_size = 0;
    for (size_t i = 0; i < 209; i++) {
        char name[64];
        int name_length =
            snprintf(name, sizeof(name),
                     i < 36 ? "flow_lm.fixture_%03zu.weight" : "mimi.fixture_%03zu.weight", i);
        assert(name_length > 0 && (size_t)name_length < sizeof(name));
        size_t aligned = (payload_size + 7U) & ~(size_t)7U;
        memset(payload + payload_size, 0, aligned - payload_size);
        payload_size = aligned;
        uint8_t type = i < 36 ? PHONON_PACK_Q8_0 : PHONON_PACK_BF16;
        unsigned char name_size_le[2] = {(unsigned char)name_length,
                                         (unsigned char)((unsigned int)name_length >> 8)};
        append(entries, &index_size, name_size_le, 2);
        append(entries, &index_size, &type, 1);
        unsigned char rank = i < 36 ? 2 : 1;
        append(entries, &index_size, &rank, 1);
        unsigned char dim[8];
        if (rank == 2) {
            put_u64(dim, 1);
            append(entries, &index_size, dim, 8);
            put_u64(dim, 32);
            append(entries, &index_size, dim, 8);
        } else {
            put_u64(dim, 1);
            append(entries, &index_size, dim, 8);
        }
        unsigned char range[16];
        put_u64(range, payload_size);
        put_u64(range + 8, type == PHONON_PACK_Q8_0 ? 34 : 2);
        append(entries, &index_size, range, sizeof(range));
        append(entries, &index_size, name, (size_t)name_length);
        if (type == PHONON_PACK_Q8_0) {
            for (size_t block = 0; block < 1; block++) {
                payload[payload_size++] = 0x00;
                payload[payload_size++] = 0x3c;
                memset(payload + payload_size, 1, 32);
                payload_size += 32;
            }
        } else {
            payload[payload_size++] = 0x80;
            payload[payload_size++] = 0x3f;
        }
    }
    const char *model = "f6acc8f43cb296be31341b4428a6f01b6402659cd0039b624c4aea224936cf95";
    const char *config = "3e89e90b9e7ae5a74c6b4e694960c37fd2ec85906d84adfcc1440c64307aa4f1";
    int meta_len =
        snprintf((char *)metadata, sizeof(metadata),
                 "{\"config_sha256\":\"%s\",\"model\":\"gradium/"
                 "phonon-40m-en\",\"revision\":\"cc738b7e4d182c2ba1bafb8062aa4846af000637\","
                 "\"source\":\"model.q8.gguf\",\"source_sha256\":\"%s\",\"tensors\":209}",
                 config, model);
    assert(meta_len > 0);
    size_t metadata_size = (size_t)meta_len;
    metadata_size += (8U - (192U + 4U + metadata_size + index_size) % 8U) % 8U;
    memset(metadata + meta_len, ' ', metadata_size - (size_t)meta_len);
    memcpy(header, "PHNQ8P1\0", 8);
    put_u32(header + 8, 1);
    put_u32(header + 12, 209);
    put_u64(header + 16, index_size);
    put_u64(header + 24, payload_size);
    hex_bytes(model, header + 32, 32);
    hex_bytes(config, header + 64, 32);
    hash_bytes(entries, index_size, header + 96);
    hash_bytes(payload, payload_size, header + 128);
    hash_bytes(metadata, metadata_size, header + 160);
    const char *path = ".build/phonon-synthetic.kpack";
    FILE *file = fopen(path, "wb");
    assert(file && fwrite(header, 1, sizeof(header), file) == sizeof(header));
    unsigned char length_bytes[4];
    put_u32(length_bytes, (uint32_t)metadata_size);
    assert(fwrite(length_bytes, 1, 4, file) == 4 &&
           fwrite(metadata, 1, metadata_size, file) == metadata_size);
    assert(fwrite(entries, 1, index_size, file) == index_size &&
           fwrite(payload, 1, payload_size, file) == payload_size);
    assert(fclose(file) == 0);
    return path;
}

static void reject_mutation(const unsigned char *source, size_t size, const char *suffix,
                            size_t offset, const unsigned char *replacement,
                            size_t replacement_size) {
    unsigned char *copy = malloc(size);
    assert(copy);
    memcpy(copy, source, size);
    assert(offset <= size && replacement_size <= size - offset);
    memcpy(copy + offset, replacement, replacement_size);
    uint32_t metadata_size = (uint32_t)copy[192] | (uint32_t)copy[193] << 8 |
                             (uint32_t)copy[194] << 16 | (uint32_t)copy[195] << 24;
    size_t entries = 196U + metadata_size;
    uint64_t index_size = 0;
    for (unsigned int i = 0; i < 8; i++)
        index_size |= (uint64_t)copy[16 + i] << (8U * i);
    if (offset >= entries && offset < entries + index_size) {
        char hash[65];
        assert(speech_sha256_hex(copy + entries, (size_t)index_size, hash));
        for (size_t i = 0; i < 32; i++) {
            unsigned char high =
                (unsigned char)(hash[i * 2] <= '9' ? hash[i * 2] - '0' : hash[i * 2] - 'a' + 10);
            unsigned char low =
                (unsigned char)(hash[i * 2 + 1] <= '9' ? hash[i * 2 + 1] - '0'
                                                       : hash[i * 2 + 1] - 'a' + 10);
            copy[96 + i] = (unsigned char)((high << 4) | low);
        }
    }
    if (offset >= 196 && offset < entries) {
        char hash[65];
        assert(speech_sha256_hex(copy + 196, entries - 196, hash));
        for (size_t i = 0; i < 32; i++) {
            unsigned char high =
                (unsigned char)(hash[i * 2] <= '9' ? hash[i * 2] - '0' : hash[i * 2] - 'a' + 10);
            unsigned char low =
                (unsigned char)(hash[i * 2 + 1] <= '9' ? hash[i * 2 + 1] - '0'
                                                       : hash[i * 2 + 1] - 'a' + 10);
            copy[160 + i] = (unsigned char)((high << 4) | low);
        }
    }
    char path[256];
    assert(snprintf(path, sizeof(path), ".build/phonon-invalid-%s.kpack", suffix) > 0);
    FILE *file = fopen(path, "wb");
    assert(file && fwrite(copy, 1, size, file) == size && fclose(file) == 0);
    PhononPack *pack = NULL;
    assert(!phonon_pack_open_for_tests(path, &pack) && pack == NULL);
    assert(unlink(path) == 0);
    free(copy);
}

static void test_rejected_pack_variants(const char *path) {
    FILE *file = fopen(path, "rb");
    assert(file && fseek(file, 0, SEEK_END) == 0);
    long signed_size = ftell(file);
    assert(signed_size > 256 && fseek(file, 0, SEEK_SET) == 0);
    size_t size = (size_t)signed_size;
    unsigned char *data = malloc(size);
    assert(data && fread(data, 1, size, file) == size && fclose(file) == 0);
    assert(!phonon_pack_open_for_tests(path, NULL));
    char truncated[256];
    assert(snprintf(truncated, sizeof(truncated), ".build/phonon-truncated.kpack") > 0);
    file = fopen(truncated, "wb");
    assert(file && fwrite(data, 1, size - 1, file) == size - 1 && fclose(file) == 0);
    PhononPack *pack = NULL;
    assert(!phonon_pack_open_for_tests(truncated, &pack) && pack == NULL);
    assert(unlink(truncated) == 0);
    const unsigned char unsupported_version[4] = {2, 0, 0, 0};
    reject_mutation(data, size, "version", 8, unsupported_version, sizeof(unsupported_version));
    const unsigned char wrong_source = (unsigned char)(data[32] ^ 1U);
    reject_mutation(data, size, "source", 32, &wrong_source, 1);
    const unsigned char wrong_config = (unsigned char)(data[64] ^ 1U);
    reject_mutation(data, size, "config", 64, &wrong_config, 1);
    const unsigned char changed_metadata = 'X';
    reject_mutation(data, size, "metadata", 196 + 20, &changed_metadata, 1);
    uint32_t metadata_size = (uint32_t)data[192] | (uint32_t)data[193] << 8 |
                             (uint32_t)data[194] << 16 | (uint32_t)data[195] << 24;
    size_t first_entry = 196U + metadata_size;
    assert(first_entry + 4 < size);
    unsigned int rank = data[first_entry + 3];
    assert(rank > 0 && rank <= PHONON_PACK_MAX_RANK);
    unsigned char overflow_dimension[8];
    memset(overflow_dimension, 0xff, sizeof(overflow_dimension));
    reject_mutation(data, size, "overflow", first_entry + 4, overflow_dimension,
                    sizeof(overflow_dimension));
    uint16_t first_name = (uint16_t)(data[first_entry] | (uint16_t)data[first_entry + 1] << 8);
    size_t second_entry = first_entry + 4 + (size_t)rank * 8 + 16 + first_name;
    assert(second_entry + 4 < size);
    unsigned int second_rank = data[second_entry + 3];
    size_t second_offset = second_entry + 4 + (size_t)second_rank * 8;
    const unsigned char zero_offset[8] = {0};
    reject_mutation(data, size, "overlap", second_offset, zero_offset, sizeof(zero_offset));
    const unsigned char unsupported_dtype = 9;
    reject_mutation(data, size, "dtype", first_entry + 2, &unsupported_dtype, 1);
    const unsigned char wrong_length[8] = {0};
    reject_mutation(data, size, "length", first_entry + 4 + (size_t)rank * 8 + 8, wrong_length,
                    sizeof(wrong_length));
    unsigned char bad_row_shape[16];
    put_u64(bad_row_shape, 2);
    put_u64(bad_row_shape + 8, 16);
    reject_mutation(data, size, "row-shape", first_entry + 4, bad_row_shape, sizeof(bad_row_shape));
    uint64_t payload_size = 0;
    for (unsigned int i = 0; i < 8; i++)
        payload_size |= (uint64_t)data[24 + i] << (8U * i);
    size_t payload_start = size - (size_t)payload_size;
    const unsigned char corrupt_payload = (unsigned char)(data[payload_start] ^ 1U);
    reject_mutation(data, size, "payload", payload_start, &corrupt_payload, 1);
    const unsigned char changed_name = '/';
    reject_mutation(data, size, "name", first_entry + 4 + rank * 8 + 16, &changed_name, 1);
    free(data);
}

static void test_pinned_index_rejects_changed_name(const char *path) {
    FILE *file = fopen(path, "rb");
    assert(file && fseek(file, 0, SEEK_END) == 0);
    long file_size = ftell(file);
    assert(file_size > 256 && fseek(file, 0, SEEK_SET) == 0);
    size_t size = (size_t)file_size;
    unsigned char *data = malloc(size);
    assert(data && fread(data, 1, size, file) == size && fclose(file) == 0);
    uint32_t metadata_size = (uint32_t)data[192] | (uint32_t)data[193] << 8 |
                             (uint32_t)data[194] << 16 | (uint32_t)data[195] << 24;
    size_t entries = 196U + metadata_size;
    uint64_t index_size = 0;
    for (unsigned int i = 0; i < 8; i++)
        index_size |= (uint64_t)data[16 + i] << (8U * i);
    assert(entries + index_size <= size);
    size_t cursor = entries, name_start = 0;
    for (size_t i = 0; i < 209; i++) {
        uint16_t name_size = (uint16_t)(data[cursor] | (uint16_t)data[cursor + 1] << 8);
        unsigned int rank = data[cursor + 3];
        size_t current_name = cursor + 4 + (size_t)rank * 8 + 16;
        assert(current_name + name_size <= entries + index_size);
        if (data[cursor + 2] == PHONON_PACK_Q8_0) {
            assert(rank == 2);
            name_start = current_name;
            break;
        }
        cursor = current_name + name_size;
    }
    assert(name_start > entries);
    assert(data[name_start] == 'f');
    data[name_start] = 'F';
    hash_bytes(data + entries, (size_t)index_size, data + 96);
    const char *mutated_path = ".build/phonon-pinned-index-mutation.kpack";
    file = fopen(mutated_path, "wb");
    assert(file && fwrite(data, 1, size, file) == size && fclose(file) == 0);
    PhononPack *pack = NULL;
    assert(!phonon_pack_open(mutated_path, &pack) && pack == NULL);
    assert(unlink(mutated_path) == 0);
    free(data);
}

/* Published Q8 blocks here encode 32 ones with unit scale. Each projection
 * must be 32 regardless of preparation, sharing, or another pack's lifetime. */
static void *shared_linear(void *context) {
    const PhononTensorView *view = context;
    float input[32], output;
    for (size_t i = 0; i < 32; i++)
        input[i] = 1;
    for (size_t i = 0; i < 32; i++) {
        assert(phonon_linear(view, input, 1, 32, NULL, &output));
        assert(output == 32);
    }
    return NULL;
}
static void test_materialization(const char *path, PhononPack *pack) {
    PhononTensorView before, after;
    assert(phonon_pack_tensor(pack, "flow_lm.fixture_000.weight", &before));
    assert(!before.linear_data);
    shared_linear(&before);
    /* Fail after one successful allocation; no partial cache may escape. */
    assert(!phonon_pack_prepare_linears_for_tests(pack, 32 * sizeof(float)));
    assert(phonon_pack_tensor(pack, "flow_lm.fixture_000.weight", &after));
    assert(!after.linear_data);
    shared_linear(&after);
    assert(phonon_pack_prepare_linears(pack));
    assert(phonon_pack_tensor(pack, "flow_lm.fixture_000.weight", &after));
    assert(after.linear_data && after.data == before.data);
    for (size_t i = 0; i < 32; i++)
        assert(after.linear_data[i] == 1);
    assert(phonon_pack_prepare_linears(pack));
    PhononTensorView again;
    assert(phonon_pack_tensor(pack, "flow_lm.fixture_000.weight", &again));
    assert(again.linear_data == after.linear_data);
    PhononPack *other = NULL;
    assert(phonon_pack_open_for_tests(path, &other));
    assert(phonon_pack_prepare_linears(other));
    assert(phonon_pack_tensor(other, "flow_lm.fixture_000.weight", &again));
    assert(again.linear_data != after.linear_data);
    phonon_pack_close(other);
    pthread_t threads[2];
    for (size_t i = 0; i < 2; i++)
        assert(!pthread_create(threads + i, NULL, shared_linear, &after));
    for (size_t i = 0; i < 2; i++)
        assert(!pthread_join(threads[i], NULL));
    uint16_t bf16[] = {0x3f80, 0xc000};
    PhononTensorView small = {
        .dtype = PHONON_PACK_BF16, .rank = 2, .shape = {1, 2}, .data = bf16, .byte_count = 4};
    float *converted = NULL;
    assert(phonon_tensor_expand_linear(&small, &converted));
    assert(converted[0] == 1 && converted[1] == -2);
    float input[] = {3, 4}, ordinary, cached;
    assert(phonon_linear(&small, input, 1, 2, NULL, &ordinary));
    small.linear_data = converted;
    assert(phonon_linear(&small, input, 1, 2, NULL, &cached));
    assert(ordinary == -5 && cached == ordinary);
    free(converted);
    uint16_t bad[] = {0x7f80};
    PhononTensorView invalid = {
        .dtype = PHONON_PACK_BF16, .rank = 2, .shape = {1, 1}, .data = bad, .byte_count = 2};
    float *expanded = (float *)(uintptr_t)1;
    assert(!phonon_tensor_expand_linear(&invalid, &expanded) && !expanded);
    assert(!phonon_pack_prepare_linears(NULL));
}

int main(void) {
    const char *synthetic_path = make_synthetic_pack();
    PhononPack *synthetic = NULL;
    assert(phonon_pack_open_for_tests(synthetic_path, &synthetic));
    assert(phonon_pack_tensor_count(synthetic) == 209);
    test_rejected_pack_variants(synthetic_path);
    test_materialization(synthetic_path, synthetic);
    PhononTensorView bf16_view;
    assert(phonon_pack_tensor(synthetic, "mimi.fixture_036.weight", &bf16_view));
    float decoded[1];
    assert(!phonon_bf16_to_float(bf16_view.data, 1, (float *)(uintptr_t)bf16_view.data));
    assert(phonon_bf16_to_float(bf16_view.data, 1, decoded) && decoded[0] == 1.0f);
    phonon_pack_close(synthetic);
    for (size_t i = 0; i < 8; i++) {
        assert(phonon_pack_open_for_tests(synthetic_path, &synthetic));
        phonon_pack_close(synthetic);
        synthetic = NULL;
    }
    const char *pack_path = getenv("PHONON_NATIVE_PACK");
    const char *oracle_path = getenv("PHONON_Q8_ORACLE");
    const char *bf16_oracle_path = getenv("PHONON_BF16_ORACLE");
    if (!pack_path || !oracle_path || !bf16_oracle_path) {
        puts("SKIP real Phonon pack checks: run make phonon-native-check");
        return 0;
    }
    PhononPack *pack = NULL;
    assert(phonon_pack_open(pack_path, &pack));
    assert(phonon_pack_prepare_linears(pack));
    assert(phonon_pack_tensor_count(pack) == 209);
    test_pinned_index_rejects_changed_name(pack_path);
    size_t q8_count = 0, bf16_count = 0, materialized_count = 0, materialized_bytes = 0;
    for (size_t i = 0; i < phonon_pack_tensor_count(pack); i++) {
        const char *name = NULL;
        size_t name_length = 0;
        PhononTensorView view;
        assert(phonon_pack_tensor_at(pack, i, &name, &name_length, &view));
        assert(name && name_length > 2 && name[name_length - 1] != 0 && view.rank > 0);
        char lookup_name[1025];
        assert(name_length < sizeof(lookup_name));
        memcpy(lookup_name, name, name_length);
        lookup_name[name_length] = 0;
        PhononTensorView lookup_view;
        assert(phonon_pack_tensor(pack, lookup_name, &lookup_view));
        assert(lookup_view.data == view.data && lookup_view.byte_count == view.byte_count);
        size_t elements = 1;
        for (unsigned int d = 0; d < view.rank; d++) {
            assert(view.shape[d] && elements <= SIZE_MAX / view.shape[d]);
            elements *= (size_t)view.shape[d];
        }
        if (view.linear_data) {
            materialized_count++;
            materialized_bytes += elements * sizeof(float);
        }
        if (view.dtype == PHONON_PACK_Q8_0) {
            q8_count++;
            assert(view.byte_count == elements / 32 * 34 && elements % 32 == 0);
        } else {
            bf16_count++;
            assert(view.byte_count == elements * 2);
            float *decoded = malloc(elements * sizeof(*decoded));
            assert(decoded);
            assert(phonon_bf16_to_float(view.data, elements, decoded));
            free(decoded);
        }
    }
    assert(q8_count == 36 && bf16_count == 173);
    assert(materialized_count == 75 && materialized_bytes == 177702912);

    FILE *oracle = fopen(oracle_path, "rb");
    assert(oracle);
    unsigned char magic[8];
    assert(fread(magic, 1, 8, oracle) == 8 && memcmp(magic, "PHNPROJ1", 8) == 0);
    uint32_t matrix_count = read_u32(oracle);
    assert(matrix_count == 36);
    for (uint32_t matrix = 0; matrix < matrix_count; matrix++) {
        uint16_t name_size;
        assert(fread(&name_size, sizeof(name_size), 1, oracle) == 1 && name_size > 0 &&
               name_size <= 1024);
        char name[1025];
        assert(fread(name, 1, name_size, oracle) == name_size);
        name[name_size] = 0;
        uint32_t rows = read_u32(oracle), columns = read_u32(oracle);
        assert(rows > 0 && columns > 0 && columns % 32 == 0);
        PhononTensorView view;
        assert(phonon_pack_tensor(pack, name, &view));
        assert(view.dtype == PHONON_PACK_Q8_0 && view.rank == 2);
        assert(view.shape[0] == rows && view.shape[1] == columns);
        float *input = malloc(columns * sizeof(*input));
        float *expected = malloc(rows * sizeof(*expected));
        float *actual = malloc(rows * sizeof(*actual));
        assert(input && expected && actual);
        for (uint32_t i = 0; i < columns; i++)
            input[i] = (float)((int)(i * 17U % 101U) - 50) / 37.0f;
        assert(fread(expected, sizeof(*expected), rows, oracle) == rows);
        assert(phonon_q8_matvec(view.data, input, rows, columns, actual));
        for (uint32_t row = 0; row < rows; row++) {
            float allowance = 1.0e-5f + 1.0e-3f * fabsf(expected[row]);
            assert(isfinite(actual[row]) && fabsf(actual[row] - expected[row]) <= allowance);
        }
        assert(view.linear_data);
        assert(phonon_linear(&view, input, rows, columns, NULL, actual));
        for (uint32_t row = 0; row < rows; row++) {
            float allowance = 1.0e-5f + 1.0e-3f * fabsf(expected[row]);
            assert(isfinite(actual[row]) && fabsf(actual[row] - expected[row]) <= allowance);
        }
        free(actual);
        free(expected);
        free(input);
    }
    assert(fgetc(oracle) == EOF);
    fclose(oracle);
    FILE *bf16_oracle = fopen(bf16_oracle_path, "rb");
    assert(bf16_oracle);
    assert(fread(magic, 1, 8, bf16_oracle) == 8 && memcmp(magic, "PHNBF161", 8) == 0);
    assert(read_u32(bf16_oracle) == 173);
    for (size_t i = 0; i < 173; i++) {
        uint16_t name_size;
        assert(fread(&name_size, sizeof(name_size), 1, bf16_oracle) == 1 && name_size > 0 &&
               name_size <= 1024);
        char name[1025];
        assert(fread(name, 1, name_size, bf16_oracle) == name_size);
        name[name_size] = 0;
        uint32_t sample_count = read_u32(bf16_oracle);
        assert(sample_count > 0 && sample_count <= 16);
        PhononTensorView view;
        assert(phonon_pack_tensor(pack, name, &view) && view.dtype == PHONON_PACK_BF16);
        float actual[16], expected[16];
        assert(phonon_bf16_to_float(view.data, sample_count, actual));
        assert(fread(expected, sizeof(*expected), sample_count, bf16_oracle) == sample_count);
        for (uint32_t i = 0; i < sample_count; i++) {
            assert(actual[i] == expected[i]);
            if (view.linear_data)
                assert(view.linear_data[i] == expected[i]);
        }
    }
    assert(fgetc(bf16_oracle) == EOF);
    fclose(bf16_oracle);
    phonon_pack_close(pack);
    puts("Phonon native pack: 209 tensors, 36 Q8 projections matched oracle");
    return 0;
}

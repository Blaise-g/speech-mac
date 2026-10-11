#include "context_encoder.h"
#include "kokoro.h"
#include "model_pack.h"

#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define PACK_PATH "tests/fixtures/minimal_valid_pack.kpack"
#define HEADER_SIZE 40U

static int check(bool condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "test_model_pack: %s\n", message);
        return 0;
    }
    return 1;
}

static uint32_t read_u32_le(const unsigned char *bytes) {
    return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) | ((uint32_t)bytes[2] << 16) |
           ((uint32_t)bytes[3] << 24);
}

static void write_u32_le(unsigned char *bytes, uint32_t value) {
    for (unsigned int index = 0; index < 4; index++) {
        bytes[index] = (unsigned char)(value >> (index * 8));
    }
}

static void write_u64_le(unsigned char *bytes, uint64_t value) {
    for (unsigned int index = 0; index < 8; index++) {
        bytes[index] = (unsigned char)(value >> (index * 8));
    }
}

static unsigned char *read_file(const char *path, size_t *size) {
    FILE *file = fopen(path, "rb");
    if (file == NULL || fseek(file, 0, SEEK_END) != 0) {
        if (file != NULL) {
            fclose(file);
        }
        return NULL;
    }
    long file_size = ftell(file);
    if (file_size < 0 || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return NULL;
    }
    unsigned char *data = malloc((size_t)file_size);
    if (data == NULL || fread(data, 1, (size_t)file_size, file) != (size_t)file_size) {
        free(data);
        fclose(file);
        return NULL;
    }
    fclose(file);
    *size = (size_t)file_size;
    return data;
}

static int write_temp(const unsigned char *data, size_t size, char path[64]) {
    memcpy(path, "/tmp/kokoro-pack-test-XXXXXX", sizeof("/tmp/kokoro-pack-test-XXXXXX"));
    int descriptor = mkstemp(path);
    if (descriptor < 0) {
        return 0;
    }
    size_t offset = 0;
    while (offset < size) {
        ssize_t written = write(descriptor, data + offset, size - offset);
        if (written < 0 && errno == EINTR) {
            continue;
        }
        if (written <= 0) {
            close(descriptor);
            unlink(path);
            return 0;
        }
        offset += (size_t)written;
    }
    if (close(descriptor) != 0) {
        unlink(path);
        return 0;
    }
    return 1;
}

static int rejected_pack(const unsigned char *data, size_t size, KokoroStatus expected_status,
                         const char *description) {
    char path[64];
    if (!write_temp(data, size, path)) {
        fprintf(stderr, "test_model_pack: could not write temporary pack for %s\n", description);
        return 0;
    }
    KokoroPack *pack = NULL;
    KokoroError error = {0};
    KokoroStatus status = kokoro_pack_open(path, &pack, &error);
    unlink(path);
    kokoro_pack_close(pack);
    if (status != expected_status) {
        fprintf(stderr, "test_model_pack: %s returned %s (%s), expected %s\n", description,
                kokoro_status_name(status), error.message, kokoro_status_name(expected_status));
        return 0;
    }
    return 1;
}

int main(void) {
    size_t pack_size = 0;
    unsigned char *original = read_file(PACK_PATH, &pack_size);
    if (!check(original != NULL, "could not read checked-in structural pack fixture")) {
        return 1;
    }
    if (!check(pack_size > HEADER_SIZE, "pack fixture is shorter than a header")) {
        free(original);
        return 1;
    }

    KokoroPack *pack = NULL;
    KokoroError error = {0};
    KokoroStatus status = kokoro_pack_open(PACK_PATH, &pack, &error);
    if (!check(status == KOKORO_OK, error.message)) {
        free(original);
        return 1;
    }
    KokoroTensorView tensor;
    status = kokoro_pack_tensor(pack, "voice.af_heart.style", &tensor, &error);
    if (!check(status == KOKORO_OK && tensor.rank == 3 && tensor.shape[0] == 1 &&
                   tensor.shape[1] == 1 && tensor.shape[2] == 256 && tensor.byte_count == 1024,
               "voice tensor view has the wrong shape or byte size")) {
        kokoro_pack_close(pack);
        free(original);
        return 1;
    }
    if (!check((uintptr_t)tensor.data % _Alignof(float) == 0,
               "float tensor view is not naturally aligned")) {
        kokoro_pack_close(pack);
        free(original);
        return 1;
    }
    status = kokoro_pack_tensor(pack, "bert.missing.weight", &tensor, &error);
    if (!check(status == KOKORO_INCOMPATIBLE_PACK, "missing tensor lookup was not rejected")) {
        kokoro_pack_close(pack);
        free(original);
        return 1;
    }
    KokoroContextEncoder *encoder = NULL;
    status = kokoro_context_encoder_create(pack, &encoder, &error);
    if (!check(status == KOKORO_INCOMPATIBLE_PACK && encoder == NULL,
               "context encoder accepted a pack with missing required tensors")) {
        kokoro_context_encoder_free(encoder);
        kokoro_pack_close(pack);
        free(original);
        return 1;
    }
    kokoro_pack_close(pack);

    if (!rejected_pack(original, pack_size - 1, KOKORO_MALFORMED_PACK, "truncated payload") ||
        !rejected_pack(original, pack_size + 1, KOKORO_MALFORMED_PACK, "trailing byte")) {
        free(original);
        return 1;
    }

    unsigned char *changed = malloc(pack_size + 1);
    if (!check(changed != NULL, "could not allocate malformed pack test data")) {
        free(original);
        return 1;
    }
    uint32_t index_size = read_u32_le(original + 16);
    size_t metadata_start = HEADER_SIZE + index_size;
    size_t metadata_size = read_u32_le(original + 20);
    size_t payload_start = metadata_start + metadata_size;
    memcpy(changed, original, payload_start);
    changed[payload_start] = ' ';
    memcpy(changed + payload_start + 1, original + payload_start, pack_size - payload_start);
    write_u32_le(changed + 20, (uint32_t)(metadata_size + 1));
    int passed =
        rejected_pack(changed, pack_size + 1, KOKORO_MALFORMED_PACK, "unaligned tensor payload");

    memcpy(changed, original, pack_size);
    write_u32_le(changed + 8, 2);
    passed &= rejected_pack(changed, pack_size, KOKORO_INCOMPATIBLE_PACK, "unsupported version");

    memcpy(changed, original, pack_size);
    changed[0] = 'X';
    passed &= rejected_pack(changed, pack_size, KOKORO_MALFORMED_PACK, "invalid magic");

    memcpy(changed, original, pack_size);
    changed[metadata_start] = '!';
    passed &=
        rejected_pack(changed, pack_size, KOKORO_INCOMPATIBLE_PACK, "malformed JSON metadata");

    memcpy(changed, original, pack_size);
    size_t first_name = HEADER_SIZE + 8 + 8 + 16;
    changed[first_name] = 'x';
    passed &= rejected_pack(changed, pack_size, KOKORO_MALFORMED_PACK, "invalid tensor name");

    memcpy(changed, original, pack_size);
    write_u64_le(changed + HEADER_SIZE + 8, 2);
    passed &=
        rejected_pack(changed, pack_size, KOKORO_MALFORMED_PACK, "shape and byte-size mismatch");

    memcpy(changed, original, pack_size);
    changed[payload_start] ^= 1;
    passed &= rejected_pack(changed, pack_size, KOKORO_INCOMPATIBLE_PACK, "payload hash mismatch");

    free(changed);
    free(original);
    if (!passed) {
        return 1;
    }
    puts("model pack parser checks passed");
    return 0;
}

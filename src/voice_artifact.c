#include "voice_artifact.h"

#include "kokoro_internal.h"
#include "sha256.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define VOICE_HEADER_SIZE 192U
#define VOICE_VERSION 1U
#define VOICE_STYLE_ROWS 510U
#define VOICE_STYLE_WIDTH 256U
#define VOICE_PAYLOAD_SIZE (VOICE_STYLE_ROWS * VOICE_STYLE_WIDTH * sizeof(float))
#define VOICE_FILE_SIZE (VOICE_HEADER_SIZE + VOICE_PAYLOAD_SIZE)

static const unsigned char voice_magic[8] = {'K', 'V', 'O', 'I', 'C', 'E', '1', '\0'};
static const char model_revision[] = "f3ff3571791e39611d31c381e3a41a3af07b4987";
static const char source_sha256[] =
    "ced7e284aba12472891be1da3ab34db84cc05cc02b5889535796dbf2d8b0cb34";
static const char payload_sha256[] =
    "162b035ed91cfc48b6046982184c645f72edcdd1b82843347f605d7bf7b15716";
static const char artifact_sha256[] =
    "2c49b32ed630afb2366ab40b19bcb666e58e98f721748e6c856f214a2807773b";

struct KokoroVoiceArtifact {
    int descriptor;
    const unsigned char *mapping;
    size_t mapping_size;
};

static uint32_t read_u32_le(const unsigned char *bytes) {
    return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) | ((uint32_t)bytes[2] << 16) |
           ((uint32_t)bytes[3] << 24);
}

static void voice_artifact_release(KokoroVoiceArtifact *artifact) {
    if (artifact == NULL) {
        return;
    }
    if (artifact->mapping != NULL) {
        (void)munmap((void *)artifact->mapping, artifact->mapping_size);
    }
    if (artifact->descriptor >= 0) {
        (void)close(artifact->descriptor);
    }
    free(artifact);
}

void kokoro_voice_artifact_close(KokoroVoiceArtifact *artifact) {
    voice_artifact_release(artifact);
}

KokoroStatus kokoro_voice_artifact_open(const char *path, const char *voice_name,
                                        KokoroVoiceArtifact **result, KokoroError *error) {
    if (result != NULL) {
        *result = NULL;
    }
    if (path == NULL || voice_name == NULL || result == NULL) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "voice artifact path, name, and output are required");
        return KOKORO_INVALID_ARGUMENT;
    }
    if (strcmp(voice_name, "am_adam") != 0) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT, "unsupported separate voice: %s",
                         voice_name);
        return KOKORO_INVALID_ARGUMENT;
    }

    int descriptor = open(path, O_RDONLY);
    if (descriptor < 0) {
        kokoro_error_set(error, KOKORO_IO_ERROR, "cannot open am_adam voice artifact: %s",
                         strerror(errno));
        return KOKORO_IO_ERROR;
    }
    struct stat info;
    if (fstat(descriptor, &info) != 0 || !S_ISREG(info.st_mode) || info.st_size < 0 ||
        (uintmax_t)info.st_size != VOICE_FILE_SIZE) {
        kokoro_error_set(error, KOKORO_MALFORMED_PACK,
                         "am_adam voice artifact has the wrong file size or type");
        (void)close(descriptor);
        return KOKORO_MALFORMED_PACK;
    }
    const unsigned char *mapping =
        mmap(NULL, VOICE_FILE_SIZE, PROT_READ, MAP_PRIVATE, descriptor, 0);
    if (mapping == MAP_FAILED) {
        kokoro_error_set(error, KOKORO_IO_ERROR, "cannot map am_adam voice artifact: %s",
                         strerror(errno));
        (void)close(descriptor);
        return KOKORO_IO_ERROR;
    }
    if (memcmp(mapping, voice_magic, sizeof(voice_magic)) != 0) {
        kokoro_error_set(error, KOKORO_MALFORMED_PACK, "invalid voice artifact magic");
        (void)munmap((void *)mapping, VOICE_FILE_SIZE);
        (void)close(descriptor);
        return KOKORO_MALFORMED_PACK;
    }
    if (read_u32_le(mapping + 8) != VOICE_VERSION ||
        read_u32_le(mapping + 12) != VOICE_STYLE_ROWS ||
        read_u32_le(mapping + 16) != VOICE_STYLE_WIDTH ||
        read_u32_le(mapping + 20) != VOICE_PAYLOAD_SIZE ||
        memcmp(mapping + 24, model_revision, sizeof(model_revision) - 1) != 0 ||
        memcmp(mapping + 64, source_sha256, sizeof(source_sha256) - 1) != 0 ||
        memcmp(mapping + 128, payload_sha256, sizeof(payload_sha256) - 1) != 0) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK,
                         "am_adam voice artifact provenance or style shape is incompatible");
        (void)munmap((void *)mapping, VOICE_FILE_SIZE);
        (void)close(descriptor);
        return KOKORO_INCOMPATIBLE_PACK;
    }
    char actual_payload_sha[65];
    char actual_artifact_sha[65];
    if (!kokoro_sha256_hex(mapping + VOICE_HEADER_SIZE, VOICE_PAYLOAD_SIZE, actual_payload_sha) ||
        !kokoro_sha256_hex(mapping, VOICE_FILE_SIZE, actual_artifact_sha)) {
        kokoro_error_set(error, KOKORO_IO_ERROR, "cannot calculate am_adam voice SHA-256");
        (void)munmap((void *)mapping, VOICE_FILE_SIZE);
        (void)close(descriptor);
        return KOKORO_IO_ERROR;
    }
    if (strcmp(actual_payload_sha, payload_sha256) != 0 ||
        strcmp(actual_artifact_sha, artifact_sha256) != 0) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK,
                         "am_adam voice artifact SHA-256 does not match the pin");
        (void)munmap((void *)mapping, VOICE_FILE_SIZE);
        (void)close(descriptor);
        return KOKORO_INCOMPATIBLE_PACK;
    }
    const float *styles = (const float *)(const void *)(mapping + VOICE_HEADER_SIZE);
    for (size_t index = 0; index < (size_t)VOICE_STYLE_ROWS * VOICE_STYLE_WIDTH; index++) {
        if (!isfinite(styles[index])) {
            kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK,
                             "am_adam voice style contains a non-finite value");
            (void)munmap((void *)mapping, VOICE_FILE_SIZE);
            (void)close(descriptor);
            return KOKORO_INCOMPATIBLE_PACK;
        }
    }
    KokoroVoiceArtifact *artifact = calloc(1, sizeof(*artifact));
    if (artifact == NULL) {
        kokoro_error_set(error, KOKORO_OUT_OF_MEMORY, "cannot allocate voice artifact mapping");
        (void)munmap((void *)mapping, VOICE_FILE_SIZE);
        (void)close(descriptor);
        return KOKORO_OUT_OF_MEMORY;
    }
    artifact->descriptor = descriptor;
    artifact->mapping = mapping;
    artifact->mapping_size = VOICE_FILE_SIZE;
    *result = artifact;
    kokoro_error_set(error, KOKORO_OK, NULL);
    return KOKORO_OK;
}

KokoroStatus kokoro_voice_artifact_tensor(const KokoroVoiceArtifact *artifact,
                                          KokoroTensorView *view, KokoroError *error) {
    if (artifact == NULL || view == NULL) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "voice artifact and tensor view are required");
        return KOKORO_INVALID_ARGUMENT;
    }
    *view = (KokoroTensorView){
        .type = KOKORO_TENSOR_F32,
        .rank = 3,
        .shape = {VOICE_STYLE_ROWS, 1, VOICE_STYLE_WIDTH},
        .data = artifact->mapping + VOICE_HEADER_SIZE,
        .byte_count = VOICE_PAYLOAD_SIZE,
    };
    kokoro_error_set(error, KOKORO_OK, NULL);
    return KOKORO_OK;
}

#include "voice_artifact.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define AM_ADAM_ARTIFACT_PATH "models/voices/am_adam.kvoice"

static int check(int condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "test_voice_artifact: %s\n", message);
        return 0;
    }
    return 1;
}

static int write_temp(const void *data, size_t size, char path[64]) {
    memcpy(path, "/tmp/kokoro-voice-test-XXXXXX", sizeof("/tmp/kokoro-voice-test-XXXXXX"));
    int descriptor = mkstemp(path);
    if (descriptor < 0) {
        return 0;
    }
    size_t offset = 0;
    while (offset < size) {
        ssize_t written = write(descriptor, (const unsigned char *)data + offset, size - offset);
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

static unsigned char *read_artifact(size_t *size) {
    FILE *file = fopen(AM_ADAM_ARTIFACT_PATH, "rb");
    if (file == NULL || fseek(file, 0, SEEK_END) != 0) {
        if (file != NULL) {
            fclose(file);
        }
        return NULL;
    }
    long file_size = ftell(file);
    if (file_size <= 0 || fseek(file, 0, SEEK_SET) != 0) {
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

static int test_rejected_artifacts(void) {
    KokoroError error = {0};
    KokoroVoiceArtifact *artifact = NULL;
    KokoroStatus status =
        kokoro_voice_artifact_open("/no/such/kokoro-voice", "am_adam", &artifact, &error);
    if (!check(status == KOKORO_IO_ERROR && artifact == NULL,
               "missing voice artifact did not report an I/O error")) {
        return 0;
    }
    status = kokoro_voice_artifact_open(AM_ADAM_ARTIFACT_PATH, "unknown", &artifact, &error);
    if (!check(status == KOKORO_INVALID_ARGUMENT && artifact == NULL,
               "unsupported separate voice was not rejected")) {
        return 0;
    }
    char path[64];
    static const char short_data[] = "bad";
    if (!check(write_temp(short_data, sizeof(short_data), path),
               "could not write invalid fixture")) {
        return 0;
    }
    status = kokoro_voice_artifact_open(path, "am_adam", &artifact, &error);
    unlink(path);
    return check(status == KOKORO_MALFORMED_PACK && artifact == NULL,
                 "truncated voice artifact was not rejected");
}

int main(void) {
    if (!test_rejected_artifacts()) {
        return 1;
    }
    size_t artifact_size = 0;
    unsigned char *data = read_artifact(&artifact_size);
    if (data == NULL) {
        puts("SKIP am_adam artifact checks: run the locked voice conversion first");
        return 0;
    }

    KokoroVoiceArtifact *artifact = NULL;
    KokoroError error = {0};
    KokoroStatus status =
        kokoro_voice_artifact_open(AM_ADAM_ARTIFACT_PATH, "am_adam", &artifact, &error);
    int passed = check(status == KOKORO_OK, error.message);
    KokoroTensorView view = {0};
    if (passed) {
        status = kokoro_voice_artifact_tensor(artifact, &view, &error);
        passed = check(status == KOKORO_OK && view.rank == 3 && view.shape[0] == 510 &&
                           view.shape[1] == 1 && view.shape[2] == 256 &&
                           view.byte_count == 510U * 256U * sizeof(float),
                       "loaded style view has the wrong shape or byte size");
    }
    kokoro_voice_artifact_close(artifact);

    if (passed) {
        data[24] ^= 1;
        char path[64];
        passed =
            check(write_temp(data, artifact_size, path), "could not write incompatible fixture");
        if (passed) {
            status = kokoro_voice_artifact_open(path, "am_adam", &artifact, &error);
            unlink(path);
            passed = check(status == KOKORO_INCOMPATIBLE_PACK && artifact == NULL,
                           "wrong model revision was not rejected");
        }
        data[24] ^= 1;
    }
    if (passed) {
        data[artifact_size - 1] ^= 1;
        char path[64];
        passed = check(write_temp(data, artifact_size, path), "could not write corrupt fixture");
        if (passed) {
            status = kokoro_voice_artifact_open(path, "am_adam", &artifact, &error);
            unlink(path);
            passed = check(status == KOKORO_INCOMPATIBLE_PACK && artifact == NULL,
                           "corrupt voice payload was not rejected");
        }
    }
    free(data);
    if (passed) {
        puts("am_adam voice artifact compatibility checks passed");
    }
    return passed ? 0 : 1;
}

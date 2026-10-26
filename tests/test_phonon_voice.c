#include "phonon_voice.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "failed %s:%d: %s\n", __FILE__, __LINE__, #x);                         \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)

static int write_voice(const char *path, const char *signature, size_t frames, int rank2,
                       int truncate_payload) {
    char header[512];
    int n = rank2 ? snprintf(header, sizeof(header),
                             "{\"speaker_wavs\":{\"data_offsets\":[0,%zu],\"shape\":[512,%zu],"
                             "\"dtype\":\"F32\"},"
                             "\"__metadata__\":{\"epoch\":\"300\",\"sig\":\"%s\"}}",
                             512U * frames * 4U, frames, signature)
                  : snprintf(header, sizeof(header),
                             "{\"__metadata__\":{\"sig\":\"%s\",\"epoch\":\"300\"},"
                             "\"speaker_wavs\":{\"dtype\":\"F32\",\"shape\":[1,512,%zu],"
                             "\"data_offsets\":[0,%zu]}}",
                             signature, frames, 512U * frames * 4U);
    if (n <= 0 || (size_t)n >= sizeof(header))
        return 0;
    size_t header_size = ((size_t)n + 7U) & ~7U;
    FILE *f = fopen(path, "wb");
    if (!f)
        return 0;
    uint64_t len = header_size;
    if (fwrite(&len, sizeof(len), 1, f) != 1 || fwrite(header, 1, (size_t)n, f) != (size_t)n) {
        fclose(f);
        return 0;
    }
    for (size_t i = (size_t)n; i < header_size; ++i)
        fputc(' ', f);
    size_t count = (size_t)512 * frames - (truncate_payload ? 1U : 0U);
    for (size_t i = 0; i < count; ++i) {
        float value = (float)i / 1000.0f;
        if (fwrite(&value, sizeof(value), 1, f) != 1) {
            fclose(f);
            return 0;
        }
    }
    return fclose(f) == 0;
}

static int write_raw(const char *path, const char *header, size_t frames, int nan_payload) {
    size_t n = strlen(header), size = (n + 7U) & ~7U;
    FILE *f = fopen(path, "wb");
    if (!f)
        return 0;
    uint64_t len = size;
    if (fwrite(&len, sizeof(len), 1, f) != 1 || fwrite(header, 1, n, f) != n) {
        fclose(f);
        return 0;
    }
    for (size_t i = n; i < size; ++i)
        fputc(' ', f);
    size_t count = 512U * frames;
    for (size_t i = 0; i < count; ++i) {
        float value = nan_payload && i == 0 ? NAN : (float)i / 1000.0f;
        if (fwrite(&value, 4, 1, f) != 1) {
            fclose(f);
            return 0;
        }
    }
    return fclose(f) == 0;
}

int main(void) {
    const char *path = ".build/test-phonon-voice.safetensors";
    CHECK(write_voice(path, "ba7a5a45", 125, 0, 0));
    PhononVoice *voice = NULL;
    CHECK(phonon_voice_open(path, &voice));
    CHECK(phonon_voice_frames(voice) == 125);
    const float *data = phonon_voice_latents(voice);
    CHECK(data && data[0] == 0.0f && data[1] == 0.125f && data[512] == 0.001f);
    phonon_voice_close(voice);
    CHECK(write_voice(path, "wrong", 125, 0, 0));
    CHECK(!phonon_voice_open(path, &voice) && voice == NULL);
    CHECK(write_voice(path, "ba7a5a45", 125, 0, 1));
    CHECK(!phonon_voice_open(path, &voice) && voice == NULL);
    CHECK(write_voice(path, "ba7a5a45", 64, 1, 0));
    CHECK(phonon_voice_open(path, &voice));
    CHECK(phonon_voice_frames(voice) == 64);
    phonon_voice_close(voice);
    const char *duplicate =
        "{\"__metadata__\":{\"sig\":\"ba7a5a45\",\"epoch\":\"300\"},\"speaker_wavs\":{\"dtype\":"
        "\"F32\",\"dtype\":\"F32\",\"shape\":[1,512,2],\"data_offsets\":[0,4096]}}";
    const char *bad_shape =
        "{\"__metadata__\":{\"sig\":\"ba7a5a45\",\"epoch\":\"300\"},\"speaker_wavs\":{\"dtype\":"
        "\"F32\",\"shape\":[1,511,2],\"data_offsets\":[0,4096]}}";
    const char *bad_dtype =
        "{\"__metadata__\":{\"sig\":\"ba7a5a45\",\"epoch\":\"300\"},\"speaker_wavs\":{\"dtype\":"
        "\"I32\",\"shape\":[1,512,2],\"data_offsets\":[0,4096]}}";
    CHECK(write_raw(path, duplicate, 2, 0));
    CHECK(!phonon_voice_open(path, &voice) && voice == NULL);
    CHECK(write_raw(path, bad_shape, 2, 0));
    CHECK(!phonon_voice_open(path, &voice) && voice == NULL);
    CHECK(write_raw(path, bad_dtype, 2, 0));
    CHECK(!phonon_voice_open(path, &voice) && voice == NULL);
    const char *valid2 =
        "{\"__metadata__\":{\"sig\":\"ba7a5a45\",\"epoch\":\"300\"},\"speaker_wavs\":{\"dtype\":"
        "\"F32\",\"shape\":[1,512,2],\"data_offsets\":[0,4096]}}";
    CHECK(write_raw(path, valid2, 2, 1));
    CHECK(!phonon_voice_open(path, &voice) && voice == NULL);
    const char *escaped = "{\"__metad\\u0061ta__\":{\"s\\u0069g\":\"ba\\u0037a5a45\",\"epoch\":"
                          "\"300\",\"model_ext\":\"ba7a5a45@300\"},\"speaker_wavs\":{\"data_"
                          "offsets\":[0,4096],\"shape\":[1,512,2],\"dtype\":\"F32\"}}";
    CHECK(write_raw(path, escaped, 2, 0));
    CHECK(phonon_voice_open(path, &voice));
    phonon_voice_close(voice);
    const char *bad_surrogate =
        "{\"__metadata__\":{\"sig\":\"ba7a5a45\",\"epoch\":\"300\",\"extra\":\"\\uD800\"},"
        "\"speaker_wavs\":{\"dtype\":\"F32\",\"shape\":[1,512,2],\"data_offsets\":[0,4096]}}";
    const char *bad_metadata_type =
        "{\"__metadata__\":{\"sig\":\"ba7a5a45\",\"epoch\":\"300\",\"extra\":1},\"speaker_wavs\":{"
        "\"dtype\":\"F32\",\"shape\":[1,512,2],\"data_offsets\":[0,4096]}}";
    const char *bad_model_ext =
        "{\"__metadata__\":{\"sig\":\"ba7a5a45\",\"epoch\":\"300\",\"model_ext\":\"other@1\"},"
        "\"speaker_wavs\":{\"dtype\":\"F32\",\"shape\":[1,512,2],\"data_offsets\":[0,4096]}}";
    CHECK(write_raw(path, bad_surrogate, 2, 0));
    CHECK(!phonon_voice_open(path, &voice) && voice == NULL);
    CHECK(write_raw(path, bad_metadata_type, 2, 0));
    CHECK(!phonon_voice_open(path, &voice) && voice == NULL);
    CHECK(write_raw(path, bad_model_ext, 2, 0));
    CHECK(!phonon_voice_open(path, &voice) && voice == NULL);
    /* Source loader distinguishes names/layout, never width alone. */
    const char *names[] = {"emb", "audio_prompt"};
    for (size_t kind = 0; kind < 2; ++kind) {
        for (int rank = 2; rank <= 3; ++rank) {
            char projected[512];
            snprintf(projected, sizeof(projected),
                     "{\"__metadata__\":{\"model_ext\":\"ba7a5a45@300\"},\"%s\":{\"dtype\":\"F32\","
                     "\"shape\":%s,\"data_offsets\":[0,4096]}}",
                     names[kind], rank == 2 ? "[2,512]" : "[1,2,512]");
            CHECK(write_raw(path, projected, 2, 0));
            CHECK(phonon_voice_open(path, &voice));
            CHECK(phonon_voice_is_projected(voice));
            CHECK(phonon_voice_frames(voice) == 2);
            CHECK(phonon_voice_latents(voice)[1] == .001f);
            CHECK(phonon_voice_latents(voice)[512] == .512f);
            phonon_voice_close(voice);
            CHECK(write_raw(path, projected, 2, 1));
            CHECK(!phonon_voice_open(path, &voice));
        }
    }
    const char *invalid_projected[] = {
        "{\"emb\":{\"dtype\":\"F32\",\"shape\":[2,512],\"data_offsets\":[0,4096]}}",
        "{\"__metadata__\":{\"model_ext\":\"8aa279a6@100\"},\"emb\":{\"dtype\":\"F32\",\"shape\":["
        "2,512],\"data_offsets\":[0,4096]}}",
        "{\"__metadata__\":{\"model_ext\":\"ba7a5a45@300\",\"sig\":\"wrong\"},\"emb\":{\"dtype\":"
        "\"F32\",\"shape\":[2,512],\"data_offsets\":[0,4096]}}",
        "{\"__metadata__\":{\"model_ext\":\"ba7a5a45@300\"},\"emb\":{\"dtype\":\"F32\",\"shape\":["
        "512,2],\"data_offsets\":[0,4096]}}",
        "{\"__metadata__\":{\"model_ext\":\"ba7a5a45@300\"},\"unknown\":{\"dtype\":\"F32\","
        "\"shape\":[2,512],\"data_offsets\":[0,4096]}}"};
    for (size_t i = 0; i < sizeof(invalid_projected) / sizeof(invalid_projected[0]); ++i) {
        CHECK(write_raw(path, invalid_projected[i], 2, 0));
        CHECK(!phonon_voice_open(path, &voice) && !voice);
    }
    remove(path);
    puts("phonon voice safetensors compatibility: ok");
    return 0;
}

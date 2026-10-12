#include "wav_writer.h"

#include "kokoro_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>

static int write_u16(FILE *file, uint16_t value) {
    unsigned char bytes[2] = {(unsigned char)(value & 0xffU),
                              (unsigned char)((value >> 8U) & 0xffU)};
    return fwrite(bytes, 1, sizeof(bytes), file) == sizeof(bytes);
}

static int write_u32(FILE *file, uint32_t value) {
    unsigned char bytes[4] = {
        (unsigned char)(value & 0xffU), (unsigned char)((value >> 8U) & 0xffU),
        (unsigned char)((value >> 16U) & 0xffU), (unsigned char)((value >> 24U) & 0xffU)};
    return fwrite(bytes, 1, sizeof(bytes), file) == sizeof(bytes);
}

static KokoroStatus validate_wav_input(const float *samples, size_t sample_count,
                                       uint32_t sample_rate_hz, KokoroError *error) {
    if (samples == NULL || sample_count == 0 || sample_rate_hz == 0 ||
        sample_count > (UINT32_MAX - 36U) / 2U || sample_rate_hz > UINT32_MAX / 2U) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT, "invalid WAV output arguments");
        return KOKORO_INVALID_ARGUMENT;
    }
    for (size_t index = 0; index < sample_count; index++) {
        if (!isfinite(samples[index])) {
            kokoro_error_set(error, KOKORO_NUMERIC_ERROR, "WAV samples contain a non-finite value");
            return KOKORO_NUMERIC_ERROR;
        }
    }
    return KOKORO_OK;
}

static KokoroStatus write_validated_wav(FILE *file, const float *samples, size_t sample_count,
                                        uint32_t sample_rate_hz, KokoroError *error) {
    uint32_t data_size = (uint32_t)(sample_count * 2U);
    int valid = fwrite("RIFF", 1, 4, file) == 4 && write_u32(file, 36U + data_size) &&
                fwrite("WAVEfmt ", 1, 8, file) == 8 && write_u32(file, 16U) &&
                write_u16(file, 1U) && write_u16(file, 1U) && write_u32(file, sample_rate_hz) &&
                write_u32(file, sample_rate_hz * 2U) && write_u16(file, 2U) &&
                write_u16(file, 16U) && fwrite("data", 1, 4, file) == 4 &&
                write_u32(file, data_size);
    for (size_t index = 0; index < sample_count && valid; index++) {
        float sample = samples[index];
        long quantized = sample <= -1.0f  ? -32768L
                         : sample >= 1.0f ? 32767L
                                          : lrintf(sample * 32767.0f);
        valid = write_u16(file, (uint16_t)(int16_t)quantized);
    }
    if (!valid) {
        kokoro_error_set(error, KOKORO_IO_ERROR, "failed writing WAV output stream");
        return KOKORO_IO_ERROR;
    }
    kokoro_error_set(error, KOKORO_OK, NULL);
    return KOKORO_OK;
}

KokoroStatus kokoro_wav_write_pcm16_stream(FILE *file, const float *samples, size_t sample_count,
                                           uint32_t sample_rate_hz, KokoroError *error) {
    if (file == NULL) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT, "invalid WAV output arguments");
        return KOKORO_INVALID_ARGUMENT;
    }
    KokoroStatus status = validate_wav_input(samples, sample_count, sample_rate_hz, error);
    if (status != KOKORO_OK) {
        return status;
    }
    return write_validated_wav(file, samples, sample_count, sample_rate_hz, error);
}

KokoroStatus kokoro_wav_write_pcm16(const char *path, const float *samples, size_t sample_count,
                                    uint32_t sample_rate_hz, KokoroError *error) {
    if (path == NULL || path[0] == '\0') {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT, "invalid WAV output path");
        return KOKORO_INVALID_ARGUMENT;
    }
    KokoroStatus status = validate_wav_input(samples, sample_count, sample_rate_hz, error);
    if (status != KOKORO_OK) {
        return status;
    }
    FILE *file = fopen(path, "wb");
    if (file == NULL) {
        kokoro_error_set(error, KOKORO_IO_ERROR, "cannot open WAV output '%s'", path);
        return KOKORO_IO_ERROR;
    }
    status = write_validated_wav(file, samples, sample_count, sample_rate_hz, error);
    if (fclose(file) != 0 && status == KOKORO_OK) {
        kokoro_error_set(error, KOKORO_IO_ERROR, "failed writing WAV output '%s'", path);
        return KOKORO_IO_ERROR;
    }
    return status;
}

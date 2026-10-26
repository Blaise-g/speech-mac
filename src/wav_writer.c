#include "wav_writer.h"

#include "speech_error_internal.h"

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

static SpeechIOStatus validate_wav_input(const float *samples, size_t sample_count,
                                         uint32_t sample_rate_hz, SpeechError *error) {
    if (samples == NULL || sample_count == 0 || sample_rate_hz == 0 ||
        sample_count > (UINT32_MAX - 36U) / 2U || sample_rate_hz > UINT32_MAX / 2U) {
        speech_error_set(error, SPEECH_IO_INVALID_ARGUMENT, "invalid WAV output arguments");
        return SPEECH_IO_INVALID_ARGUMENT;
    }
    for (size_t index = 0; index < sample_count; index++) {
        if (!isfinite(samples[index])) {
            speech_error_set(error, SPEECH_IO_NUMERIC_ERROR,
                             "WAV samples contain a non-finite value");
            return SPEECH_IO_NUMERIC_ERROR;
        }
    }
    return SPEECH_IO_OK;
}

static SpeechIOStatus write_validated_wav(FILE *file, const float *samples, size_t sample_count,
                                          uint32_t sample_rate_hz, SpeechError *error) {
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
        speech_error_set(error, SPEECH_IO_ERROR, "failed writing WAV output stream");
        return SPEECH_IO_ERROR;
    }
    speech_error_set(error, SPEECH_IO_OK, NULL);
    return SPEECH_IO_OK;
}

SpeechIOStatus speech_wav_write_pcm16_stream(FILE *file, const float *samples, size_t sample_count,
                                             uint32_t sample_rate_hz, SpeechError *error) {
    if (file == NULL) {
        speech_error_set(error, SPEECH_IO_INVALID_ARGUMENT, "invalid WAV output arguments");
        return SPEECH_IO_INVALID_ARGUMENT;
    }
    SpeechIOStatus status = validate_wav_input(samples, sample_count, sample_rate_hz, error);
    if (status != SPEECH_IO_OK) {
        return status;
    }
    return write_validated_wav(file, samples, sample_count, sample_rate_hz, error);
}

SpeechIOStatus speech_wav_write_pcm16(const char *path, const float *samples, size_t sample_count,
                                      uint32_t sample_rate_hz, SpeechError *error) {
    if (path == NULL || path[0] == '\0') {
        speech_error_set(error, SPEECH_IO_INVALID_ARGUMENT, "invalid WAV output path");
        return SPEECH_IO_INVALID_ARGUMENT;
    }
    SpeechIOStatus status = validate_wav_input(samples, sample_count, sample_rate_hz, error);
    if (status != SPEECH_IO_OK) {
        return status;
    }
    FILE *file = fopen(path, "wb");
    if (file == NULL) {
        speech_error_set(error, SPEECH_IO_ERROR, "cannot open WAV output '%s'", path);
        return SPEECH_IO_ERROR;
    }
    status = write_validated_wav(file, samples, sample_count, sample_rate_hz, error);
    if (fclose(file) != 0 && status == SPEECH_IO_OK) {
        speech_error_set(error, SPEECH_IO_ERROR, "failed writing WAV output '%s'", path);
        return SPEECH_IO_ERROR;
    }
    return status;
}

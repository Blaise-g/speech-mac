#include "phonon_voice.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VOICE_MAX_FILE (16U * 1024U * 1024U)
#define VOICE_MAX_HEADER (1024U * 1024U)
#define VOICE_MAX_FRAMES 4096U
#define VOICE_WIDTH 512U

struct PhononVoice {
    size_t frames;
    int projected;
    float *latents;
};

typedef struct {
    const unsigned char *p, *end;
} Json;
typedef struct {
    const unsigned char *p;
    size_t n;
    int escaped;
} JsonString;
typedef struct {
    int have_sig, have_epoch;
    int sig_ok, epoch_ok;
    int have_model_ext, model_ext_ok;
} Metadata;
typedef struct {
    int have_dtype, have_shape, have_offsets;
    int dtype_ok, shape_ok, offsets_ok;
    size_t frames;
    uint64_t offset_start, offset_end;
} Tensor;

static void ws(Json *j) {
    while (j->p < j->end && (*j->p == ' ' || *j->p == '\n' || *j->p == '\r' || *j->p == '\t'))
        ++j->p;
}

static int string(Json *j, JsonString *out) {
    if (j->p == j->end || *j->p++ != '"')
        return 0;
    const unsigned char *start = j->p;
    int escaped = 0;
    while (j->p < j->end) {
        unsigned char c = *j->p++;
        if (c == '"') {
            out->p = start;
            out->n = (size_t)(j->p - start - 1);
            out->escaped = escaped;
            return 1;
        }
        if (c < 0x20)
            return 0;
        if (c == '\\') {
            escaped = 1;
            if (j->p == j->end)
                return 0;
            c = *j->p++;
            if (c == 'u') {
                uint32_t codepoint = 0;
                for (size_t i = 0; i < 4; ++i) {
                    if (j->p == j->end ||
                        !((*j->p >= '0' && *j->p <= '9') || (*j->p >= 'a' && *j->p <= 'f') ||
                          (*j->p >= 'A' && *j->p <= 'F')))
                        return 0;
                    unsigned char digit = *j->p++;
                    codepoint = (codepoint << 4) | (digit <= '9'   ? digit - '0'
                                                    : digit <= 'F' ? digit - 'A' + 10
                                                                   : digit - 'a' + 10);
                }
                if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
                    if ((size_t)(j->end - j->p) < 6 || j->p[0] != '\\' || j->p[1] != 'u')
                        return 0;
                    j->p += 2;
                    uint32_t low = 0;
                    for (size_t i = 0; i < 4; ++i) {
                        unsigned char d = *j->p++;
                        if (!((d >= '0' && d <= '9') || (d >= 'a' && d <= 'f') ||
                              (d >= 'A' && d <= 'F')))
                            return 0;
                        low = (low << 4) | (d <= '9'   ? d - '0'
                                            : d <= 'F' ? d - 'A' + 10
                                                       : d - 'a' + 10);
                    }
                    if (low < 0xdc00 || low > 0xdfff)
                        return 0;
                } else if (codepoint >= 0xdc00 && codepoint <= 0xdfff)
                    return 0;
            } else if (!(c == '"' || c == '\\' || c == '/' || c == 'b' || c == 'f' || c == 'n' ||
                         c == 'r' || c == 't'))
                return 0;
        } else if (c >= 0x80) {
            size_t extra;
            if (c >= 0xc2 && c <= 0xdf)
                extra = 1;
            else if (c >= 0xe0 && c <= 0xef)
                extra = 2;
            else if (c >= 0xf0 && c <= 0xf4)
                extra = 3;
            else
                return 0;
            if ((size_t)(j->end - j->p) < extra)
                return 0;
            unsigned char second = j->p[0];
            if ((second & 0xc0) != 0x80 || (c == 0xe0 && second < 0xa0) ||
                (c == 0xed && second >= 0xa0) || (c == 0xf0 && second < 0x90) ||
                (c == 0xf4 && second >= 0x90))
                return 0;
            for (size_t i = 1; i < extra; ++i)
                if ((j->p[i] & 0xc0) != 0x80)
                    return 0;
            j->p += extra;
        }
    }
    return 0;
}

static int eq(JsonString s, const char *text) {
    size_t i = 0, o = 0, wanted = strlen(text);
    while (i < s.n) {
        unsigned char c = s.p[i++];
        if (c == '\\') {
            c = s.p[i++];
            if (c == 'u') {
                uint32_t cp = 0;
                for (size_t k = 0; k < 4; ++k) {
                    unsigned char d = s.p[i++];
                    cp = (cp << 4) | (d <= '9' ? d - '0' : d <= 'F' ? d - 'A' + 10 : d - 'a' + 10);
                }
                if (cp > 0x7f)
                    return 0;
                c = (unsigned char)cp;
            } else {
                const char *esc = "\"\\/bfnrt";
                const char *found = strchr(esc, c);
                if (!found)
                    return 0;
                const char values[] = {'"', '\\', '/', '\b', '\f', '\n', '\r', '\t'};
                c = (unsigned char)values[found - esc];
            }
        }
        if (o >= wanted || (unsigned char)text[o++] != c)
            return 0;
    }
    return o == wanted;
}

static int uint_value(Json *j, uint64_t *value) {
    ws(j);
    if (j->p == j->end || *j->p < '0' || *j->p > '9')
        return 0;
    if (*j->p == '0' && j->p + 1 < j->end && j->p[1] >= '0' && j->p[1] <= '9')
        return 0;
    uint64_t n = 0;
    while (j->p < j->end && *j->p >= '0' && *j->p <= '9') {
        unsigned int digit = (unsigned int)(*j->p++ - '0');
        if (n > (UINT64_MAX - digit) / 10)
            return 0;
        n = n * 10 + digit;
    }
    *value = n;
    return 1;
}

static int skip_value(Json *j, unsigned int depth);

static int skip_array(Json *j, unsigned int depth) {
    if (depth > 32 || j->p == j->end || *j->p++ != '[')
        return 0;
    ws(j);
    if (j->p < j->end && *j->p == ']') {
        ++j->p;
        return 1;
    }
    for (;;) {
        if (!skip_value(j, depth + 1))
            return 0;
        ws(j);
        if (j->p == j->end)
            return 0;
        if (*j->p++ == ']')
            return 1;
        if (j->p[-1] != ',')
            return 0;
    }
}

static int skip_object(Json *j, unsigned int depth) {
    if (depth > 32 || j->p == j->end || *j->p++ != '{')
        return 0;
    ws(j);
    if (j->p < j->end && *j->p == '}') {
        ++j->p;
        return 1;
    }
    for (;;) {
        JsonString key;
        if (!string(j, &key))
            return 0;
        ws(j);
        if (j->p == j->end || *j->p++ != ':')
            return 0;
        if (!skip_value(j, depth + 1))
            return 0;
        ws(j);
        if (j->p == j->end)
            return 0;
        if (*j->p++ == '}')
            return 1;
        if (j->p[-1] != ',')
            return 0;
        ws(j);
    }
}

static int skip_number(Json *j) {
    const unsigned char *start = j->p;
    if (j->p < j->end && *j->p == '-')
        ++j->p;
    if (j->p == j->end)
        return 0;
    if (*j->p == '0')
        ++j->p;
    else {
        if (*j->p < '1' || *j->p > '9')
            return 0;
        while (j->p < j->end && *j->p >= '0' && *j->p <= '9')
            ++j->p;
    }
    if (j->p < j->end && *j->p == '.') {
        ++j->p;
        if (j->p == j->end || *j->p < '0' || *j->p > '9')
            return 0;
        while (j->p < j->end && *j->p >= '0' && *j->p <= '9')
            ++j->p;
    }
    if (j->p < j->end && (*j->p == 'e' || *j->p == 'E')) {
        ++j->p;
        if (j->p < j->end && (*j->p == '+' || *j->p == '-'))
            ++j->p;
        if (j->p == j->end || *j->p < '0' || *j->p > '9')
            return 0;
        while (j->p < j->end && *j->p >= '0' && *j->p <= '9')
            ++j->p;
    }
    return j->p > start;
}

static int skip_value(Json *j, unsigned int depth) {
    ws(j);
    if (j->p == j->end || depth > 32)
        return 0;
    if (*j->p == '"') {
        JsonString s;
        return string(j, &s);
    }
    if (*j->p == '{')
        return skip_object(j, depth + 1);
    if (*j->p == '[')
        return skip_array(j, depth + 1);
    if (*j->p == '-' || (*j->p >= '0' && *j->p <= '9'))
        return skip_number(j);
    const char *literals[] = {"true", "false", "null"};
    for (size_t i = 0; i < 3; ++i) {
        size_t n = strlen(literals[i]);
        if ((size_t)(j->end - j->p) >= n && memcmp(j->p, literals[i], n) == 0) {
            j->p += n;
            return 1;
        }
    }
    return 0;
}

static int parse_metadata(Json *j, Metadata *m) {
    if (j->p == j->end || *j->p++ != '{')
        return 0;
    ws(j);
    if (j->p < j->end && *j->p == '}') {
        ++j->p;
        return 1;
    }
    for (;;) {
        JsonString key;
        if (!string(j, &key))
            return 0;
        ws(j);
        if (j->p == j->end || *j->p++ != ':')
            return 0;
        ws(j);
        if (eq(key, "sig") || eq(key, "epoch")) {
            JsonString value;
            int *seen = eq(key, "sig") ? &m->have_sig : &m->have_epoch;
            int *match = eq(key, "sig") ? &m->sig_ok : &m->epoch_ok;
            const char *expected = eq(key, "sig") ? "ba7a5a45" : "300";
            if (*seen || !string(j, &value))
                return 0;
            *seen = 1;
            *match = eq(value, expected);
        } else if (eq(key, "model_ext")) {
            JsonString value;
            if (m->have_model_ext || !string(j, &value))
                return 0;
            m->have_model_ext = 1;
            m->model_ext_ok = eq(value, "ba7a5a45@300");
        } else {
            JsonString value;
            if (!string(j, &value))
                return 0;
        }
        ws(j);
        if (j->p == j->end)
            return 0;
        if (*j->p++ == '}')
            return 1;
        if (j->p[-1] != ',')
            return 0;
        ws(j);
    }
}

static int parse_shape(Json *j, size_t *frames, int projected) {
    if (j->p == j->end || *j->p++ != '[')
        return 0;
    uint64_t values[3] = {0};
    size_t n = 0;
    for (;;) {
        if (n == 3 || !uint_value(j, &values[n++]))
            return 0;
        ws(j);
        if (j->p == j->end)
            return 0;
        if (*j->p++ == ']')
            break;
        if (j->p[-1] != ',')
            return 0;
    }
    uint64_t t;
    if (n == 3 && values[0] == 1 && values[projected ? 2 : 1] == VOICE_WIDTH)
        t = values[projected ? 1 : 2];
    else if (n == 2 && values[projected ? 1 : 0] == VOICE_WIDTH)
        t = values[projected ? 0 : 1];
    else
        return 0;
    if (!t || t > VOICE_MAX_FRAMES)
        return 0;
    *frames = (size_t)t;
    return 1;
}

static int parse_offsets(Json *j, uint64_t *start, uint64_t *end) {
    uint64_t a, b;
    if (j->p == j->end || *j->p++ != '[' || !uint_value(j, &a))
        return 0;
    ws(j);
    if (j->p == j->end || *j->p++ != ',' || !uint_value(j, &b))
        return 0;
    ws(j);
    if (j->p == j->end || *j->p++ != ']')
        return 0;
    *start = a;
    *end = b;
    return 1;
}

static int parse_tensor(Json *j, Tensor *t, int projected) {
    if (j->p == j->end || *j->p++ != '{')
        return 0;
    ws(j);
    if (j->p < j->end && *j->p == '}')
        return 0;
    for (;;) {
        JsonString key;
        if (!string(j, &key))
            return 0;
        ws(j);
        if (j->p == j->end || *j->p++ != ':')
            return 0;
        ws(j);
        if (eq(key, "dtype")) {
            JsonString value;
            if (t->have_dtype || !string(j, &value))
                return 0;
            t->have_dtype = 1;
            t->dtype_ok = eq(value, "F32");
        } else if (eq(key, "shape")) {
            if (t->have_shape || !parse_shape(j, &t->frames, projected))
                return 0;
            t->have_shape = 1;
            t->shape_ok = 1;
        } else if (eq(key, "data_offsets")) {
            if (t->have_offsets || !parse_offsets(j, &t->offset_start, &t->offset_end))
                return 0;
            t->have_offsets = 1;
            t->offsets_ok = 1;
        } else
            return 0;
        ws(j);
        if (j->p == j->end)
            return 0;
        if (*j->p++ == '}')
            return t->have_dtype && t->have_shape && t->have_offsets && t->dtype_ok &&
                   t->shape_ok && t->offsets_ok && t->offset_start == 0 &&
                   t->offset_end == (uint64_t)VOICE_WIDTH * t->frames * sizeof(float);
        if (j->p[-1] != ',')
            return 0;
        ws(j);
    }
}

static int parse_header(const unsigned char *data, size_t n, size_t *frames, int *projected) {
    if (memchr(data, 0, n))
        return 0;
    Json j = {data, data + n};
    Metadata metadata = {0};
    Tensor tensor = {0};
    int have_metadata = 0, have_tensor = 0;
    ws(&j);
    if (j.p == j.end || *j.p++ != '{')
        return 0;
    ws(&j);
    for (;;) {
        JsonString key;
        if (!string(&j, &key))
            return 0;
        ws(&j);
        if (j.p == j.end || *j.p++ != ':')
            return 0;
        ws(&j);
        if (eq(key, "__metadata__")) {
            if (have_metadata || !parse_metadata(&j, &metadata))
                return 0;
            have_metadata = 1;
        } else if (eq(key, "speaker_wavs") || eq(key, "emb") || eq(key, "audio_prompt")) {
            *projected = !eq(key, "speaker_wavs");
            if (have_tensor || !parse_tensor(&j, &tensor, *projected))
                return 0;
            have_tensor = 1;
        } else
            return 0;
        ws(&j);
        if (j.p == j.end)
            return 0;
        if (*j.p++ == '}')
            break;
        if (j.p[-1] != ',')
            return 0;
        ws(&j);
    }
    ws(&j);
    if (j.p != j.end || !have_metadata ||
        (!*projected && (!metadata.have_sig || !metadata.have_epoch)) ||
        (metadata.have_sig && !metadata.sig_ok) || (metadata.have_epoch && !metadata.epoch_ok) ||
        (*projected && !metadata.have_model_ext) ||
        (metadata.have_model_ext && !metadata.model_ext_ok) || !have_tensor)
        return 0;
    *frames = tensor.frames;
    return 1;
}

static uint64_t le64(const unsigned char *p) {
    uint64_t value = 0;
    for (unsigned int i = 0; i < 8; ++i)
        value |= (uint64_t)p[i] << (8U * i);
    return value;
}

void phonon_voice_close(PhononVoice *voice) {
    if (!voice)
        return;
    free(voice->latents);
    free(voice);
}

size_t phonon_voice_frames(const PhononVoice *voice) { return voice ? voice->frames : 0; }

const float *phonon_voice_latents(const PhononVoice *voice) {
    return voice ? voice->latents : NULL;
}

int phonon_voice_open(const char *path, PhononVoice **result) {
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
    if (length < 8 || (unsigned long)length > VOICE_MAX_FILE || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return 0;
    }
    size_t file_size = (size_t)length;
    unsigned char *bytes = malloc(file_size);
    if (!bytes || fread(bytes, 1, file_size, file) != file_size) {
        free(bytes);
        fclose(file);
        return 0;
    }
    fclose(file);
    uint64_t header_size = le64(bytes);
    if (!header_size || header_size > VOICE_MAX_HEADER || header_size > file_size - 8) {
        free(bytes);
        return 0;
    }
    size_t frames = 0;
    int projected = 0;
    if (!parse_header(bytes + 8, (size_t)header_size, &frames, &projected)) {
        free(bytes);
        return 0;
    }
    size_t data_start = 8 + (size_t)header_size;
    size_t data_bytes = (size_t)VOICE_WIDTH * frames * sizeof(float);
    if (data_bytes != file_size - data_start) {
        free(bytes);
        return 0;
    }
    PhononVoice *voice = calloc(1, sizeof(*voice));
    if (!voice) {
        free(bytes);
        return 0;
    }
    voice->latents = malloc(data_bytes);
    if (!voice->latents) {
        phonon_voice_close(voice);
        free(bytes);
        return 0;
    }
    voice->frames = frames;
    voice->projected = projected;
    const unsigned char *data = bytes + data_start;
    for (size_t channel = 0; channel < VOICE_WIDTH; ++channel) {
        for (size_t frame = 0; frame < frames; ++frame) {
            size_t source =
                (projected ? frame * VOICE_WIDTH + channel : channel * frames + frame) * 4;
            uint32_t bits = (uint32_t)data[source] | (uint32_t)data[source + 1] << 8 |
                            (uint32_t)data[source + 2] << 16 | (uint32_t)data[source + 3] << 24;
            float value;
            memcpy(&value, &bits, sizeof(value));
            if (!isfinite(value)) {
                phonon_voice_close(voice);
                free(bytes);
                return 0;
            }
            voice->latents[frame * VOICE_WIDTH + channel] = value;
        }
    }
    free(bytes);
    *result = voice;
    return 1;
}

int phonon_voice_is_projected(const PhononVoice *voice) { return voice && voice->projected; }

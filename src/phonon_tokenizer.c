#include "phonon_tokenizer.h"
#include <stdio.h>

#include "phonon_tokenizer_vocab.inc"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
typedef struct {
    char *s;
    size_t n;
} Piece;
static int reserve(void **p, size_t *cap, size_t need, size_t unit) {
    if (need > SIZE_MAX / unit)
        return 0;
    if (need > *cap) {
        size_t c = *cap ? *cap : 32;
        while (c < need) {
            if (c > SIZE_MAX / 2) {
                c = need;
                break;
            }
            c *= 2;
        }
        void *q = realloc(*p, c * unit);
        if (!q)
            return 0;
        *p = q;
        *cap = c;
    }
    return 1;
}
static int utf8_ok(const unsigned char *s, size_t n) {
    for (size_t i = 0; i < n;) {
        unsigned c = s[i];
        size_t z = c < 128 ? 1
                           : (c >= 194 && c <= 223
                                  ? 2
                                  : (c >= 224 && c <= 239 ? 3 : (c >= 240 && c <= 244 ? 4 : 0)));
        if (!z || z > n - i)
            return 0;
        if (z > 1) {
            for (size_t j = 1; j < z; j++)
                if ((s[i + j] & 192) != 128)
                    return 0;
            if ((c == 224 && s[i + 1] < 160) || (c == 237 && s[i + 1] >= 160) ||
                (c == 240 && s[i + 1] < 144) || (c == 244 && s[i + 1] >= 144))
                return 0;
        }
        i += z;
    }
    return 1;
}
static int byte_for_id(uint32_t id) {
    return id < 4000 && phonon_byte_value[id] < 256 ? (int)phonon_byte_value[id] : -1;
}
void phonon_token_ids_free(PhononTokenIds *x) {
    if (x) {
        free(x->ids);
        x->ids = NULL;
        x->length = 0;
    }
}
int phonon_tokenizer_encode(const char *text, size_t n, PhononTokenIds *out) {
    if (!out)
        return 0;
    out->ids = NULL;
    out->length = 0;
    if (!text || n > PHONON_TOKENIZER_MAX_INPUT_BYTES || memchr(text, 0, n) ||
        !utf8_ok((const unsigned char *)text, n))
        return 0;
    if (n == 0)
        return 1;
    if (n > (SIZE_MAX - 4) / 3 || n > INT32_MAX - 4)
        return 0;
    size_t tn = n * 3 + 4;
    char *t = malloc(tn);
    if (!t)
        return 0;
    size_t z = 0;
    memcpy(t + z, "▁", 3);
    z += 3;
    for (size_t i = 0; i < n;) {
        unsigned char c = text[i];
        size_t k = c < 128 ? 1 : (c < 224 ? 2 : (c < 240 ? 3 : 4));
        if (k == 1 && c == ' ') {
            memcpy(t + z, "▁", 3);
            z += 3;
        } else {
            memcpy(t + z, text + i, k);
            z += k;
        }
        i += k;
    }
    t[z] = 0;
    double *best = malloc((z + 1) * sizeof(*best));
    int32_t *prev = malloc((z + 1) * sizeof(*prev)), *tok = malloc((z + 1) * sizeof(*tok));
    if (!best || !prev || !tok) {
        free(t);
        free(best);
        free(prev);
        free(tok);
        return 0;
    }
    for (size_t i = 0; i <= z; i++) {
        best[i] = -1e300;
        prev[i] = -1;
        tok[i] = -1;
    }
    best[0] = 0;
    double min_score = phonon_scores[0];
    for (size_t id = 1; id < 4000; id++)
        if (phonon_scores[id] < min_score)
            min_score = phonon_scores[id];
    for (size_t i = 0; i < z;) {
        unsigned char c = (unsigned char)t[i];
        size_t scalar = c < 128 ? 1 : (c < 224 ? 2 : (c < 240 ? 3 : 4));
        int has_single = 0;
        if (best[i] > -1e299) {
            size_t bucket = (unsigned char)t[i];
            for (size_t ci = phonon_candidate_offsets[bucket];
                 ci < phonon_candidate_offsets[bucket + 1]; ci++) {
                uint32_t id = phonon_candidate_ids[ci];
                size_t l = strlen(phonon_vocab[id]);
                if (!l || l > PHONON_VOCAB_MAX_BYTES || l > z - i ||
                    memcmp(t + i, phonon_vocab[id], l))
                    continue;
                int crosses_metaspace = 0;
                for (size_t k = i + 1; k + 2 < i + l; k++)
                    if ((unsigned char)t[k] == 0xe2 && (unsigned char)t[k + 1] == 0x96 &&
                        (unsigned char)t[k + 2] == 0x81) {
                        crosses_metaspace = 1;
                        break;
                    }
                if (crosses_metaspace)
                    continue;
                double v = best[i] + phonon_scores[id];
                if (v > best[i + l]) {
                    best[i + l] = v;
                    prev[i + l] = (int32_t)i;
                    tok[i + l] = (int32_t)id;
                }
                if (l == scalar)
                    has_single = 1;
            }
            if (!has_single && best[i] + min_score - 10.0 > best[i + scalar]) {
                best[i + scalar] = best[i] + min_score - 10.0;
                prev[i + scalar] = (int32_t)i;
                tok[i + scalar] = 0;
            } else if (!has_single && prev[i + scalar] < 0) {
                best[i + scalar] = best[i] + min_score - 10.0;
                prev[i + scalar] = (int32_t)i;
                tok[i + scalar] = 0;
            }
        }
        i += scalar;
    }
    size_t path_count = 0;
    for (size_t i = z; i;) {
        if (prev[i] < 0) {
            free(t);
            free(best);
            free(prev);
            free(tok);
            return 0;
        }
        path_count++;
        i = (size_t)prev[i];
    }
    uint32_t *ids = malloc((z + 1) * sizeof(*ids));
    if (!ids) {
        free(t);
        free(best);
        free(prev);
        free(tok);
        return 0;
    }
    size_t used = 0;
    for (size_t i = z; i;) {
        size_t start = (size_t)prev[i];
        if (tok[i] == 0) {
            for (size_t j = i; j > start; j--)
                ids[used++] = phonon_byte_id[(unsigned char)t[j - 1]];
        } else {
            ids[used++] = (uint32_t)tok[i];
        }
        i = start;
    }
    for (size_t i = 0; i < used / 2; i++) {
        uint32_t tmp = ids[i];
        ids[i] = ids[used - i - 1];
        ids[used - i - 1] = tmp;
    }
    out->ids = ids;
    out->length = used;
    (void)path_count;
    free(t);
    free(best);
    free(prev);
    free(tok);
    return 1;
}
int phonon_tokenizer_decode(const uint32_t *ids, size_t n, char **out, size_t *length) {
    if (out)
        *out = NULL;
    if (length)
        *length = 0;
    if (!out || !length || (!ids && n) || n > PHONON_TOKENIZER_MAX_IDS)
        return 0;
    size_t cap = 0, used = 0;
    size_t decoded_tokens = 0;
    char *b = NULL;
    for (size_t i = 0; i < n; i++) {
        if (ids[i] >= 4000)
            goto fail;
        int byte_value = byte_for_id(ids[i]);
        if (byte_value >= 0) {
            size_t run = i;
            while (run < n && (ids[run] < 204 || byte_for_id(ids[run]) >= 0)) {
                if (ids[run] >= 4000)
                    break;
                run++;
            }
            size_t count = 0;
            for (size_t j = i; j < run; j++)
                if (byte_for_id(ids[j]) >= 0)
                    count++;
            unsigned char *raw = calloc(count ? count : 1, 1);
            if (!raw)
                goto fail;
            size_t raw_i = 0;
            for (size_t j = i; j < run; j++) {
                int byte = byte_for_id(ids[j]);
                if (byte >= 0)
                    raw[raw_i++] = (unsigned char)byte;
            }
            if (utf8_ok(raw, count)) {
                if (!reserve((void **)&b, &cap, used + count + 1, 1)) {
                    free(raw);
                    goto fail;
                }
                memcpy(b + used, raw, count);
                used += count;
            } else {
                if (count > (SIZE_MAX - used) / 3 ||
                    !reserve((void **)&b, &cap, used + count * 3 + 1, 1)) {
                    free(raw);
                    goto fail;
                }
                for (size_t j = 0; j < count; j++) {
                    b[used++] = (char)0xef;
                    b[used++] = (char)0xbf;
                    b[used++] = (char)0xbd;
                }
            }
            free(raw);
            i = run - 1;
            decoded_tokens++;
            continue;
        }
        const char *p = phonon_vocab[ids[i]];
        if (ids[i] < 204)
            continue;
        size_t l = strlen(p);
        if (l == 6 && !memcmp(p, "<0x", 3) && p[5] == '>') {
            unsigned x;
            sscanf(p + 3, "%2x", &x);
            if (!reserve((void **)&b, &cap, used + 1, 1))
                goto fail;
            b[used++] = (char)x;
            continue;
        }
        if (!reserve((void **)&b, &cap, used + l + 1, 1))
            goto fail;
        for (size_t j = 0; j < l; j++) {
            if ((unsigned char)p[j] == 0xe2 && j + 2 < l && (unsigned char)p[j + 1] == 0x96 &&
                (unsigned char)p[j + 2] == 0x81) {
                if (decoded_tokens != 0)
                    b[used++] = ' ';
                j += 2;
            } else
                b[used++] = p[j];
        }
        decoded_tokens++;
    }
    if (!reserve((void **)&b, &cap, used + 1, 1))
        goto fail;
    b[used] = 0;
    *out = b;
    *length = used;
    return 1;
fail:
    free(b);
    return 0;
}

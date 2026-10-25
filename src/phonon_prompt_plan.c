#include "phonon_prompt_plan.h"
#include "phonon_prompt_unicode.inc"
#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
static int scalar(const unsigned char *s, size_t n, unsigned int *cp, size_t *width) {
    if (!n)
        return 0;
    unsigned c = s[0];
    if (c < 0x80) {
        *cp = c;
        *width = 1;
        return 1;
    }
    size_t w = c >= 0xc2 && c <= 0xdf   ? 2
               : c >= 0xe0 && c <= 0xef ? 3
               : c >= 0xf0 && c <= 0xf4 ? 4
                                        : 0;
    if (!w || w > n)
        return 0;
    for (size_t i = 1; i < w; i++)
        if ((s[i] & 0xc0) != 0x80)
            return 0;
    if ((c == 0xe0 && s[1] < 0xa0) || (c == 0xed && s[1] >= 0xa0) || (c == 0xf0 && s[1] < 0x90) ||
        (c == 0xf4 && s[1] >= 0x90))
        return 0;
    *cp = w == 2   ? ((c & 0x1f) << 6) | (s[1] & 0x3f)
          : w == 3 ? ((c & 0xf) << 12) | ((s[1] & 0x3f) << 6) | (s[2] & 0x3f)
                   : ((c & 7) << 18) | ((s[1] & 0x3f) << 12) | ((s[2] & 0x3f) << 6) | (s[3] & 0x3f);
    *width = w;
    return 1;
}
static int in_ranges(unsigned int cp, const unsigned int ranges[][2], size_t count) {
    size_t lo = 0, hi = count;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (cp < ranges[mid][0])
            hi = mid;
        else if (cp > ranges[mid][1])
            lo = mid + 1;
        else
            return 1;
    }
    return 0;
}
static const PhononUpperMap *upper_mapping(unsigned int cp) {
    size_t lo = 0, hi = phonon_upper_map_count;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (cp < phonon_upper_map[mid].code)
            hi = mid;
        else if (cp > phonon_upper_map[mid].code)
            lo = mid + 1;
        else
            return &phonon_upper_map[mid];
    }
    return NULL;
}
static int valid_utf8(const unsigned char *s, size_t n) {
    size_t i = 0, w;
    unsigned int cp;
    while (i < n) {
        if (!scalar(s + i, n - i, &cp, &w))
            return 0;
        i += w;
    }
    return 1;
}
static int ws(const unsigned char *s, size_t n, size_t *k) {
    unsigned int cp;
    if (!scalar(s, n, &cp, k))
        return 0;
    return in_ranges(cp, phonon_space_ranges, phonon_space_ranges_count);
}
static int alnum_end(const unsigned char *s, size_t n) {
    if (!n)
        return 0;
    size_t start = n - 1;
    while (start && (s[start] & 0xc0) == 0x80)
        start--;
    unsigned int cp;
    size_t width;
    if (!scalar(s + start, n - start, &cp, &width) || start + width != n)
        return 0;
    return in_ranges(cp, phonon_alnum_ranges, phonon_alnum_ranges_count);
}
int phonon_prepare_text_prompt(const char *text, size_t n, char **out, size_t *olen, size_t *tail) {
    if (out)
        *out = NULL;
    if (olen)
        *olen = 0;
    if (tail)
        *tail = 0;
    if (!text || !out || !olen || !tail || n > PHONON_TOKENIZER_MAX_INPUT_BYTES ||
        memchr(text, 0, n) || !valid_utf8((const unsigned char *)text, n))
        return 0;
    size_t a = 0, b = n, k;
    while (a < b && ws((const unsigned char *)text + a, b - a, &k))
        a += k;
    while (b > a) {
        size_t p = b - 1;
        while (p > a && ((unsigned char)text[p] & 0xc0) == 0x80)
            p--;
        size_t q;
        if (ws((const unsigned char *)text + p, b - p, &q) && p + q == b)
            b = p;
        else
            break;
    }
    if (a == b) {
        char *e = calloc(1, 1);
        if (!e)
            return 0;
        *out = e;
        *tail = 3;
        return 1;
    }
    char *buf = malloc(n * 3 + 8);
    if (!buf)
        return 0;
    size_t w = 0, words = 0, i = a;
    int pending = 0, first_scalar = 1;
    while (i < b) {
        if (ws((const unsigned char *)text + i, b - i, &k)) {
            pending = 1;
            i += k;
            continue;
        }
        if (pending && w)
            buf[w++] = ' ';
        pending = 0;
        size_t q;
        ws((const unsigned char *)text + i, b - i, &q);
        if (q == 0)
            q = 1;
        unsigned int cp;
        size_t scalar_width;
        if (!scalar((const unsigned char *)text + i, b - i, &cp, &scalar_width)) {
            free(buf);
            return 0;
        }
        const PhononUpperMap *mapping = first_scalar ? upper_mapping(cp) : NULL;
        if (mapping) {
            memcpy(buf + w, mapping->bytes, mapping->length);
            w += mapping->length;
        } else if (first_scalar && cp >= 'a' && cp <= 'z')
            buf[w++] = (char)(cp - 32);
        else {
            memcpy(buf + w, text + i, q);
            w += q;
        }
        first_scalar = 0;
        i += scalar_width;
        if (i == b || ws((const unsigned char *)text + i, b - i, &k))
            words++;
    }
    if (words <= 4)
        *tail = 3;
    else
        *tail = 1;
    if (alnum_end((const unsigned char *)buf, w))
        buf[w++] = '.';
    buf[w] = 0;
    *out = buf;
    *olen = w;
    return 1;
}
void phonon_prompt_plan_free(PhononPromptPlan *p) {
    if (!p)
        return;
    for (size_t i = 0; i < p->count; i++) {
        free(p->chunks[i].text);
        phonon_token_ids_free(&p->chunks[i].tokens);
    }
    free(p->chunks);
    p->chunks = NULL;
    p->count = 0;
}
static int add_chunk(PhononPromptPlan *p, const char *t, size_t n, double rate) {
    if (p->count == SIZE_MAX / sizeof(*p->chunks) - 1)
        return 0;
    PhononPromptChunk *c = realloc(p->chunks, (p->count + 1) * sizeof(*c));
    if (!c)
        return 0;
    p->chunks = c;
    c = &p->chunks[p->count];
    memset(c, 0, sizeof(*c));
    c->text = malloc(n + 1);
    if (!c->text)
        return 0;
    memcpy(c->text, t, n);
    c->text[n] = 0;
    char *prepared = NULL;
    size_t pn = 0;
    if (!phonon_prepare_text_prompt(t, n, &prepared, &pn, &c->frames_after_eos) ||
        !phonon_tokenizer_encode(prepared, pn, &c->tokens)) {
        goto fail;
    }
    free(prepared);
    prepared = NULL;
    double frames = ceil(((double)c->tokens.length / 3.0 + 2.0) * rate);
    if (!isfinite(frames) || frames < 0 || frames > PHONON_PLAN_MAX_FRAMES)
        goto fail;
    c->frame_budget = (size_t)frames;
    if (c->tokens.length > SIZE_MAX - PHONON_PLAN_PROMPT_HEADROOM - c->frame_budget)
        goto fail;
    c->seq_budget = c->tokens.length + PHONON_PLAN_PROMPT_HEADROOM + c->frame_budget;
    if (c->seq_budget > PHONON_PLAN_MAX_CONTEXT)
        goto fail;
    p->count++;
    return 1;
fail:
    free(prepared);
    free(c->text);
    phonon_token_ids_free(&c->tokens);
    memset(c, 0, sizeof(*c));
    return 0;
}
typedef struct {
    size_t start, end, tokens;
    char end_mark;
} Word;
static char ending_punct(const char *text, size_t start, size_t end) {
    static const char *const closers[] = {"\"", "'", ")", "]", "”", "’", "»"};
    for (;;) {
        int removed = 0;
        for (size_t i = 0; i < sizeof(closers) / sizeof(closers[0]); i++) {
            size_t n = strlen(closers[i]);
            if (end - start >= n && !memcmp(text + end - n, closers[i], n)) {
                end -= n;
                removed = 1;
                break;
            }
        }
        if (!removed)
            break;
    }
    return end > start ? text[end - 1] : 0;
}
static int add_fitted(PhononPromptPlan *out, const char *text, size_t length, double rate,
                      size_t max_tokens, size_t word_limit, size_t depth) {
    if (depth > 64 || max_tokens == 0 || word_limit < max_tokens)
        return 0;
    char *prepared = NULL;
    size_t prepared_len = 0, tail = 0;
    if (!phonon_prepare_text_prompt(text, length, &prepared, &prepared_len, &tail))
        return 0;
    PhononTokenIds ids = {0};
    if (!phonon_tokenizer_encode(prepared, prepared_len, &ids)) {
        free(prepared);
        return 0;
    }
    size_t total = ids.length;
    phonon_token_ids_free(&ids);
    free(prepared);
    if (total <= max_tokens)
        return add_chunk(out, text, length, rate);
    Word *words = calloc(length + 1, sizeof(*words));
    if (!words)
        return 0;
    size_t count = 0, i = 0, k;
    while (i < length) {
        if (ws((const unsigned char *)text + i, length - i, &k)) {
            i += k;
            continue;
        }
        size_t start = i;
        while (i < length && !ws((const unsigned char *)text + i, length - i, &k))
            i += k;
        words[count].start = start;
        words[count].end = i;
        PhononTokenIds wi = {0};
        if (!phonon_tokenizer_encode(text + start, i - start, &wi)) {
            free(words);
            return 0;
        }
        words[count].tokens = wi.length;
        phonon_token_ids_free(&wi);
        words[count].end_mark = ending_punct(text, start, i);
        count++;
    }
    if (count >= 2) {
        size_t cumulative = 0;
        for (size_t j = 0; j < count; j++)
            cumulative += words[j].tokens;
        size_t min_side = (max_tokens + 3) / 4;
        int chosen = -1, chosen_kind = 3;
        size_t best_diff = SIZE_MAX;
        for (int kind = 0; kind < 3 && chosen < 0; kind++) {
            best_diff = SIZE_MAX;
            for (size_t j = 0, left = 0; j + 1 < count; j++) {
                left += words[j].tokens;
                size_t right = cumulative - left;
                if (kind < 2 && (left < min_side || right < min_side))
                    continue;
                char c = words[j].end_mark;
                int matches = kind == 0   ? (c == '.' || c == '!' || c == '?')
                              : kind == 1 ? (c == ',' || c == ';' || c == ':')
                                          : 1;
                if (matches) {
                    size_t diff = left > right ? left - right : right - left;
                    if (diff < best_diff) {
                        best_diff = diff;
                        chosen = (int)j;
                        chosen_kind = kind;
                    }
                }
            }
        }
        (void)chosen_kind;
        if (chosen < 0)
            chosen = (int)(count / 2) - 1;
        if (chosen < 0)
            chosen = 0;
        size_t cut = words[chosen].end;
        size_t right = words[chosen + 1].start;
        char *a = malloc(cut + 1), *b = malloc(length - right + 1);
        free(words);
        if (!a || !b) {
            free(a);
            free(b);
            return 0;
        }
        memcpy(a, text, cut);
        a[cut] = 0;
        memcpy(b, text + right, length - right);
        b[length - right] = 0;
        int ok = add_fitted(out, a, cut, rate, max_tokens, word_limit, depth + 1) &&
                 add_fitted(out, b, length - right, rate, max_tokens, word_limit, depth + 1);
        free(a);
        free(b);
        return ok;
    }
    free(words);
    if (count == 1 && total <= word_limit)
        return add_chunk(out, text, length, rate);
    if (count == 1) {
        size_t chars = 0;
        for (i = 0; i < length;) {
            size_t width;
            unsigned int cp;
            if (!scalar((const unsigned char *)text + i, length - i, &cp, &width))
                return 0;
            i += width;
            chars++;
        }
        if (chars <= 1)
            return 0;
        size_t target = chars / 2;
        i = 0;
        for (size_t j = 0; j < target; j++) {
            size_t width;
            unsigned int cp;
            if (!scalar((const unsigned char *)text + i, length - i, &cp, &width))
                return 0;
            i += width;
        }
        return add_fitted(out, text, i, rate, max_tokens, word_limit, depth + 1) &&
               add_fitted(out, text + i, length - i, rate, max_tokens, word_limit, depth + 1);
    }
    return 0;
}
static int group_reserve(char **text, size_t *capacity, size_t need) {
    if (need > PHONON_TOKENIZER_MAX_INPUT_BYTES + 1)
        return 0;
    if (need <= *capacity)
        return 1;
    size_t cap = *capacity ? *capacity : 64;
    while (cap < need) {
        if (cap > (PHONON_TOKENIZER_MAX_INPUT_BYTES + 1) / 2) {
            cap = PHONON_TOKENIZER_MAX_INPUT_BYTES + 1;
            break;
        }
        cap *= 2;
    }
    char *next = realloc(*text, cap);
    if (!next)
        return 0;
    *text = next;
    *capacity = cap;
    return 1;
}
static size_t group_trim(char *text, size_t length) {
    size_t start = 0, width;
    while (start < length && ws((const unsigned char *)text + start, length - start, &width))
        start += width;
    size_t end = length;
    while (end > start) {
        size_t pos = end - 1;
        while (pos > start && ((unsigned char)text[pos] & 0xc0) == 0x80)
            pos--;
        if (ws((const unsigned char *)text + pos, end - pos, &width) && pos + width == end)
            end = pos;
        else
            break;
    }
    size_t kept = end - start;
    if (start)
        memmove(text, text + start, kept);
    text[kept] = 0;
    return kept;
}
int phonon_prompt_plan(const char *text, size_t n, size_t max, double rate, PhononPromptPlan *out) {
    if (!out)
        return 0;
    out->chunks = NULL;
    out->count = 0;
    if (!text || !n || !max || max > PHONON_PLAN_MAX_FIT_TOKENS || !isfinite(rate) || rate <= 0 ||
        rate > 100)
        return 0;
    char *prepared = NULL;
    char *group = NULL;
    size_t group_cap = 0;
    size_t pn = 0, tail = 0;
    if (!phonon_prepare_text_prompt(text, n, &prepared, &pn, &tail))
        return 0;
    if (!pn) {
        free(prepared);
        return 0;
    }
    PhononTokenIds ids = {0}, markers = {0};
    if (!phonon_tokenizer_encode(prepared, pn, &ids) ||
        !phonon_tokenizer_encode(".!...?", 6, &markers))
        goto fail;
    size_t *bounds = malloc((ids.length + 2) * sizeof(*bounds));
    if (!bounds)
        goto fail;
    size_t nb = 0;
    bounds[nb++] = 0;
    int previous_eos = 0;
    for (size_t i = 0; i < ids.length; i++) {
        int eos = 0;
        for (size_t j = markers.length > 1 ? 1 : 0; j < markers.length; j++)
            if (ids.ids[i] == markers.ids[j]) {
                eos = 1;
                break;
            }
        if (eos)
            previous_eos = 1;
        else {
            if (previous_eos)
                bounds[nb++] = i;
            previous_eos = 0;
        }
    }
    bounds[nb++] = ids.length;
    size_t group_len = 0, group_tokens = 0;
    for (size_t w = 0; w + 1 < nb; w++) {
        size_t begin = bounds[w], end = bounds[w + 1];
        if (end <= begin)
            continue;
        char *sentence = NULL;
        size_t sentence_len = 0;
        if (!phonon_tokenizer_decode(ids.ids + begin, end - begin, &sentence, &sentence_len)) {
            free(bounds);
            goto fail;
        }
        if (group_len && group_tokens + end - begin > max) {
            group_len = group_trim(group, group_len);
            if (!add_fitted(out, group, group_len, rate, PHONON_PLAN_MAX_FIT_TOKENS,
                            PHONON_PLAN_MAX_FIT_TOKENS, 0)) {
                free(sentence);
                free(bounds);
                goto fail;
            }
            group_len = 0;
            group_tokens = 0;
        }
        size_t separator = group_len ? 1 : 0;
        if (sentence_len > SIZE_MAX - group_len - separator - 1 ||
            !group_reserve(&group, &group_cap, group_len + separator + sentence_len + 1)) {
            free(sentence);
            free(bounds);
            goto fail;
        }
        if (separator)
            group[group_len++] = ' ';
        memcpy(group + group_len, sentence, sentence_len);
        group_len += sentence_len;
        group[group_len] = 0;
        group_tokens += end - begin;
        free(sentence);
    }
    if (group_len) {
        group_len = group_trim(group, group_len);
        if (!add_fitted(out, group, group_len, rate, PHONON_PLAN_MAX_FIT_TOKENS,
                        PHONON_PLAN_MAX_FIT_TOKENS, 0)) {
            free(bounds);
            goto fail;
        }
    }
    free(group);
    free(bounds);
    free(prepared);
    phonon_token_ids_free(&ids);
    phonon_token_ids_free(&markers);
    return out->count != 0;
fail:
    free(group);
    free(prepared);
    phonon_token_ids_free(&ids);
    phonon_token_ids_free(&markers);
    phonon_prompt_plan_free(out);
    return 0;
}
int phonon_prompt_fit(PhononPromptPlan *plan, size_t max_tokens, double frame_rate) {
    if (!plan || !plan->chunks || plan->count == 0 || max_tokens == 0 ||
        max_tokens > PHONON_PLAN_MAX_FIT_TOKENS || !isfinite(frame_rate) || frame_rate <= 0 ||
        frame_rate > 100)
        return 0;
    PhononPromptPlan fitted = {0};
    for (size_t i = 0; i < plan->count; i++) {
        const char *text = plan->chunks[i].text;
        if (!add_fitted(&fitted, text, strlen(text), frame_rate, max_tokens, max_tokens, 0)) {
            phonon_prompt_plan_free(&fitted);
            return 0;
        }
    }
    phonon_prompt_plan_free(plan);
    *plan = fitted;
    return 1;
}

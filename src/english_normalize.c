#include "english_normalize.h"

#include "speech_error_internal.h"
#include "utf8.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#ifdef PHONON_ENGLISH_NORMALIZE_TEST_ALLOCATOR
static _Thread_local long allocation_failure_pending = -1;
static _Thread_local long allocation_failure_active = -1;
static _Thread_local bool allocation_failure_triggered;
void phonon_english_normalize_test_fail_after(long allocation_index) {
    allocation_failure_pending = allocation_index;
}
bool phonon_english_normalize_test_failure_triggered(void) { return allocation_failure_triggered; }
static bool fail_test_allocation(void) {
    if (allocation_failure_active < 0)
        return false;
    if (allocation_failure_active-- == 0) {
        allocation_failure_active = -1;
        allocation_failure_triggered = true;
        errno = ENOMEM;
        return true;
    }
    return false;
}
#endif

static void *normalizer_malloc(size_t size) {
#ifdef PHONON_ENGLISH_NORMALIZE_TEST_ALLOCATOR
    if (fail_test_allocation())
        return NULL;
#endif
    return malloc(size);
}
static void *normalizer_calloc(size_t count, size_t size) {
#ifdef PHONON_ENGLISH_NORMALIZE_TEST_ALLOCATOR
    if (fail_test_allocation())
        return NULL;
#endif
    return calloc(count, size);
}
static void *normalizer_realloc(void *pointer, size_t size) {
#ifdef PHONON_ENGLISH_NORMALIZE_TEST_ALLOCATOR
    if (fail_test_allocation())
        return NULL;
#endif
    return realloc(pointer, size);
}
#define malloc normalizer_malloc
#define calloc normalizer_calloc
#define realloc normalizer_realloc

static bool ascii_alpha(unsigned char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}
static bool ascii_digit(unsigned char c) { return c >= '0' && c <= '9'; }
static bool ascii_upper(unsigned char c) { return c >= 'A' && c <= 'Z'; }
static bool ascii_lower(unsigned char c) { return c >= 'a' && c <= 'z'; }
static unsigned char ascii_to_upper(unsigned char c) {
    return ascii_lower(c) ? (unsigned char)(c - ('a' - 'A')) : c;
}
typedef struct {
    uint32_t *items;
    size_t length;
    size_t capacity;
} Codepoints;

static bool reserve(Codepoints *buffer, size_t additional) {
    if (additional > SIZE_MAX - buffer->length)
        return false;
    size_t required = buffer->length + additional;
    if (required <= buffer->capacity)
        return true;
    size_t capacity = buffer->capacity == 0 ? 64 : buffer->capacity;
    while (capacity < required) {
        if (capacity > SIZE_MAX / 2) {
            capacity = required;
            break;
        }
        capacity *= 2;
    }
    if (capacity > SIZE_MAX / sizeof(*buffer->items))
        return false;
    uint32_t *items = realloc(buffer->items, capacity * sizeof(*items));
    if (items == NULL)
        return false;
    buffer->items = items;
    buffer->capacity = capacity;
    return true;
}

static bool push(Codepoints *buffer, uint32_t codepoint) {
    if (!reserve(buffer, 1))
        return false;
    buffer->items[buffer->length++] = codepoint;
    return true;
}

static bool ascii_punctuation(uint32_t c) {
    return (c >= '!' && c <= '/') || (c >= ':' && c <= '@') || (c >= '[' && c <= '`') ||
           (c >= '{' && c <= '~');
}

static bool whitespace(uint32_t c) {
    return c == ' ' || (c >= '\t' && c <= '\r') || c == 0x85 || c == 0xa0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200a) || c == 0x2028 || c == 0x2029 || c == 0x202f ||
           c == 0x205f || c == 0x3000;
}

static bool emoji(uint32_t c) {
    return (c >= 0x1f000 && c <= 0x1faff) || (c >= 0x2600 && c <= 0x27bf) ||
           (c >= 0x2b00 && c <= 0x2bff) || c == 0xfe0f || c == 0x200d;
}

static uint32_t quote_fold(uint32_t c) {
    switch (c) {
    case 0xbb:
    case 0xab:
    case 0x2039:
    case 0x203a:
    case 0x201c:
    case 0x201d:
    case 0x201e:
    case 0x201f:
    case 0x2033:
    case 0x2036:
    case 0x2e42:
    case 0x275d:
    case 0x275e:
    case 0x2760:
    case 0x301d:
    case 0x301e:
    case 0x301f:
    case 0xff02:
        return '"';
    case 0x2018:
    case 0x2019:
    case 0x201a:
    case 0x201b:
    case 0x2032:
    case 0x2035:
    case 0x275b:
    case 0x275c:
    case 0x2761:
    case 0xff07:
    case 0x02bc:
    case 0x02bb:
    case 0x02b9:
    case 0x00b4:
    case 0x0060:
        return '\'';
    default:
        return c;
    }
}

static bool sink_push(Codepoints *out, uint32_t c) {
    if (c == '.' || c == ',') {
        while (out->length > 0) {
            uint32_t last = out->items[out->length - 1];
            bool keep = last == '"' || last == '\'' || last == '@' || last == '+' || last == '%' ||
                        last == '#' || last == '&' ||
                        (last == '$' && out->length >= 2 && out->items[out->length - 2] >= '0' &&
                         out->items[out->length - 2] <= '9');
            if (!(whitespace(last) || (ascii_punctuation(last) && !keep)))
                break;
            out->length--;
        }
    }
    return push(out, c);
}

static bool sink_space(Codepoints *out) {
    return out->length == 0 || whitespace(out->items[out->length - 1]) || push(out, ' ');
}

static bool sink_spoken(Codepoints *out, const char *word) {
    if (!sink_space(out))
        return false;
    for (const unsigned char *p = (const unsigned char *)word; *p != 0; p++)
        if (!sink_push(out, *p))
            return false;
    return sink_space(out);
}

static bool encode_utf8(const Codepoints *input, char **output, size_t *length) {
    if (input->length > (SIZE_MAX - 1) / 4)
        return false;
    size_t capacity = input->length * 4 + 1;
    char *text = malloc(capacity);
    if (text == NULL)
        return false;
    size_t n = 0;
    for (size_t i = 0; i < input->length; i++) {
        uint32_t c = input->items[i];
        if (c < 0x80)
            text[n++] = (char)c;
        else if (c < 0x800) {
            text[n++] = (char)(0xc0 | (c >> 6));
            text[n++] = (char)(0x80 | (c & 63));
        } else if (c < 0x10000) {
            text[n++] = (char)(0xe0 | (c >> 12));
            text[n++] = (char)(0x80 | ((c >> 6) & 63));
            text[n++] = (char)(0x80 | (c & 63));
        } else {
            text[n++] = (char)(0xf0 | (c >> 18));
            text[n++] = (char)(0x80 | ((c >> 12) & 63));
            text[n++] = (char)(0x80 | ((c >> 6) & 63));
            text[n++] = (char)(0x80 | (c & 63));
        }
    }
    text[n] = '\0';
    *output = text;
    *length = n;
    return true;
}

static const char *abbreviation(const char *word, size_t *matched) {
    static const struct {
        const char *short_form;
        const char *long_form;
    } entries[] = {{"Lt", "Lieutenant"},
                   {"Lt.", "Lieutenant"},
                   {"Col.", "Colonel"},
                   {"Sgt.", "Sergeant"},
                   {"Sgt", "Sergeant"},
                   {"Capt.", "Captain"},
                   {"Cpl.", "Corporal"},
                   {"Gen.", "General"},
                   {"Adm.", "Admiral"},
                   {"Sen.", "Senator"},
                   {"Sens.", "Senators"},
                   {"Rep.", "Representative"},
                   {"Reps.", "Representatives"},
                   {"Gov.", "Governor"},
                   {"Pres.", "President"},
                   {"Det.", "Detective"},
                   {"Insp.", "Inspector"},
                   {"Supt.", "Superintendent"},
                   {"Ft.", "Fort"},
                   {"Mt.", "Mount"},
                   {"Jan.", "January"},
                   {"Feb.", "February"},
                   {"Mar.", "March"},
                   {"Apr.", "April"},
                   {"Jun.", "June"},
                   {"Jul.", "July"},
                   {"Aug.", "August"},
                   {"Sep.", "September"},
                   {"Sept.", "September"},
                   {"Oct.", "October"},
                   {"Nov.", "November"},
                   {"Dec.", "December"},
                   {"Mon.", "Monday"},
                   {"Tue.", "Tuesday"},
                   {"Tues.", "Tuesday"},
                   {"Wed.", "Wednesday"},
                   {"Thu.", "Thursday"},
                   {"Thur.", "Thursday"},
                   {"Thurs.", "Thursday"},
                   {"Fri.", "Friday"},
                   {"Sat.", "Saturday"},
                   {"Sun.", "Sunday"},
                   {"vol.", "volume"},
                   {"Vol.", "Volume"},
                   {"ed.", "edition"},
                   {"approx.", "approximately"},
                   {"Approx.", "Approximately"},
                   {"dept.", "department"},
                   {"Dept.", "Department"},
                   {"vs.", "versus"},
                   {"w/", "with"},
                   {"W/", "With"},
                   {"w/o", "without"},
                   {"W/o", "Without"}};
    size_t n = strlen(word);
    const char *best = NULL;
    for (size_t i = 0; i < sizeof(entries) / sizeof(entries[0]); i++) {
        size_t m = strlen(entries[i].short_form);
        if (m > n || m <= *matched || memcmp(word, entries[i].short_form, m) != 0)
            continue;
        bool suffix = true;
        for (size_t j = m; j < n; j++)
            if (word[j] != '!' && word[j] != '?' && word[j] != '.' && word[j] != ',' &&
                word[j] != ':')
                suffix = false;
        if (suffix) {
            *matched = m;
            best = entries[i].long_form;
        }
    }
    return best;
}

static char *rewrite_abbreviation(const char *word) {
    size_t matched = 0;
    const char *long_form = abbreviation(word, &matched);
    if (long_form == NULL)
        return NULL;
    size_t n = strlen(word), m = strlen(long_form), tail = n - matched;
    if (m > SIZE_MAX - tail - 1)
        return NULL;
    char *out = malloc(m + tail + 1);
    if (out == NULL)
        return NULL;
    memcpy(out, long_form, m);
    memcpy(out + m, word + matched, tail + 1);
    return out;
}

static char *rewrite_elongation(const char *word) {
    size_t n = strlen(word), end = n;
    while (end && strchr("!?.,:;", word[end - 1]))
        end--;
    if (end < 3)
        return NULL;
    bool triple = false, all_upper = true;
    for (size_t i = 0; i < end; i++) {
        unsigned char c = (unsigned char)word[i];
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')))
            return NULL;
        if (c < 'A' || c > 'Z')
            all_upper = false;
        unsigned char low = c >= 'A' && c <= 'Z' ? c + ('a' - 'A') : c;
        if (i >= 2 && word[i - 1] == (char)c && word[i - 2] == (char)c && strchr("aeiouyhw", low))
            triple = true;
    }
    if (all_upper || !triple)
        return NULL;
    char *out = malloc(n + 1);
    if (out == NULL)
        return NULL;
    size_t o = 0;
    for (size_t i = 0; i < end; i++) {
        unsigned char c = (unsigned char)word[i];
        unsigned char low = c >= 'A' && c <= 'Z' ? c + ('a' - 'A') : c;
        bool long_run = false;
        size_t first = i > 2 ? i - 2 : 0, last = i + 3 < end ? i + 3 : end;
        for (size_t j = first; j + 2 < last; j++)
            if (word[j] == (char)c && word[j + 1] == (char)c && word[j + 2] == (char)c)
                long_run = true;
        bool repeated = i > 0 && word[i - 1] == (char)c && strchr("aeiouyhw", low) != NULL;
        if (!(repeated && long_run))
            out[o++] = (char)c;
    }
    memcpy(out + o, word + end, n - end + 1);
    return out;
}

static bool decimal_group(const char *s, size_t n);
static char *rewrite_number(const char *word);

static char *append_suffix_copy(const char *base, const char *suffix, size_t suffix_len) {
    size_t base_len = strlen(base);
    if (base_len > SIZE_MAX - suffix_len - 1)
        return NULL;
    char *out = malloc(base_len + suffix_len + 1);
    if (out == NULL)
        return NULL;
    memcpy(out, base, base_len);
    memcpy(out + base_len, suffix, suffix_len);
    out[base_len + suffix_len] = 0;
    return out;
}

static char *rewrite_currency(const char *word) {
    if (word[0] == '-') {
        size_t symbol_length =
            word[1] == '$' ? 1
            : ((unsigned char)word[1] == 0xe2 && (unsigned char)word[2] == 0x82 &&
               (unsigned char)word[3] == 0xac)
                ? 3
            : ((unsigned char)word[1] == 0xc2 &&
               ((unsigned char)word[2] == 0xa3 || (unsigned char)word[2] == 0xa5 ||
                (unsigned char)word[2] == 0xa2))
                ? 2
                : 0;
        if (symbol_length != 0) {
            size_t n = strlen(word);
            char *shifted = malloc(n + 1);
            if (shifted == NULL)
                return NULL;
            memcpy(shifted, word + 1, symbol_length);
            shifted[symbol_length] = '-';
            memcpy(shifted + symbol_length + 1, word + symbol_length + 1, n - symbol_length);
            char *spoken = rewrite_currency(shifted);
            free(shifted);
            return spoken;
        }
    }
    size_t n = strlen(word), end = n;
    while (end && strchr("!?.,:;", word[end - 1]))
        end--;
    if (end == 0)
        return NULL;
    int currency = 0;
    size_t amount_start = 0, amount_end = end;
    if (word[0] == '$') {
        currency = 1;
        amount_start = 1;
    } else if (word[end - 1] == '$') {
        currency = 1;
        amount_end = end - 1;
    } else if (end >= 3 && (unsigned char)word[0] == 0xe2 && (unsigned char)word[1] == 0x82 &&
               (unsigned char)word[2] == 0xac) {
        currency = 2;
        amount_start = 3;
    } else if (end >= 3 && (unsigned char)word[end - 3] == 0xe2 &&
               (unsigned char)word[end - 2] == 0x82 && (unsigned char)word[end - 1] == 0xac) {
        currency = 2;
        amount_end = end - 3;
    } else if (end >= 2 && (unsigned char)word[0] == 0xc2 && (unsigned char)word[1] == 0xa3) {
        currency = 3;
        amount_start = 2;
    } else if (end >= 2 && (unsigned char)word[end - 2] == 0xc2 &&
               (unsigned char)word[end - 1] == 0xa3) {
        currency = 3;
        amount_end = end - 2;
    } else if (end >= 2 && (unsigned char)word[0] == 0xc2 && (unsigned char)word[1] == 0xa5) {
        currency = 4;
        amount_start = 2;
    } else if (end >= 2 && (unsigned char)word[end - 2] == 0xc2 &&
               (unsigned char)word[end - 1] == 0xa5) {
        currency = 4;
        amount_end = end - 2;
    } else if (end >= 2 && (unsigned char)word[end - 2] == 0xc2 &&
               (unsigned char)word[end - 1] == 0xa2) {
        currency = 5;
        amount_end = end - 2;
    } else if (end >= 2 && (unsigned char)word[0] == 0xc2 && (unsigned char)word[1] == 0xa2) {
        currency = 5;
        amount_start = 2;
    }
    if (currency == 0 || amount_start >= amount_end)
        return NULL;
    size_t amount_len = amount_end - amount_start;
    if (amount_len == SIZE_MAX)
        return NULL;
    char *amount = malloc(amount_len + 1);
    if (amount == NULL)
        return NULL;
    memcpy(amount, word + amount_start, amount_len);
    amount[amount_len] = 0;
    const char *name = currency == 1   ? "dollars"
                       : currency == 2 ? "euros"
                       : currency == 3 ? "pounds"
                       : currency == 4 ? "yen"
                                       : "cents";
    size_t tail_len = n - end;
    if ((amount[0] == '0' && amount[1] >= '0' && amount[1] <= '9') ||
        (strchr(amount, '-') != NULL && amount[0] != '-')) {
        free(amount);
        return NULL;
    }
    static const struct {
        const char *short_form;
        const char *scale;
    } scales[] = {{"thousand", "thousand"}, {"million", "million"}, {"billion", "billion"},
                  {"trillion", "trillion"}, {"bn", "billion"},      {"BN", "billion"},
                  {"mn", "million"},        {"tn", "trillion"},     {"k", "thousand"},
                  {"K", "thousand"},        {"m", "million"},       {"M", "million"},
                  {"b", "billion"},         {"B", "billion"}};
    if (currency != 5)
        for (size_t i = 0; i < sizeof(scales) / sizeof(scales[0]); i++) {
            size_t amount_n = strlen(amount), scale_n = strlen(scales[i].short_form);
            if (amount_n <= scale_n ||
                strcmp(amount + amount_n - scale_n, scales[i].short_form) != 0)
                continue;
            char *prefix = malloc(amount_n - scale_n + 1);
            if (prefix == NULL) {
                free(amount);
                return NULL;
            }
            memcpy(prefix, amount, amount_n - scale_n);
            prefix[amount_n - scale_n] = 0;
            char *number = rewrite_number(prefix);
            free(prefix);
            if (number == NULL) {
                free(amount);
                return NULL;
            }
            size_t number_length = strlen(number), scale_length = strlen(scales[i].scale),
                   name_length = strlen(name);
            if (number_length > SIZE_MAX - scale_length - name_length - 3) {
                free(number);
                free(amount);
                return NULL;
            }
            size_t base_length = number_length + scale_length + name_length + 2;
            char *base = malloc(base_length + 1);
            if (base == NULL) {
                free(number);
                free(amount);
                return NULL;
            }
            memcpy(base, number, number_length);
            base[number_length] = ' ';
            memcpy(base + number_length + 1, scales[i].scale, scale_length);
            base[number_length + 1 + scale_length] = ' ';
            memcpy(base + number_length + 2 + scale_length, name, name_length + 1);
            free(number);
            free(amount);
            char *result = append_suffix_copy(base, word + end, tail_len);
            free(base);
            return result;
        }
    char *decimal = strchr(amount, '.');
    if (decimal != NULL && strlen(decimal + 1) == 2 && decimal_group(decimal + 1, 2) &&
        currency != 5) {
        char cents[3] = {decimal[1], decimal[2], 0};
        *decimal = 0;
        char *number = rewrite_number(amount);
        if (number == NULL) {
            free(amount);
            return NULL;
        }
        const char *dollars = strcmp(amount, "1") == 0 ? (currency == 1   ? "dollar"
                                                          : currency == 3 ? "pound"
                                                                          : name)
                                                       : name;
        const char *cents_value = cents[0] == '0' ? cents + 1 : cents;
        size_t number_length = strlen(number), dollars_length = strlen(dollars),
               cents_length = strlen(cents_value);
        size_t base_length;
        if (strcmp(cents, "00") == 0) {
            if (number_length > SIZE_MAX - dollars_length - 2) {
                free(number);
                free(amount);
                return NULL;
            }
            base_length = number_length + dollars_length + 1;
        } else if (strcmp(amount, "0") == 0) {
            base_length = cents_length + 6;
        } else {
            if (number_length > SIZE_MAX - dollars_length - cents_length - 13) {
                free(number);
                free(amount);
                errno = ENOMEM;
                return NULL;
            }
            base_length = number_length + dollars_length + cents_length + 12;
        }
        char *base = malloc(base_length + 1);
        if (base == NULL) {
            free(number);
            free(amount);
            return NULL;
        }
        if (strcmp(cents, "00") == 0) {
            memcpy(base, number, number_length);
            base[number_length] = ' ';
            memcpy(base + number_length + 1, dollars, dollars_length + 1);
        } else if (strcmp(amount, "0") == 0) {
            memcpy(base, cents_value, cents_length);
            memcpy(base + cents_length, " cents", 7);
        } else {
            size_t at = 0;
            memcpy(base + at, number, number_length);
            at += number_length;
            base[at++] = ' ';
            memcpy(base + at, dollars, dollars_length);
            at += dollars_length;
            memcpy(base + at, " and ", 5);
            at += 5;
            memcpy(base + at, cents_value, cents_length);
            at += cents_length;
            memcpy(base + at, " cents", 7);
        }
        free(number);
        free(amount);
        char *result = append_suffix_copy(base, word + end, tail_len);
        free(base);
        return result;
    }
    if (amount[0] == '0' && amount[1] >= '0' && amount[1] <= '9') {
        free(amount);
        return NULL;
    }
    bool single_cent = currency == 5 && strcmp(amount, "1") == 0;
    char *number = rewrite_number(amount);
    free(amount);
    if (number == NULL)
        return NULL;
    size_t number_len = strlen(number), name_len = strlen(name);
    if (number_len > SIZE_MAX - name_len - 2) {
        free(number);
        return NULL;
    }
    char *base = malloc(number_len + name_len + 2);
    if (base == NULL) {
        free(number);
        return NULL;
    }
    memcpy(base, number, number_len);
    base[number_len] = ' ';
    memcpy(base + number_len + 1, name, name_len + 1);
    free(number);
    char *out = append_suffix_copy(base, word + end, tail_len);
    free(base);
    if (single_cent && out != NULL) {
        char *plural = strstr(out, " cents");
        if (plural != NULL)
            memmove(plural + 5, plural + 6, strlen(plural + 6) + 1);
    }
    return out;
}

static bool unicode_alphabetic_at(const char *text);
static bool decimal_group(const char *s, size_t n);

static bool decimal_group(const char *s, size_t n);

static char *rewrite_number(const char *word) {
    size_t n = strlen(word), end = n;
    while (end && strchr("!?.,:;", word[end - 1]))
        end--;
    if (end == 0)
        return NULL;
    char *dash = memchr(word, '-', end);
    if (dash != NULL && memchr(dash + 1, '-', (size_t)(word + end - dash - 1)) == NULL) {
        size_t left_n = (size_t)(dash - word), right_n = (size_t)(word + end - dash - 1);
        char left[16], right[16];
        if (left_n < sizeof(left) && right_n < sizeof(right)) {
            memcpy(left, word, left_n);
            left[left_n] = 0;
            memcpy(right, dash + 1, right_n);
            right[right_n] = 0;
            bool digits_ok = decimal_group(left, left_n) && decimal_group(right, right_n);
            bool small = left_n >= 2 && left_n <= 3 && right_n >= 2 && right_n <= 3 &&
                         left[0] != '0' && right[0] != '0' &&
                         strtoul(left, NULL, 10) < strtoul(right, NULL, 10);
            bool year = left_n == 4 && right_n == 4 &&
                        (left[0] == '1' || (left[0] == '2' && left[1] == '0')) &&
                        (right[0] == '1' || (right[0] == '2' && right[1] == '0')) &&
                        strcmp(left, right) < 0;
            bool year_short = left_n == 4 && right_n == 2 &&
                              (left[0] == '1' || (left[0] == '2' && left[1] == '0')) && digits_ok &&
                              strtoul(left + 2, NULL, 10) < strtoul(right, NULL, 10);
            if (digits_ok && (small || year || year_short)) {
                size_t tail_length = n - end;
                size_t left_length = strlen(left), right_length = strlen(right);
                if (left_length > SIZE_MAX - right_length - tail_length - 6)
                    return NULL;
                size_t range_length = left_length + 4 + right_length + tail_length;
                char *range = malloc(range_length + 1);
                if (range == NULL)
                    return NULL;
                size_t used = 0;
                memcpy(range + used, left, left_length);
                used += left_length;
                memcpy(range + used, " to ", 4);
                used += 4;
                memcpy(range + used, right, right_length);
                used += right_length;
                memcpy(range + used, word + end, tail_length);
                used += tail_length;
                range[used] = 0;
                return range;
            }
        }
        const char *tail = dash + 1;
        if (*tail && unicode_alphabetic_at(tail)) {
            char head[256];
            size_t head_n = left_n < sizeof(head) ? left_n : 0;
            if (head_n) {
                memcpy(head, word, head_n);
                head[head_n] = 0;
                char *spoken = rewrite_number(head);
                if (spoken != NULL) {
                    size_t spoken_n = strlen(spoken), rest_n = n - left_n;
                    char *joined = malloc(spoken_n + rest_n + 1);
                    if (joined) {
                        memcpy(joined, spoken, spoken_n);
                        memcpy(joined + spoken_n, word + left_n, rest_n + 1);
                    }
                    free(spoken);
                    return joined;
                }
            }
        }
    }
    size_t start = word[0] == '-' ? 1 : 0;
    if (start == end)
        return NULL;
    size_t dot = end, digits = 0, commas = 0;
    for (size_t i = start; i < end; i++) {
        if (word[i] >= '0' && word[i] <= '9')
            digits++;
        else if (word[i] == ',' && dot == end)
            commas++;
        else if (word[i] == '.' && dot == end)
            dot = i;
        else
            return NULL;
    }
    if (digits == 0)
        return NULL;
    size_t integer_end = dot;
    size_t integer_digits = 0;
    for (size_t i = start; i < integer_end; i++)
        if (word[i] >= '0' && word[i] <= '9')
            integer_digits++;
    if (dot < end && dot + 1 == end)
        return NULL;
    if (commas == 0 && dot == end && word[0] != '-' && integer_digits > 1 && word[start] == '0') {
        static const char *const names[] = {"zero", "one", "two",   "three", "four",
                                            "five", "six", "seven", "eight", "nine"};
        if (integer_digits > (SIZE_MAX - 1) / 6)
            return NULL;
        size_t cap = integer_digits * 6 + (n - end) + 1;
        char *spelled = malloc(cap);
        if (spelled == NULL)
            return NULL;
        size_t out_n = 0;
        for (size_t i = start; i < integer_end; i++) {
            if (i != start)
                spelled[out_n++] = ' ';
            const char *name = names[word[i] - '0'];
            size_t len = strlen(name);
            memcpy(spelled + out_n, name, len);
            out_n += len;
        }
        memcpy(spelled + out_n, word + end, n - end);
        out_n += n - end;
        spelled[out_n] = 0;
        return spelled;
    }
    if (commas) {
        for (size_t i = start; i < integer_end; i++)
            if (word[i] != ',' && (word[i] < '0' || word[i] > '9'))
                return NULL;
    } else {
        for (size_t i = start; i < integer_end; i++)
            if (word[i] < '0' || word[i] > '9')
                return NULL;
    }
    if (integer_digits > 12)
        return NULL;
    unsigned long long value = 0;
    for (size_t i = start; i < integer_end; i++)
        if (word[i] != ',')
            value = value * 10 + (unsigned)(word[i] - '0');
    if (n > SIZE_MAX - 256)
        return NULL;
    size_t result_capacity = n + 256;
    char *result = calloc(result_capacity, 1);
    if (result == NULL)
        return NULL;
    size_t used = 0;
    if (word[0] == '-') {
        memcpy(result, "minus ", 6);
        used = 6;
    }
    if (value < 1000 || (value >= 1900 && value < 2100)) {
        int wrote = snprintf(result + used, result_capacity - used, "%llu", value);
        if (wrote < 0) {
            free(result);
            return NULL;
        }
        used += (size_t)wrote;
    } else {
        unsigned long long billions = value / 1000000000ULL;
        unsigned long long millions = (value / 1000000ULL) % 1000;
        unsigned long long thousands = (value / 1000ULL) % 1000;
        unsigned long long rest = value % 1000;
        unsigned long long groups[] = {billions, millions, thousands};
        const char *scales[] = {"billion", "million", "thousand"};
        for (size_t i = 0; i < 3; i++)
            if (groups[i]) {
                int wrote =
                    snprintf(result + used, result_capacity - used, "%s%llu %s",
                             used && result[used - 1] != ' ' ? " " : "", groups[i], scales[i]);
                if (wrote < 0 || (size_t)wrote >= result_capacity - used) {
                    free(result);
                    return NULL;
                }
                used += (size_t)wrote;
            }
        if (rest) {
            int wrote =
                snprintf(result + used, result_capacity - used, "%s%llu", used ? " " : "", rest);
            if (wrote < 0) {
                free(result);
                return NULL;
            }
            used += (size_t)wrote;
        }
    }
    if (dot < end) {
        size_t decimal_length = end - dot - 1;
        if (decimal_length > result_capacity - used - 8) {
            free(result);
            return NULL;
        }
        memcpy(result + used, " point ", 7);
        used += 7;
        memcpy(result + used, word + dot + 1, decimal_length);
        used += decimal_length;
    }
    size_t tail = n - end;
    if (used > SIZE_MAX - tail - 1) {
        free(result);
        return NULL;
    }
    char *out = malloc(used + tail + 1);
    if (out == NULL) {
        free(result);
        return NULL;
    }
    memcpy(out, result, used);
    memcpy(out + used, word + end, tail);
    out[used + tail] = 0;
    free(result);
    return out;
}

static bool decimal_group(const char *s, size_t n) {
    if (n == 0)
        return false;
    for (size_t i = 0; i < n; i++)
        if (s[i] < '0' || s[i] > '9')
            return false;
    return true;
}

static char *rewrite_dashed_digits(const char *word) {
    size_t n = strlen(word), end = n;
    while (end && strchr("!?.,:;", word[end - 1]))
        end--;
    size_t separators = 0;
    for (size_t i = 0; i < end; i++)
        if (word[i] == '-' || word[i] == '_')
            separators++;
    if (separators < 2 || separators == SIZE_MAX)
        return NULL;
    size_t groups_cap = separators + 1;
    if (groups_cap > SIZE_MAX / (3 * sizeof(size_t)) || n > (SIZE_MAX - 1) / 12)
        return NULL;
    size_t *storage = malloc(groups_cap * 3 * sizeof(size_t));
    if (storage == NULL)
        return NULL;
    size_t *starts = storage, *lengths = storage + groups_cap, *seps = storage + groups_cap * 2;
    size_t groups = 0, start = 0;
    for (size_t i = 0; i < end; i++) {
        if (word[i] != '-' && word[i] != '_')
            continue;
        if (i == start || !decimal_group(word + start, i - start)) {
            free(storage);
            return NULL;
        }
        starts[groups] = start;
        lengths[groups] = i - start;
        seps[groups++] = i;
        start = i + 1;
    }
    if (start == end || !decimal_group(word + start, end - start)) {
        free(storage);
        return NULL;
    }
    starts[groups] = start;
    lengths[groups] = end - start;
    seps[groups++] = end;
    if (separators == 2 && groups == 3 && word[seps[0]] == '-' && word[seps[1]] == '-') {
        size_t a = lengths[0], b = lengths[1], c = lengths[2];
        unsigned long x = strtoul(word + starts[1], NULL, 10),
                      y = strtoul(word + starts[2], NULL, 10);
        unsigned long day = x > y ? x : y, month = x > y ? y : x;
        bool ymd = a == 4 && b <= 2 && c <= 2 && day >= 1 && day <= 31 && month >= 1 && month <= 12;
        bool dmy = (c == 4 || c == 2) && a <= 2 && b <= 2 && x >= 1 && x <= 31 && y >= 1 && y <= 12;
        bool mdy = (c == 4 || c == 2) && a <= 2 && b <= 2 && y >= 1 && y <= 31 && x >= 1 && x <= 12;
        if (ymd || dmy || mdy) {
            free(storage);
            return NULL;
        }
    }
    size_t capacity = n * 12 + 1;
    char *out = malloc(capacity);
    if (out == NULL) {
        free(storage);
        return NULL;
    }
    size_t used = 0;
    for (size_t g = 0; g < groups; g++) {
        size_t len = lengths[g];
        if (len >= 4) {
            for (size_t j = 0; j < len; j++) {
                if (used)
                    out[used++] = ' ';
                out[used++] = word[starts[g] + j];
            }
        } else {
            if (used)
                out[used++] = ' ';
            memcpy(out + used, word + starts[g], len);
            used += len;
        }
        if (g + 1 < groups && word[seps[g]] == '_') {
            memcpy(out + used, " underscore", 11);
            used += 11;
        }
    }
    memcpy(out + used, word + end, n - end);
    used += n - end;
    out[used] = 0;
    free(storage);
    return out;
}

#include "english_alnum_ranges.inc"
#include "english_alpha_ranges.inc"
#include "english_folds.inc"
#include "english_tlds.inc"

static bool append_text(char *out, size_t capacity, size_t *used, const char *text) {
    size_t n = strlen(text);
    if (n > capacity - *used - 1)
        return false;
    memcpy(out + *used, text, n);
    *used += n;
    out[*used] = 0;
    return true;
}

static bool email_separator(char c) { return c == '-' || c == '_' || c == '.' || c == '+'; }

static bool is_tld(const char *s, size_t n) {
    char label[128];
    if (n == 0 || n >= sizeof(label))
        return false;
    for (size_t i = 0; i < n; i++)
        label[i] = (char)ascii_to_upper((unsigned char)s[i]);
    label[n] = 0;
    size_t low = 0, high = sizeof(ENGLISH_TLDS) / sizeof(ENGLISH_TLDS[0]);
    while (low < high) {
        size_t mid = low + (high - low) / 2;
        int cmp = strcmp(label, ENGLISH_TLDS[mid]);
        if (cmp == 0)
            return true;
        if (cmp < 0)
            high = mid;
        else
            low = mid + 1;
    }
    return false;
}

static bool unicode_alphanumeric(uint32_t codepoint) {
    size_t low = 0, high = sizeof(ENGLISH_ALNUM_RANGES) / sizeof(ENGLISH_ALNUM_RANGES[0]);
    while (low < high) {
        size_t mid = low + (high - low) / 2;
        if (codepoint < ENGLISH_ALNUM_RANGES[mid].first)
            high = mid;
        else if (codepoint > ENGLISH_ALNUM_RANGES[mid].last)
            low = mid + 1;
        else
            return true;
    }
    return false;
}

static bool unicode_alphabetic(uint32_t codepoint) {
    size_t low = 0, high = sizeof(ENGLISH_ALPHA_RANGES) / sizeof(ENGLISH_ALPHA_RANGES[0]);
    while (low < high) {
        size_t mid = low + (high - low) / 2;
        if (codepoint < ENGLISH_ALPHA_RANGES[mid].first)
            high = mid;
        else if (codepoint > ENGLISH_ALPHA_RANGES[mid].last)
            low = mid + 1;
        else
            return true;
    }
    return false;
}

static bool unicode_alphabetic_at(const char *text) {
    size_t offset = 0;
    uint32_t cp;
    return speech_utf8_decode_next((const unsigned char *)text, strlen(text), &offset, &cp) &&
           unicode_alphabetic(cp);
}

static bool url_alphanumeric(const char *text, size_t length, size_t *width) {
    size_t offset = 0;
    uint32_t cp;
    if (!speech_utf8_decode_next((const unsigned char *)text, length, &offset, &cp))
        return false;
    *width = offset;
    return unicode_alphanumeric(cp);
}

static bool speak_url_part(char *out, size_t cap, size_t *used, const char *part, size_t n) {
    size_t start = 0, i = 0;
    while (i <= n) {
        unsigned char c = i < n ? (unsigned char)part[i] : 0;
        bool separator = c == '-' || c == '/' || (c >= '0' && c <= '9') || i == n;
        if (i < n && !separator) {
            size_t off = 0;
            uint32_t cp;
            if (!speech_utf8_decode_next((const unsigned char *)part + i, n - i, &off, &cp))
                return false;
            i += off;
            continue;
        }
        if (i > start) {
            size_t len = i - start;
            if (len == 3 && strncasecmp(part + start, "www", 3) == 0) {
                if (!append_text(out, cap, used, " W-W-W"))
                    return false;
            } else if (len == 2 && ascii_alpha((unsigned char)part[start]) &&
                       ascii_alpha((unsigned char)part[start + 1])) {
                if (!append_text(out, cap, used, " "))
                    return false;
                char pair[4] = {(char)ascii_to_upper((unsigned char)part[start]), '-',
                                (char)ascii_to_upper((unsigned char)part[start + 1]), 0};
                if (!append_text(out, cap, used, pair))
                    return false;
            } else {
                if (!append_text(out, cap, used, " ") || len >= cap - *used - 1)
                    return false;
                memcpy(out + *used, part + start, len);
                *used += len;
                out[*used] = 0;
            }
        }
        if (i < n) {
            const char *spoken = c == '-' ? " dash" : c == '/' ? " slash" : NULL;
            if (spoken) {
                if (!append_text(out, cap, used, spoken))
                    return false;
            } else {
                if (!append_text(out, cap, used, " "))
                    return false;
                char d[2] = {(char)c, 0};
                if (!append_text(out, cap, used, d))
                    return false;
            }
            i++;
        }
        if (i == n)
            break;
        start = i;
    }
    return true;
}

static char *rewrite_url(const char *word);

static char *rewrite_email(const char *word) {
    const char *at = strchr(word, '@');
    if (at == NULL || strchr(at + 1, '@') != NULL)
        return NULL;
    if (at[1] == '.') {
        const char *domain = at + 1;
        while (*domain == '.')
            domain++;
        size_t domain_n = strlen(domain);
        if (domain_n > SIZE_MAX - 4)
            return NULL;
        char *url_input = malloc(domain_n + 4);
        if (url_input == NULL)
            return NULL;
        memcpy(url_input, "at.", 3);
        memcpy(url_input + 3, domain, domain_n + 1);
        char *url = rewrite_url(url_input);
        free(url_input);
        if (url == NULL)
            return NULL;
        size_t user_n = (size_t)(at - word), url_n = strlen(url);
        char *out = malloc(user_n + url_n + 2);
        if (out) {
            if (user_n) {
                memcpy(out, word, user_n);
                out[user_n] = ' ';
            }
            memcpy(out + user_n + (user_n ? 1 : 0), url, url_n + 1);
        }
        free(url);
        return out;
    }
    if (at == word)
        return NULL;
    size_t n = strlen(word), end = n;
    while (end && strchr("!?.,:;", word[end - 1]))
        end--;
    size_t domain_n = (size_t)(word + end - at - 1);
    if (domain_n < 3)
        return NULL;
    for (const char *p = word; p < at; p++)
        if (!(ascii_alpha((unsigned char)*p) || ascii_digit((unsigned char)*p) ||
              email_separator(*p)))
            return NULL;
    const char *domain = at + 1, *p = domain;
    size_t labels = 0;
    while (p < word + end) {
        const char *begin = p;
        while (p < word + end &&
               (ascii_alpha((unsigned char)*p) || ascii_digit((unsigned char)*p) || *p == '-'))
            p++;
        if (p == begin)
            return NULL;
        labels++;
        if (p == word + end)
            break;
        if (*p++ != '.' || p == word + end)
            return NULL;
    }
    if (labels < 2)
        return NULL;
    if (n > (SIZE_MAX - 64) / 16)
        return NULL;
    size_t cap = n * 16 + 64, used = 0;
    char *out = calloc(cap, 1);
    if (out == NULL)
        return NULL;
    p = word;
    while (p < at) {
        const char *sep = NULL;
        switch (*p) {
        case '-':
            sep = "dash";
            break;
        case '_':
            sep = "underscore";
            break;
        case '.':
            sep = "dot";
            break;
        case '+':
            sep = "plus";
            break;
        }
        if (sep) {
            if (used && !append_text(out, cap, &used, " "))
                goto fail;
            if (!append_text(out, cap, &used, sep))
                goto fail;
            p++;
        } else {
            const char *begin = p;
            while (p < at && !email_separator(*p))
                p++;
            if (used && !append_text(out, cap, &used, " "))
                goto fail;
            size_t len = (size_t)(p - begin);
            if (len >= cap - used - 1)
                goto fail;
            memcpy(out + used, begin, len);
            used += len;
            out[used] = 0;
        }
    }
    if (!append_text(out, cap, &used, " at"))
        goto fail;
    p = domain;
    size_t label_index = 0;
    while (p < word + end) {
        if (*p == '.') {
            if (!append_text(out, cap, &used, " dot"))
                goto fail;
            p++;
            label_index++;
            continue;
        }
        if (*p == '-') {
            if (!append_text(out, cap, &used, " dash"))
                goto fail;
            p++;
            continue;
        }
        const char *begin = p;
        while (p < word + end && *p != '.' && *p != '-')
            p++;
        size_t len = (size_t)(p - begin);
        bool tld = label_index == labels - 1 && len == 2 && ascii_alpha((unsigned char)begin[0]) &&
                   ascii_alpha((unsigned char)begin[1]);
        if (tld) {
            char pair[4] = {(char)ascii_to_upper((unsigned char)begin[0]), '-',
                            (char)ascii_to_upper((unsigned char)begin[1]), 0};
            if (!append_text(out, cap, &used, " ") || !append_text(out, cap, &used, pair))
                goto fail;
        } else {
            if (!append_text(out, cap, &used, " ") || len >= cap - used - 1)
                goto fail;
            memcpy(out + used, begin, len);
            used += len;
            out[used] = 0;
        }
    }
    if (!append_text(out, cap, &used, word + end))
        goto fail;
    return out;
fail:
    free(out);
    return NULL;
}

static char *rewrite_url(const char *word) {
    size_t n = strlen(word), end = n;
    while (end && strchr("!?.,:;", word[end - 1]))
        end--;
    const char *p = word;
    size_t cap;
    if (n > (SIZE_MAX - 128) / 16)
        return NULL;
    cap = n * 16 + 128;
    char *out = calloc(cap, 1);
    if (out == NULL)
        return NULL;
    size_t used = 0;
    bool scheme = false;
    if (strncmp(p, "https://", 8) == 0) {
        scheme = true;
        p += 8;
        if (!append_text(out, cap, &used, "H-T-T-P-S colon slash slash"))
            goto fail;
    } else if (strncmp(p, "http://", 7) == 0) {
        scheme = true;
        p += 7;
        if (!append_text(out, cap, &used, "H-T-T-P colon slash slash"))
            goto fail;
    }
    const char *domain_start = p;
    while (p < word + end) {
        size_t width = 0;
        if (*p == '-' || *p == '.') {
            p++;
            continue;
        }
        if (!url_alphanumeric(p, (size_t)(word + end - p), &width))
            break;
        p += width;
    }
    if (p == domain_start)
        goto fail;
    const char *domain_end = p;
    size_t count = 1;
    for (const char *q = domain_start; q < domain_end; q++)
        if (*q == '.')
            count++;
    if (count < 2 || domain_start[0] == '.' || domain_end[-1] == '.')
        goto fail;
    for (const char *q = domain_start + 1; q < domain_end; q++)
        if (q[-1] == '.' && *q == '.')
            goto fail;
    const char *tld = domain_end;
    while (tld > domain_start && tld[-1] != '.')
        tld--;
    size_t tld_length = (size_t)(domain_end - tld);
    if (!is_tld(tld, tld_length))
        goto fail;
    bool capital = false, lower = false;
    for (const char *q = tld; q < domain_end; q++) {
        capital |= ascii_upper((unsigned char)*q);
        lower |= ascii_lower((unsigned char)*q);
    }
    const char *first_end = domain_start;
    while (first_end < domain_end && *first_end != '.')
        first_end++;
    bool www = first_end - domain_start == 3 && strncasecmp(domain_start, "www", 3) == 0;
    if (capital && lower && !scheme && !www)
        goto fail;
    const char *label = domain_start;
    while (label < domain_end) {
        const char *label_end = label;
        while (label_end < domain_end && *label_end != '.')
            label_end++;
        if (label_end == label)
            goto fail;
        if (label != domain_start && !append_text(out, cap, &used, " dot"))
            goto fail;
        if (!speak_url_part(out, cap, &used, label, (size_t)(label_end - label)))
            goto fail;
        label = label_end + 1;
    }
    if (p < word + end) {
        if (*p != '/')
            goto fail;
        for (const char *path_char = p; path_char < word + end;) {
            if (*path_char == '-' || *path_char == '/') {
                path_char++;
                continue;
            }
            size_t width = 0;
            if (!url_alphanumeric(path_char, (size_t)(word + end - path_char), &width))
                goto fail;
            path_char += width;
        }
        if (!speak_url_part(out, cap, &used, p, (size_t)(word + end - p)))
            goto fail;
    }
    if (!append_text(out, cap, &used, word + end))
        goto fail;
    return out;
fail:
    free(out);
    return NULL;
}

static char *rewrite_word_english(const char *word, bool *allocation_failed);

static char *rewrite_slash_word(const char *word, bool *allocation_failed) {
    size_t n = strlen(word), end = n;
    while (end && ascii_punctuation((unsigned char)word[end - 1]) && word[end - 1] != '/')
        end--;
    if (memchr(word, '/', end) == NULL)
        return NULL;
    static const char *const units[] = {"km", "kg",  "mg", "ml",  "mL",   "dL",  "mol", "kJ",
                                        "kW", "kWh", "Hz", "min", "hr",   "hrs", "sec", "ft",
                                        "lb", "lbs", "oz", "mi",  "mph",  "kph", "yd",  "cm",
                                        "mm", "yr",  "wk", "mo",  "kcal", "rpm"};
    size_t parts = 0, start = 0;
    for (size_t i = 0; i <= end; i++)
        if (i == end || word[i] == '/') {
            if (i == start)
                return NULL;
            size_t off = start, count = 0;
            bool alpha = true;
            while (off < i) {
                size_t width = 0;
                if (!unicode_alphabetic_at(word + off)) {
                    alpha = false;
                    break;
                }
                uint32_t cp;
                if (!speech_utf8_decode_next((const unsigned char *)word + off, i - off, &width,
                                             &cp)) {
                    alpha = false;
                    break;
                }
                off += width;
                count++;
            }
            if (!alpha || count < 2)
                return NULL;
            size_t len = i - start;
            for (size_t u = 0; u < sizeof(units) / sizeof(units[0]); u++)
                if (strlen(units[u]) == len && memcmp(word + start, units[u], len) == 0)
                    return NULL;
            parts++;
            start = i + 1;
        }
    if (n > (SIZE_MAX - 64) / 16)
        return NULL;
    size_t cap = n * 16 + 64, used = 0;
    char *out = calloc(cap, 1);
    if (out == NULL) {
        *allocation_failed = true;
        return NULL;
    }
    start = 0;
    size_t index = 0;
    for (size_t i = 0; i <= end; i++)
        if (i == end || word[i] == '/') {
            if (index++ && !append_text(out, cap, &used, " "))
                goto fail;
            size_t len = i - start;
            char *part = malloc(len + 1);
            if (part == NULL) {
                *allocation_failed = true;
                goto fail;
            }
            memcpy(part, word + start, len);
            part[len] = 0;
            bool failed = false;
            errno = 0;
            char *rewritten = rewrite_word_english(part, &failed);
            if (failed || errno == ENOMEM) {
                *allocation_failed = true;
                free(part);
                free(rewritten);
                goto fail;
            }
            const char *piece = rewritten == NULL ? part : rewritten;
            if (!append_text(out, cap, &used, piece)) {
                free(part);
                free(rewritten);
                goto fail;
            }
            free(part);
            free(rewritten);
            start = i + 1;
        }
    if (!append_text(out, cap, &used, word + end))
        goto fail;
    (void)parts;
    return out;
fail:
    free(out);
    return NULL;
}

static char *rewrite_word_english(const char *word, bool *allocation_failed) {
    errno = 0;
    char *out = rewrite_number(word);
    if (out != NULL)
        return out;
    if (errno == ENOMEM) {
        *allocation_failed = true;
        return NULL;
    }
    errno = 0;
    out = rewrite_currency(word);
    if (out != NULL)
        return out;
    if (errno == ENOMEM) {
        *allocation_failed = true;
        return NULL;
    }
    errno = 0;
    out = rewrite_dashed_digits(word);
    if (out != NULL)
        return out;
    if (errno == ENOMEM) {
        *allocation_failed = true;
        return NULL;
    }
    errno = 0;
    out = rewrite_email(word);
    if (out != NULL)
        return out;
    if (errno == ENOMEM) {
        *allocation_failed = true;
        return NULL;
    }
    errno = 0;
    out = rewrite_url(word);
    if (out != NULL)
        return out;
    if (errno == ENOMEM) {
        *allocation_failed = true;
        return NULL;
    }
    errno = 0;
    out = rewrite_abbreviation(word);
    if (out != NULL)
        return out;
    if (errno == ENOMEM) {
        *allocation_failed = true;
        return NULL;
    }
    errno = 0;
    out = rewrite_elongation(word);
    if (out != NULL)
        return out;
    if (errno == ENOMEM) {
        *allocation_failed = true;
        return NULL;
    }
    errno = 0;
    out = rewrite_slash_word(word, allocation_failed);
    if (out != NULL)
        return out;
    if (*allocation_failed || errno == ENOMEM) {
        *allocation_failed = true;
        return NULL;
    }
    if (strchr(word, '@') != NULL || strchr(word, '+') != NULL) {
        size_t n = strlen(word);
        if (n > (SIZE_MAX - 32) / 16)
            return NULL;
        size_t cap = n * 16 + 32, used = 0;
        out = calloc(cap, 1);
        if (out == NULL) {
            *allocation_failed = true;
            return NULL;
        }
        const char *start = word;
        const char *end = word + n;
        while (end > word && strchr("!?.,:;", end[-1]) != NULL)
            end--;
        for (const char *p = word;; p++) {
            if (*p != '@' && *p != '+' && p != end)
                continue;
            size_t part_len = (size_t)(p - start);
            if (part_len) {
                char *part = malloc(part_len + 1);
                if (part == NULL) {
                    *allocation_failed = true;
                    free(out);
                    return NULL;
                }
                memcpy(part, start, part_len);
                part[part_len] = 0;
                bool part_failed = false;
                errno = 0;
                char *spoken = rewrite_word_english(part, &part_failed);
                if (part_failed || errno == ENOMEM) {
                    *allocation_failed = true;
                    free(part);
                    free(spoken);
                    free(out);
                    return NULL;
                }
                const char *piece = spoken == NULL ? part : spoken;
                if (used && !append_text(out, cap, &used, " ")) {
                    free(part);
                    free(spoken);
                    free(out);
                    return NULL;
                }
                if (!append_text(out, cap, &used, piece)) {
                    free(part);
                    free(spoken);
                    free(out);
                    return NULL;
                }
                free(part);
                free(spoken);
            }
            if (p < end && (*p == '@' || *p == '+')) {
                const char *spoken = *p == '@' ? "at" : "plus";
                if (used && !append_text(out, cap, &used, " ")) {
                    free(out);
                    return NULL;
                }
                if (!append_text(out, cap, &used, spoken)) {
                    free(out);
                    return NULL;
                }
                start = p + 1;
            } else
                break;
        }
        if (!append_text(out, cap, &used, end)) {
            free(out);
            return NULL;
        }
        return out;
    }
    return NULL;
}

static bool append_text(char *out, size_t capacity, size_t *used, const char *text);

static char *rewrite_digit_run(const char *input, size_t *consumed) {
    const char *p = input;
    bool plus = *p == '+';
    size_t token_count = 0, bytes = 0;
    bool years = true, leading_zero = false;
    for (;;) {
        if (p != input) {
            if (*p != ' ')
                break;
            p++;
        }
        const char *token = p;
        while (*p && *p != ' ')
            p++;
        const char *body_end = p;
        while (body_end > token && strchr("!?.,:;", body_end[-1]) != NULL)
            body_end--;
        const char *body = token;
        if (token_count == 0 && plus)
            body++;
        size_t length = (size_t)(body_end - body);
        if (length == 0 || length > 8)
            break;
        bool has_digit = false, valid = true;
        for (const char *c = body; c < body_end; c++) {
            if (ascii_digit((unsigned char)*c))
                has_digit = true;
            else if (!ascii_upper((unsigned char)*c))
                valid = false;
        }
        if (!valid || !has_digit)
            break;
        token_count++;
        bytes = (size_t)(p - input);
        years = years && length == 4 && (memcmp(body, "19", 2) == 0 || memcmp(body, "20", 2) == 0);
        leading_zero = leading_zero || *body == '0';
        if (body_end < p)
            break;
        if (*p == 0)
            break;
    }
    bool long_run = token_count >= 3 && !years;
    if (!(long_run || (token_count == 2 && (plus || leading_zero))))
        return NULL;
    if (bytes > SIZE_MAX / 6 - token_count - 16)
        return NULL;
    size_t capacity = bytes * 6 + token_count * 2 + 32;
    char *out = calloc(capacity, 1);
    if (out == NULL) {
        errno = ENOMEM;
        return NULL;
    }
    static const char *const digits[] = {"zero", "one", "two",   "three", "four",
                                         "five", "six", "seven", "eight", "nine"};
    p = input;
    size_t used = 0;
    if (plus) {
        if (!append_text(out, capacity, &used, "plus ")) {
            free(out);
            return NULL;
        }
    }
    for (size_t token_index = 0; token_index < token_count; token_index++) {
        if (token_index) {
            if (!append_text(out, capacity, &used, ", ")) {
                free(out);
                return NULL;
            }
        }
        if (token_index)
            p++;
        const char *token = p;
        while (*p && *p != ' ')
            p++;
        const char *body_end = p;
        while (body_end > token && strchr("!?.,:;", body_end[-1]) != NULL)
            body_end--;
        const char *body = token + ((token_index == 0 && plus) ? 1 : 0);
        for (const char *c = body; c < body_end; c++) {
            if (c > body && !append_text(out, capacity, &used, " ")) {
                free(out);
                return NULL;
            }
            const char *word = ascii_digit((unsigned char)*c) ? digits[*c - '0'] : NULL;
            if (word != NULL) {
                if (!append_text(out, capacity, &used, word)) {
                    free(out);
                    return NULL;
                }
            } else {
                char one[2] = {*c, 0};
                if (!append_text(out, capacity, &used, one)) {
                    free(out);
                    return NULL;
                }
            }
        }
        for (const char *suffix = body_end; suffix < p; suffix++) {
            char one[2] = {*suffix, 0};
            if (!append_text(out, capacity, &used, one)) {
                free(out);
                return NULL;
            }
        }
    }
    *consumed = bytes;
    return out;
}

static char *rewrite_currency_scale_span(const char *input, size_t *consumed) {
    if (!(input[0] == '$' || (unsigned char)input[0] == 0xc2 || (unsigned char)input[0] == 0xe2))
        return NULL;
    const char *first_end = strchr(input, ' ');
    if (first_end == NULL)
        return NULL;
    const char *second = first_end + 1;
    const char *second_end = strchr(second, ' ');
    size_t second_len = second_end == NULL ? strlen(second) : (size_t)(second_end - second);
    size_t first_len = (size_t)(first_end - input), body_len = second_len;
    while (body_len && strchr("!?.,:;", second[body_len - 1]))
        body_len--;
    static const char *const scales[] = {"thousand", "million", "billion", "trillion", "bn",
                                         "BN",       "mn",      "tn",      "k",        "K",
                                         "m",        "M",       "b",       "B"};
    bool scale = false;
    for (size_t i = 0; i < sizeof(scales) / sizeof(scales[0]); i++)
        if (strlen(scales[i]) == body_len && memcmp(second, scales[i], body_len) == 0)
            scale = true;
    if (!scale || first_len > SIZE_MAX - second_len - 1)
        return NULL;
    size_t combined_len = first_len + second_len;
    char *combined = malloc(combined_len + 1);
    if (combined == NULL) {
        errno = ENOMEM;
        return NULL;
    }
    memcpy(combined, input, first_len);
    memcpy(combined + first_len, second, second_len);
    combined[combined_len] = 0;
    char *spoken = rewrite_currency(combined);
    free(combined);
    if (spoken != NULL)
        *consumed = first_len + 1 + second_len;
    return spoken;
}

static bool ascii_word_equal(const char *word, size_t length, const char *expected) {
    size_t n = strlen(expected);
    if (length != n)
        return false;
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)word[i];
        if (c >= 'A' && c <= 'Z')
            c = (unsigned char)(c + ('a' - 'A'));
        if (c != (unsigned char)expected[i])
            return false;
    }
    return true;
}

static char *rewrite_ending_in(const char *input, const char *previous_item, size_t previous_length,
                               size_t *consumed) {
    if (!ascii_word_equal(previous_item, previous_length, "ending") &&
        !ascii_word_equal(previous_item, previous_length, "ends") &&
        !ascii_word_equal(previous_item, previous_length, "ended"))
        return NULL;
    const char *space = strchr(input, ' ');
    if (space == NULL)
        return NULL;
    size_t first = (size_t)(space - input);
    if (first != 2 || memcmp(input, "in", 2) != 0)
        return NULL;
    const char *code = space + 1;
    const char *finish = strchr(code, ' ');
    if (finish == NULL)
        finish = code + strlen(code);
    size_t n = (size_t)(finish - code), body = n;
    while (body && strchr("!?.,:;", code[body - 1]))
        body--;
    if (body < 2 || code[0] != '-')
        return NULL;
    bool digit = false;
    for (size_t i = 1; i < body; i++) {
        unsigned char c = (unsigned char)code[i];
        if (ascii_digit(c))
            digit = true;
        else if (!ascii_upper(c) && c != '-')
            return NULL;
    }
    if (!digit)
        return NULL;
    static const char *const names[] = {"zero", "one", "two",   "three", "four",
                                        "five", "six", "seven", "eight", "nine"};
    if (body > (SIZE_MAX - 64) / 6)
        return NULL;
    size_t cap = body * 6 + 64, used = 0;
    char *out = calloc(cap, 1);
    if (out == NULL) {
        errno = ENOMEM;
        return NULL;
    }
    if (!append_text(out, cap, &used, "in ")) {
        free(out);
        return NULL;
    }
    bool first_spoken = true;
    for (size_t i = 1; i < body; i++)
        if (code[i] != '-') {
            if (!first_spoken && !append_text(out, cap, &used, " ")) {
                free(out);
                return NULL;
            }
            const char *name = ascii_digit((unsigned char)code[i]) ? names[code[i] - '0'] : NULL;
            if (name) {
                if (!append_text(out, cap, &used, name)) {
                    free(out);
                    return NULL;
                }
            } else {
                char one[2] = {code[i], 0};
                if (!append_text(out, cap, &used, one)) {
                    free(out);
                    return NULL;
                }
            }
            first_spoken = false;
        }
    if (!append_text(out, cap, &used, code + body)) {
        free(out);
        return NULL;
    }
    *consumed = first + 1 + n;
    return out;
}

static bool rewrite_words(const char *input, char **output, size_t *length,
                          bool *allocation_failed) {
    size_t input_length = strlen(input);
    if (input_length > SIZE_MAX - 64)
        return false;
    size_t capacity = input_length + 64, used = 0;
    char *buffer = malloc(capacity);
    if (buffer == NULL)
        return false;
    const char *p = input;
    size_t previous_item_start = 0, previous_item_length = 0;
    bool previous_item_present = false;
    while (*p) {
        size_t scale_bytes = 0;
        errno = 0;
        char *scaled = rewrite_currency_scale_span(p, &scale_bytes);
        if (scaled != NULL) {
            size_t scaled_length = strlen(scaled);
            if (scaled_length > SIZE_MAX - used - 2) {
                free(scaled);
                free(buffer);
                return false;
            }
            if (used + scaled_length + 2 > capacity) {
                size_t next_capacity = capacity;
                while (next_capacity < used + scaled_length + 2) {
                    if (next_capacity > SIZE_MAX / 2) {
                        next_capacity = used + scaled_length + 2;
                        break;
                    }
                    next_capacity *= 2;
                }
                char *next = realloc(buffer, next_capacity);
                if (next == NULL) {
                    free(scaled);
                    free(buffer);
                    return false;
                }
                buffer = next;
                capacity = next_capacity;
            }
            memcpy(buffer + used, scaled, scaled_length);
            previous_item_start = used;
            previous_item_length = scaled_length;
            previous_item_present = true;
            used += scaled_length;
            free(scaled);
            p += scale_bytes;
            if (*p == ' ') {
                buffer[used++] = ' ';
                p++;
            }
            continue;
        }
        if (errno == ENOMEM) {
            free(buffer);
            *allocation_failed = true;
            return false;
        }
        size_t digit_bytes = 0;
        errno = 0;
        char *digit_run = rewrite_digit_run(p, &digit_bytes);
        if (digit_run != NULL) {
            size_t digit_length = strlen(digit_run);
            if (digit_length > SIZE_MAX - used - 2) {
                free(digit_run);
                free(buffer);
                return false;
            }
            if (used + digit_length + 2 > capacity) {
                size_t next_capacity = capacity;
                while (next_capacity < used + digit_length + 2) {
                    if (next_capacity > SIZE_MAX / 2) {
                        next_capacity = used + digit_length + 2;
                        break;
                    }
                    next_capacity *= 2;
                }
                char *next = realloc(buffer, next_capacity);
                if (next == NULL) {
                    free(digit_run);
                    free(buffer);
                    return false;
                }
                buffer = next;
                capacity = next_capacity;
            }
            memcpy(buffer + used, digit_run, digit_length);
            previous_item_start = used;
            previous_item_length = digit_length;
            previous_item_present = true;
            used += digit_length;
            free(digit_run);
            p += digit_bytes;
            if (*p == ' ') {
                buffer[used++] = ' ';
                p++;
            }
            continue;
        }
        if (errno == ENOMEM) {
            free(buffer);
            *allocation_failed = true;
            return false;
        }
        size_t ending_bytes = 0;
        errno = 0;
        char *ending =
            rewrite_ending_in(p, previous_item_present ? buffer + previous_item_start : "",
                              previous_item_length, &ending_bytes);
        if (ending != NULL) {
            size_t ending_length = strlen(ending);
            if (ending_length > SIZE_MAX - used - 2) {
                free(ending);
                free(buffer);
                return false;
            }
            if (used + ending_length + 2 > capacity) {
                size_t next_capacity = capacity;
                while (next_capacity < used + ending_length + 2) {
                    if (next_capacity > SIZE_MAX / 2) {
                        next_capacity = used + ending_length + 2;
                        break;
                    }
                    next_capacity *= 2;
                }
                char *next = realloc(buffer, next_capacity);
                if (next == NULL) {
                    free(ending);
                    free(buffer);
                    return false;
                }
                buffer = next;
                capacity = next_capacity;
            }
            if (used && buffer[used - 1] != ' ')
                buffer[used++] = ' ';
            previous_item_start = used;
            previous_item_length = ending_length;
            previous_item_present = true;
            memcpy(buffer + used, ending, ending_length);
            used += ending_length;
            free(ending);
            p += ending_bytes;
            if (*p == ' ') {
                buffer[used++] = ' ';
                p++;
            }
            continue;
        }
        if (errno == ENOMEM) {
            free(buffer);
            *allocation_failed = true;
            return false;
        }
        if ((p == input || p[-1] == ' ') && strncmp(p, "Lt ", 3) == 0 && p[3] != ' ') {
            size_t second_end = 3;
            while (p[second_end] != 0 && p[second_end] != ' ')
                second_end++;
            size_t body_end = second_end;
            while (body_end > 3 && (strchr("!?.:,;", p[body_end - 1]) != NULL ||
                                    (body_end - 3 >= 3 && (unsigned char)p[body_end - 3] == 0xe2 &&
                                     (unsigned char)p[body_end - 2] == 0x80 &&
                                     (unsigned char)p[body_end - 1] == 0xa6)))
                body_end -= body_end - 3 >= 3 && (unsigned char)p[body_end - 3] == 0xe2 ? 3 : 1;
            if (body_end - 3 != 3 || memcmp(p + 3, "Col", 3) != 0)
                goto no_rank_span;
            size_t expanded_length = 18 + second_end - body_end;
            char *expanded = malloc(expanded_length + 1);
            if (expanded == NULL) {
                *allocation_failed = true;
                free(buffer);
                return false;
            }
            memcpy(expanded, "Lieutenant Colonel", 18);
            memcpy(expanded + 18, p + body_end, second_end - body_end);
            expanded[expanded_length] = 0;
            if (used + expanded_length + 2 > capacity) {
                size_t next_capacity = used + expanded_length + 2;
                char *next = realloc(buffer, next_capacity);
                if (next == NULL) {
                    *allocation_failed = true;
                    free(expanded);
                    free(buffer);
                    return false;
                }
                buffer = next;
                capacity = next_capacity;
            }
            memcpy(buffer + used, expanded, expanded_length);
            previous_item_start = used;
            previous_item_length = expanded_length;
            previous_item_present = true;
            used += expanded_length;
            free(expanded);
            p += second_end;
            if (*p == ' ') {
                buffer[used++] = ' ';
                p++;
            }
            continue;
        }
    no_rank_span:;
        const char *start = p;
        while (*p && *p != ' ')
            p++;
        size_t n = (size_t)(p - start);
        if (n > SIZE_MAX - 1) {
            free(buffer);
            return false;
        }
        char *word = malloc(n + 1);
        if (word == NULL) {
            free(buffer);
            return false;
        }
        memcpy(word, start, n);
        word[n] = 0;
        char *rewritten = rewrite_word_english(word, allocation_failed);
        if (*allocation_failed) {
            free(word);
            free(rewritten);
            free(buffer);
            return false;
        }
        const char *piece = rewritten == NULL ? word : rewritten;
        size_t piece_n = strlen(piece), extra = piece_n + (*p ? 1 : 0);
        if (extra > SIZE_MAX - used) {
            free(word);
            free(rewritten);
            free(buffer);
            return false;
        }
        if (used + extra + 1 > capacity) {
            size_t next_capacity = capacity;
            while (next_capacity < used + extra + 1) {
                if (next_capacity > SIZE_MAX / 2) {
                    next_capacity = used + extra + 1;
                    break;
                }
                next_capacity *= 2;
            }
            char *next = realloc(buffer, next_capacity);
            if (next == NULL) {
                free(word);
                free(rewritten);
                free(buffer);
                return false;
            }
            buffer = next;
            capacity = next_capacity;
        }
        previous_item_start = used;
        previous_item_length = piece_n;
        previous_item_present = true;
        memcpy(buffer + used, piece, piece_n);
        used += piece_n;
        if (*p)
            buffer[used++] = ' ';
        free(word);
        free(rewritten);
        if (*p)
            p++;
    }
    size_t compact = 0;
    for (size_t i = 0; i < used; i++) {
        if (buffer[i] == ' ' && (compact == 0 || buffer[compact - 1] == ' '))
            continue;
        buffer[compact++] = buffer[i];
    }
    while (compact && buffer[compact - 1] == ' ')
        compact--;
    buffer[compact] = 0;
    *output = buffer;
    *length = compact;
    return true;
}

static uint32_t fold_unicode(uint32_t c) {
    size_t low = 0, high = sizeof(ENGLISH_FOLDS) / sizeof(ENGLISH_FOLDS[0]);
    while (low < high) {
        size_t mid = low + (high - low) / 2;
        if (ENGLISH_FOLDS[mid].from == c)
            return ENGLISH_FOLDS[mid].to;
        if (c < ENGLISH_FOLDS[mid].from)
            high = mid;
        else
            low = mid + 1;
    }
    return c;
}

static const char *greek_name(uint32_t c) {
    static const char *const names[] = {"alpha", "beta",    "gamma",   "delta", "epsilon", "zeta",
                                        "eta",   "theta",   "iota",    "kappa", "lambda",  "mu",
                                        "nu",    "xi",      "omicron", "pi",    "rho",     "sigma",
                                        "tau",   "upsilon", "phi",     "chi",   "psi",     "omega"};
    size_t i;
    if (c >= 0x03b1 && c <= 0x03c1)
        i = c - 0x03b1;
    else if (c == 0x03c2)
        i = 17;
    else if (c >= 0x03c3 && c <= 0x03c9)
        i = c - 0x03b1 - 1;
    else if (c >= 0x0391 && c <= 0x03a1)
        i = c - 0x0391;
    else if (c >= 0x03a3 && c <= 0x03a9)
        i = c - 0x0391 - 1;
    else
        return NULL;
    return names[i];
}

static const char *spoken_symbol(uint32_t c) {
    switch (c) {
    case 0x00d7:
        return "times";
    case 0x00f7:
        return "divided by";
    case 0x2264:
        return "less than or equal to";
    case 0x2265:
        return "greater than or equal to";
    case 0x2260:
        return "is not equal to";
    case 0x2248:
        return "approximately";
    case 0x221a:
        return "square root of";
    case 0x221e:
        return "infinity";
    case 0x2211:
        return "sigma";
    case 0x2030:
        return "per mille";
    default:
        return NULL;
    }
}

static uint32_t subscript(uint32_t c) {
    if (c >= 0x2080 && c <= 0x2089)
        return '0' + c - 0x2080;
    switch (c) {
    case 0x208a:
        return '+';
    case 0x208b:
        return '-';
    case 0x2090:
        return 'a';
    case 0x2091:
        return 'e';
    case 0x2092:
        return 'o';
    case 0x2093:
        return 'x';
    case 0x2095:
        return 'h';
    case 0x2096:
        return 'k';
    case 0x2097:
        return 'l';
    case 0x2098:
        return 'm';
    case 0x2099:
        return 'n';
    case 0x209a:
        return 'p';
    case 0x209b:
        return 's';
    case 0x209c:
        return 't';
    case 0x1d62:
        return 'i';
    case 0x2c7c:
        return 'j';
    default:
        return c;
    }
}

static uint32_t superscript(uint32_t c) {
    if (c == 0x2070)
        return '0';
    if (c >= 0x2074 && c <= 0x2079)
        return '4' + c - 0x2074;
    switch (c) {
    case 0x00b9:
        return '1';
    case 0x00b2:
        return '2';
    case 0x00b3:
        return '3';
    case 0x207b:
        return '-';
    case 0x207a:
        return '+';
    case 0x207f:
        return 'n';
    default:
        return c;
    }
}

SpeechIOStatus phonon_english_normalize(const char *language, const char *text, size_t text_length,
                                        char **normalized, size_t *normalized_length,
                                        SpeechError *error) {
#ifdef PHONON_ENGLISH_NORMALIZE_TEST_ALLOCATOR
    allocation_failure_active = allocation_failure_pending;
    allocation_failure_pending = -1;
    allocation_failure_triggered = false;
#endif
    if (normalized != NULL)
        *normalized = NULL;
    if (normalized_length != NULL)
        *normalized_length = 0;
    if (language == NULL || strcmp(language, "en") != 0 || text == NULL || normalized == NULL ||
        normalized_length == NULL || memchr(text, '\0', text_length) != NULL ||
        !speech_utf8_is_valid((const unsigned char *)text, text_length)) {
        speech_error_set(error, SPEECH_IO_INVALID_ARGUMENT,
                         "English normalization requires language en and valid non-NUL UTF-8 text");
#ifdef PHONON_ENGLISH_NORMALIZE_TEST_ALLOCATOR
        allocation_failure_active = -1;
#endif
        return SPEECH_IO_INVALID_ARGUMENT;
    }
    Codepoints output = {0};
    size_t offset = 0;
    uint32_t previous = UINT32_MAX;
    while (offset < text_length) {
        size_t codepoint_start = offset;
        uint32_t c;
        if (!speech_utf8_decode_next((const unsigned char *)text, text_length, &offset, &c))
            goto oom;
        c = quote_fold(c);
        c = fold_unicode(c);
        uint32_t sub = subscript(c);
        uint32_t sup = superscript(c);
        const char *greek = greek_name(c);
        const char *symbol = spoken_symbol(c);
        bool ok = true;
        if (c == 0x2605 || c == 0x2606) {
            size_t full = c == 0x2605 ? 1 : 0, empty = c == 0x2606 ? 1 : 0;
            while (offset < text_length) {
                size_t peek = offset;
                uint32_t next;
                if (!speech_utf8_decode_next((const unsigned char *)text, text_length, &peek,
                                             &next) ||
                    (next != 0x2605 && next != 0x2606))
                    break;
                if (next == 0x2605)
                    full++;
                else
                    empty++;
                offset = peek;
                c = next;
            }
            bool half = false;
            size_t peek = offset;
            uint32_t next;
            if (offset < text_length &&
                speech_utf8_decode_next((const unsigned char *)text, text_length, &peek, &next) &&
                next == 0x00bd) {
                half = true;
                offset = peek;
                c = next;
            }
            char rating[64];
            snprintf(rating, sizeof(rating), half ? "%zu.5" : "%zu", full);
            char spoken[128];
            if (empty)
                snprintf(spoken, sizeof(spoken), "%s out of %zu stars", rating,
                         full + empty + (half ? 1U : 0U));
            else
                snprintf(spoken, sizeof(spoken), "%s stars", rating);
            if (!sink_spoken(&output, spoken))
                goto oom;
            previous = c;
            continue;
        }
        if (sup != c) {
            size_t run_start = codepoint_start, run_end = offset, count = 1;
            char first = (char)sup;
            uint32_t last_source = c;
            while (run_end < text_length) {
                size_t peek = run_end;
                uint32_t next;
                if (!speech_utf8_decode_next((const unsigned char *)text, text_length, &peek,
                                             &next))
                    break;
                uint32_t mapped = superscript(next);
                if (mapped == next)
                    break;
                count++;
                last_source = next;
                run_end = peek;
            }
            bool attached =
                previous != UINT32_MAX && (unicode_alphanumeric(previous) || previous == ')');
            if (attached && count == 1 && first == '2')
                ok = sink_spoken(&output, "squared");
            else if (attached && count == 1 && first == '3')
                ok = sink_spoken(&output, "cubed");
            else {
                if (attached) {
                    ok = sink_spoken(&output, "to the power of");
                    if (first == '-')
                        ok = ok && sink_spoken(&output, "minus");
                }
                size_t scan = run_start;
                bool skip_minus = attached && first == '-';
                while (ok && scan < run_end) {
                    uint32_t codepoint;
                    size_t next_offset = scan;
                    if (!speech_utf8_decode_next((const unsigned char *)text, text_length,
                                                 &next_offset, &codepoint)) {
                        ok = false;
                        break;
                    }
                    uint32_t mapped = superscript(codepoint);
                    if (skip_minus)
                        skip_minus = false;
                    else
                        ok = sink_push(&output, mapped);
                    scan = next_offset;
                }
            }
            if (!ok)
                goto oom;
            offset = run_end;
            previous = last_source;
            continue;
        }
        switch (c) {
        case 0x2010:
        case 0x2011:
        case 0x2012:
        case 0x2015:
            ok = sink_push(&output, '-');
            break;
        case 0x2026:
            ok = sink_push(&output, '.');
            break;
        case '=':
            ok = sink_spoken(&output, "equals");
            break;
        case 0x00b1:
            ok = sink_spoken(&output, "plus or minus");
            break;
        case '~': {
            size_t peek = offset;
            uint32_t next;
            ok = speech_utf8_decode_next((const unsigned char *)text, text_length, &peek, &next) &&
                         next >= '0' && next <= '9'
                     ? sink_spoken(&output, "about")
                     : sink_push(&output, c);
            break;
        }
        case 0x03c0:
            ok = sink_spoken(&output, "pi");
            break;
        case 0x2665:
        case 0x2764:
            ok = sink_spoken(&output, "love");
            break;
        case '*': {
            size_t peek = offset;
            uint32_t next;
            bool before = previous == UINT32_MAX || whitespace(previous);
            bool after =
                speech_utf8_decode_next((const unsigned char *)text, text_length, &peek, &next) &&
                next >= '0' && next <= '9';
            ok = before && after ? sink_spoken(&output, "star") : sink_space(&output);
            break;
        }
        case 0x20ac: {
            size_t peek = offset;
            uint32_t next = 0;
            size_t i = output.length;
            while (i && whitespace(output.items[i - 1]))
                i--;
            bool has_next =
                speech_utf8_decode_next((const unsigned char *)text, text_length, &peek, &next);
            bool spaced_digit = i > 0 && output.items[i - 1] >= '0' && output.items[i - 1] <= '9' &&
                                (!has_next || !(next >= '0' && next <= '9'));
            if (spaced_digit) {
                while (output.length && whitespace(output.items[output.length - 1]))
                    output.length--;
                ok = sink_push(&output, c);
            } else
                ok = sink_push(&output, c);
            break;
        }
        case 0x00b0:
        case 0x00ba: {
            size_t end = output.length;
            if (end && output.items[end - 1] == '.')
                end--;
            bool own_n = false, capital_n = false;
            if (end && (output.items[end - 1] == 'n' || output.items[end - 1] == 'N')) {
                size_t n_at = end - 1;
                own_n = n_at == 0 || !unicode_alphanumeric(output.items[n_at - 1]);
                capital_n = output.items[n_at] == 'N';
                if (own_n) {
                    if (output.length && output.items[output.length - 1] == '.')
                        output.length--;
                    output.length--;
                }
            }
            if (own_n)
                ok = sink_spoken(&output, capital_n ? "Number" : "number");
            else if (c == 0x00ba)
                ok = sink_push(&output, c);
            else {
                bool one = output.length && output.items[output.length - 1] == '1' &&
                           (output.length < 2 || !((output.items[output.length - 2] >= '0' &&
                                                    output.items[output.length - 2] <= '9') ||
                                                   output.items[output.length - 2] == '.' ||
                                                   output.items[output.length - 2] == ','));
                ok = sink_spoken(&output, one ? "degree" : "degrees");
                size_t peek = offset;
                uint32_t unit = 0;
                bool has_unit =
                    speech_utf8_decode_next((const unsigned char *)text, text_length, &peek, &unit);
                if (has_unit && (unit == 'C' || unit == 'F')) {
                    size_t after = peek;
                    uint32_t next = 0;
                    bool has_after = speech_utf8_decode_next((const unsigned char *)text,
                                                             text_length, &after, &next);
                    if (!has_after || !unicode_alphanumeric(next)) {
                        offset = peek;
                        c = unit;
                        ok = ok && sink_spoken(&output, unit == 'C' ? "Celsius" : "Fahrenheit");
                    }
                }
            }
            break;
        }
        case '@':
        case '+':
            ok = sink_push(&output, c);
            break;
        case ':': {
            size_t look = offset;
            uint32_t next = 0;
            bool has_next =
                look < text_length &&
                speech_utf8_decode_next((const unsigned char *)text, text_length, &look, &next);
            if (previous != UINT32_MAX && !whitespace(previous) && has_next && !whitespace(next)) {
                ok = sink_push(&output, ':');
            } else {
                while (output.length && whitespace(output.items[output.length - 1]))
                    output.length--;
                ok = sink_push(&output, ',') && sink_space(&output);
            }
            break;
        }
        case ';':
        case '(':
        case ')':
            while (output.length && whitespace(output.items[output.length - 1]))
                output.length--;
            ok = sink_push(&output, ',') && sink_space(&output);
            break;
        case 0x2013:
        case 0x2014:
        case '[':
        case ']':
        case '{':
        case '}':
        case 0x2022:
        case 0x2023:
        case 0x25e6:
        case 0x00b7:
        case 0x2190:
        case 0x2191:
        case 0x2192:
        case 0x2193:
        case 0x27a1:
        case 0x279c:
            ok = sink_space(&output);
            break;
        default:
            if (greek != NULL)
                ok = sink_spoken(&output, greek);
            else if (symbol != NULL)
                ok = sink_spoken(&output, symbol);
            else if (sub != c)
                ok = sink_push(&output, sub);
            else if (emoji(c) || c < 0x20 || (c >= 0x7f && c <= 0x9f) || whitespace(c))
                ok = sink_space(&output);
            else
                ok = sink_push(&output, c);
            break;
        }
        if (!ok)
            goto oom;
        previous = c;
        (void)previous;
    }
    while (output.length && whitespace(output.items[output.length - 1]))
        output.length--;
    char *characters = NULL;
    size_t character_length = 0;
    if (!encode_utf8(&output, &characters, &character_length))
        goto oom;
    free(output.items);
    output.items = NULL;
    bool rewrite_oom = false;
    if (!rewrite_words(characters, normalized, normalized_length, &rewrite_oom)) {
        free(characters);
        goto oom;
    }
    free(characters);
    while (*normalized_length && (*normalized)[*normalized_length - 1] == '\"')
        (*normalized_length)--;
    if (*normalized_length >= 2 && (*normalized)[*normalized_length - 1] == '\'' &&
        strchr(".!?,;", (*normalized)[*normalized_length - 2]) != NULL)
        (*normalized_length)--;
    (*normalized)[*normalized_length] = 0;
    speech_error_set(error, SPEECH_IO_OK, NULL);
#ifdef PHONON_ENGLISH_NORMALIZE_TEST_ALLOCATOR
    allocation_failure_active = -1;
#endif
    return SPEECH_IO_OK;
oom:
    free(output.items);
    speech_error_set(error, SPEECH_IO_OUT_OF_MEMORY,
                     "cannot allocate English normalization output");
#ifdef PHONON_ENGLISH_NORMALIZE_TEST_ALLOCATOR
    allocation_failure_active = -1;
#endif
    return SPEECH_IO_OUT_OF_MEMORY;
}

void phonon_english_normalize_free(char *normalized) { free(normalized); }

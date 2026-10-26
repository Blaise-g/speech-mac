#include "english_normalize.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int expect(const char *input, const char *expected) {
    char *actual = NULL;
    size_t length = 0;
    SpeechError error;
    SpeechIOStatus status =
        phonon_english_normalize("en", input, strlen(input), &actual, &length, &error);
    if (status != SPEECH_IO_OK || length != strlen(expected) || strcmp(actual, expected) != 0) {
        fprintf(stderr, "normalize(%s): status=%s actual=%s expected=%s\n", input,
                speech_status_name(status), actual == NULL ? "<null>" : actual, expected);
        phonon_english_normalize_free(actual);
        return 1;
    }
    phonon_english_normalize_free(actual);
    return 0;
}

static int expect_long_inputs(void) {
    const size_t payload = 4096;
    const char prefix[] = "https://www.example.com/";
    size_t input_len = sizeof(prefix) - 1 + payload;
    char *input = malloc(input_len + 1);
    size_t expected_prefix_len =
        strlen("H-T-T-P-S colon slash slash W-W-W dot example dot com slash ");
    char *expected = malloc(expected_prefix_len + payload + 1);
    if (input == NULL || expected == NULL) {
        free(input);
        free(expected);
        return 1;
    }
    memcpy(input, prefix, sizeof(prefix) - 1);
    memset(input + sizeof(prefix) - 1, 'p', payload);
    input[input_len] = 0;
    memcpy(expected, "H-T-T-P-S colon slash slash W-W-W dot example dot com slash ",
           expected_prefix_len);
    memset(expected + expected_prefix_len, 'p', payload);
    expected[expected_prefix_len + payload] = 0;
    int failed = expect(input, expected);
    free(input);
    free(expected);

    input = malloc(payload + 1);
    if (input == NULL)
        return failed + 1;
    memset(input, 'x', payload);
    input[payload] = 0;
    failed += expect(input, input);
    free(input);

    input = malloc(payload + 3);
    expected = malloc(payload + 10);
    if (input == NULL || expected == NULL) {
        free(input);
        free(expected);
        return failed + 1;
    }
    input[0] = '1';
    input[1] = '.';
    memset(input + 2, '7', payload);
    input[payload + 2] = 0;
    memcpy(expected, "1 point ", 8);
    memset(expected + 8, '7', payload);
    expected[payload + 8] = 0;
    failed += expect(input, expected);
    free(input);
    free(expected);

    const size_t punctuation = 1000;
    input = malloc(punctuation + 6);
    expected = malloc(punctuation + 10);
    if (input == NULL || expected == NULL) {
        free(input);
        free(expected);
        return failed + 1;
    }
    memcpy(input, "20-25", 5);
    memset(input + 5, '!', punctuation);
    input[punctuation + 5] = 0;
    memcpy(expected, "20 to 25", 8);
    memset(expected + 8, '!', punctuation);
    expected[punctuation + 8] = 0;
    failed += expect(input, expected);
    free(input);
    free(expected);
    return failed;
}

int main(void) {
    int failures = expect_long_inputs();
    failures += expect("Hello, world!", "Hello, world!");
    failures += expect("«hello» ‹world›", "\"hello\" \"world");
    failures += expect("Turn 90°, then n°5, or N° 12; but 10n°.",
                       "Turn 90 degrees, then number 5, or Number 12, but 10n degrees.");
    failures += expect("100°C and 32°F", "100 degrees Celsius and 32 degrees Fahrenheit");
    failures += expect("a‐b‑c‒d―e", "a-b-c-d-e");
    failures += expect("a–b—c", "a b c");
    failures += expect("foo (bar) [baz] {qux}", "foo, bar, baz qux");
    failures += expect("• ‣ ◦ · a→b←c↑d↓e ➡ ➜", "a b c d e");
    failures += expect("wait… a…b", "wait. a.b");
    failures += expect("and/or foo/bar mg/kg kJ/mol a/b foo/bar.",
                       "and or foo bar mg/kg kJ/mol a/b foo bar.");
    failures += expect("user@host @home", "user at host at home");
    failures += expect("1500+20", "1 thousand 500 plus 20");
    failures += expect("C++.", "C plus plus.");
    failures += expect("foo@bar.com", "foo at bar dot com");
    failures += expect("foo@.com", "foo A-T dot com");
    failures += expect("a.b+tag@example.com", "a dot b plus tag at example dot com");
    failures += expect("first_last@x.co.uk.", "first underscore last at x dot co dot U-K.");
    failures += expect("https://www.example.com/path/to-page/",
                       "H-T-T-P-S colon slash slash W-W-W dot example dot com slash path slash T-O "
                       "dash page slash");
    failures += expect("example.com.", "example dot com.");
    failures += expect("http://sub.domain.co.uk",
                       "H-T-T-P colon slash slash sub dot domain dot C-O dot U-K");
    failures += expect("abc123.io/v2", "abc 1 2 3 dot I-O slash v 2");
    failures += expect("example.html", "example.html");
    failures += expect("www.example.Com", "W-W-W dot example dot Com");
    failures += expect("time: 10:30", "time, 10:30");
    failures += expect("a   b\t\tc\n\nd", "a b c d");
    failures += expect("hello 😀 flag 🇫🇷 sun ☀", "hello flag sun");
    failures +=
        expect("1920×1080, 6÷2, x≤3, y≥4, a≠b, π≈3.14",
               "1920 times 1 thousand 80, 6 divided by 2, x less than or equal to 3, y greater "
               "than or equal to 4, a is not equal to b, pi approximately 3 point 14");
    failures +=
        expect("é² α² C⁴ A²³", "é squared alpha squared C to the power of 4 A to the power of 23");
    failures +=
        expect("x² + y³ = 10⁻⁶", "x squared plus y cubed equals 10 to the power of minus 6");
    failures +=
        expect("Rated ★★★★☆, or ★★★½. I ❤️ NY, dial *67, it is 4.5%.",
               "Rated 4 out of 5 stars, or 3 point 5 stars. I love NY, dial star 67, it is 4.5%.");
    failures += expect("C₈H₁₈ and ¹³C", "C8H18 and 13C");
    failures += expect("café résumé", "café résumé");
    failures += expect("Lt. Col. Smith visited Dec.,", "Lieutenant Colonel Smith visited Dec,");
    failures += expect("Lt Col Vann", "Lieutenant Colonel Vann");
    failures += expect("250 €", "250 euros");
    failures += expect("-$500", "minus 500 dollars");
    failures += expect("$20-25 $007m $5€", "$20-25 $007m $5€");
    failures +=
        expect("$2m $2bn $2 million $4.59",
               "2 million dollars 2 billion dollars 2 million dollars 4 dollars and 59 cents");
    failures += expect("$1,234.56 $4.00 30¢ 1¢",
                       "1 thousand 234 dollars and 56 cents 4 dollars 30 cents 1 cent");
    failures += expect("20-25 1445-1450 2024-25 90-60", "20 to 25 1445 to 1450 2024 to 25 90-60");
    failures += expect("7,000-ton", "7 thousand-ton");
    failures += expect("007 0600. 1,2,3 1,999 1234567890123",
                       "zero zero seven, zero six zero zero. 123 1999 1234567890123");
    failures += expect("The card ending in -4098.", "The card ending in four zero nine eight.");
    failures += expect("123 456 789 and +123 456, 2000 2024",
                       "one two three, four five six, seven eight nine and plus one two three, "
                       "four five six, 2000 2024");
    failures += expect("999,999,999,999 1,000,000,000,000",
                       "999 billion 999 million 999 thousand 999 1,000,000,000,000");
    failures +=
        expect("1234 1234567 3.14 -12.00 2024",
               "1 thousand 234 1 million 234 thousand 567 3 point 14 minus 12 point 00 2024");
    failures += expect("555-123-4567", "555 123 4 5 6 7");
    failures += expect("4111-1111-1111-1111", "4 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1");
    failures += expect("555_123_4567", "555 underscore 123 underscore 4 5 6 7");
    failures += expect("1_2_3", "1 underscore 2 underscore 3");
    failures += expect("2024-05-12", "2024-05-12");
    failures += expect("2024-99-99", "2 0 2 4 99 99");
    failures += expect("sooo Nooooo, Hiii Whyyy Ewww, Ahhh! good Zzz Brrr WWW Mmm",
                       "so No, Hi Why Ew, Ah! good Zzz Brrr WWW Mmm");
    char *result = NULL;
    size_t result_length = 0;
    SpeechError error;
    const char malformed[] = {(char)0xc0, (char)0x80};
    const char embedded_nul[] = {'a', 0, 'b'};
    if (phonon_english_normalize("en", malformed, sizeof(malformed), &result, &result_length,
                                 &error) != SPEECH_IO_INVALID_ARGUMENT ||
        result != NULL)
        failures++;
    if (phonon_english_normalize("en", embedded_nul, sizeof(embedded_nul), &result, &result_length,
                                 &error) != SPEECH_IO_INVALID_ARGUMENT ||
        result != NULL)
        failures++;
    if (phonon_english_normalize("en", "😀", strlen("😀"), &result, &result_length, &error) !=
            SPEECH_IO_OK ||
        result == NULL || result_length != 0)
        failures++;
    phonon_english_normalize_free(result);
    result = NULL;
    if (phonon_english_normalize("fr", "hello", 5, &result, &result_length, &error) !=
            SPEECH_IO_INVALID_ARGUMENT ||
        result != NULL)
        failures++;
    if (phonon_english_normalize("en", "", 0, &result, &result_length, &error) != SPEECH_IO_OK ||
        result == NULL || result_length != 0 || result[0] != '\0')
        failures++;
    phonon_english_normalize_free(result);
#ifdef PHONON_ENGLISH_NORMALIZE_TEST_ALLOCATOR
    const char allocation_input[] = "$1.25, foo@example.com 123-456-7890 Lt Col";
    bool reached_success = false;
    size_t injected_failures = 0;
    for (long fail_at = 0; fail_at < 256; fail_at++) {
        result = NULL;
        result_length = 0;
        phonon_english_normalize_test_fail_after(fail_at);
        SpeechIOStatus status = phonon_english_normalize(
            "en", allocation_input, sizeof(allocation_input) - 1, &result, &result_length, &error);
        if (status == SPEECH_IO_OK) {
            if (phonon_english_normalize_test_failure_triggered() || result == NULL ||
                strcmp(result,
                       "1 dollar and 25 cents, foo at example dot com 123 456 7 8 9 0 Lieutenant "
                       "Colonel") != 0) {
                fprintf(stderr,
                        "allocation failure %ld was swallowed or changed output: trigger=%d "
                        "actual=%s\n",
                        fail_at, phonon_english_normalize_test_failure_triggered(),
                        result == NULL ? "<null>" : result);
                phonon_english_normalize_free(result);
                return EXIT_FAILURE;
            }
            reached_success = true;
            phonon_english_normalize_free(result);
            break;
        }
        if (status != SPEECH_IO_OUT_OF_MEMORY ||
            !phonon_english_normalize_test_failure_triggered() || result != NULL ||
            result_length != 0) {
            fprintf(stderr, "allocation failure %ld returned %s with output %s\n", fail_at,
                    speech_status_name(status), result == NULL ? "<null>" : "non-null");
            phonon_english_normalize_free(result);
            return EXIT_FAILURE;
        }
        injected_failures++;
    }
    if (!reached_success || injected_failures < 10) {
        fprintf(stderr, "allocation injection covered only %zu failures\n", injected_failures);
        return EXIT_FAILURE;
    }
#endif
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

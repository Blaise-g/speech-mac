#include "english_normalize.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    char *line = NULL;
    size_t capacity = 0;
    ssize_t line_length;
    while ((line_length = getline(&line, &capacity, stdin)) >= 0) {
        size_t length = (size_t)line_length;
        if (length > 0 && line[length - 1] == '\n')
            line[--length] = '\0';
        char *normalized = NULL;
        size_t normalized_length = 0;
        KokoroError error;
        KokoroStatus status =
            phonon_english_normalize("en", line, length, &normalized, &normalized_length, &error);
        if (status != KOKORO_OK) {
            fprintf(stderr, "%s\n", error.message);
            free(line);
            return EXIT_FAILURE;
        }
        if (fwrite(normalized, 1, normalized_length, stdout) != normalized_length ||
            fputc('\n', stdout) == EOF) {
            phonon_english_normalize_free(normalized);
            free(line);
            return EXIT_FAILURE;
        }
        phonon_english_normalize_free(normalized);
    }
    free(line);
    return ferror(stdin) ? EXIT_FAILURE : EXIT_SUCCESS;
}

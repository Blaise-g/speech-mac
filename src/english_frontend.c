#include "english_frontend.h"

#include "kokoro_internal.h"

#include <errno.h>
#include <spawn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

static KokoroStatus read_stream(FILE *stream, char **contents, size_t *length, KokoroError *error) {
    char *buffer = NULL;
    size_t used = 0;
    size_t capacity = 0;
    unsigned char block[8192];
    for (;;) {
        size_t count = fread(block, 1, sizeof(block), stream);
        if (count > 0) {
            if (count > SIZE_MAX - used - 1) {
                free(buffer);
                kokoro_error_set(error, KOKORO_OUT_OF_MEMORY, "eSpeak NG output is too large");
                return KOKORO_OUT_OF_MEMORY;
            }
            size_t required = used + count + 1;
            if (required > capacity) {
                size_t grown = capacity == 0 ? sizeof(block) : capacity;
                while (grown < required) {
                    if (grown > SIZE_MAX / 2) {
                        grown = required;
                        break;
                    }
                    grown *= 2;
                }
                char *next = realloc(buffer, grown);
                if (next == NULL) {
                    free(buffer);
                    kokoro_error_set(error, KOKORO_OUT_OF_MEMORY,
                                     "cannot allocate eSpeak NG output");
                    return KOKORO_OUT_OF_MEMORY;
                }
                buffer = next;
                capacity = grown;
            }
            memcpy(buffer + used, block, count);
            used += count;
        }
        if (count < sizeof(block)) {
            if (ferror(stream)) {
                free(buffer);
                kokoro_error_set(error, KOKORO_IO_ERROR, "cannot read eSpeak NG output");
                return KOKORO_IO_ERROR;
            }
            break;
        }
    }
    if (buffer == NULL) {
        buffer = malloc(1);
        if (buffer == NULL) {
            kokoro_error_set(error, KOKORO_OUT_OF_MEMORY, "cannot allocate eSpeak NG output");
            return KOKORO_OUT_OF_MEMORY;
        }
    }
    buffer[used] = '\0';
    if (memchr(buffer, '\0', used) != NULL) {
        free(buffer);
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "eSpeak NG output contains an embedded NUL byte");
        return KOKORO_INVALID_ARGUMENT;
    }
    *contents = buffer;
    *length = used;
    return KOKORO_OK;
}

static void normalize_espeak_markers(char *phonemes, size_t *length) {
    size_t source = 0;
    size_t target = 0;
    size_t last_codepoint_start = 0;
    while (source < *length) {
        if (*length - source >= 3 && (unsigned char)phonemes[source] == 0xe2U &&
            (unsigned char)phonemes[source + 1] == 0x80U &&
            (unsigned char)phonemes[source + 2] == 0x8dU) {
            source += 3;
        } else if (*length - source >= 2 && (unsigned char)phonemes[source] == 0xccU &&
                   (unsigned char)phonemes[source + 1] == 0xa9U && target > 0) {
            memmove(phonemes + last_codepoint_start + 2, phonemes + last_codepoint_start,
                    target - last_codepoint_start);
            phonemes[last_codepoint_start] = (char)0xc9;
            phonemes[last_codepoint_start + 1] = (char)0x99;
            target += 2;
            source += 2;
        } else {
            unsigned char first = (unsigned char)phonemes[source];
            size_t codepoint_length = first < 0x80U   ? 1U
                                      : first < 0xe0U ? 2U
                                      : first < 0xf0U ? 3U
                                                      : 4U;
            if (codepoint_length > *length - source) {
                codepoint_length = 1;
            }
            last_codepoint_start = target;
            memmove(phonemes + target, phonemes + source, codepoint_length);
            target += codepoint_length;
            source += codepoint_length;
        }
    }
    phonemes[target] = '\0';
    *length = target;
}

KokoroStatus kokoro_english_frontend(const char *program_path, const char *text, size_t text_length,
                                     char **phonemes, size_t *phoneme_length, KokoroError *error) {
    if (phonemes != NULL) {
        *phonemes = NULL;
    }
    if (phoneme_length != NULL) {
        *phoneme_length = 0;
    }
    if (program_path == NULL || program_path[0] == '\0' || text == NULL || text_length == 0 ||
        phonemes == NULL || phoneme_length == NULL) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "eSpeak NG path, nonempty text, and output pointers are required");
        return KOKORO_INVALID_ARGUMENT;
    }

    FILE *input = tmpfile();
    FILE *output = tmpfile();
    if (input == NULL || output == NULL) {
        int saved_errno = errno;
        if (input != NULL) {
            fclose(input);
        }
        if (output != NULL) {
            fclose(output);
        }
        kokoro_error_set(error, KOKORO_IO_ERROR, "cannot create eSpeak NG temporary streams: %s",
                         strerror(saved_errno));
        return KOKORO_IO_ERROR;
    }
    KokoroStatus status = KOKORO_OK;
    if (fwrite(text, 1, text_length, input) != text_length || fflush(input) != 0 ||
        fseek(input, 0, SEEK_SET) != 0) {
        kokoro_error_set(error, KOKORO_IO_ERROR, "cannot prepare text for eSpeak NG");
        status = KOKORO_IO_ERROR;
        goto cleanup;
    }

    posix_spawn_file_actions_t actions;
    int spawn_error = posix_spawn_file_actions_init(&actions);
    if (spawn_error != 0) {
        kokoro_error_set(error, KOKORO_IO_ERROR, "cannot prepare eSpeak NG process: %s",
                         strerror(spawn_error));
        status = KOKORO_IO_ERROR;
        goto cleanup;
    }
    int input_descriptor = fileno(input);
    int output_descriptor = fileno(output);
    spawn_error = posix_spawn_file_actions_adddup2(&actions, input_descriptor, STDIN_FILENO);
    if (spawn_error == 0) {
        spawn_error = posix_spawn_file_actions_adddup2(&actions, output_descriptor, STDOUT_FILENO);
    }
    if (spawn_error == 0 && input_descriptor > STDERR_FILENO) {
        spawn_error = posix_spawn_file_actions_addclose(&actions, input_descriptor);
    }
    if (spawn_error == 0 && output_descriptor > STDERR_FILENO) {
        spawn_error = posix_spawn_file_actions_addclose(&actions, output_descriptor);
    }
    if (spawn_error != 0) {
        (void)posix_spawn_file_actions_destroy(&actions);
        kokoro_error_set(error, KOKORO_IO_ERROR, "cannot prepare eSpeak NG streams: %s",
                         strerror(spawn_error));
        status = KOKORO_IO_ERROR;
        goto cleanup;
    }

    char *const arguments[] = {(char *)program_path, "-q", "--ipa=3", "-v", "en-us", NULL};
    pid_t process;
    spawn_error = posix_spawnp(&process, program_path, &actions, NULL, arguments, environ);
    (void)posix_spawn_file_actions_destroy(&actions);
    if (spawn_error != 0) {
        kokoro_error_set(error, KOKORO_IO_ERROR, "cannot start eSpeak NG: %s",
                         strerror(spawn_error));
        status = KOKORO_IO_ERROR;
        goto cleanup;
    }
    int wait_status;
    pid_t waited;
    do {
        waited = waitpid(process, &wait_status, 0);
    } while (waited < 0 && errno == EINTR);
    if (waited < 0) {
        kokoro_error_set(error, KOKORO_IO_ERROR, "cannot wait for eSpeak NG: %s", strerror(errno));
        status = KOKORO_IO_ERROR;
        goto cleanup;
    }
    if (!WIFEXITED(wait_status) || WEXITSTATUS(wait_status) != 0) {
        if (WIFEXITED(wait_status)) {
            kokoro_error_set(error, KOKORO_IO_ERROR, "eSpeak NG exited with status %d",
                             WEXITSTATUS(wait_status));
        } else if (WIFSIGNALED(wait_status)) {
            kokoro_error_set(error, KOKORO_IO_ERROR, "eSpeak NG terminated by signal %d",
                             WTERMSIG(wait_status));
        } else {
            kokoro_error_set(error, KOKORO_IO_ERROR, "eSpeak NG did not complete successfully");
        }
        status = KOKORO_IO_ERROR;
        goto cleanup;
    }
    if (fseek(output, 0, SEEK_SET) != 0) {
        kokoro_error_set(error, KOKORO_IO_ERROR, "cannot read eSpeak NG output");
        status = KOKORO_IO_ERROR;
        goto cleanup;
    }
    status = read_stream(output, phonemes, phoneme_length, error);
    if (status == KOKORO_OK) {
        normalize_espeak_markers(*phonemes, phoneme_length);
    }
    if (status == KOKORO_OK && *phoneme_length == 0) {
        free(*phonemes);
        *phonemes = NULL;
        *phoneme_length = 0;
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "eSpeak NG produced no phoneme output for this text");
        status = KOKORO_INVALID_ARGUMENT;
    }

cleanup:
    fclose(output);
    fclose(input);
    if (status == KOKORO_OK) {
        kokoro_error_set(error, KOKORO_OK, NULL);
    }
    return status;
}

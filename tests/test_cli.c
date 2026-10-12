#include "cli.h"

#include "english_frontend.h"
#include "ipa_tokenizer.h"
#include "istft.h"
#include "synthesis.h"
#include "wav_writer.h"

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define MODEL_PACK_PATH "models/kokoro-v1_0-af_heart.kpack"
#define VOICES_DIR "models/voices"
#define OUTPUT_PATH ".build/test-cli-output with spaces.wav"

typedef struct {
    float *samples;
    size_t sample_count;
    size_t sample_capacity;
    size_t open_count;
    size_t start_count;
    size_t drain_count;
    size_t wake_count;
    size_t abort_count;
    size_t close_count;
} CliStreamPlayback;

static int append_float_samples(float **samples, size_t *count, size_t *capacity,
                                const float *chunk, size_t chunk_count);

static int fail(const char *message) {
    fprintf(stderr, "test_cli: %s\n", message);
    return 0;
}

static int stream_text(FILE *stream, char *buffer, size_t capacity) {
    if (fflush(stream) != 0 || fseek(stream, 0, SEEK_SET) != 0) {
        return 0;
    }
    size_t count = fread(buffer, 1, capacity - 1, stream);
    buffer[count] = '\0';
    return !ferror(stream);
}

static int call_cli(int argc, char **argv, const KokoroCliConfig *base, int *exit_code,
                    char out[4096], char err[4096]) {
    FILE *out_stream = tmpfile();
    FILE *err_stream = tmpfile();
    if (out_stream == NULL || err_stream == NULL) {
        if (out_stream != NULL) {
            fclose(out_stream);
        }
        if (err_stream != NULL) {
            fclose(err_stream);
        }
        return fail("could not create capture streams");
    }
    KokoroCliConfig config = *base;
    config.stdout_stream = out_stream;
    config.stderr_stream = err_stream;
    *exit_code = kokoro_cli_run(argc, argv, &config);
    int valid = stream_text(out_stream, out, 4096) && stream_text(err_stream, err, 4096);
    fclose(out_stream);
    fclose(err_stream);
    return valid;
}

static int test_help_and_argument_errors(const KokoroCliConfig *config) {
    char out[4096];
    char err[4096];
    int exit_code;
    char *help_args[] = {"kokoro", "say", "--help"};
    if (!call_cli(3, help_args, config, &exit_code, out, err) || exit_code != 0 ||
        strstr(out, "--file PATH") == NULL || strstr(out, "--ipa") == NULL ||
        strstr(out, "af_heart (default)") == NULL || strstr(out, "afplay") == NULL ||
        strstr(out, "--stream") == NULL) {
        return fail("say help omitted source selection or default details");
    }
    char *version_args[] = {"kokoro", "--version"};
    if (!call_cli(2, version_args, config, &exit_code, out, err) || exit_code != 0 ||
        strstr(out, "kokoro 0.1.0") == NULL) {
        return fail("version output is missing");
    }
    char *bare_help_args[] = {"kokoro", "say", "--help"};
    if (!call_cli(3, bare_help_args, config, &exit_code, out, err) || exit_code != 0) {
        return fail("say --help should show usage");
    }
    char *unknown_voice_args[] = {"kokoro", "say", "--ipa", "həˈloʊ", "--voice", "unknown"};
    if (!call_cli(6, unknown_voice_args, config, &exit_code, out, err) || exit_code != 2 ||
        strstr(err, "unknown voice") == NULL) {
        return fail("unknown voice was not rejected with a clear error");
    }
    const char *invalid_speeds[] = {"nan", "inf", "0.49", "2.01", "1e999"};
    for (size_t index = 0; index < sizeof(invalid_speeds) / sizeof(invalid_speeds[0]); index++) {
        char *speed_args[] = {"kokoro", "say",     "--ipa",
                              "həˈloʊ", "--speed", (char *)invalid_speeds[index]};
        if (!call_cli(6, speed_args, config, &exit_code, out, err) || exit_code != 2 ||
            strstr(err, "finite") == NULL) {
            return fail("non-finite or out-of-range speed was not rejected");
        }
    }
    char *unknown_option_args[] = {"kokoro", "say", "--ipa", "həˈloʊ", "--shell"};
    if (!call_cli(5, unknown_option_args, config, &exit_code, out, err) || exit_code != 2 ||
        strstr(err, "unknown option") == NULL) {
        return fail("unknown option was not rejected");
    }
    char *stream_output_args[] = {"kokoro",   "say", "--ipa",    "həˈloʊ",
                                  "--stream", "-o",  OUTPUT_PATH};
    (void)remove(OUTPUT_PATH);
    if (!call_cli(7, stream_output_args, config, &exit_code, out, err) || exit_code != 2 ||
        strstr(err, "--stream cannot be combined with --output") == NULL ||
        access(OUTPUT_PATH, F_OK) == 0) {
        return fail("streaming output was not rejected before creating a WAV");
    }
    return 1;
}

static int test_model_artifact_errors(const KokoroCliConfig *base) {
    char out[4096];
    char err[4096];
    int exit_code;
    char *args[] = {"kokoro", "say", "--ipa", "həˈloʊ"};
    KokoroCliConfig config = *base;
    config.model_pack_path = "/no/such/kokoro-model.kpack";
    if (!call_cli(4, args, &config, &exit_code, out, err) || exit_code != 1 ||
        strstr(err, "cannot load af_heart model/voice artifacts") == NULL ||
        strstr(err, "cannot open") == NULL) {
        return fail("missing model pack was not reported clearly by the CLI");
    }

    char model_path[] = "/tmp/kokoro-invalid-model-XXXXXX";
    int descriptor = mkstemp(model_path);
    if (descriptor < 0 || close(descriptor) != 0) {
        if (descriptor >= 0) {
            unlink(model_path);
        }
        return fail("could not create malformed model-pack fixture");
    }
    config.model_pack_path = model_path;
    int passed = call_cli(4, args, &config, &exit_code, out, err) && exit_code == 1 &&
                 strstr(err, "cannot load af_heart model/voice artifacts") != NULL &&
                 strstr(err, "truncated pack header") != NULL;
    unlink(model_path);
    return passed ? 1 : fail("malformed model pack was not reported clearly by the CLI");
}

static int test_voice_artifact_errors(const KokoroCliConfig *base) {
    char voices_directory[] = "/tmp/kokoro-cli-voices-XXXXXX";
    if (mkdtemp(voices_directory) == NULL) {
        return fail("could not create temporary voice directory");
    }
    KokoroCliConfig config = *base;
    config.voices_directory = voices_directory;
    char *args[] = {"kokoro", "say", "--ipa", "həˈloʊ", "--voice", "am_adam"};
    char out[4096];
    char err[4096];
    int exit_code;
    int passed = call_cli(6, args, &config, &exit_code, out, err) && exit_code == 1 &&
                 strstr(err, "cannot load am_adam model/voice artifacts") != NULL &&
                 strstr(err, "cannot open am_adam voice artifact") != NULL;
    char artifact_path[256];
    (void)snprintf(artifact_path, sizeof(artifact_path), "%s/am_adam.kvoice", voices_directory);
    FILE *artifact = fopen(artifact_path, "wb");
    if (artifact == NULL) {
        rmdir(voices_directory);
        return fail("could not create malformed voice-artifact fixture");
    }
    int write_result = fwrite("bad", 1, 3, artifact) == 3;
    int close_result = fclose(artifact);
    int wrote_fixture = write_result && close_result == 0;
    if (!wrote_fixture) {
        unlink(artifact_path);
        rmdir(voices_directory);
        return fail("could not write malformed voice-artifact fixture");
    }
    passed = passed && call_cli(6, args, &config, &exit_code, out, err) && exit_code == 1 &&
             strstr(err, "cannot load am_adam model/voice artifacts") != NULL &&
             strstr(err, "wrong file size or type") != NULL;
    unlink(artifact_path);
    rmdir(voices_directory);
    return passed ? 1 : fail("missing or malformed voice artifact was not reported clearly");
}

static int verify_wav(const char *path) {
    unsigned char header[44];
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return 0;
    }
    int valid = fread(header, 1, sizeof(header), file) == sizeof(header) &&
                memcmp(header, "RIFF", 4) == 0 && memcmp(header + 8, "WAVEfmt ", 8) == 0 &&
                header[22] == 1 && header[23] == 0 && header[24] == 0xC0 && header[25] == 0x5D &&
                header[26] == 0 && header[27] == 0 && header[34] == 16 && header[28] == 0x80 &&
                header[29] == 0xBB && header[30] == 0 && header[31] == 0 && header[32] == 2 &&
                header[33] == 0 && header[35] == 0 && memcmp(header + 36, "data", 4) == 0;
    if (valid && fseek(file, 0, SEEK_END) == 0) {
        long file_size = ftell(file);
        uint32_t riff_size = (uint32_t)header[4] | ((uint32_t)header[5] << 8) |
                             ((uint32_t)header[6] << 16) | ((uint32_t)header[7] << 24);
        uint32_t data_size = (uint32_t)header[40] | ((uint32_t)header[41] << 8) |
                             ((uint32_t)header[42] << 16) | ((uint32_t)header[43] << 24);
        valid = file_size >= 44 && data_size > 0 && data_size % 2 == 0 &&
                (uint64_t)file_size == UINT64_C(44) + data_size && riff_size == data_size + 36U;
    } else {
        valid = 0;
    }
    fclose(file);
    return valid;
}

static int write_player_stub(char player_path[64], char log_path[64], int exit_status) {
    memcpy(player_path, "/tmp/kokoro-player-XXXXXX", sizeof("/tmp/kokoro-player-XXXXXX"));
    int descriptor = mkstemp(player_path);
    if (descriptor < 0) {
        return 0;
    }
    if (close(descriptor) != 0) {
        unlink(player_path);
        return 0;
    }
    memcpy(log_path, "/tmp/kokoro-player-log-XXXXXX", sizeof("/tmp/kokoro-player-log-XXXXXX"));
    descriptor = mkstemp(log_path);
    if (descriptor < 0) {
        unlink(player_path);
        return 0;
    }
    if (close(descriptor) != 0) {
        unlink(player_path);
        unlink(log_path);
        return 0;
    }
    FILE *script = fopen(player_path, "w");
    if (script == NULL) {
        unlink(player_path);
        unlink(log_path);
        return 0;
    }
    int valid = fprintf(script,
                        "#!/bin/sh\n"
                        "printf 'argc=%%s\\n' \"$#\" > '%s'\n"
                        "printf 'path=%%s\\n' \"$1\" >> '%s'\n"
                        "if test -f \"$1\"; then printf 'exists=1\\n' >> '%s'; "
                        "else printf 'exists=0\\n' >> '%s'; fi\n"
                        "printf 'mode=%%s\\n' \"$(stat -f %%Lp \"$1\")\" >> '%s'\n"
                        "exit %d\n",
                        log_path, log_path, log_path, log_path, log_path, exit_status) > 0;
    if (fclose(script) != 0 || chmod(player_path, 0700) != 0) {
        valid = 0;
    }
    return valid;
}

static int verify_player_log(const char *log_path, char wav_path[256]) {
    FILE *log = fopen(log_path, "r");
    if (log == NULL) {
        return 0;
    }
    char argc_line[64];
    char path_line[300];
    char exists_line[64];
    char mode_line[64];
    int valid = fgets(argc_line, sizeof(argc_line), log) != NULL &&
                fgets(path_line, sizeof(path_line), log) != NULL &&
                fgets(exists_line, sizeof(exists_line), log) != NULL &&
                fgets(mode_line, sizeof(mode_line), log) != NULL;
    fclose(log);
    if (!valid || strcmp(argc_line, "argc=1\n") != 0 || strcmp(exists_line, "exists=1\n") != 0 ||
        strcmp(mode_line, "mode=600\n") != 0 || strncmp(path_line, "path=", 5) != 0) {
        return 0;
    }
    size_t path_length = strlen(path_line + 5);
    if (path_length == 0 || path_length >= 256 || path_line[5 + path_length - 1] != '\n') {
        return 0;
    }
    memcpy(wav_path, path_line + 5, path_length - 1);
    wav_path[path_length - 1] = '\0';
    return access(wav_path, F_OK) != 0;
}

static int test_voice_listing(const KokoroCliConfig *base) {
    char empty_directory[] = "/tmp/kokoro-empty-voices-XXXXXX";
    if (mkdtemp(empty_directory) == NULL) {
        return fail("could not create empty voice directory");
    }
    KokoroCliConfig config = *base;
    config.voices_directory = empty_directory;
    char *args[] = {"kokoro", "--list-voices"};
    char out[4096];
    char err[4096];
    int exit_code;
    int passed = call_cli(2, args, &config, &exit_code, out, err) && exit_code == 0 &&
                 strcmp(out, "af_heart\n") == 0;
    rmdir(empty_directory);
    if (!passed) {
        return fail("voice listing included an uninstalled voice or omitted af_heart");
    }
    int has_am_adam = access("models/voices/am_adam.kvoice", R_OK) == 0;
    return call_cli(2, args, base, &exit_code, out, err) && exit_code == 0 &&
                   strcmp(out, has_am_adam ? "af_heart\nam_adam\n" : "af_heart\n") == 0
               ? 1
               : fail("voice listing did not match the installed supported artifacts");
}

static int test_synthesis_and_playback(const KokoroCliConfig *base) {
    (void)remove(OUTPUT_PATH);
    char *output_args[] = {"kokoro",   "say", "--ipa", "həˈloʊ", "-v",
                           "af_heart", "-s",  "1.25",  "-o",     OUTPUT_PATH};
    char out[4096];
    char err[4096];
    int exit_code;
    if (!call_cli(10, output_args, base, &exit_code, out, err) || exit_code != 0 ||
        !verify_wav(OUTPUT_PATH)) {
        (void)remove(OUTPUT_PATH);
        return fail("prepared phonemes did not write a valid mono 24 kHz WAV");
    }
    (void)remove(OUTPUT_PATH);

    char *adam_output_args[] = {"kokoro",  "say",     "--ipa",    "həˈloʊ",
                                "--voice", "am_adam", "--output", OUTPUT_PATH};
    if (!call_cli(8, adam_output_args, base, &exit_code, out, err) || exit_code != 0 ||
        !verify_wav(OUTPUT_PATH)) {
        (void)remove(OUTPUT_PATH);
        return fail("am_adam selection did not write a valid WAV");
    }
    (void)remove(OUTPUT_PATH);

    char player_path[64];
    char log_path[64];
    if (!write_player_stub(player_path, log_path, 0)) {
        return fail("could not create playback stub");
    }
    KokoroCliConfig config = *base;
    config.player_path = player_path;
    char *play_args[] = {"kokoro", "say", "--ipa", "həˈloʊ"};
    int success = call_cli(4, play_args, &config, &exit_code, out, err) && exit_code == 0;
    char played_path[256];
    success = success && verify_player_log(log_path, played_path);
    unlink(player_path);
    unlink(log_path);
    if (!success) {
        return fail("completed WAV was not passed as one secure temporary argument and removed");
    }

    if (!write_player_stub(player_path, log_path, 7)) {
        return fail("could not create failing playback stub");
    }
    config.player_path = player_path;
    int failed_playback = call_cli(4, play_args, &config, &exit_code, out, err) && exit_code == 1 &&
                          strstr(err, "afplay exited with status 7") != NULL &&
                          verify_player_log(log_path, played_path);
    unlink(player_path);
    unlink(log_path);
    return failed_playback ? 1
                           : fail("playback error was unclear or left its temporary WAV behind");
}

static int write_espeak_stub(char espeak_path[64], char input_log[64], const char *ipa_output) {
    memcpy(espeak_path, "/tmp/kokoro-espeak-XXXXXX", sizeof("/tmp/kokoro-espeak-XXXXXX"));
    int descriptor = mkstemp(espeak_path);
    if (descriptor < 0) {
        return 0;
    }
    if (close(descriptor) != 0) {
        unlink(espeak_path);
        return 0;
    }
    memcpy(input_log, "/tmp/kokoro-espeak-input-XXXXXX", sizeof("/tmp/kokoro-espeak-input-XXXXXX"));
    descriptor = mkstemp(input_log);
    if (descriptor < 0) {
        unlink(espeak_path);
        return 0;
    }
    if (close(descriptor) != 0) {
        unlink(espeak_path);
        unlink(input_log);
        return 0;
    }
    FILE *script = fopen(espeak_path, "w");
    if (script == NULL) {
        unlink(espeak_path);
        unlink(input_log);
        return 0;
    }
    int valid = fprintf(script,
                        "#!/bin/sh\n"
                        "if test \"$#\" -ne 4 || test \"$1\" != \"-q\" || "
                        "test \"$2\" != \"--ipa=3\" || test \"$3\" != \"-v\" || "
                        "test \"$4\" != \"en-us\"; then exit 91; fi\n"
                        "cat > '%s'\n"
                        "printf '%%s' '%s'\n",
                        input_log, ipa_output) > 0;
    if (fclose(script) != 0 || chmod(espeak_path, 0700) != 0) {
        valid = 0;
    }
    return valid;
}

static int file_equals(const char *path, const char *expected) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return 0;
    }
    size_t expected_length = strlen(expected);
    char *contents = malloc(expected_length + 1);
    if (contents == NULL) {
        fclose(file);
        return 0;
    }
    size_t count = fread(contents, 1, expected_length + 1, file);
    int valid = count == expected_length && !ferror(file) &&
                memcmp(contents, expected, expected_length) == 0;
    valid = fclose(file) == 0 && valid;
    free(contents);
    return valid;
}

static int write_bytes(const char *path, const unsigned char *bytes, size_t length) {
    FILE *file = fopen(path, "wb");
    if (file == NULL) {
        return 0;
    }
    int wrote_all = fwrite(bytes, 1, length, file) == length;
    int closed = fclose(file) == 0;
    return wrote_all && closed;
}

static int read_wav_payload(const char *path, unsigned char **payload, size_t *payload_length) {
    *payload = NULL;
    *payload_length = 0;
    FILE *file = fopen(path, "rb");
    if (file == NULL || fseek(file, 0, SEEK_END) != 0) {
        if (file != NULL) {
            fclose(file);
        }
        return 0;
    }
    long file_size = ftell(file);
    if (file_size < 44 || fseek(file, 44, SEEK_SET) != 0) {
        fclose(file);
        return 0;
    }
    size_t length = (size_t)(file_size - 44);
    unsigned char *data = malloc(length);
    if (data == NULL) {
        fclose(file);
        return 0;
    }
    int valid = fread(data, 1, length, file) == length && !ferror(file);
    valid = fclose(file) == 0 && valid;
    if (!valid) {
        free(data);
        return 0;
    }
    *payload = data;
    *payload_length = length;
    return 1;
}

static int read_text_file(const char *path, char **text, size_t *length) {
    *text = NULL;
    *length = 0;
    FILE *file = fopen(path, "rb");
    if (file == NULL || fseek(file, 0, SEEK_END) != 0) {
        if (file != NULL) {
            fclose(file);
        }
        return 0;
    }
    long file_size = ftell(file);
    if (file_size < 0 || fseek(file, 0, SEEK_SET) != 0 || (uint64_t)file_size >= SIZE_MAX) {
        fclose(file);
        return 0;
    }
    size_t size = (size_t)file_size;
    char *buffer = malloc(size + 1);
    if (buffer == NULL) {
        fclose(file);
        return 0;
    }
    int valid = fread(buffer, 1, size, file) == size && !ferror(file);
    valid = fclose(file) == 0 && valid;
    if (!valid) {
        free(buffer);
        return 0;
    }
    buffer[size] = '\0';
    *text = buffer;
    *length = size;
    return 1;
}

static int append_float_samples(float **samples, size_t *count, size_t *capacity,
                                const float *chunk, size_t chunk_count) {
    if (chunk == NULL || chunk_count == 0 || chunk_count > SIZE_MAX - *count) {
        return 0;
    }
    size_t required = *count + chunk_count;
    if (required > *capacity) {
        size_t grown = *capacity == 0 ? chunk_count : *capacity;
        while (grown < required) {
            if (grown > SIZE_MAX / 2) {
                grown = required;
                break;
            }
            grown *= 2;
        }
        if (grown > SIZE_MAX / sizeof(**samples)) {
            return 0;
        }
        float *next = realloc(*samples, grown * sizeof(*next));
        if (next == NULL) {
            return 0;
        }
        *samples = next;
        *capacity = grown;
    }
    memcpy(*samples + *count, chunk, chunk_count * sizeof(*chunk));
    *count = required;
    return 1;
}

static KokoroStatus cli_stream_open(void *context, void **playback, KokoroError *error) {
    (void)error;
    CliStreamPlayback *fake = context;
    fake->open_count++;
    *playback = fake;
    return KOKORO_OK;
}

static KokoroStatus cli_stream_submit(void *handle, const float *samples, size_t sample_count,
                                      KokoroError *error) {
    CliStreamPlayback *fake = handle;
    if (!append_float_samples(&fake->samples, &fake->sample_count, &fake->sample_capacity, samples,
                              sample_count)) {
        if (error != NULL) {
            error->code = KOKORO_OUT_OF_MEMORY;
            (void)snprintf(error->message, sizeof(error->message),
                           "cannot retain fake streamed samples");
        }
        return KOKORO_OUT_OF_MEMORY;
    }
    return KOKORO_OK;
}

static KokoroStatus cli_stream_start(void *handle, KokoroError *error) {
    (void)error;
    CliStreamPlayback *fake = handle;
    fake->start_count++;
    return KOKORO_OK;
}

static KokoroStatus cli_stream_drain(void *handle, KokoroError *error) {
    (void)error;
    CliStreamPlayback *fake = handle;
    fake->drain_count++;
    return KOKORO_OK;
}

static void cli_stream_wake(void *handle) {
    CliStreamPlayback *fake = handle;
    fake->wake_count++;
}

static void cli_stream_abort(void *handle) {
    CliStreamPlayback *fake = handle;
    fake->abort_count++;
}

static KokoroStatus cli_stream_close(void *handle, KokoroError *error) {
    (void)error;
    CliStreamPlayback *fake = handle;
    fake->close_count++;
    return KOKORO_OK;
}

static const KokoroStreamPlaybackOps cli_stream_ops = {
    .open = cli_stream_open,
    .submit = cli_stream_submit,
    .start = cli_stream_start,
    .drain = cli_stream_drain,
    .wake = cli_stream_wake,
    .abort = cli_stream_abort,
    .close = cli_stream_close,
};

static int test_stream_cli(const KokoroCliConfig *base) {
    const char *phonemes = "həlˈoʊ";
    KokoroIpaChunks chunks = {0};
    KokoroError error = {0};
    KokoroStatus status = kokoro_ipa_encode_chunks(phonemes, false, &chunks, &error);
    if (status != KOKORO_OK) {
        kokoro_ipa_chunks_free(&chunks);
        return fail("could not tokenize IPA for streaming CLI parity");
    }
    KokoroEngine *engine = NULL;
    status = kokoro_engine_create(base->model_pack_path, "af_heart", NULL, &engine, &error);
    float *expected = NULL;
    size_t expected_count = 0;
    size_t expected_capacity = 0;
    if (status == KOKORO_OK) {
        for (size_t index = 0; index < chunks.count; index++) {
            KokoroSynthesisOptions options = {
                .voice_style_index = chunks.items[index].token_count - 3,
                .speed = 1.25f,
                .random_seed = base->random_seed + (uint64_t)index,
            };
            KokoroSynthesisOutput output = {0};
            status = kokoro_engine_synthesize(engine, chunks.items[index].token_ids,
                                              chunks.items[index].token_count, &options, &output,
                                              &error);
            if (status == KOKORO_OK &&
                !append_float_samples(&expected, &expected_count, &expected_capacity,
                                      output.samples, output.sample_count)) {
                status = KOKORO_OUT_OF_MEMORY;
            }
            kokoro_synthesis_output_free(&output);
            if (status != KOKORO_OK) {
                break;
            }
        }
    }
    kokoro_engine_free(engine);
    kokoro_ipa_chunks_free(&chunks);
    if (status != KOKORO_OK || expected_count == 0) {
        free(expected);
        return fail("could not synthesize expected stream CLI samples");
    }

    CliStreamPlayback playback = {0};
    KokoroCliConfig config = *base;
    config.stream_playback_ops = &cli_stream_ops;
    config.stream_playback_context = &playback;
    char *args[] = {"kokoro", "say", "--ipa", "həlˈoʊ", "--speed", "1.25", "--stream"};
    char out[4096];
    char err[4096];
    int exit_code;
    int passed = call_cli(7, args, &config, &exit_code, out, err) && exit_code == 0 &&
                 out[0] == '\0' && err[0] == '\0' && playback.open_count == 1 &&
                 playback.start_count == 1 && playback.drain_count == 1 &&
                 playback.wake_count == 0 && playback.abort_count == 0 &&
                 playback.close_count == 1 && playback.sample_count == expected_count &&
                 memcmp(playback.samples, expected, expected_count * sizeof(*expected)) == 0;
    free(playback.samples);
    free(expected);
    return passed ? 1 : fail("--stream did not deliver exact native float audio through playback");
}

static int test_text_sources_and_ipa_bypass(const KokoroCliConfig *base) {
    char espeak_path[64];
    char input_log[64];
    if (!write_espeak_stub(espeak_path, input_log, "həlˈo‍ʊ")) {
        return fail("could not create eSpeak NG stub");
    }
    KokoroCliConfig config = *base;
    config.espeak_path = espeak_path;
    const char *output_path = ".build/test-cli-english-source.wav";
    char out[4096];
    char err[4096];
    int exit_code;

    char *positional_args[] = {"kokoro", "say", "Hello from $HOME; and the terminal.", "-o",
                               (char *)output_path};
    int passed = call_cli(5, positional_args, &config, &exit_code, out, err) && exit_code == 0 &&
                 verify_wav(output_path) &&
                 file_equals(input_log, "Hello from $HOME; and the terminal.");
    (void)remove(output_path);
    if (!passed) {
        unlink(espeak_path);
        unlink(input_log);
        return fail("positional English source was not passed safely to eSpeak NG");
    }

    const char *special_path = "/tmp/kokoro file '$; input.txt";
    (void)remove(special_path);
    const char file_text[] = "File input — café.";
    if (!write_bytes(special_path, (const unsigned char *)file_text, sizeof(file_text) - 1)) {
        unlink(espeak_path);
        unlink(input_log);
        return fail("could not write UTF-8 source file");
    }
    char *file_args[] = {"kokoro",           "say", "--file", (char *)special_path, "--output",
                         (char *)output_path};
    passed = call_cli(6, file_args, &config, &exit_code, out, err) && exit_code == 0 &&
             verify_wav(output_path) && file_equals(input_log, file_text);
    (void)remove(special_path);
    (void)remove(output_path);
    if (!passed) {
        unlink(espeak_path);
        unlink(input_log);
        return fail("UTF-8 file source or metacharacter path was not handled safely");
    }

    const char stdin_text[] = "Text from standard input.";
    FILE *stdin_file = tmpfile();
    if (stdin_file == NULL ||
        fwrite(stdin_text, 1, sizeof(stdin_text) - 1, stdin_file) != sizeof(stdin_text) - 1 ||
        fseek(stdin_file, 0, SEEK_SET) != 0) {
        if (stdin_file != NULL) {
            fclose(stdin_file);
        }
        unlink(espeak_path);
        unlink(input_log);
        return fail("could not create standard-input fixture");
    }
    config.stdin_stream = stdin_file;
    char *stdin_args[] = {"kokoro", "say", "--output", (char *)output_path};
    passed = call_cli(4, stdin_args, &config, &exit_code, out, err) && exit_code == 0 &&
             verify_wav(output_path) && file_equals(input_log, stdin_text);
    fclose(stdin_file);
    config.stdin_stream = NULL;
    (void)remove(output_path);
    if (!passed) {
        unlink(espeak_path);
        unlink(input_log);
        return fail("standard input was not consumed as the default source");
    }

    char *conflict_args[] = {"kokoro", "say", "hello", "-f", (char *)special_path};
    if (!call_cli(5, conflict_args, &config, &exit_code, out, err) || exit_code != 2 ||
        strstr(err, "exactly one text source") == NULL) {
        unlink(espeak_path);
        unlink(input_log);
        return fail("conflicting positional and file sources were not rejected");
    }

    char invalid_path[] = "/tmp/kokoro-invalid-text-XXXXXX";
    int descriptor = mkstemp(invalid_path);
    const unsigned char invalid_utf8[] = {0xc3, 0x28};
    if (descriptor < 0 || close(descriptor) != 0 ||
        !write_bytes(invalid_path, invalid_utf8, sizeof(invalid_utf8))) {
        if (descriptor >= 0) {
            unlink(invalid_path);
        }
        unlink(espeak_path);
        unlink(input_log);
        return fail("could not create invalid UTF-8 source fixture");
    }
    char *invalid_args[] = {"kokoro", "say", "-f", invalid_path};
    passed = call_cli(4, invalid_args, &config, &exit_code, out, err) && exit_code == 2 &&
             strstr(err, "not valid UTF-8") != NULL;
    unlink(invalid_path);
    if (!passed) {
        unlink(espeak_path);
        unlink(input_log);
        return fail("invalid UTF-8 file input was not rejected clearly");
    }

    char empty_path[] = "/tmp/kokoro-empty-text-XXXXXX";
    descriptor = mkstemp(empty_path);
    if (descriptor < 0 || close(descriptor) != 0) {
        if (descriptor >= 0) {
            unlink(empty_path);
        }
        unlink(espeak_path);
        unlink(input_log);
        return fail("could not create empty text fixture");
    }
    char *empty_args[] = {"kokoro", "say", "--file", empty_path};
    passed = call_cli(4, empty_args, &config, &exit_code, out, err) && exit_code == 2 &&
             strstr(err, "text input is empty") != NULL;
    unlink(empty_path);
    if (!passed) {
        unlink(espeak_path);
        unlink(input_log);
        return fail("empty text file was not rejected clearly");
    }

    KokoroCliConfig bypass_config = config;
    bypass_config.model_pack_path = "/no/such/model.kpack";
    bypass_config.espeak_path = "/no/such/espeak-ng";
    char *ipa_args[] = {"kokoro", "say", "--ipa", "həˈloʊ"};
    passed = call_cli(4, ipa_args, &bypass_config, &exit_code, out, err) && exit_code == 1 &&
             strstr(err, "cannot load af_heart model/voice artifacts") != NULL &&
             strstr(err, "English frontend failed") == NULL;
    unlink(espeak_path);
    unlink(input_log);
    return passed ? 1 : fail("--ipa did not bypass eSpeak NG");
}

static int test_sentence_chunk_wav_combination(const KokoroCliConfig *base) {
    char espeak_path[64];
    char input_log[64];
    if (!write_espeak_stub(espeak_path, input_log, "həlˈo‍ʊ\nn̩")) {
        return fail("could not create multi-sentence eSpeak NG stub");
    }
    KokoroCliConfig config = *base;
    config.espeak_path = espeak_path;

    const char english_text[] = "Hello from the terminal.";
    char *phonemes = NULL;
    size_t phoneme_length = 0;
    KokoroError frontend_error = {0};
    KokoroStatus frontend_status =
        kokoro_english_frontend(espeak_path, english_text, sizeof(english_text) - 1, &phonemes,
                                &phoneme_length, &frontend_error);
    KokoroIpaChunks mapped_chunks = {0};
    if (frontend_status == KOKORO_OK) {
        frontend_status = kokoro_ipa_encode_chunks(phonemes, true, &mapped_chunks, &frontend_error);
    }
    const int64_t expected_first[] = {0, 50, 83, 54, 156, 57, 135, 0};
    const int64_t expected_second[] = {0, 16, 83, 56, 0};
    int mapping_passed =
        frontend_status == KOKORO_OK && phoneme_length == strlen("həlˈoʊ\nən") &&
        memcmp(phonemes, "həlˈoʊ\nən", phoneme_length) == 0 && mapped_chunks.count == 2 &&
        mapped_chunks.items[0].token_count == sizeof(expected_first) / sizeof(expected_first[0]) &&
        memcmp(mapped_chunks.items[0].token_ids, expected_first, sizeof(expected_first)) == 0 &&
        mapped_chunks.items[1].token_count ==
            sizeof(expected_second) / sizeof(expected_second[0]) &&
        memcmp(mapped_chunks.items[1].token_ids, expected_second, sizeof(expected_second)) == 0;
    free(phonemes);
    kokoro_ipa_chunks_free(&mapped_chunks);
    if (!mapping_passed) {
        unlink(espeak_path);
        unlink(input_log);
        return fail(frontend_error.message[0] == '\0'
                        ? "eSpeak output did not map to the expected Kokoro token IDs"
                        : frontend_error.message);
    }

    const char *combined_path = ".build/test-cli-sentences.wav";
    const char *first_path = ".build/test-cli-sentence-first.wav";
    const char *second_path = ".build/test-cli-sentence-second.wav";
    (void)remove(combined_path);
    (void)remove(first_path);
    (void)remove(second_path);
    char out[4096];
    char err[4096];
    int exit_code;
    char *text_args[] = {"kokoro", "say", "The first sentence. Here is another one.", "-o",
                         (char *)combined_path};
    int passed = call_cli(5, text_args, &config, &exit_code, out, err) && exit_code == 0 &&
                 verify_wav(combined_path);
    char *first_args[] = {"kokoro", "say", "--ipa", "həlˈoʊ", "-o", (char *)first_path};
    passed = passed && call_cli(6, first_args, base, &exit_code, out, err) && exit_code == 0 &&
             verify_wav(first_path);
    KokoroCliConfig second_config = *base;
    second_config.random_seed = base->random_seed + 1;
    char *second_args[] = {"kokoro", "say", "--ipa", " ən", "-o", (char *)second_path};
    passed = passed && call_cli(6, second_args, &second_config, &exit_code, out, err) &&
             exit_code == 0 && verify_wav(second_path);

    unsigned char *combined = NULL;
    unsigned char *first = NULL;
    unsigned char *second = NULL;
    size_t combined_length = 0;
    size_t first_length = 0;
    size_t second_length = 0;
    passed = passed && read_wav_payload(combined_path, &combined, &combined_length) &&
             read_wav_payload(first_path, &first, &first_length) &&
             read_wav_payload(second_path, &second, &second_length) &&
             combined_length == first_length + second_length &&
             memcmp(combined, first, first_length) == 0 &&
             memcmp(combined + first_length, second, second_length) == 0;
    free(combined);
    free(first);
    free(second);
    (void)remove(combined_path);
    (void)remove(first_path);
    (void)remove(second_path);
    unlink(espeak_path);
    unlink(input_log);
    return passed ? 1 : fail("sentence chunks were not combined in order with summed samples");
}

static int test_long_english_chunk_sample_sum(const KokoroCliConfig *base) {
    const char *source_path = "artifacts/11-af-heart-english.txt";
    const char *actual_path = ".build/test-cli-long-english.wav";
    const char *expected_path = ".build/test-cli-long-expected.wav";
    char *source = NULL;
    size_t source_length = 0;
    if (!read_text_file(source_path, &source, &source_length)) {
        return fail("could not read the long English source fixture");
    }
    char *phonemes = NULL;
    size_t phoneme_length = 0;
    KokoroError error = {0};
    KokoroStatus status = kokoro_english_frontend(base->espeak_path, source, source_length,
                                                  &phonemes, &phoneme_length, &error);
    free(source);
    if (status != KOKORO_OK) {
        return fail(error.message);
    }
    (void)phoneme_length;
    KokoroIpaChunks chunks = {0};
    status = kokoro_ipa_encode_chunks(phonemes, true, &chunks, &error);
    free(phonemes);
    if (status != KOKORO_OK || chunks.count < 2) {
        kokoro_ipa_chunks_free(&chunks);
        return fail(status == KOKORO_OK ? "long English input produced fewer than two chunks"
                                        : error.message);
    }

    KokoroEngine *engine = NULL;
    status = kokoro_engine_create(base->model_pack_path, "af_heart", NULL, &engine, &error);
    float *expected_samples = NULL;
    size_t expected_count = 0;
    size_t expected_capacity = 0;
    if (status == KOKORO_OK) {
        for (size_t index = 0; index < chunks.count; index++) {
            KokoroSynthesisOptions options = {
                .voice_style_index = chunks.items[index].token_count - 3,
                .speed = 1.0f,
                .random_seed = base->random_seed + (uint64_t)index,
            };
            KokoroSynthesisOutput output = {0};
            status = kokoro_engine_synthesize(engine, chunks.items[index].token_ids,
                                              chunks.items[index].token_count, &options, &output,
                                              &error);
            if (status == KOKORO_OK &&
                !append_float_samples(&expected_samples, &expected_count, &expected_capacity,
                                      output.samples, output.sample_count)) {
                error.code = KOKORO_OUT_OF_MEMORY;
                (void)snprintf(error.message, sizeof(error.message),
                               "cannot allocate expected long-input audio");
                status = KOKORO_OUT_OF_MEMORY;
            }
            kokoro_synthesis_output_free(&output);
            if (status != KOKORO_OK) {
                break;
            }
        }
    }
    kokoro_engine_free(engine);
    if (status != KOKORO_OK || expected_count == 0) {
        free(expected_samples);
        kokoro_ipa_chunks_free(&chunks);
        return fail(error.message);
    }

    (void)remove(actual_path);
    (void)remove(expected_path);
    char out[4096];
    char err[4096];
    int exit_code;
    char *cli_args[] = {"kokoro", "say", "-f", (char *)source_path, "-o", (char *)actual_path};
    int passed = call_cli(6, cli_args, base, &exit_code, out, err) && exit_code == 0 &&
                 verify_wav(actual_path) &&
                 kokoro_wav_write_pcm16(expected_path, expected_samples, expected_count,
                                        KOKORO_ISTFT_SAMPLE_RATE_HZ, &error) == KOKORO_OK &&
                 verify_wav(expected_path);
    unsigned char *actual_payload = NULL;
    unsigned char *expected_payload = NULL;
    size_t actual_length = 0;
    size_t expected_length = 0;
    passed = passed && read_wav_payload(actual_path, &actual_payload, &actual_length) &&
             read_wav_payload(expected_path, &expected_payload, &expected_length) &&
             actual_length == expected_count * 2 && expected_length == actual_length &&
             memcmp(actual_payload, expected_payload, expected_length) == 0;
    free(actual_payload);
    free(expected_payload);
    free(expected_samples);
    kokoro_ipa_chunks_free(&chunks);
    (void)remove(actual_path);
    (void)remove(expected_path);
    return passed
               ? 1
               : fail("long English WAV did not match every ordered chunk and summed sample count");
}

int main(void) {
    KokoroCliConfig config = {
        .model_pack_path = MODEL_PACK_PATH,
        .voices_directory = VOICES_DIR,
        .player_path = "afplay",
        .espeak_path = "espeak-ng",
        .random_seed = 104729,
    };
    if (!test_help_and_argument_errors(&config)) {
        return 1;
    }
    if (!test_model_artifact_errors(&config)) {
        return 1;
    }
    if (access(MODEL_PACK_PATH, R_OK) != 0) {
        puts("SKIP model-backed CLI checks: convert the pinned model pack first");
        return 0;
    }
    if (!test_voice_listing(&config)) {
        return 1;
    }
    if (!test_voice_artifact_errors(&config)) {
        return 1;
    }
    if (!test_synthesis_and_playback(&config)) {
        return 1;
    }
    if (!test_stream_cli(&config)) {
        return 1;
    }
    if (!test_text_sources_and_ipa_bypass(&config)) {
        return 1;
    }
    if (!test_sentence_chunk_wav_combination(&config)) {
        return 1;
    }
    if (!test_long_english_chunk_sample_sum(&config)) {
        return 1;
    }
    puts("English and IPA CLI WAV, source-selection, chunking, and playback checks passed");
    return 0;
}

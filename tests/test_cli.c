#include "cli.h"

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
        strstr(out, "--ipa TEXT") == NULL || strstr(out, "af_heart (default)") == NULL ||
        strstr(out, "afplay") == NULL) {
        return fail("say help omitted required input or default details");
    }
    char *version_args[] = {"kokoro", "--version"};
    if (!call_cli(2, version_args, config, &exit_code, out, err) || exit_code != 0 ||
        strstr(out, "kokoro 0.1.0") == NULL) {
        return fail("version output is missing");
    }
    char *missing_ipa_args[] = {"kokoro", "say"};
    if (!call_cli(2, missing_ipa_args, config, &exit_code, out, err) || exit_code != 0) {
        return fail("bare say should show usage help");
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

int main(void) {
    KokoroCliConfig config = {
        .model_pack_path = MODEL_PACK_PATH,
        .voices_directory = VOICES_DIR,
        .player_path = "afplay",
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
    puts("IPA CLI WAV output and playback cleanup checks passed");
    return 0;
}

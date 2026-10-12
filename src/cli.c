#include "cli.h"

#include "ipa_tokenizer.h"
#include "istft.h"
#include "synthesis.h"
#include "voice_artifact.h"
#include "wav_writer.h"

#include <errno.h>
#include <math.h>
#include <spawn.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

#define DEFAULT_MODEL_PACK "models/kokoro-v1_0-af_heart.kpack"
#define DEFAULT_VOICES_DIR "models/voices"
#define DEFAULT_PLAYER "afplay"
#define KOKORO_VERSION "0.1.0"

typedef struct {
    const char *model_pack_path;
    const char *voices_directory;
    const char *player_path;
    uint64_t random_seed;
    FILE *out;
    FILE *err;
} CliConfig;

static CliConfig cli_config(const KokoroCliConfig *config) {
    return (CliConfig){
        .model_pack_path = config != NULL && config->model_pack_path != NULL
                               ? config->model_pack_path
                               : DEFAULT_MODEL_PACK,
        .voices_directory = config != NULL && config->voices_directory != NULL
                                ? config->voices_directory
                                : DEFAULT_VOICES_DIR,
        .player_path =
            config != NULL && config->player_path != NULL ? config->player_path : DEFAULT_PLAYER,
        .random_seed = config != NULL && config->random_seed != 0 ? config->random_seed : 104729,
        .out = config != NULL && config->stdout_stream != NULL ? config->stdout_stream : stdout,
        .err = config != NULL && config->stderr_stream != NULL ? config->stderr_stream : stderr,
    };
}

static void print_help(FILE *stream) {
    fputs("Usage:\n"
          "  kokoro say --ipa TEXT [-v VOICE] [-s SPEED] [-o WAV]\n"
          "  kokoro --list-voices\n"
          "\n"
          "Synthesize prepared IPA/phoneme input with the native Kokoro v1.0 model.\n"
          "\n"
          "Options for `say`:\n"
          "  --ipa TEXT       Prepared phonemes from the pinned Kokoro vocabulary\n"
          "  -v, --voice NAME Voice: af_heart (default) or am_adam\n"
          "  -s, --speed NUM  Speed from 0.5 through 2.0 (default: 1.0)\n"
          "  -o, --output WAV Write mono 24 kHz WAV instead of playing it\n"
          "  --help           Show this help\n"
          "  --version        Show version\n"
          "\n"
          "Without --output, the completed WAV is played with macOS afplay.\n"
          "Model pack: models/kokoro-v1_0-af_heart.kpack\n"
          "Separate am_adam style: models/voices/am_adam.kvoice\n",
          stream);
}

static int report_error(const CliConfig *config, const char *message) {
    fprintf(config->err, "kokoro: %s\n", message);
    return 1;
}

static char *voice_artifact_path(const char *voices_directory) {
    static const char filename[] = "am_adam.kvoice";
    size_t directory_length = strlen(voices_directory);
    bool separator = directory_length == 0 || voices_directory[directory_length - 1] != '/';
    if (directory_length > SIZE_MAX - sizeof(filename) - (separator ? 1U : 0U)) {
        return NULL;
    }
    size_t path_size = directory_length + (separator ? 1U : 0U) + sizeof(filename);
    char *path = malloc(path_size);
    if (path == NULL) {
        return NULL;
    }
    (void)snprintf(path, path_size, "%s%s%s", voices_directory, separator ? "/" : "", filename);
    return path;
}

static int list_voices(const CliConfig *config) {
    char *adam_path = voice_artifact_path(config->voices_directory);
    if (adam_path == NULL) {
        return report_error(config, "cannot allocate voice artifact path");
    }
    KokoroPack *pack = NULL;
    KokoroError error = {0};
    bool model_compatible = kokoro_pack_open(config->model_pack_path, &pack, &error) == KOKORO_OK;
    KokoroTensorView af_heart_view;
    if (model_compatible) {
        model_compatible = kokoro_pack_voice_style(pack, NULL, &af_heart_view, &error) == KOKORO_OK;
    }
    bool adam_compatible = false;
    KokoroVoiceArtifact *adam = NULL;
    if (model_compatible &&
        kokoro_voice_artifact_open(adam_path, "am_adam", &adam, &error) == KOKORO_OK) {
        KokoroTensorView adam_view;
        adam_compatible = kokoro_voice_artifact_tensor(adam, &adam_view, &error) == KOKORO_OK;
    }
    int result = 0;
    if (model_compatible) {
        if (fprintf(config->out, "af_heart\n") < 0) {
            result = 1;
        }
        if (adam_compatible && fprintf(config->out, "am_adam\n") < 0) {
            result = 1;
        }
    } else if (fputs("No supported voices installed.\n", config->out) == EOF) {
        result = 1;
    }
    kokoro_voice_artifact_close(adam);
    kokoro_pack_close(pack);
    free(adam_path);
    return result;
}

static bool parse_speed(const char *text, float *speed) {
    if (text == NULL || text[0] == '\0') {
        return false;
    }
    errno = 0;
    char *end = NULL;
    float value = strtof(text, &end);
    if (errno == ERANGE || end == text || *end != '\0' || !isfinite(value) || value < 0.5f ||
        value > 2.0f) {
        return false;
    }
    *speed = value;
    return true;
}

static bool run_player(const char *player_path, const char *wav_path, char error_message[256]) {
    char *const arguments[] = {(char *)player_path, (char *)wav_path, NULL};
    pid_t process;
    int spawn_error = posix_spawnp(&process, player_path, NULL, NULL, arguments, environ);
    if (spawn_error != 0) {
        (void)snprintf(error_message, 256, "cannot start afplay: %s", strerror(spawn_error));
        return false;
    }
    int wait_status;
    pid_t waited;
    do {
        waited = waitpid(process, &wait_status, 0);
    } while (waited < 0 && errno == EINTR);
    if (waited < 0) {
        (void)snprintf(error_message, 256, "cannot wait for afplay: %s", strerror(errno));
        return false;
    }
    if (WIFEXITED(wait_status) && WEXITSTATUS(wait_status) == 0) {
        return true;
    }
    if (WIFEXITED(wait_status)) {
        (void)snprintf(error_message, 256, "afplay exited with status %d",
                       WEXITSTATUS(wait_status));
    } else if (WIFSIGNALED(wait_status)) {
        (void)snprintf(error_message, 256, "afplay terminated by signal %d", WTERMSIG(wait_status));
    } else {
        (void)snprintf(error_message, 256, "afplay did not complete successfully");
    }
    return false;
}

static int write_and_play(const CliConfig *config, const float *samples, size_t sample_count) {
    char wav_path[] = "/tmp/kokoro-say-XXXXXX";
    int descriptor = mkstemp(wav_path);
    if (descriptor < 0) {
        fprintf(config->err, "kokoro: cannot create temporary WAV: %s\n", strerror(errno));
        return 1;
    }
    FILE *file = fdopen(descriptor, "wb");
    if (file == NULL) {
        int saved_errno = errno;
        (void)close(descriptor);
        (void)unlink(wav_path);
        fprintf(config->err, "kokoro: cannot open temporary WAV stream: %s\n",
                strerror(saved_errno));
        return 1;
    }
    KokoroError error = {0};
    KokoroStatus status = kokoro_wav_write_pcm16_stream(file, samples, sample_count,
                                                        KOKORO_ISTFT_SAMPLE_RATE_HZ, &error);
    if (status == KOKORO_OK && fflush(file) != 0) {
        status = KOKORO_IO_ERROR;
        (void)snprintf(error.message, sizeof(error.message), "failed flushing temporary WAV");
    }
    if (fclose(file) != 0 && status == KOKORO_OK) {
        status = KOKORO_IO_ERROR;
        (void)snprintf(error.message, sizeof(error.message), "failed closing temporary WAV");
    }
    if (status != KOKORO_OK) {
        (void)unlink(wav_path);
        fprintf(config->err, "kokoro: cannot write temporary WAV: %s\n", error.message);
        return 1;
    }

    char playback_error[256];
    bool played = run_player(config->player_path, wav_path, playback_error);
    int cleanup_error = unlink(wav_path);
    if (!played) {
        fprintf(config->err, "kokoro: %s\n", playback_error);
    }
    if (cleanup_error != 0) {
        fprintf(config->err, "kokoro: cannot remove temporary WAV: %s\n", strerror(errno));
        return 1;
    }
    return played ? 0 : 1;
}

static int fail_arguments(const CliConfig *config, const char *message) {
    fprintf(config->err, "kokoro: %s\n", message);
    fputs("Run `kokoro say --help` for usage.\n", config->err);
    return 2;
}

int kokoro_cli_run(int argc, char **argv, const KokoroCliConfig *input_config) {
    CliConfig config = cli_config(input_config);
    if (argc <= 1 || strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        print_help(config.out);
        return 0;
    }
    if (strcmp(argv[1], "--version") == 0) {
        fprintf(config.out, "kokoro %s\n", KOKORO_VERSION);
        return 0;
    }
    if (strcmp(argv[1], "--list-voices") == 0) {
        return list_voices(&config);
    }
    if (strcmp(argv[1], "say") != 0) {
        return fail_arguments(&config, "expected the `say` command or --list-voices");
    }
    if (argc == 2 || strcmp(argv[2], "--help") == 0 || strcmp(argv[2], "-h") == 0) {
        print_help(config.out);
        return 0;
    }
    if (strcmp(argv[2], "--version") == 0) {
        fprintf(config.out, "kokoro %s\n", KOKORO_VERSION);
        return 0;
    }
    if (strcmp(argv[2], "--list-voices") == 0 && argc == 3) {
        return list_voices(&config);
    }

    const char *phonemes = NULL;
    const char *voice = "af_heart";
    const char *output_path = NULL;
    float speed = 1.0f;
    for (int index = 2; index < argc; index++) {
        const char *option = argv[index];
        if (strcmp(option, "--ipa") == 0) {
            if (phonemes != NULL || index + 1 >= argc) {
                return fail_arguments(&config, "--ipa requires one nonempty phoneme string");
            }
            phonemes = argv[++index];
        } else if (strcmp(option, "-v") == 0 || strcmp(option, "--voice") == 0) {
            if (index + 1 >= argc) {
                return fail_arguments(&config, "--voice requires a name");
            }
            voice = argv[++index];
        } else if (strcmp(option, "-s") == 0 || strcmp(option, "--speed") == 0) {
            if (index + 1 >= argc || !parse_speed(argv[index + 1], &speed)) {
                return fail_arguments(&config, "--speed must be finite and between 0.5 and 2.0");
            }
            index++;
        } else if (strcmp(option, "-o") == 0 || strcmp(option, "--output") == 0) {
            if (output_path != NULL || index + 1 >= argc || argv[index + 1][0] == '\0') {
                return fail_arguments(&config, "--output requires one path");
            }
            output_path = argv[++index];
        } else if (strcmp(option, "--list-voices") == 0) {
            return fail_arguments(&config,
                                  "--list-voices cannot be combined with synthesis options");
        } else if (strcmp(option, "--help") == 0 || strcmp(option, "--version") == 0) {
            return fail_arguments(&config, "help and version flags must be used alone");
        } else {
            return fail_arguments(&config, "unknown option");
        }
    }
    if (phonemes == NULL || phonemes[0] == '\0') {
        return fail_arguments(&config, "say requires --ipa with a nonempty phoneme string");
    }
    if (strcmp(voice, "af_heart") != 0 && strcmp(voice, "am_adam") != 0) {
        return fail_arguments(&config, "unknown voice; choose af_heart or am_adam");
    }

    int64_t *token_ids = NULL;
    size_t token_count = 0;
    KokoroError error = {0};
    KokoroStatus status = kokoro_ipa_encode(phonemes, &token_ids, &token_count, &error);
    if (status != KOKORO_OK) {
        fprintf(config.err, "kokoro: invalid IPA input: %s\n", error.message);
        return 2;
    }
    char *adam_path = NULL;
    const char *voice_path = NULL;
    if (strcmp(voice, "am_adam") == 0) {
        adam_path = voice_artifact_path(config.voices_directory);
        if (adam_path == NULL) {
            free(token_ids);
            return report_error(&config, "cannot allocate voice artifact path");
        }
        voice_path = adam_path;
    }
    KokoroEngine *engine = NULL;
    status = kokoro_engine_create(config.model_pack_path, voice, voice_path, &engine, &error);
    free(adam_path);
    if (status != KOKORO_OK) {
        free(token_ids);
        fprintf(config.err, "kokoro: cannot load %s model/voice artifacts: %s\n", voice,
                error.message);
        return 1;
    }
    KokoroSynthesisOptions options = {
        .voice_style_index = token_count - 3,
        .speed = speed,
        .random_seed = config.random_seed,
    };
    KokoroSynthesisOutput output = {0};
    status = kokoro_engine_synthesize(engine, token_ids, token_count, &options, &output, &error);
    free(token_ids);
    kokoro_engine_free(engine);
    if (status != KOKORO_OK) {
        fprintf(config.err, "kokoro: synthesis failed: %s\n", error.message);
        kokoro_synthesis_output_free(&output);
        return 1;
    }
    int result = output_path != NULL
                     ? (kokoro_wav_write_pcm16(output_path, output.samples, output.sample_count,
                                               KOKORO_ISTFT_SAMPLE_RATE_HZ, &error) == KOKORO_OK
                            ? 0
                            : 1)
                     : write_and_play(&config, output.samples, output.sample_count);
    if (result != 0 && output_path != NULL) {
        fprintf(config.err, "kokoro: cannot write WAV output: %s\n", error.message);
    }
    kokoro_synthesis_output_free(&output);
    return result;
}

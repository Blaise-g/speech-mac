#include "cli.h"

#include "english_frontend.h"
#include "ipa_tokenizer.h"
#include "istft.h"
#include "kokoro_internal.h"
#include "stream.h"
#include "stream_internal.h"
#include "synthesis.h"
#include "utf8.h"
#include "voice_artifact.h"
#include "wav_writer.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <signal.h>
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
#define DEFAULT_ESPEAK "espeak-ng"
#define KOKORO_VERSION "0.1.0"

typedef struct {
    const char *model_pack_path;
    const char *voices_directory;
    const char *player_path;
    const char *espeak_path;
    const KokoroStreamPlaybackOps *stream_playback_ops;
    void *stream_playback_context;
    uint64_t random_seed;
    FILE *in;
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
        .espeak_path =
            config != NULL && config->espeak_path != NULL ? config->espeak_path : DEFAULT_ESPEAK,
        .stream_playback_ops = config != NULL ? config->stream_playback_ops : NULL,
        .stream_playback_context = config != NULL ? config->stream_playback_context : NULL,
        .random_seed = config != NULL && config->random_seed != 0 ? config->random_seed : 104729,
        .in = config != NULL && config->stdin_stream != NULL ? config->stdin_stream : stdin,
        .out = config != NULL && config->stdout_stream != NULL ? config->stdout_stream : stdout,
        .err = config != NULL && config->stderr_stream != NULL ? config->stderr_stream : stderr,
    };
}

static void print_help(FILE *stream) {
    fputs("Usage:\n"
          "  kokoro say TEXT [-v VOICE] [-s SPEED] [-o WAV | --stream]\n"
          "  kokoro say -f, --file PATH [--ipa] [-v VOICE] [-s SPEED] [-o WAV | --stream]\n"
          "  ... | kokoro say [--ipa] [-v VOICE] [-s SPEED] [-o WAV | --stream]\n"
          "  kokoro --list-voices\n"
          "\n"
          "Synthesize English text with eSpeak NG and the native Kokoro v1.0 model.\n"
          "\n"
          "Options for `say`:\n"
          "  -f, --file PATH  Read UTF-8 text from a file (default source: standard input)\n"
          "  --ipa            Treat the selected source as prepared Kokoro phonemes\n"
          "  -v, --voice NAME Voice: af_heart (default) or am_adam\n"
          "  -s, --speed NUM  Speed from 0.5 through 2.0 (default: 1.0)\n"
          "  -o, --output WAV Write mono 24 kHz WAV instead of playing it\n"
          "  --stream         Play each complete synthesized sentence as it becomes ready\n"
          "  --help           Show this help\n"
          "  --version        Show version\n"
          "\n"
          "Without --output or --stream, the completed WAV is played with macOS afplay.\n"
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

static volatile sig_atomic_t stream_interrupted;

static void record_stream_interrupt(int signal_number) {
    (void)signal_number;
    stream_interrupted = 1;
}

static int run_stream(const CliConfig *config, KokoroEngine *engine, const KokoroIpaChunks *chunks,
                      float speed, uint64_t random_seed) {
    struct sigaction action = {0};
    struct sigaction previous_action;
    stream_interrupted = 0;
    action.sa_handler = record_stream_interrupt;
    (void)sigemptyset(&action.sa_mask);
    if (sigaction(SIGINT, &action, &previous_action) != 0) {
        fprintf(config->err, "kokoro: cannot install Ctrl-C handler: %s\n", strerror(errno));
        return 1;
    }
    KokoroStream *stream = NULL;
    KokoroError error = {0};
    KokoroStatus status =
        config->stream_playback_ops == NULL
            ? kokoro_stream_start(engine, chunks, speed, random_seed, &stream, &error)
            : kokoro_stream_start_with(engine, chunks, speed, random_seed,
                                       config->stream_playback_ops, config->stream_playback_context,
                                       NULL, NULL, &stream, &error);
    if (status == KOKORO_OK) {
        status = kokoro_stream_wait(stream, &stream_interrupted, &error);
    }
    kokoro_stream_free(stream);
    int restore_error = sigaction(SIGINT, &previous_action, NULL) == 0 ? 0 : errno;
    if (status == KOKORO_CANCELLED) {
        fprintf(config->err, "kokoro: streaming playback cancelled\n");
        return 130;
    }
    if (status != KOKORO_OK) {
        fprintf(config->err, "kokoro: streaming playback failed: %s\n", error.message);
        return 1;
    }
    if (restore_error != 0) {
        fprintf(config->err, "kokoro: cannot restore Ctrl-C handler: %s\n",
                strerror(restore_error));
        return 1;
    }
    return 0;
}

typedef enum { INPUT_SOURCE_NONE, INPUT_SOURCE_POSITIONAL, INPUT_SOURCE_FILE } InputSourceKind;

typedef struct {
    char *text;
    size_t length;
} TextInput;

typedef struct {
    float *samples;
    size_t count;
    size_t capacity;
} SampleBuffer;

static bool valid_utf8(const unsigned char *text, size_t length) {
    return memchr(text, '\0', length) == NULL && kokoro_utf8_is_valid(text, length);
}

static bool text_is_blank(const char *text, size_t length) {
    for (size_t index = 0; index < length; index++) {
        if (text[index] != ' ' && text[index] != '\t' && text[index] != '\n' &&
            text[index] != '\r' && text[index] != '\v' && text[index] != '\f') {
            return false;
        }
    }
    return true;
}

static bool read_stream_text(FILE *stream, TextInput *input, char message[256]) {
    char *buffer = NULL;
    size_t used = 0;
    size_t capacity = 0;
    unsigned char block[8192];
    for (;;) {
        size_t count = fread(block, 1, sizeof(block), stream);
        if (count > 0) {
            if (count > SIZE_MAX - used - 1) {
                free(buffer);
                (void)snprintf(message, 256, "text input is too large");
                return false;
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
                    (void)snprintf(message, 256, "cannot allocate text input");
                    return false;
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
                (void)snprintf(message, 256, "cannot read text input");
                return false;
            }
            break;
        }
    }
    if (buffer == NULL) {
        buffer = malloc(1);
        if (buffer == NULL) {
            (void)snprintf(message, 256, "cannot allocate text input");
            return false;
        }
    }
    buffer[used] = '\0';
    if (memchr(buffer, '\0', used) != NULL || !valid_utf8((const unsigned char *)buffer, used)) {
        free(buffer);
        (void)snprintf(message, 256, "text input is not valid UTF-8 or contains a NUL byte");
        return false;
    }
    if (used == 0 || text_is_blank(buffer, used)) {
        free(buffer);
        (void)snprintf(message, 256, "text input is empty");
        return false;
    }
    input->text = buffer;
    input->length = used;
    return true;
}

static bool load_text_input(const CliConfig *config, InputSourceKind source_kind,
                            const char *source_value, TextInput *input, char message[256]) {
    if (source_kind == INPUT_SOURCE_POSITIONAL) {
        size_t length = strlen(source_value);
        if (length == 0 || text_is_blank(source_value, length)) {
            (void)snprintf(message, 256, "text input is empty");
            return false;
        }
        if (!valid_utf8((const unsigned char *)source_value, length)) {
            (void)snprintf(message, 256, "text input is not valid UTF-8");
            return false;
        }
        input->text = malloc(length + 1);
        if (input->text == NULL) {
            (void)snprintf(message, 256, "cannot allocate text input");
            return false;
        }
        memcpy(input->text, source_value, length + 1);
        input->length = length;
        return true;
    }
    if (source_kind == INPUT_SOURCE_FILE) {
        FILE *file = fopen(source_value, "rb");
        if (file == NULL) {
            (void)snprintf(message, 256, "cannot open text file '%s': %s", source_value,
                           strerror(errno));
            return false;
        }
        bool read_ok = read_stream_text(file, input, message);
        if (fclose(file) != 0 && read_ok) {
            free(input->text);
            input->text = NULL;
            input->length = 0;
            (void)snprintf(message, 256, "cannot close text file '%s': %s", source_value,
                           strerror(errno));
            return false;
        }
        return read_ok;
    }
    return read_stream_text(config->in, input, message);
}

static bool append_samples(SampleBuffer *buffer, const float *samples, size_t count,
                           KokoroError *error) {
    const size_t wav_sample_limit = (size_t)((UINT32_MAX - 36U) / 2U);
    if (samples == NULL || count == 0 || buffer->count > wav_sample_limit ||
        count > wav_sample_limit - buffer->count || count > SIZE_MAX - buffer->count) {
        kokoro_error_set(error, KOKORO_INVALID_ARGUMENT,
                         "combined audio exceeds the WAV sample-count limit");
        return false;
    }
    size_t required = buffer->count + count;
    if (required > buffer->capacity) {
        size_t grown = buffer->capacity == 0 ? count : buffer->capacity;
        while (grown < required) {
            if (grown > SIZE_MAX / 2) {
                grown = required;
                break;
            }
            grown *= 2;
        }
        if (grown > SIZE_MAX / sizeof(*buffer->samples)) {
            kokoro_error_set(error, KOKORO_OUT_OF_MEMORY, "combined audio is too large");
            return false;
        }
        float *next = realloc(buffer->samples, grown * sizeof(*next));
        if (next == NULL) {
            kokoro_error_set(error, KOKORO_OUT_OF_MEMORY, "cannot allocate combined audio");
            return false;
        }
        buffer->samples = next;
        buffer->capacity = grown;
    }
    memcpy(buffer->samples + buffer->count, samples, count * sizeof(*samples));
    buffer->count = required;
    return true;
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
    if (argc > 2 && (strcmp(argv[2], "--help") == 0 || strcmp(argv[2], "-h") == 0)) {
        print_help(config.out);
        return 0;
    }
    if (argc > 2 && strcmp(argv[2], "--version") == 0) {
        if (argc == 3) {
            fprintf(config.out, "kokoro %s\n", KOKORO_VERSION);
            return 0;
        }
        return fail_arguments(&config, "help and version flags must be used alone");
    }
    if (argc == 3 && strcmp(argv[2], "--list-voices") == 0) {
        return list_voices(&config);
    }

    InputSourceKind source_kind = INPUT_SOURCE_NONE;
    const char *source_value = NULL;
    const char *voice = "af_heart";
    const char *output_path = NULL;
    bool ipa_mode = false;
    bool stream_mode = false;
    bool after_double_dash = false;
    float speed = 1.0f;
    for (int index = 2; index < argc; index++) {
        const char *option = argv[index];
        if (!after_double_dash && strcmp(option, "--") == 0) {
            after_double_dash = true;
        } else if (!after_double_dash && strcmp(option, "--ipa") == 0) {
            if (ipa_mode) {
                return fail_arguments(&config, "--ipa may be specified only once");
            }
            ipa_mode = true;
        } else if (!after_double_dash && strcmp(option, "--stream") == 0) {
            if (stream_mode) {
                return fail_arguments(&config, "--stream may be specified only once");
            }
            stream_mode = true;
        } else if (!after_double_dash &&
                   (strcmp(option, "-f") == 0 || strcmp(option, "--file") == 0)) {
            if (index + 1 >= argc || argv[index + 1][0] == '\0') {
                return fail_arguments(&config, "--file requires a path");
            }
            if (source_kind != INPUT_SOURCE_NONE) {
                return fail_arguments(&config, "provide exactly one text source");
            }
            source_kind = INPUT_SOURCE_FILE;
            source_value = argv[++index];
        } else if (!after_double_dash &&
                   (strcmp(option, "-v") == 0 || strcmp(option, "--voice") == 0)) {
            if (index + 1 >= argc) {
                return fail_arguments(&config, "--voice requires a name");
            }
            voice = argv[++index];
        } else if (!after_double_dash &&
                   (strcmp(option, "-s") == 0 || strcmp(option, "--speed") == 0)) {
            if (index + 1 >= argc || !parse_speed(argv[index + 1], &speed)) {
                return fail_arguments(&config, "--speed must be finite and between 0.5 and 2.0");
            }
            index++;
        } else if (!after_double_dash &&
                   (strcmp(option, "-o") == 0 || strcmp(option, "--output") == 0)) {
            if (output_path != NULL || index + 1 >= argc || argv[index + 1][0] == '\0') {
                return fail_arguments(&config, "--output requires one path");
            }
            output_path = argv[++index];
        } else if (!after_double_dash && strcmp(option, "--list-voices") == 0) {
            return fail_arguments(&config,
                                  "--list-voices cannot be combined with synthesis options");
        } else if (!after_double_dash &&
                   (strcmp(option, "--help") == 0 || strcmp(option, "--version") == 0)) {
            return fail_arguments(&config, "help and version flags must be used alone");
        } else if (!after_double_dash && option[0] == '-') {
            return fail_arguments(&config, "unknown option");
        } else {
            if (source_kind != INPUT_SOURCE_NONE) {
                return fail_arguments(&config, "provide exactly one text source");
            }
            source_kind = INPUT_SOURCE_POSITIONAL;
            source_value = option;
        }
    }
    if (strcmp(voice, "af_heart") != 0 && strcmp(voice, "am_adam") != 0) {
        return fail_arguments(&config, "unknown voice; choose af_heart or am_adam");
    }
    if (stream_mode && output_path != NULL) {
        return fail_arguments(&config, "--stream cannot be combined with --output");
    }

    TextInput input = {0};
    char input_message[256];
    if (!load_text_input(&config, source_kind, source_value, &input, input_message)) {
        fprintf(config.err, "kokoro: invalid text input: %s\n", input_message);
        return 2;
    }
    char *phonemes = NULL;
    size_t phoneme_length = 0;
    KokoroError error = {0};
    KokoroStatus status = KOKORO_OK;
    if (ipa_mode) {
        phonemes = input.text;
        phoneme_length = input.length;
        input.text = NULL;
    } else {
        status = kokoro_english_frontend(config.espeak_path, input.text, input.length, &phonemes,
                                         &phoneme_length, &error);
    }
    free(input.text);
    if (status != KOKORO_OK) {
        fprintf(config.err, "kokoro: English frontend failed: %s\n", error.message);
        return 2;
    }
    (void)phoneme_length;
    KokoroIpaChunks chunks = {0};
    status = kokoro_ipa_encode_chunks(phonemes, !ipa_mode, &chunks, &error);
    free(phonemes);
    if (status != KOKORO_OK) {
        fprintf(config.err, "kokoro: %s input: %s\n",
                ipa_mode ? "invalid IPA" : "unmappable eSpeak", error.message);
        return 2;
    }

    char *adam_path = NULL;
    const char *voice_path = NULL;
    if (strcmp(voice, "am_adam") == 0) {
        adam_path = voice_artifact_path(config.voices_directory);
        if (adam_path == NULL) {
            kokoro_ipa_chunks_free(&chunks);
            return report_error(&config, "cannot allocate voice artifact path");
        }
        voice_path = adam_path;
    }
    KokoroEngine *engine = NULL;
    status = kokoro_engine_create(config.model_pack_path, voice, voice_path, &engine, &error);
    free(adam_path);
    if (status != KOKORO_OK) {
        kokoro_ipa_chunks_free(&chunks);
        fprintf(config.err, "kokoro: cannot load %s model/voice artifacts: %s\n", voice,
                error.message);
        return 1;
    }
    SampleBuffer combined = {0};
    int result = 0;
    if (stream_mode) {
        result = run_stream(&config, engine, &chunks, speed, config.random_seed);
        kokoro_engine_free(engine);
        kokoro_ipa_chunks_free(&chunks);
        return result;
    }
    for (size_t index = 0; index < chunks.count; index++) {
        KokoroSynthesisOptions options = {
            .voice_style_index = chunks.items[index].token_count - 3,
            .speed = speed,
            .random_seed = config.random_seed + (uint64_t)index,
        };
        KokoroSynthesisOutput output = {0};
        status =
            kokoro_engine_synthesize(engine, chunks.items[index].token_ids,
                                     chunks.items[index].token_count, &options, &output, &error);
        if (status != KOKORO_OK ||
            !append_samples(&combined, output.samples, output.sample_count, &error)) {
            fprintf(config.err, "kokoro: synthesis failed: %s\n", error.message);
            result = 1;
            kokoro_synthesis_output_free(&output);
            break;
        }
        kokoro_synthesis_output_free(&output);
    }
    kokoro_engine_free(engine);
    kokoro_ipa_chunks_free(&chunks);
    if (result == 0) {
        result = output_path != NULL
                     ? (kokoro_wav_write_pcm16(output_path, combined.samples, combined.count,
                                               KOKORO_ISTFT_SAMPLE_RATE_HZ, &error) == KOKORO_OK
                            ? 0
                            : 1)
                     : write_and_play(&config, combined.samples, combined.count);
        if (result != 0 && output_path != NULL) {
            fprintf(config.err, "kokoro: cannot write WAV output: %s\n", error.message);
        }
    }
    free(combined.samples);
    return result;
}

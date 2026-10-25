#define _POSIX_C_SOURCE 200809L
#include "speech_cli.h"
#include "speech.h"
#include "speech_stream.h"
#include "wav_writer.h"
#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <spawn.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;

static int output_status(FILE *out) { return fflush(out) == 0 && !ferror(out) ? 0 : 1; }

static int player(void *context, const char *path) {
    (void)context;
    pid_t pid;
    char *args[] = {"/usr/bin/afplay", (char *)path, NULL};
    if (posix_spawn(&pid, args[0], NULL, NULL, args, environ))
        return 0;
    int status;
    while (waitpid(pid, &status, 0) < 0)
        if (errno != EINTR)
            return 0;
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}
static int name_ok(const char *name) {
    if (!*name || strlen(name) > 128)
        return 0;
    for (const unsigned char *p = (const unsigned char *)name; *p; ++p)
        if (!((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') || (*p >= '0' && *p <= '9') ||
              *p == '-' || *p == '_'))
            return 0;
    return 1;
}
static char *voice_path(const char *directory, const char *name) {
    size_t n = strlen(directory) + strlen(name) + 16;
    char *path = malloc(n);
    if (path)
        snprintf(path, n, "%s/%s.safetensors", directory, name);
    return path;
}
static char *read_text(FILE *file, size_t *length) {
    char *text = malloc(SPEECH_MAX_INPUT_BYTES + 1);
    if (!text)
        return NULL;
    *length = fread(text, 1, SPEECH_MAX_INPUT_BYTES + 1, file);
    if (ferror(file) || !*length || *length > SPEECH_MAX_INPUT_BYTES || memchr(text, 0, *length)) {
        free(text);
        return NULL;
    }
    text[*length] = 0;
    return text;
}
int speech_cli_run(int argc, char **argv, const SpeechCliConfig *config) {
    FILE *out = config && config->output ? config->output : stdout;
    FILE *err = config && config->error ? config->error : stderr;
    const char *directory =
        config && config->voices_directory ? config->voices_directory : "voices";
    const char *pack =
        config && config->pack_path ? config->pack_path : "models/phonon-40m-en.kpack";
    if (argc == 2 && !strcmp(argv[1], "--help")) {
        fprintf(out,
                "speech-mac say [--file PATH | --stdin | TEXT] [--output WAV | --stream]\n"
                "  --voice NAME | --voice-file PATH   compatible local voice (default: default)\n"
                "  --model PATH --voices-dir PATH --language en --seed UINT64\n"
                "  --duration-delta FLOAT   native duration conditioning, default 0\n"
                "speech-mac voices [--voices-dir PATH]\n"
                "Without --output, play the completed mono 24 kHz PCM16 WAV locally.\n"
                "--stream plays progressive frames with bounded AudioQueue buffering.\n");
        return output_status(out);
    }
    if (argc == 2 && !strcmp(argv[1], "--version")) {
        fprintf(out, "speech-mac 0.1.0\n");
        return output_status(out);
    }
    if (argc < 2 || (strcmp(argv[1], "say") && strcmp(argv[1], "voices"))) {
        fprintf(err, "Use speech-mac say or voices; see --help.\n");
        return 2;
    }
    const char *text_arg = NULL, *input_path = NULL, *output_path = NULL, *name = "default",
               *explicit_voice = NULL;
    int stdin_selected = 0, end_options = 0, streaming = 0;
    unsigned seen = 0;
    SpeechOptions options = speech_options_default();
    for (int i = 2; i < argc; ++i) {
        const char *arg = argv[i];
        unsigned bit = 0;
        if (!end_options && !strcmp(arg, "--")) {
            end_options = 1;
            continue;
        }
        if (end_options) {
            if (text_arg)
                goto invalid;
            text_arg = arg;
            continue;
        }
        if (!strcmp(arg, "--speed") || !strcmp(arg, "-s") || !strcmp(arg, "--ipa")) {
            fprintf(err,
                    "%s is a removed Kokoro option; supply ordinary English text and native "
                    "--duration-delta (not speed).\n",
                    arg);
            return 2;
        }
        if (!strcmp(arg, "--stream")) {
            if (streaming)
                goto invalid;
            streaming = 1;
            continue;
        }
        if (!strcmp(arg, "--stdin")) {
            bit = 1;
            if (seen & bit)
                goto invalid;
            seen |= bit;
            stdin_selected = 1;
            continue;
        }
        const char *keys[] = {"--file",       "--output",   "--voice", "--voice-file",    "--model",
                              "--voices-dir", "--language", "--seed",  "--duration-delta"};
        size_t key = 0;
        while (key < 9 && strcmp(arg, keys[key]))
            ++key;
        if (key == 9) {
            if (arg[0] == '-' || text_arg)
                goto invalid;
            text_arg = arg;
            continue;
        }
        bit = 2U << key;
        if ((seen & bit) || ++i == argc)
            goto invalid;
        seen |= bit;
        const char *value = argv[i];
        switch (key) {
        case 0:
            input_path = value;
            break;
        case 1:
            output_path = value;
            break;
        case 2:
            name = value;
            break;
        case 3:
            explicit_voice = value;
            break;
        case 4:
            pack = value;
            break;
        case 5:
            directory = value;
            break;
        case 6:
            if (strcmp(value, "en")) {
                fprintf(err, "Only --language en is supported.\n");
                return 2;
            }
            break;
        case 7: {
            uint64_t n = 0;
            if (!*value)
                goto invalid;
            for (const char *p = value; *p; ++p) {
                if (*p < '0' || *p > '9' || n > (UINT64_MAX - (unsigned)(*p - '0')) / 10)
                    goto invalid;
                n = n * 10 + (unsigned)(*p - '0');
            }
            options.seed = n;
            break;
        }
        case 8: {
            char *end;
            errno = 0;
            options.duration_delta = strtof(value, &end);
            if (errno || end == value || *end || !isfinite(options.duration_delta) ||
                value[0] == ' ' || value[0] == '\t' || value[0] == '\n')
                goto invalid;
            break;
        }
        }
    }
    if (!strcmp(argv[1], "voices")) {
        if (seen & ~(2U << 5) || text_arg || streaming)
            goto invalid;
        DIR *dir = opendir(directory);
        if (!dir) {
            fprintf(err, "Cannot open installed voices directory %s.\n", directory);
            return 1;
        }
        struct dirent *entry;
        while ((entry = readdir(dir))) {
            size_t n = strlen(entry->d_name);
            const char *suffix = ".safetensors";
            if (n <= strlen(suffix) || strcmp(entry->d_name + n - strlen(suffix), suffix))
                continue;
            char label[129];
            size_t length = n - strlen(suffix);
            if (length >= sizeof(label))
                continue;
            memcpy(label, entry->d_name, length);
            label[length] = 0;
            if (!name_ok(label))
                continue;
            char *path = voice_path(directory, label);
            PhononVoice *voice = NULL;
            if (path && phonon_voice_open(path, &voice)) {
                fprintf(out, "%s\n", label);
                phonon_voice_close(voice);
            }
            free(path);
        }
        closedir(dir);
        return output_status(out);
    }
    if (streaming && output_path)
        goto invalid;
    if ((!!text_arg + !!input_path + stdin_selected) > 1 || ((seen & (2U << 2)) && explicit_voice))
        goto invalid;
    if (!explicit_voice && !name_ok(name))
        goto invalid;
    char *owned_path = explicit_voice ? NULL : voice_path(directory, name);
    const char *path = explicit_voice ? explicit_voice : owned_path;
    PhononVoice *voice = NULL;
    if (!path || !phonon_voice_open(path, &voice)) {
        fprintf(err,
                "Cannot load compatible Phonon voice '%s'. Use voices or --voice-file with a "
                "compatible safetensors artifact; Kokoro voice names are unsupported.\n",
                path ? path : name);
        free(owned_path);
        return 1;
    }
    free(owned_path);
    char *text = NULL;
    size_t length = 0;
    if (text_arg) {
        length = strlen(text_arg);
        if (length && length <= SPEECH_MAX_INPUT_BYTES)
            text = strdup(text_arg);
    } else {
        FILE *input = input_path ? fopen(input_path, "rb")
                                 : (config && config->input ? config->input : stdin);
        if (input)
            text = read_text(input, &length);
        if (input_path && input)
            fclose(input);
    }
    if (!text) {
        fprintf(err, "Input must be nonempty UTF-8 text, no NUL, at most 1 MiB.\n");
        phonon_voice_close(voice);
        return 2;
    }
    SpeechEngine *engine = NULL;
    float *pcm = NULL;
    size_t count = 0;
    SpeechResult result;
    SpeechStatus status = speech_engine_create(pack, &engine);
    if (status == SPEECH_OK && streaming) {
        SpeechStream *stream = NULL;
        KokoroError error = {0};
        const KokoroStreamPlaybackOps *ops = config ? config->stream_playback : NULL;
        KokoroStatus played =
            ops ? speech_stream_start_with(engine, voice, text, length, &options, ops,
                                           config->stream_context, &stream, &error)
                : speech_stream_start(engine, voice, text, length, &options, &stream, &error);
        if (played == KOKORO_OK)
            played = speech_stream_wait(stream, config ? config->interrupted : NULL, &error);
        speech_stream_free(stream);
        free(text);
        speech_engine_destroy(engine);
        phonon_voice_close(voice);
        if (played != KOKORO_OK)
            fprintf(err, "Streaming failed (%d): %s\n", played, error.message);
        return played == KOKORO_OK ? 0 : played == KOKORO_CANCELLED ? 130 : 1;
    }
    if (status == SPEECH_OK)
        status =
            speech_collect(engine, voice, text, length, &options,
                           SPEECH_MAX_REQUEST_FRAMES * SPEECH_FRAME_SAMPLES, &pcm, &count, &result);
    free(text);
    speech_engine_destroy(engine);
    phonon_voice_close(voice);
    if (status != SPEECH_OK) {
        fprintf(err,
                "Synthesis failed (%d); check the native Phonon pack, UTF-8 input and generation "
                "limits.\n",
                status);
        free(pcm);
        return 1;
    }
    char temporary[] = "/tmp/speech-mac-XXXXXX";
    int fd = -1;
    FILE *file = NULL;
    if (output_path)
        file = fopen(output_path, "wb");
    else {
        fd = mkstemp(temporary);
        if (fd >= 0) {
            file = fdopen(fd, "wb");
            if (!file)
                close(fd);
        }
    }
    KokoroError error;
    int ok = file && kokoro_wav_write_pcm16_stream(file, pcm, count, SPEECH_SAMPLE_RATE, &error) ==
                         KOKORO_OK;
    if (file && fclose(file))
        ok = 0;
    free(pcm);
    if (ok && !output_path)
        ok = (config && config->play ? config->play : player)(config ? config->play_context : NULL,
                                                              temporary);
    if (!output_path && fd >= 0 && unlink(temporary) != 0) {
        fprintf(err, "Cannot remove temporary WAV %s: %s\n", temporary, strerror(errno));
        ok = 0;
    }
    if (!ok)
        fprintf(err, "Cannot write or play completed WAV. Use --output PATH to save audio.\n");
    return ok ? 0 : 1;
invalid:
    fprintf(err, "Invalid, duplicate or conflicting option/input. See speech-mac --help; seed must "
                 "be uint64 and duration-delta finite.\n");
    return 2;
}

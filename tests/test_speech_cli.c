#define _POSIX_C_SOURCE 200809L
#include "speech.h"
#include "speech_cli.h"
#include "wav_writer.h"
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "failed %s:%d: %s\n", __FILE__, __LINE__, #x);                         \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)
static int run(size_t n, char **args) {
    FILE *sink = tmpfile();
    CHECK(sink);
    SpeechCliConfig c = {.output = sink, .error = sink, .voices_directory = "/no/such/voices"};
    int status = speech_cli_run((int)n, args, &c);
    fclose(sink);
    return status;
}
static unsigned u32(const unsigned char *p) {
    return (unsigned)p[0] | (unsigned)p[1] << 8 | (unsigned)p[2] << 16 | (unsigned)p[3] << 24;
}
static unsigned char *wav(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    CHECK(f);
    CHECK(!fseek(f, 0, SEEK_END));
    long n = ftell(f);
    CHECK(n > 44);
    CHECK(!fseek(f, 0, SEEK_SET));
    unsigned char *data = malloc((size_t)n);
    CHECK(data);
    CHECK(fread(data, 1, (size_t)n, f) == (size_t)n);
    fclose(f);
    *size = (size_t)n;
    CHECK(!memcmp(data, "RIFF", 4) && !memcmp(data + 8, "WAVEfmt ", 8));
    CHECK(u32(data + 4) == *size - 8 && u32(data + 16) == 16);
    CHECK(data[20] == 1 && data[21] == 0 && data[22] == 1 && data[23] == 0);
    CHECK(u32(data + 24) == 24000 && u32(data + 28) == 48000 && data[32] == 2 && data[34] == 16);
    CHECK(!memcmp(data + 36, "data", 4) && u32(data + 40) == *size - 44);
    return data;
}
/* Safetensors source schema: bound projected voice; no numerical engine needed. */
static void tiny_voice(const char *path) {
    const char *header = "{\"__metadata__\":{\"model_ext\":\"ba7a5a45@300\"},\"emb\":{\"dtype\":"
                         "\"F32\",\"shape\":[1,512],\"data_offsets\":[0,2048]}}";
    FILE *f = fopen(path, "wb");
    CHECK(f);
    uint64_t n = strlen(header);
    CHECK(fwrite(&n, 8, 1, f) == 1);
    CHECK(fwrite(header, 1, (size_t)n, f) == n);
    float data[512] = {0};
    CHECK(fwrite(data, sizeof(data), 1, f) == 1);
    CHECK(!fclose(f));
}
static void weightless(void) {
    CHECK(!mkdir(".build/speech-cli-unit-voices", 0700) || errno == EEXIST);
    tiny_voice(".build/speech-cli-unit-voices/default.safetensors");
    FILE *bad = fopen(".build/speech-cli-unit-voices/bad.safetensors", "wb");
    CHECK(bad);
    fputs("obsolete", bad);
    fclose(bad);
    FILE *out = tmpfile(), *err = tmpfile(), *input = tmpfile();
    CHECK(out && err && input);
    SpeechCliConfig c = {.pack_path = "/no/such/model",
                         .voices_directory = ".build/speech-cli-unit-voices",
                         .output = out,
                         .error = err,
                         .input = input};
    char *list[] = {"speech-mac", "voices"};
    CHECK(!speech_cli_run(2, list, &c));
    CHECK(!fseek(out, 0, SEEK_SET));
    char names[64] = {0};
    CHECK(fread(names, 1, sizeof(names) - 1, out) == 8);
    CHECK(!strcmp(names, "default\n"));
    FILE *readonly = fopen("tests/test_speech_cli.c", "rb");
    CHECK(readonly);
    c.output = readonly;
    char *help[] = {"speech-mac", "--help"}, *version[] = {"speech-mac", "--version"};
    CHECK(speech_cli_run(2, help, &c) == 1);
    clearerr(readonly);
    CHECK(speech_cli_run(2, version, &c) == 1);
    clearerr(readonly);
    CHECK(speech_cli_run(2, list, &c) == 1);
    fclose(readonly);
    c.output = out;
    char *literal[] = {"speech-mac", "say", "--", "--ipa"};
    CHECK(speech_cli_run(4, literal, &c) ==
          1); /* reaches missing model, not removed-option parser */
    char *in[] = {"speech-mac", "say", "--stdin"};
    CHECK(speech_cli_run(3, in, &c) == 2); /* empty */
    fputc(0, input);
    CHECK(!fseek(input, 0, SEEK_SET));
    CHECK(speech_cli_run(3, in, &c) == 2); /* embedded NUL */
    fclose(input);
    input = tmpfile();
    CHECK(input);
    c.input = input;
    char block[4096];
    memset(block, 'x', sizeof(block));
    for (size_t n = 0; n < SPEECH_MAX_INPUT_BYTES / sizeof(block); ++n)
        CHECK(fwrite(block, 1, sizeof(block), input) == sizeof(block));
    fputc('x', input);
    CHECK(!fseek(input, 0, SEEK_SET));
    CHECK(speech_cli_run(3, in, &c) == 2); /* hard input cap */
    CHECK(!ftruncate(fileno(input), SPEECH_MAX_INPUT_BYTES));
    CHECK(!fseek(input, 0, SEEK_SET));
    CHECK(speech_cli_run(3, in, &c) == 1); /* exact cap accepted, reaches missing model */
    char *missing[] = {"speech-mac", "say", "--file", "/no/such/text"};
    CHECK(speech_cli_run(4, missing, &c) == 2);
    char *old[] = {"speech-mac", "say", "hello", "--voice", "af_heart"};
    CHECK(speech_cli_run(5, old, &c) == 1);
    CHECK(!fseek(err, 0, SEEK_SET));
    char messages[4096] = {0};
    CHECK(fread(messages, 1, sizeof(messages) - 1, err) > 0);
    CHECK(strstr(messages, "Kokoro voice names are unsupported"));
    fclose(input);
    fclose(out);
    fclose(err);
    unlink(".build/speech-cli-unit-voices/default.safetensors");
    unlink(".build/speech-cli-unit-voices/bad.safetensors");
    rmdir(".build/speech-cli-unit-voices");
    const float pcm[] = {-2, -1, 0, 1, 2};
    SpeechError error;
    CHECK(speech_wav_write_pcm16(".build/cli-unit.wav", pcm, 5, 24000, &error) == SPEECH_IO_OK);
    size_t size;
    unsigned char *data = wav(".build/cli-unit.wav", &size);
    CHECK(size == 54);
    const unsigned char expected[] = {0, 128, 0, 128, 0, 0, 255, 127, 255, 127};
    CHECK(!memcmp(data + 44, expected, sizeof(expected)));
    free(data);
    float nonfinite[] = {NAN};
    CHECK(speech_wav_write_pcm16(".build/cli-unit.wav", nonfinite, 1, 24000, &error) !=
          SPEECH_IO_OK);
    CHECK(speech_wav_write_pcm16(".build/cli-unit.wav", pcm, 0, 24000, &error) != SPEECH_IO_OK);
    unlink(".build/cli-unit.wav");
    puts("weightless CLI catalog/input/WAV contracts: ok");
}
typedef struct {
    int calls;
    char path[256];
    int result;
    int unlink_failure;
} Playback;
static int play(void *context, const char *path) {
    Playback *p = context;
    struct stat st;
    CHECK(!stat(path, &st));
    CHECK((st.st_mode & 0777) == 0600);
    size_t n;
    unsigned char *data = wav(path, &n);
    free(data);
    CHECK(n > 44);
    CHECK(strlen(path) < sizeof(p->path));
    strcpy(p->path, path);
    p->calls++;
    if (p->unlink_failure) {
        CHECK(!unlink(path));
        CHECK(!mkdir(path, 0700));
    }
    return p->result;
}
static void actual(void) {
    const char *pack = getenv("PHONON_CLI_PACK"), *voice = getenv("PHONON_DEFAULT_VOICE"),
               *projected = getenv("PHONON_PROJECTED_VOICE");
    if (!pack) {
        puts("SKIP actual CLI: set PHONON_CLI_PACK, PHONON_DEFAULT_VOICE, PHONON_PROJECTED_VOICE");
        return;
    }
    CHECK(voice && projected);
    CHECK(!mkdir(".build/speech-cli-voices", 0700) || errno == EEXIST);
    const char *link = ".build/speech-cli-voices/default.safetensors";
    unlink(link);
    CHECK(!symlink(voice, link));
    CHECK(!symlink(projected, ".build/speech-cli-voices/projected.safetensors") || errno == EEXIST);
    FILE *input = tmpfile(), *sink = tmpfile();
    CHECK(input && sink);
    fputs("Hello.", input);
    CHECK(!fseek(input, 0, SEEK_SET));
    Playback playback = {.result = 1};
    SpeechCliConfig c = {.pack_path = pack,
                         .voices_directory = ".build/speech-cli-voices",
                         .input = input,
                         .output = sink,
                         .error = sink,
                         .play = play,
                         .play_context = &playback};
    char *pos[] = {"speech-mac",        "say", "Hello.", "--seed", "123", "--output",
                   ".build/cli-pos.wav"};
    CHECK(!speech_cli_run(7, pos, &c));
    CHECK(!playback.calls);
    FILE *text = fopen(".build/cli-input.txt", "wb");
    CHECK(text);
    fputs("Hello.", text);
    fclose(text);
    char *file[] = {"speech-mac", "say",    "--file",   ".build/cli-input.txt",
                    "--seed",     "123",    "--output", ".build/cli-file.wav",
                    "--voice",    "default"};
    CHECK(!speech_cli_run(10, file, &c));
    char *in[] = {"speech-mac",           "say",          "--stdin",    "--seed", "123", "--output",
                  ".build/cli-stdin.wav", "--voice-file", (char *)voice};
    CHECK(!speech_cli_run(9, in, &c));
    size_t n, m;
    unsigned char *a = wav(".build/cli-pos.wav", &n), *b = wav(".build/cli-file.wav", &m);
    CHECK(n == m && !memcmp(a, b, n));
    free(b);
    b = wav(".build/cli-stdin.wav", &m);
    CHECK(n == m && !memcmp(a, b, n));
    free(b);
    char *proj[] = {
        "speech-mac", "say",      "Hello.", "--seed", "123", "--output", ".build/cli-projected.wav",
        "--voice",    "projected"};
    CHECK(!speech_cli_run(9, proj, &c));
    b = wav(".build/cli-projected.wav", &m);
    CHECK(n == m);
    double power = 0, square = 0;
    double maximum = 0;
    for (size_t i = 44; i < n; i += 2) {
        int16_t x = (int16_t)(a[i] | a[i + 1] << 8), y = (int16_t)(b[i] | b[i + 1] << 8);
        double d = (double)x - y;
        square += d * d;
        power += (double)x * x;
        if (fabs(d) > maximum)
            maximum = fabs(d);
    }
    CHECK(sqrt(square / power) <= .001 && maximum / 32768. <= .002);
    free(a);
    free(b);
    char *say[] = {"speech-mac", "say", "Hello.", "--seed", "123"};
    CHECK(!speech_cli_run(5, say, &c));
    CHECK(playback.calls == 1 && access(playback.path, F_OK) != 0);
    playback.result = 0;
    CHECK(speech_cli_run(5, say, &c) == 1);
    CHECK(playback.calls == 2 && access(playback.path, F_OK) != 0);
    playback.result = 1;
    playback.unlink_failure = 1;
    CHECK(speech_cli_run(5, say, &c) == 1);
    CHECK(playback.calls == 3);
    CHECK(!rmdir(playback.path));
    playback.unlink_failure = 0;
    char *controls[] = {"speech-mac",
                        "say",
                        "Hello.",
                        "--seed",
                        "18446744073709551615",
                        "--duration-delta",
                        "-0.1",
                        "--language",
                        "en",
                        "--output",
                        ".build/cli-controls.wav"};
    CHECK(!speech_cli_run(11, controls, &c));
    b = wav(".build/cli-controls.wav", &m);
    free(b);
    FILE *invalid = fopen(".build/cli-invalid.txt", "wb");
    CHECK(invalid);
    fputc(255, invalid);
    fclose(invalid);
    char *malformed[] = {"speech-mac", "say",
                         "--file",     ".build/cli-invalid.txt",
                         "--output",   ".build/cli-invalid.wav"};
    unlink(".build/cli-invalid.wav");
    CHECK(speech_cli_run(6, malformed, &c) == 1);
    CHECK(access(".build/cli-invalid.wav", F_OK) != 0);
    unlink(".build/cli-invalid.txt");
    char *list[] = {"speech-mac", "voices"};
    CHECK(!speech_cli_run(2, list, &c));
    fclose(input);
    fclose(sink);
    unlink(link);
    unlink(".build/speech-cli-voices/projected.safetensors");
    rmdir(".build/speech-cli-voices");
    puts("actual CLI positional/file/stdin/default/named/local/projected/native "
         "controls/WAV/playback cleanup: ok (listening unperformed)");
}
int main(void) {
    char *help[] = {"speech-mac", "--help"}, *version[] = {"speech-mac", "--version"};
    CHECK(run(2, help) == 0);
    CHECK(run(2, version) == 0);
    char *bad[][7] = {{"speech-mac", "say", "--speed", "1"},
                      {"speech-mac", "say", "-s", "1"},
                      {"speech-mac", "say", "--ipa", "x"},
                      {"speech-mac", "say", "--stream", "--output", "x"},
                      {"speech-mac", "say", "Hello.", "--stream", "--stream"},
                      {"speech-mac", "voices", "--stream"},
                      {"speech-mac", "say", "--language", "fr"},
                      {"speech-mac", "say", "--seed", "-1"},
                      {"speech-mac", "say", "--seed", "18446744073709551616"},
                      {"speech-mac", "say", "--seed", "1junk"},
                      {"speech-mac", "say", "--duration-delta", "nan"},
                      {"speech-mac", "say", "--duration-delta", "inf"},
                      {"speech-mac", "say", "--duration-delta", "1junk"},
                      {"speech-mac", "say", "--duration-delta", "1e999"},
                      {"speech-mac", "say", "--seed", "1", "--seed", "2"},
                      {"speech-mac", "say", "--stdin", "--file", "x"},
                      {"speech-mac", "say", "--stdin", "text"},
                      {"speech-mac", "say", "--voice", "default", "--voice-file", "x"},
                      {"speech-mac", "say", "--voice", "../escape"},
                      {"speech-mac", "voices", "--seed", "1"},
                      {"speech-mac", "say", "--output"},
                      {"speech-mac", "say", "one", "two"}};
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); ++i) {
        size_t n = 0;
        while (n < 7 && bad[i][n])
            ++n;
        CHECK(run(n, bad[i]) == 2);
    }
    char *valid[] = {"speech-mac",       "say",  "--seed", "18446744073709551615",
                     "--duration-delta", "-0.1", "Hello"};
    CHECK(run(7, valid) == 1); /* parsed successfully, missing compatible voice */
    weightless();
    actual();
    puts("speech CLI validation: ok");
    return 0;
}

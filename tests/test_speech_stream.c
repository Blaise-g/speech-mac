#define _POSIX_C_SOURCE 200809L
#include "audio_queue_playback.h"
#include "kokoro_internal.h"
#include "speech_cli.h"
#include "speech_stream.h"
#include "speech_stream_test.h"
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define FAILURE_STATUS ((OSStatus) - 71)
#include "audio_queue_fake.h"
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "failed %s:%d: %s\n", __FILE__, __LINE__, #x);                         \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)
static size_t waits, drain_waits;
static void observe_wait(void) {
    MockAudio *a = active_mock_audio;
    pthread_mutex_lock(&a->mutex);
    waits++;
    if (a->graceful_stop_count)
        drain_waits++;
    pthread_cond_broadcast(&a->condition);
    pthread_mutex_unlock(&a->mutex);
}
static bool blocked_submit(const MockAudio *a) {
    return waits && a->enqueue_count == 3 && !a->graceful_stop_count;
}
static bool blocked_drain(const MockAudio *a) { return drain_waits && a->graceful_stop_count > 0; }
static KokoroAudioQueueApi api;
static SpeechStream *start(MockAudio *a, SpeechEngine *e, PhononVoice *v, SpeechOptions *o) {
    active_mock_audio = a;
    waits = drain_waits = 0;
    SpeechStream *s = NULL;
    CHECK(speech_stream_start_with(e, v, "Hello.", 6, o, kokoro_audio_queue_playback_ops(), &api,
                                   &s, NULL) == KOKORO_OK);
    return s;
}
static void enable(MockAudio *a) {
    pthread_mutex_lock(&a->mutex);
    a->consume_enabled = true;
    pthread_cond_broadcast(&a->condition);
    pthread_mutex_unlock(&a->mutex);
}
static void finish(SpeechStream *s, KokoroStatus expected, MockAudio *a) {
    KokoroError error;
    CHECK(speech_stream_wait(s, NULL, &error) == expected);
    speech_stream_cancel(s); /* a terminal outcome cannot be overwritten */
    CHECK(speech_stream_wait(s, NULL, &error) == expected);
    speech_stream_free(s);
    CHECK(a->dispose_count == 1 && a->remove_listener_count == 1);
}
static int cancel_numerical(void *context) {
    size_t *n = context;
    return ++*n == 80;
}
static SpeechStatus controlled(const SpeechEngine *engine, const PhononVoice *voice,
                               const char *text, size_t length, const SpeechOptions *o,
                               SpeechPCMCallback callback, void *context, SpeechResult *r) {
    (void)engine;
    (void)voice;
    (void)text;
    (void)length;
    float pcm[SPEECH_FRAME_SAMPLES];
    for (size_t frame = 0; frame < 8; frame++) {
        if (o->cancel && o->cancel(o->cancel_context))
            return SPEECH_CANCELLED;
        for (size_t i = 0; i < SPEECH_FRAME_SAMPLES; i++)
            pcm[i] = (float)(frame * SPEECH_FRAME_SAMPLES + i);
        r->frames_generated++;
        r->samples_generated += SPEECH_FRAME_SAMPLES;
        int response = callback(context, pcm, SPEECH_FRAME_SAMPLES, r);
        if (response)
            return response == 1 ? SPEECH_CANCELLED : SPEECH_CALLBACK_ERROR;
    }
    r->chunks_completed = 1;
    return SPEECH_OK;
}
static int cancel_soon(void *context) {
    size_t *polls = context;
    return ++*polls == 2;
}
static bool cancel_awakened;
static void observed_wake(void *handle) {
    kokoro_audio_queue_playback_ops()->wake(handle);
    pthread_mutex_lock(&active_mock_audio->mutex);
    cancel_awakened = true;
    pthread_cond_broadcast(&active_mock_audio->condition);
    pthread_mutex_unlock(&active_mock_audio->mutex);
}
static bool awake(const MockAudio *a) {
    (void)a;
    return cancel_awakened;
}
static void *cancel_thread(void *context) {
    speech_stream_cancel(context);
    return NULL;
}
static bool property_entered, release_property;
static OSStatus delayed_property(AudioQueueRef queue, AudioQueuePropertyID property, void *data,
                                 UInt32 *size) {
    MockAudio *a = active_mock_audio;
    pthread_mutex_lock(&a->mutex);
    if (a->graceful_stop_count) {
        property_entered = true;
        pthread_cond_broadcast(&a->condition);
        while (!release_property)
            pthread_cond_wait(&a->condition, &a->mutex);
    }
    pthread_mutex_unlock(&a->mutex);
    return mock_get_property(queue, property, data, size);
}
static bool property_ready(const MockAudio *a) {
    (void)a;
    return property_entered;
}
static void cancel_during_drain_property(void) {
    MockAudio a;
    CHECK(mock_audio_init(&a, true));
    active_mock_audio = &a;
    a.hold_graceful_stop = true;
    property_entered = release_property = false;
    KokoroAudioQueueApi delayed = api;
    delayed.get_property = delayed_property;
    SpeechStream *s = NULL;
    CHECK(speech_stream_test_start(NULL, kokoro_audio_queue_playback_ops(), &delayed, &s,
                                   controlled) == KOKORO_OK);
    CHECK(wait_audio(&a, property_ready, 10000));
    speech_stream_cancel(s); /* abort makes IsRunning=false while SDK read is held */
    pthread_mutex_lock(&a.mutex);
    release_property = true;
    pthread_cond_broadcast(&a.condition);
    pthread_mutex_unlock(&a.mutex);
    CHECK(speech_stream_wait(s, NULL, NULL) == KOKORO_CANCELLED);
    SpeechStreamMetrics metrics;
    speech_stream_result(s, NULL, &metrics);
    CHECK(!metrics.drained);
    finish(s, KOKORO_CANCELLED, &a);
    mock_audio_destroy(&a);
}
static bool drain_return_entered, release_drain_return;
static KokoroStatus delayed_successful_drain(void *handle, KokoroError *error) {
    KokoroStatus status = kokoro_audio_queue_playback_ops()->drain(handle, error);
    if (status != KOKORO_OK)
        return status;
    MockAudio *a = active_mock_audio;
    pthread_mutex_lock(&a->mutex);
    drain_return_entered = true;
    pthread_cond_broadcast(&a->condition);
    while (!release_drain_return)
        pthread_cond_wait(&a->condition, &a->mutex);
    pthread_mutex_unlock(&a->mutex);
    return KOKORO_OK;
}
static bool drain_return_ready(const MockAudio *a) {
    (void)a;
    return drain_return_entered;
}
static void cancel_before_drain_publication(void) {
    MockAudio a;
    CHECK(mock_audio_init(&a, true));
    active_mock_audio = &a;
    drain_return_entered = release_drain_return = false;
    KokoroStreamPlaybackOps ops = *kokoro_audio_queue_playback_ops();
    ops.drain = delayed_successful_drain;
    SpeechStream *s = NULL;
    CHECK(speech_stream_test_start(NULL, &ops, &api, &s, controlled) == KOKORO_OK);
    CHECK(wait_audio(&a, drain_return_ready, 10000));
    speech_stream_cancel(s);
    pthread_mutex_lock(&a.mutex);
    release_drain_return = true;
    pthread_cond_broadcast(&a.condition);
    pthread_mutex_unlock(&a.mutex);
    CHECK(speech_stream_wait(s, NULL, NULL) == KOKORO_CANCELLED);
    SpeechStreamMetrics metrics;
    speech_stream_result(s, NULL, &metrics);
    CHECK(!metrics.drained);
    finish(s, KOKORO_CANCELLED, &a);
    mock_audio_destroy(&a);
}
static void sdk_cancel_races(void) {
    for (size_t mode = 0; mode < 2; mode++) {
        MockAudio a;
        CHECK(mock_audio_init(&a, false));
        active_mock_audio = &a;
        if (mode == 0)
            a.enqueue_blocked = true;
        else
            a.start_blocked = true;
        cancel_awakened = false;
        KokoroStreamPlaybackOps ops = *kokoro_audio_queue_playback_ops();
        ops.wake = observed_wake;
        SpeechStream *s = NULL;
        CHECK(speech_stream_test_start(NULL, &ops, &api, &s, controlled) == KOKORO_OK);
        CHECK(wait_audio(&a, mode == 0 ? ready_enqueue_entered : ready_start_entered, 10000));
        pthread_t thread;
        CHECK(!pthread_create(&thread, NULL, cancel_thread, s));
        CHECK(wait_audio(&a, awake, 10000)); /* cancel set before blocked SDK call returns */
        pthread_mutex_lock(&a.mutex);
        a.release_blocked_api = true;
        pthread_cond_broadcast(&a.condition);
        pthread_mutex_unlock(&a.mutex);
        CHECK(!pthread_join(thread, NULL));
        finish(s, KOKORO_CANCELLED, &a);
        CHECK(a.start_count == mode && a.immediate_stop_count > 0);
        mock_audio_destroy(&a);
    }
}
static void weightless(void) {
    api = mock_api;
    api.waiting = observe_wait;
    for (size_t mode = 0; mode < 7; mode++) {
        MockAudio a;
        CHECK(mock_audio_init(&a, mode >= 2));
        active_mock_audio = &a;
        waits = drain_waits = 0;
        if (mode == 2)
            a.hold_graceful_stop = true;
        if (mode == 3)
            a.enqueue_error_at = 2;
        if (mode == 4)
            a.dispose_error = true;
        if (mode == 6)
            a.start_error = true;
        SpeechOptions o = speech_options_default();
        size_t polls = 0;
        if (mode == 5) {
            o.cancel = cancel_soon;
            o.cancel_context = &polls;
        }
        SpeechStream *s = NULL;
        CHECK(speech_stream_test_start(&o, kokoro_audio_queue_playback_ops(), &api, &s,
                                       controlled) == KOKORO_OK);
        if (mode < 2) {
            CHECK(wait_audio(&a, blocked_submit, 10000));
            CHECK(a.start_count == 1);
            if (mode == 0)
                enable(&a);
            else
                speech_stream_cancel(s);
        }
        if (mode == 2) {
            CHECK(wait_audio(&a, blocked_drain, 10000));
            speech_stream_cancel(s);
        }
        KokoroStatus expected = mode == 1 || mode == 2 || mode == 5   ? KOKORO_CANCELLED
                                : mode == 3 || mode == 4 || mode == 6 ? KOKORO_IO_ERROR
                                                                      : KOKORO_OK;
        CHECK(speech_stream_wait(s, NULL, NULL) == expected);
        SpeechStreamMetrics m;
        speech_stream_result(s, NULL, &m);
        if (expected == KOKORO_OK) {
            CHECK(a.consumed_count == 8 * SPEECH_FRAME_SAMPLES && m.drained);
            for (size_t i = 0; i < a.consumed_count; i++)
                CHECK(a.consumed_samples[i] == (float)i);
        } else if (mode != 4)
            CHECK(!m.drained);
        finish(s, expected, &a);
        mock_audio_destroy(&a);
    }
    sdk_cancel_races();
    cancel_during_drain_property();
    cancel_before_drain_publication();
    puts("synthetic PCM controller/real AudioQueue adapter fake: bounded order, blocked "
         "submit/drain cancellation, error and stable terminal result ok (no model/device "
         "evidence)");
}
#ifdef KOKORO_STREAM_SILENT_OUTPUT
/* Test-only watchdog: a hung SDK call cannot be cancelled safely. Fail this test
 * process after 30s; ordinary runtime/cancellation handlers are unchanged. */
typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    pthread_t thread;
    bool done;
} DeviceWatchdog;
static void *device_watchdog(void *context) {
    DeviceWatchdog *watchdog = context;
    struct timespec deadline;
    deadline_after_ms(&deadline, 30000);
    pthread_mutex_lock(&watchdog->mutex);
    int status = 0;
    while (!watchdog->done && !status)
        status = pthread_cond_timedwait(&watchdog->condition, &watchdog->mutex, &deadline);
    if (!watchdog->done) {
        fprintf(stderr,
                "FAIL queue-muted AudioQueue SDK probe timed out or watchdog wait failed after 30s "
                "(wait status %d); device acceptance NOT met\n",
                status);
        fflush(stderr);
        _Exit(124);
    }
    pthread_mutex_unlock(&watchdog->mutex);
    return NULL;
}
static void watchdog_start(DeviceWatchdog *watchdog) {
    memset(watchdog, 0, sizeof(*watchdog));
    CHECK(!pthread_mutex_init(&watchdog->mutex, NULL));
    CHECK(!pthread_cond_init(&watchdog->condition, NULL));
    CHECK(!pthread_create(&watchdog->thread, NULL, device_watchdog, watchdog));
}
static void watchdog_finish(DeviceWatchdog *watchdog) {
    pthread_mutex_lock(&watchdog->mutex);
    watchdog->done = true;
    pthread_cond_broadcast(&watchdog->condition);
    pthread_mutex_unlock(&watchdog->mutex);
    CHECK(!pthread_join(watchdog->thread, NULL));
    CHECK(!pthread_cond_destroy(&watchdog->condition));
    CHECK(!pthread_mutex_destroy(&watchdog->mutex));
}
#endif
static void actual(void) {
    const char *pack = getenv("PHONON_NATIVE_PACK"), *path = getenv("PHONON_DEFAULT_VOICE");
    if (!pack) {
        puts("SKIP actual progressive stream: PHONON_NATIVE_PACK/PHONON_DEFAULT_VOICE required");
        return;
    }
    CHECK(path);
    SpeechEngine *e = NULL;
    PhononVoice *v = NULL;
    CHECK(speech_engine_create(pack, &e) == SPEECH_OK && phonon_voice_open(path, &v));
    api = mock_api;
    api.waiting = observe_wait;
    SpeechOptions o = speech_options_default();
    o.seed = 123;
    float *batch = NULL;
    size_t count = 0;
    SpeechResult result;
    CHECK(speech_collect(e, v, "Hello.", 6, &o, 8192U * 1920, &batch, &count, &result) ==
          SPEECH_OK);
    CHECK(count > 3 * 1920 && count == result.frames_generated * 1920);
    MockAudio a;
    CHECK(mock_audio_init(&a, false));
    SpeechStream *s = start(&a, e, v, &o);
    CHECK(wait_audio(&a, blocked_submit, 30000));
    CHECK(a.start_count == 1); /* start before bounded queue can fill */
    enable(&a);
    CHECK(speech_stream_wait(s, NULL, NULL) == KOKORO_OK);
    SpeechStreamMetrics m;
    SpeechResult streamed;
    speech_stream_result(s, &streamed, &m);
    CHECK(streamed.samples_generated == count && m.submitted_samples == count);
    CHECK(a.consumed_count == count && !memcmp(batch, a.consumed_samples, count * sizeof(float)));
    CHECK(m.synthesis_begin <= m.first_pcm && m.first_pcm <= m.first_submission);
    CHECK(m.first_submission <= m.started && m.started < m.synthesis_end &&
          m.synthesis_end <= m.drained);
    CHECK(a.consumer_callback_count == streamed.frames_generated);
    CHECK(m.reclaimed_buffers == streamed.frames_generated &&
          m.first_reclaimed >= m.first_submission);
    printf("actual batch/stream exact samples=%zu frames=%zu; firstPCM %.6f submission %.6f start "
           "%.6f synthesisEnd %.6f drain %.6f seconds from monotonic synthesisBegin (fake "
           "consumer; acoustic onset unknown)\n",
           count, streamed.frames_generated, m.first_pcm - m.synthesis_begin,
           m.first_submission - m.synthesis_begin, m.started - m.synthesis_begin,
           m.synthesis_end - m.synthesis_begin, m.drained - m.synthesis_begin);
    finish(s, KOKORO_OK, &a);
    mock_audio_destroy(&a);
    free(batch);
    /* Every wait is reached through real synthesis and the actual SDK adapter. */
    CHECK(mock_audio_init(&a, false));
    s = start(&a, e, v, &o);
    CHECK(wait_audio(&a, blocked_submit, 30000));
    speech_stream_cancel(s);
    finish(s, KOKORO_CANCELLED, &a);
    CHECK(a.enqueue_count == 3 && a.graceful_stop_count == 0);
    mock_audio_destroy(&a);
    CHECK(mock_audio_init(&a, true));
    a.hold_graceful_stop = true;
    s = start(&a, e, v, &o);
    CHECK(wait_audio(&a, blocked_drain, 30000));
    speech_stream_cancel(s);
    finish(s, KOKORO_CANCELLED, &a);
    mock_audio_destroy(&a);
    CHECK(mock_audio_init(&a, false));
    s = start(&a, e, v, &o);
    CHECK(wait_audio(&a, blocked_submit, 30000));
    volatile sig_atomic_t signal_flag = 1;
    CHECK(speech_stream_wait(s, &signal_flag, NULL) == KOKORO_CANCELLED);
    finish(s, KOKORO_CANCELLED, &a);
    mock_audio_destroy(&a);
    size_t polls = 0;
    o.cancel = cancel_numerical;
    o.cancel_context = &polls;
    CHECK(mock_audio_init(&a, true));
    s = start(&a, e, v, &o);
    finish(s, KOKORO_CANCELLED, &a);
    CHECK(polls == 80 && a.enqueue_count == 0);
    mock_audio_destroy(&a);
    o.cancel = NULL;
    o.cancel_context = NULL;
    for (size_t i = 0; i < 5; i++) {
        CHECK(mock_audio_init(&a, true));
        if (i == 0)
            a.enqueue_error_at = 2;
        if (i == 1)
            a.start_error = true;
        if (i == 2)
            a.graceful_stop_error = true;
        if (i == 3)
            a.property_error = true;
        if (i == 4)
            a.dispose_error = true;
        s = start(&a, e, v, &o);
        finish(s, KOKORO_IO_ERROR, &a);
        mock_audio_destroy(&a);
    }
    CHECK(mock_audio_init(&a, true));
    active_mock_audio = &a;
    FILE *sink = tmpfile();
    CHECK(sink);
    SpeechCliConfig c = {.pack_path = pack,
                         .output = sink,
                         .error = sink,
                         .stream_playback = kokoro_audio_queue_playback_ops(),
                         .stream_context = &api};
    char *args[] = {"speech-mac", "say",      "Hello.", "--voice-file",
                    (char *)path, "--stream", "--seed", "123"};
    CHECK(speech_cli_run(8, args, &c) == 0);
    CHECK(a.start_count == 1 && a.dispose_count == 1);
    fclose(sink);
    mock_audio_destroy(&a);
    CHECK(mock_audio_init(&a, true));
    active_mock_audio = &a;
    sink = tmpfile();
    CHECK(sink);
    volatile sig_atomic_t already_interrupted = 1;
    c.output = c.error = sink;
    c.interrupted = &already_interrupted;
    CHECK(speech_cli_run(8, args, &c) == 130);
    CHECK(a.dispose_count == 1);
    fclose(sink);
    mock_audio_destroy(&a);
    if (getenv("PHONON_STREAM_DEVICE")) {
#ifdef KOKORO_STREAM_SILENT_OUTPUT
        DeviceWatchdog watchdog;
        watchdog_start(&watchdog);
        CHECK(speech_stream_start(e, v, "Hello.", 6, &o, &s, NULL) == KOKORO_OK);
        KokoroError device_error = {0};
        KokoroStatus device_status = speech_stream_wait(s, NULL, &device_error);
        fprintf(stderr, "muted AudioQueue result=%d detail=%s\n", device_status,
                device_error.message);
        CHECK(device_status == KOKORO_OK);
        speech_stream_result(s, &streamed, &m);
        CHECK(m.drained && m.submitted_samples == count &&
              m.reclaimed_buffers == streamed.frames_generated);
        printf("ACTUAL queue-volume-zero AudioQueue functional start/reclaim/drain: ok, "
               "samples=%zu; acoustic onset unknown\n",
               m.submitted_samples);
        speech_stream_free(s);
        watchdog_finish(&watchdog);
#else
        CHECK(!"device probe must be compiled KOKORO_STREAM_SILENT_OUTPUT");
#endif
    } else
        puts(
            "SKIP muted device: explicitly select PHONON_STREAM_DEVICE=1 with silent-output build");
    speech_engine_destroy(e);
    phonon_voice_close(v);
    puts("actual stream order/bounds/waits/generation+submit+drain+signal "
         "cancellation/errors/resident reuse/CLI: ok");
}
int main(void) {
    SpeechStream *s = (void *)(uintptr_t)1;
    CHECK(speech_stream_start(NULL, NULL, "x", 1, NULL, &s, NULL) == KOKORO_INVALID_ARGUMENT && !s);
    weightless();
    actual();
    return 0;
}

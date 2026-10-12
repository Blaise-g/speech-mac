#include "cli.h"
#include "stream_benchmark.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <time.h>

#define STREAM_SAMPLE_RATE_HZ 24000U
#define STREAM_SEED UINT64_C(104729)

static uint64_t cli_entry_ns;
static _Atomic uint64_t event_times[KOKORO_STREAM_BENCHMARK_EVENT_COUNT];
static _Atomic uint64_t accepted_sample_count;
static _Atomic uint64_t chunk_count;
static _Atomic uint64_t queue_starvation_count;
static _Atomic uint64_t synthesis_wall_ns;
static _Atomic uint64_t synthesis_cpu_ns;
static _Atomic uint64_t backpressure_wall_ns;
static uint64_t active_synthesis_wall_ns;
static uint64_t active_synthesis_cpu_ns;
static uint64_t active_backpressure_ns;

static uint64_t clock_ns(clockid_t clock_id) {
    struct timespec value;
    if (clock_gettime(clock_id, &value) != 0) {
        perror("clock_gettime");
        exit(2);
    }
    return (uint64_t)value.tv_sec * UINT64_C(1000000000) + (uint64_t)value.tv_nsec;
}

static uint64_t process_cpu_ns(void) {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        perror("getrusage");
        exit(2);
    }
    uint64_t user_ns = (uint64_t)usage.ru_utime.tv_sec * UINT64_C(1000000000) +
                       (uint64_t)usage.ru_utime.tv_usec * UINT64_C(1000);
    uint64_t system_ns = (uint64_t)usage.ru_stime.tv_sec * UINT64_C(1000000000) +
                         (uint64_t)usage.ru_stime.tv_usec * UINT64_C(1000);
    return user_ns + system_ns;
}

void kokoro_stream_benchmark_event(KokoroStreamBenchmarkEvent event, size_t value) {
    uint64_t wall_now = clock_ns(CLOCK_MONOTONIC);
    uint64_t expected = 0;
    (void)atomic_compare_exchange_strong_explicit(&event_times[event], &expected, wall_now,
                                                  memory_order_release, memory_order_relaxed);
    switch (event) {
    case KOKORO_STREAM_BENCHMARK_SYNTHESIS_BEGIN:
        active_synthesis_wall_ns = wall_now;
        active_synthesis_cpu_ns = clock_ns(CLOCK_THREAD_CPUTIME_ID);
        break;
    case KOKORO_STREAM_BENCHMARK_SYNTHESIS_END:
        (void)atomic_fetch_add_explicit(&synthesis_wall_ns, wall_now - active_synthesis_wall_ns,
                                        memory_order_relaxed);
        (void)atomic_fetch_add_explicit(&synthesis_cpu_ns,
                                        clock_ns(CLOCK_THREAD_CPUTIME_ID) - active_synthesis_cpu_ns,
                                        memory_order_relaxed);
        break;
    case KOKORO_STREAM_BENCHMARK_CHUNK_BEGIN:
        (void)atomic_fetch_add_explicit(&chunk_count, value, memory_order_relaxed);
        break;
    case KOKORO_STREAM_BENCHMARK_BUFFER_ACCEPTED:
        (void)atomic_fetch_add_explicit(&accepted_sample_count, value, memory_order_relaxed);
        break;
    case KOKORO_STREAM_BENCHMARK_BACKPRESSURE_BEGIN:
        active_backpressure_ns = wall_now;
        break;
    case KOKORO_STREAM_BENCHMARK_BACKPRESSURE_END:
        (void)atomic_fetch_add_explicit(&backpressure_wall_ns, wall_now - active_backpressure_ns,
                                        memory_order_relaxed);
        break;
    case KOKORO_STREAM_BENCHMARK_AUDIOQUEUE_OPEN_BEGIN:
    case KOKORO_STREAM_BENCHMARK_AUDIOQUEUE_OPEN_END:
    case KOKORO_STREAM_BENCHMARK_AUDIOQUEUE_START_BEGIN:
        break;
    case KOKORO_STREAM_BENCHMARK_QUEUE_STARVED:
        (void)atomic_fetch_add_explicit(&queue_starvation_count, 1, memory_order_relaxed);
        break;
    case KOKORO_STREAM_BENCHMARK_FIRST_CHUNK_READY:
    case KOKORO_STREAM_BENCHMARK_START_REQUESTED:
    case KOKORO_STREAM_BENCHMARK_BUFFER_RECLAIMED:
    case KOKORO_STREAM_BENCHMARK_DRAIN_COMPLETE:
    case KOKORO_STREAM_BENCHMARK_EVENT_COUNT:
        break;
    }
}

static double elapsed_seconds(uint64_t timestamp) {
    if (timestamp == 0 || timestamp < cli_entry_ns) {
        return -1.0;
    }
    return (double)(timestamp - cli_entry_ns) / 1000000000.0;
}

static void print_event(const char *name, KokoroStreamBenchmarkEvent event) {
    uint64_t timestamp = atomic_load_explicit(&event_times[event], memory_order_acquire);
    double seconds = elapsed_seconds(timestamp);
    if (seconds < 0.0) {
        printf("\"%s\":null", name);
    } else {
        printf("\"%s\":%.9f", name, seconds);
    }
}

int main(int argc, char **argv) {
#if !defined(KOKORO_STREAM_BENCHMARK_TIMING) || !defined(KOKORO_STREAM_SILENT_OUTPUT)
    (void)argc;
    (void)argv;
    fputs("stream benchmark requires timing and silent-output build flags\n", stderr);
    return 64;
#endif
    KokoroCliConfig config = {
        .model_pack_path = "models/kokoro-v1_0-af_heart.kpack",
        .voices_directory = "models/voices",
        .player_path = "/usr/bin/false",
        .random_seed = STREAM_SEED,
    };
    cli_entry_ns = clock_ns(CLOCK_MONOTONIC);
    uint64_t process_cpu_before = process_cpu_ns();
    int status = kokoro_cli_run(argc, argv, &config);
    uint64_t process_exit_ns = clock_ns(CLOCK_MONOTONIC);
    uint64_t process_cpu_after = process_cpu_ns();
    if (status != 0) {
        return status;
    }

    uint64_t drain_complete = atomic_load_explicit(
        &event_times[KOKORO_STREAM_BENCHMARK_DRAIN_COMPLETE], memory_order_acquire);
    uint64_t start_requested = atomic_load_explicit(
        &event_times[KOKORO_STREAM_BENCHMARK_START_REQUESTED], memory_order_acquire);
    uint64_t queue_open_begin = atomic_load_explicit(
        &event_times[KOKORO_STREAM_BENCHMARK_AUDIOQUEUE_OPEN_BEGIN], memory_order_acquire);
    uint64_t queue_open_end = atomic_load_explicit(
        &event_times[KOKORO_STREAM_BENCHMARK_AUDIOQUEUE_OPEN_END], memory_order_acquire);
    uint64_t queue_start_begin = atomic_load_explicit(
        &event_times[KOKORO_STREAM_BENCHMARK_AUDIOQUEUE_START_BEGIN], memory_order_acquire);
    uint64_t samples = atomic_load_explicit(&accepted_sample_count, memory_order_relaxed);
    printf("{\"schema_version\":2,\"timing_origin\":\"benchmark_cli_call_entry\",");
#if defined(KOKORO_STREAM_BENCHMARK_TIMING) && defined(KOKORO_STREAM_SILENT_OUTPUT)
    printf("\"benchmark_safety_marker\":\"%s\",", KOKORO_STREAM_SILENT_BINARY_MARKER);
#endif
    printf("\"event_seconds\":{");
    print_event("first_complete_chunk_ready", KOKORO_STREAM_BENCHMARK_FIRST_CHUNK_READY);
    printf(",");
    print_event("first_buffer_accepted", KOKORO_STREAM_BENCHMARK_BUFFER_ACCEPTED);
    printf(",");
    print_event("first_playback_start_request", KOKORO_STREAM_BENCHMARK_START_REQUESTED);
    printf(",");
    print_event("first_buffer_reclaimed", KOKORO_STREAM_BENCHMARK_BUFFER_RECLAIMED);
    printf("},\"synthesis_wall_seconds\":%.9f,\"synthesis_thread_cpu_seconds\":%.9f,",
           (double)atomic_load_explicit(&synthesis_wall_ns, memory_order_relaxed) / 1e9,
           (double)atomic_load_explicit(&synthesis_cpu_ns, memory_order_relaxed) / 1e9);
    printf("\"audio_queue_open_wall_seconds\":");
    if (queue_open_begin != 0 && queue_open_end >= queue_open_begin) {
        printf("%.9f", (double)(queue_open_end - queue_open_begin) / 1e9);
    } else {
        printf("null");
    }
    printf(",\"audio_queue_open_complete_seconds\":");
    double open_complete_seconds = elapsed_seconds(queue_open_end);
    if (open_complete_seconds >= 0.0) {
        printf("%.9f", open_complete_seconds);
    } else {
        printf("null");
    }
    printf(",\"audio_queue_start_api_seconds\":");
    if (queue_start_begin != 0 && start_requested >= queue_start_begin) {
        printf("%.9f", (double)(start_requested - queue_start_begin) / 1e9);
    } else {
        printf("null");
    }
    printf(",\"submit_backpressure_wall_seconds\":%.9f,\"playback_wall_seconds\":",
           (double)atomic_load_explicit(&backpressure_wall_ns, memory_order_relaxed) / 1e9);
    if (start_requested != 0 && drain_complete >= start_requested) {
        printf("%.9f", (double)(drain_complete - start_requested) / 1e9);
    } else {
        printf("null");
    }
    printf(",\"process_wall_seconds\":%.9f,\"process_cpu_seconds\":%.9f,",
           (double)(process_exit_ns - cli_entry_ns) / 1e9,
           (double)(process_cpu_after - process_cpu_before) / 1e9);
    printf("\"generated_sample_count\":%llu,\"generated_audio_seconds\":%.9f,",
           (unsigned long long)samples, (double)samples / STREAM_SAMPLE_RATE_HZ);
    printf("\"chunk_count\":%llu,\"queue_bound_bytes\":%u,"
           "\"queue_bound_audio_seconds\":%.9f,\"queue_starvation_observations\":%llu,",
           (unsigned long long)atomic_load_explicit(&chunk_count, memory_order_relaxed),
           KOKORO_STREAM_BUFFER_FRAMES * KOKORO_STREAM_BUFFER_COUNT * (unsigned int)sizeof(float),
           (double)(KOKORO_STREAM_BUFFER_FRAMES * KOKORO_STREAM_BUFFER_COUNT) /
               STREAM_SAMPLE_RATE_HZ,
           (unsigned long long)atomic_load_explicit(&queue_starvation_count, memory_order_relaxed));
    printf("\"acoustic_onset_seconds\":null,\"acoustic_onset_status\":\"unavailable\","
           "\"audio_queue_volume\":0.0,\"audio_queue_mute_status\":\"succeeded\"}\n");
    return fflush(stdout) == 0 ? 0 : 1;
}

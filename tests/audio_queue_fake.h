/* Shared deterministic AudioQueue fake; owns copies through SDK buffer callbacks. */
typedef struct MockAudio MockAudio;

typedef struct {
    MockAudio *owner;
    AudioQueueOutputCallback output_callback;
    void *output_context;
    AudioQueuePropertyListenerProc property_callback;
    void *property_context;
    AudioQueueBufferRef buffers[KOKORO_STREAM_BUFFER_COUNT];
    AudioQueueBufferRef pending[KOKORO_STREAM_BUFFER_COUNT];
    size_t pending_head;
    size_t pending_count;
    size_t allocated_count;
    bool running;
    bool graceful_stop;
    bool processing;
    bool shutdown;
    pthread_t consumer;
} MockQueue;

struct MockAudio {
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    MockQueue *queue;
    bool queue_created, running_on_dispose;
    size_t pending_on_dispose;
    bool consume_enabled;
    bool enqueue_blocked;
    bool enqueue_entered;
    bool start_blocked;
    bool start_entered;
    bool release_blocked_api;
    int create_error;
    int listener_error;
    size_t allocation_error_at;
    size_t enqueue_error_at;
    bool start_error;
    bool graceful_stop_error;
    bool hold_graceful_stop;
    bool property_error;
    bool dispose_error;
    size_t enqueue_count;
    size_t start_count;
    size_t graceful_stop_count;
    size_t immediate_stop_count;
    size_t consumer_callback_count;
    size_t synchronous_output_callbacks;
    size_t synchronous_property_callbacks;
    size_t dispose_count;
    size_t remove_listener_count;
    bool format_valid;
    float *consumed_samples;
    size_t consumed_count;
    size_t consumed_capacity;
};

static MockAudio *active_mock_audio;

static inline int fail(const char *message) {
    fprintf(stderr, "test_stream: %s\n", message);
    return 0;
}

static inline void deadline_after_ms(struct timespec *deadline, long milliseconds) {
    (void)clock_gettime(CLOCK_REALTIME, deadline);
    deadline->tv_nsec += milliseconds * 1000000L;
    deadline->tv_sec += deadline->tv_nsec / 1000000000L;
    deadline->tv_nsec %= 1000000000L;
}

static inline bool wait_audio(MockAudio *audio, bool (*ready)(const MockAudio *), long timeout_ms) {
    struct timespec deadline;
    deadline_after_ms(&deadline, timeout_ms);
    pthread_mutex_lock(&audio->mutex);
    int wait_error = 0;
    while (!ready(audio) && wait_error == 0) {
        wait_error = pthread_cond_timedwait(&audio->condition, &audio->mutex, &deadline);
    }
    bool is_ready = ready(audio);
    pthread_mutex_unlock(&audio->mutex);
    return is_ready;
}

static inline bool ready_enqueued_three(const MockAudio *audio) {
    return audio->enqueue_count >= KOKORO_STREAM_BUFFER_COUNT;
}

static inline bool ready_consumed_first_chunk(const MockAudio *audio) {
    return audio->consumed_count >= 9000 && audio->consumer_callback_count >= 3;
}

static inline bool ready_graceful_stop(const MockAudio *audio) {
    return audio->graceful_stop_count > 0;
}

static inline bool ready_immediate_stop(const MockAudio *audio) {
    return audio->immediate_stop_count > 0;
}

static inline bool ready_enqueue_entered(const MockAudio *audio) { return audio->enqueue_entered; }

static inline bool ready_start_entered(const MockAudio *audio) { return audio->start_entered; }

static inline int append_consumed_locked(MockAudio *audio, const float *samples, size_t count) {
    if (count > SIZE_MAX - audio->consumed_count) {
        return 0;
    }
    size_t required = audio->consumed_count + count;
    if (required > audio->consumed_capacity) {
        size_t grown = audio->consumed_capacity == 0 ? 4096 : audio->consumed_capacity;
        while (grown < required) {
            if (grown > SIZE_MAX / 2) {
                grown = required;
                break;
            }
            grown *= 2;
        }
        if (grown > SIZE_MAX / sizeof(*audio->consumed_samples)) {
            return 0;
        }
        float *next = realloc(audio->consumed_samples, grown * sizeof(*next));
        if (next == NULL) {
            return 0;
        }
        audio->consumed_samples = next;
        audio->consumed_capacity = grown;
    }
    memcpy(audio->consumed_samples + audio->consumed_count, samples, count * sizeof(*samples));
    audio->consumed_count = required;
    return 1;
}

static inline void notify_property(MockQueue *queue, bool synchronous) {
    if (queue->property_callback != NULL) {
        queue->property_callback(queue->property_context, (AudioQueueRef)queue,
                                 kAudioQueueProperty_IsRunning);
        if (synchronous) {
            pthread_mutex_lock(&queue->owner->mutex);
            queue->owner->synchronous_property_callbacks++;
            pthread_cond_broadcast(&queue->owner->condition);
            pthread_mutex_unlock(&queue->owner->mutex);
        }
    }
}

static inline void *mock_consumer(void *argument) {
    MockQueue *queue = argument;
    MockAudio *audio = queue->owner;
    for (;;) {
        pthread_mutex_lock(&audio->mutex);
        while (!queue->shutdown &&
               !(queue->running && audio->consume_enabled && queue->pending_count > 0)) {
            pthread_cond_wait(&audio->condition, &audio->mutex);
        }
        if (queue->shutdown) {
            pthread_mutex_unlock(&audio->mutex);
            return NULL;
        }
        AudioQueueBufferRef buffer = queue->pending[queue->pending_head];
        queue->pending_head = (queue->pending_head + 1) % KOKORO_STREAM_BUFFER_COUNT;
        queue->pending_count--;
        queue->processing = true;
        size_t sample_count = buffer->mAudioDataByteSize / sizeof(float);
        int copied = append_consumed_locked(audio, buffer->mAudioData, sample_count);
        pthread_cond_broadcast(&audio->condition);
        pthread_mutex_unlock(&audio->mutex);

        queue->output_callback(queue->output_context, (AudioQueueRef)queue, buffer);

        pthread_mutex_lock(&audio->mutex);
        audio->consumer_callback_count++;
        queue->processing = false;
        bool stopped =
            queue->graceful_stop && queue->pending_count == 0 && !audio->hold_graceful_stop;
        if (stopped) {
            queue->running = false;
            queue->graceful_stop = false;
        }
        pthread_cond_broadcast(&audio->condition);
        pthread_mutex_unlock(&audio->mutex);
        if (stopped) {
            notify_property(queue, false);
        }
        if (!copied) {
            pthread_mutex_lock(&audio->mutex);
            queue->shutdown = true;
            pthread_cond_broadcast(&audio->condition);
            pthread_mutex_unlock(&audio->mutex);
            return NULL;
        }
    }
}

static inline OSStatus mock_new_output(const AudioStreamBasicDescription *format,
                                       AudioQueueOutputCallback callback, void *user_data,
                                       CFRunLoopRef callback_run_loop,
                                       CFStringRef callback_run_loop_mode, UInt32 flags,
                                       AudioQueueRef *queue_out) {
    (void)callback_run_loop;
    (void)callback_run_loop_mode;
    (void)flags;
    MockAudio *audio = active_mock_audio;
    pthread_mutex_lock(&audio->mutex);
    int failure = audio->create_error;
    audio->format_valid =
        format->mSampleRate == 24000.0 && format->mFormatID == kAudioFormatLinearPCM &&
        format->mFormatFlags == kAudioFormatFlagsNativeFloatPacked &&
        format->mBytesPerPacket == sizeof(float) && format->mFramesPerPacket == 1 &&
        format->mBytesPerFrame == sizeof(float) && format->mChannelsPerFrame == 1 &&
        format->mBitsPerChannel == 32;
    pthread_mutex_unlock(&audio->mutex);
    if (failure != 0) {
        return (OSStatus)failure;
    }
    MockQueue *mock_queue = calloc(1, sizeof(*mock_queue));
    if (mock_queue == NULL) {
        return FAILURE_STATUS;
    }
    mock_queue->owner = audio;
    mock_queue->output_callback = callback;
    mock_queue->output_context = user_data;
    pthread_mutex_lock(&audio->mutex);
    audio->queue = mock_queue;
    audio->queue_created = true;
    pthread_mutex_unlock(&audio->mutex);
    if (pthread_create(&mock_queue->consumer, NULL, mock_consumer, mock_queue) != 0) {
        free(mock_queue);
        return FAILURE_STATUS;
    }
    *queue_out = (AudioQueueRef)mock_queue;
    return noErr;
}

static inline OSStatus mock_add_listener(AudioQueueRef queue_ref, AudioQueuePropertyID property,
                                         AudioQueuePropertyListenerProc callback, void *user_data) {
    MockQueue *queue = (MockQueue *)queue_ref;
    if (active_mock_audio->listener_error != 0) {
        return (OSStatus)active_mock_audio->listener_error;
    }
    if (property != kAudioQueueProperty_IsRunning) {
        return FAILURE_STATUS;
    }
    queue->property_callback = callback;
    queue->property_context = user_data;
    return noErr;
}

static inline OSStatus mock_remove_listener(AudioQueueRef queue_ref, AudioQueuePropertyID property,
                                            AudioQueuePropertyListenerProc callback,
                                            void *user_data) {
    MockQueue *queue = (MockQueue *)queue_ref;
    (void)callback;
    (void)user_data;
    if (property != kAudioQueueProperty_IsRunning) {
        return FAILURE_STATUS;
    }
    pthread_mutex_lock(&queue->owner->mutex);
    queue->owner->remove_listener_count++;
    pthread_mutex_unlock(&queue->owner->mutex);
    queue->property_callback = NULL;
    queue->property_context = NULL;
    return noErr;
}

static inline OSStatus mock_allocate_buffer(AudioQueueRef queue_ref, UInt32 byte_size,
                                            AudioQueueBufferRef *buffer_out) {
    MockQueue *queue = (MockQueue *)queue_ref;
    MockAudio *audio = queue->owner;
    pthread_mutex_lock(&audio->mutex);
    size_t index = queue->allocated_count++;
    size_t fail_at = audio->allocation_error_at;
    pthread_mutex_unlock(&audio->mutex);
    if (fail_at != 0 && index + 1 == fail_at) {
        return FAILURE_STATUS;
    }
    if (index >= KOKORO_STREAM_BUFFER_COUNT) {
        return FAILURE_STATUS;
    }
    AudioQueueBufferRef buffer = malloc(sizeof(*buffer));
    if (buffer == NULL) {
        return FAILURE_STATUS;
    }
    void *audio_data = calloc(1, byte_size);
    if (audio_data == NULL) {
        free(buffer);
        return FAILURE_STATUS;
    }
    AudioQueueBuffer initial_buffer = {
        .mAudioDataBytesCapacity = byte_size,
        .mAudioData = audio_data,
        .mAudioDataByteSize = 0,
        .mUserData = NULL,
        .mPacketDescriptionCapacity = 0,
        .mPacketDescriptions = NULL,
        .mPacketDescriptionCount = 0,
    };
    memcpy(buffer, &initial_buffer, sizeof(*buffer));
    queue->buffers[index] = buffer;
    *buffer_out = buffer;
    return noErr;
}

static inline OSStatus mock_enqueue(AudioQueueRef queue_ref, AudioQueueBufferRef buffer,
                                    UInt32 packet_description_count,
                                    const AudioStreamPacketDescription *packet_descriptions) {
    (void)packet_description_count;
    (void)packet_descriptions;
    MockQueue *queue = (MockQueue *)queue_ref;
    MockAudio *audio = queue->owner;
    pthread_mutex_lock(&audio->mutex);
    audio->enqueue_count++;
    size_t enqueue_number = audio->enqueue_count;
    audio->enqueue_entered = true;
    pthread_cond_broadcast(&audio->condition);
    while (audio->enqueue_blocked && !audio->release_blocked_api) {
        pthread_cond_wait(&audio->condition, &audio->mutex);
    }
    bool fail = audio->enqueue_error_at != 0 && enqueue_number == audio->enqueue_error_at;
    if (!fail) {
        size_t tail = (queue->pending_head + queue->pending_count) % KOKORO_STREAM_BUFFER_COUNT;
        queue->pending[tail] = buffer;
        queue->pending_count++;
        pthread_cond_broadcast(&audio->condition);
    }
    pthread_mutex_unlock(&audio->mutex);
    return fail ? FAILURE_STATUS : noErr;
}

static inline OSStatus mock_start(AudioQueueRef queue_ref, const AudioTimeStamp *start_time) {
    (void)start_time;
    MockQueue *queue = (MockQueue *)queue_ref;
    MockAudio *audio = queue->owner;
    pthread_mutex_lock(&audio->mutex);
    audio->start_count++;
    audio->start_entered = true;
    pthread_cond_broadcast(&audio->condition);
    while (audio->start_blocked && !audio->release_blocked_api) {
        pthread_cond_wait(&audio->condition, &audio->mutex);
    }
    bool fail = audio->start_error;
    if (!fail) {
        queue->running = true;
        pthread_cond_broadcast(&audio->condition);
    }
    pthread_mutex_unlock(&audio->mutex);
    if (!fail) {
        /* AudioQueue may notify its running-state listener synchronously during start. */
        notify_property(queue, true);
    }
    return fail ? FAILURE_STATUS : noErr;
}

static inline OSStatus mock_stop(AudioQueueRef queue_ref, Boolean immediate) {
    MockQueue *queue = (MockQueue *)queue_ref;
    MockAudio *audio = queue->owner;
    AudioQueueBufferRef released[KOKORO_STREAM_BUFFER_COUNT];
    size_t release_count = 0;
    bool notify = false;
    pthread_mutex_lock(&audio->mutex);
    if (immediate) {
        audio->immediate_stop_count++;
        while (queue->pending_count > 0) {
            released[release_count++] = queue->pending[queue->pending_head];
            queue->pending_head = (queue->pending_head + 1) % KOKORO_STREAM_BUFFER_COUNT;
            queue->pending_count--;
        }
        queue->running = false;
        queue->graceful_stop = false;
        notify = true;
    } else {
        audio->graceful_stop_count++;
        if (audio->graceful_stop_error) {
            pthread_mutex_unlock(&audio->mutex);
            return FAILURE_STATUS;
        }
        queue->graceful_stop = true;
        if (queue->pending_count == 0 && !queue->processing && !audio->hold_graceful_stop) {
            queue->running = false;
            queue->graceful_stop = false;
            notify = true;
        }
    }
    pthread_cond_broadcast(&audio->condition);
    pthread_mutex_unlock(&audio->mutex);
    for (size_t index = 0; index < release_count; index++) {
        queue->output_callback(queue->output_context, queue_ref, released[index]);
        pthread_mutex_lock(&audio->mutex);
        audio->synchronous_output_callbacks++;
        pthread_cond_broadcast(&audio->condition);
        pthread_mutex_unlock(&audio->mutex);
    }
    if (notify) {
        notify_property(queue, true);
    }
    return noErr;
}

static inline OSStatus mock_get_property(AudioQueueRef queue_ref, AudioQueuePropertyID property,
                                         void *data, UInt32 *data_size) {
    MockQueue *queue = (MockQueue *)queue_ref;
    MockAudio *audio = queue->owner;
    if (property != kAudioQueueProperty_IsRunning || data_size == NULL ||
        *data_size < sizeof(UInt32)) {
        return FAILURE_STATUS;
    }
    pthread_mutex_lock(&audio->mutex);
    bool fail = audio->property_error;
    UInt32 running = queue->running ? 1U : 0U;
    pthread_mutex_unlock(&audio->mutex);
    if (fail) {
        return FAILURE_STATUS;
    }
    memcpy(data, &running, sizeof(running));
    *data_size = sizeof(running);
    return noErr;
}

static inline OSStatus mock_dispose(AudioQueueRef queue_ref, Boolean immediate) {
    (void)immediate;
    MockQueue *queue = (MockQueue *)queue_ref;
    MockAudio *audio = queue->owner;
    pthread_mutex_lock(&audio->mutex);
    queue->shutdown = true;
    audio->dispose_count++;
    audio->pending_on_dispose = queue->pending_count;
    audio->running_on_dispose = queue->running;
    bool fail = audio->dispose_error;
    pthread_cond_broadcast(&audio->condition);
    pthread_mutex_unlock(&audio->mutex);
    (void)pthread_join(queue->consumer, NULL);
    for (size_t index = 0; index < queue->allocated_count; index++) {
        if (queue->buffers[index] != NULL) {
            free(queue->buffers[index]->mAudioData);
            free(queue->buffers[index]);
        }
    }
    pthread_mutex_lock(&audio->mutex);
    audio->queue = NULL;
    pthread_mutex_unlock(&audio->mutex);
    free(queue);
    return fail ? FAILURE_STATUS : noErr;
}

static const KokoroAudioQueueApi mock_api = {
    .new_output = mock_new_output,
    .add_property_listener = mock_add_listener,
    .remove_property_listener = mock_remove_listener,
    .allocate_buffer = mock_allocate_buffer,
    .enqueue_buffer = mock_enqueue,
    .start = mock_start,
    .stop = mock_stop,
    .get_property = mock_get_property,
    .dispose = mock_dispose,
};

static inline int mock_audio_init(MockAudio *audio, bool consume_enabled) {
    memset(audio, 0, sizeof(*audio));
    audio->consume_enabled = consume_enabled;
    return pthread_mutex_init(&audio->mutex, NULL) == 0 &&
           pthread_cond_init(&audio->condition, NULL) == 0;
}

static inline void mock_audio_destroy(MockAudio *audio) {
    free(audio->consumed_samples);
    pthread_cond_destroy(&audio->condition);
    pthread_mutex_destroy(&audio->mutex);
}

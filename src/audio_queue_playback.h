#ifndef SPEECH_AUDIO_QUEUE_PLAYBACK_H
#define SPEECH_AUDIO_QUEUE_PLAYBACK_H

#include "stream_playback.h"

#include <AudioToolbox/AudioQueue.h>

/* Function table keeps AudioQueue lifecycle checks deterministic without opening a device. */
typedef struct {
    OSStatus (*new_output)(const AudioStreamBasicDescription *format,
                           AudioQueueOutputCallback callback, void *user_data,
                           CFRunLoopRef callback_run_loop, CFStringRef callback_run_loop_mode,
                           UInt32 flags, AudioQueueRef *queue);
    OSStatus (*add_property_listener)(AudioQueueRef queue, AudioQueuePropertyID property,
                                      AudioQueuePropertyListenerProc callback, void *user_data);
    OSStatus (*remove_property_listener)(AudioQueueRef queue, AudioQueuePropertyID property,
                                         AudioQueuePropertyListenerProc callback, void *user_data);
    OSStatus (*allocate_buffer)(AudioQueueRef queue, UInt32 byte_size, AudioQueueBufferRef *buffer);
    OSStatus (*enqueue_buffer)(AudioQueueRef queue, AudioQueueBufferRef buffer,
                               UInt32 packet_description_count,
                               const AudioStreamPacketDescription *packet_descriptions);
    OSStatus (*start)(AudioQueueRef queue, const AudioTimeStamp *start_time);
    OSStatus (*stop)(AudioQueueRef queue, Boolean immediate);
    OSStatus (*get_property)(AudioQueueRef queue, AudioQueuePropertyID property, void *data,
                             UInt32 *data_size);
    /* Optional observation-only test hook, immediately before a blocking wait. */
    void (*waiting)(void);
    OSStatus (*dispose)(AudioQueueRef queue, Boolean immediate);
} SpeechAudioQueueApi;

/* Snapshot before close; callbacks may still be reclaiming buffers. Times are
 * CLOCK_MONOTONIC seconds, zero if no callback has reclaimed a buffer. */
void speech_audio_queue_reclaimed(void *playback, size_t *count, double *first);
const SpeechStreamPlaybackOps *speech_audio_queue_playback_ops(void);
const SpeechAudioQueueApi *speech_audio_queue_default_api(void);

#endif

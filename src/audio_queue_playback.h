#ifndef KOKORO_AUDIO_QUEUE_PLAYBACK_H
#define KOKORO_AUDIO_QUEUE_PLAYBACK_H

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
    OSStatus (*dispose)(AudioQueueRef queue, Boolean immediate);
} KokoroAudioQueueApi;

const KokoroStreamPlaybackOps *kokoro_audio_queue_playback_ops(void);
const KokoroAudioQueueApi *kokoro_audio_queue_default_api(void);

#endif

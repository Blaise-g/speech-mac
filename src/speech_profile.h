#ifndef SPEECH_PROFILE_H
#define SPEECH_PROFILE_H
/* Optional producer-thread wall scopes: top-level generation stages are disjoint;
 * operator scopes nest within them. Ordinary builds evaluate no clocks/callbacks. */
#ifdef SPEECH_PROFILE
void speech_profile_begin(const char *stage);
void speech_profile_end(const char *stage);
#define SPEECH_PROFILE_BEGIN(stage) speech_profile_begin(stage)
#define SPEECH_PROFILE_END(stage) speech_profile_end(stage)
#else
#define SPEECH_PROFILE_BEGIN(stage) ((void)0)
#define SPEECH_PROFILE_END(stage) ((void)0)
#endif
#endif

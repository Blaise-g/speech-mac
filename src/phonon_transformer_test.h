#ifndef PHONON_TRANSFORMER_TEST_H
#define PHONON_TRANSFORMER_TEST_H
#include <stddef.h>
/* Compiled only with PHONON_PACK_TEST_API. Observation/execution of actual
 * attention primitives, no replacement graph or model/fixture dispatch.
 * Single query uses interleaved Q/K64 and row-major K/V with explicit row stride (at least64
 * floats). Prefill uses [count,1536] with Q in first512, [count,512] K/V and output. */
int phonon_attention_test(const float query[64], const float *keys, const float *values,
                          size_t count, size_t stride, float output[64]);
int phonon_prefill_attention_test(const float *projected, const float *keys, const float *values,
                                  size_t count, float *output);
#endif

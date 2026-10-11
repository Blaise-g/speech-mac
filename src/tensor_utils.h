#ifndef KOKORO_TENSOR_UTILS_H
#define KOKORO_TENSOR_UTILS_H

#include "model_pack.h"

KokoroStatus kokoro_tensor_bind_f32(const KokoroPack *pack, const char *name, unsigned int rank,
                                    const uint64_t *shape, const float **destination,
                                    KokoroError *error);

#endif

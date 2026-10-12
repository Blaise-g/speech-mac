#include "tensor_utils.h"

#include "kokoro_internal.h"

#include <math.h>

KokoroStatus kokoro_tensor_bind_f32(const KokoroPack *pack, const char *name, unsigned int rank,
                                    const uint64_t *shape, const float **destination,
                                    KokoroError *error) {
    KokoroTensorView view;
    KokoroStatus status = kokoro_pack_tensor(pack, name, &view, error);
    if (status != KOKORO_OK) {
        return status;
    }
    if (view.type != KOKORO_TENSOR_F32 || view.rank != rank) {
        kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK,
                         "tensor has an incompatible type or rank: %s", name);
        return KOKORO_INCOMPATIBLE_PACK;
    }
    size_t element_count = 1;
    for (unsigned int dimension = 0; dimension < rank; dimension++) {
        if (view.shape[dimension] != shape[dimension] ||
            shape[dimension] > SIZE_MAX / element_count) {
            kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK,
                             "tensor has an incompatible shape: %s", name);
            return KOKORO_INCOMPATIBLE_PACK;
        }
        element_count *= (size_t)shape[dimension];
    }
    const float *values = view.data;
    for (size_t index = 0; index < element_count; index++) {
        if (!isfinite(values[index])) {
            kokoro_error_set(error, KOKORO_INCOMPATIBLE_PACK,
                             "tensor contains a non-finite value: %s", name);
            return KOKORO_INCOMPATIBLE_PACK;
        }
    }
    *destination = values;
    return KOKORO_OK;
}

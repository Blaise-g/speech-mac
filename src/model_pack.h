#ifndef MODEL_PACK_H
#define MODEL_PACK_H

#include "kokoro.h"

#include <stdint.h>

#define KOKORO_PACK_MAX_RANK 8

typedef enum { KOKORO_TENSOR_F32 = 1, KOKORO_TENSOR_I64 = 2 } KokoroTensorType;

typedef struct {
    KokoroTensorType type;
    unsigned int rank;
    uint64_t shape[KOKORO_PACK_MAX_RANK];
    const void *data;
    size_t byte_count;
} KokoroTensorView;

typedef struct KokoroPack KokoroPack;

/* The pack maps its file read-only; all views are borrowed until it is closed. */
KokoroStatus kokoro_pack_open(const char *path, KokoroPack **pack, KokoroError *error);
void kokoro_pack_close(KokoroPack *pack);
KokoroStatus kokoro_pack_tensor(const KokoroPack *pack, const char *name, KokoroTensorView *view,
                                KokoroError *error);
KokoroStatus kokoro_pack_voice_style(const KokoroPack *pack, const KokoroTensorView *override_style,
                                     KokoroTensorView *view, KokoroError *error);

#endif

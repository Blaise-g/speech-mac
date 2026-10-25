#ifndef PHONON_PACK_H
#define PHONON_PACK_H

#include <stddef.h>
#include <stdint.h>

#define PHONON_PACK_MAX_RANK 8

typedef enum { PHONON_PACK_Q8_0 = 1, PHONON_PACK_BF16 = 2 } PhononPackDType;
typedef struct {
    PhononPackDType dtype;
    unsigned int rank;
    uint64_t shape[PHONON_PACK_MAX_RANK];
    const void *data;
    size_t byte_count;
} PhononTensorView;
typedef struct PhononPack PhononPack;

/* Tensor names and data are borrowed from pack until phonon_pack_close. */
int phonon_pack_open(const char *path, PhononPack **pack);
void phonon_pack_close(PhononPack *pack);
size_t phonon_pack_tensor_count(const PhononPack *pack);
int phonon_pack_tensor(const PhononPack *pack, const char *name, PhononTensorView *view);
int phonon_pack_tensor_at(const PhononPack *pack, size_t index, const char **name,
                          size_t *name_length, PhononTensorView *view);
/* Converts count 16-bit BF16 values into count float outputs; buffers must not overlap. */
int phonon_bf16_to_float(const uint16_t *values, size_t count, float *output);

#endif

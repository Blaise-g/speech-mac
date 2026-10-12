#include "sha256.h"

#include <CommonCrypto/CommonDigest.h>
#include <stdint.h>

int kokoro_sha256_hex(const unsigned char *data, size_t length, char output[65]) {
    CC_SHA256_CTX context;
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    if (CC_SHA256_Init(&context) != 1) {
        return 0;
    }
    size_t offset = 0;
    while (offset < length) {
        size_t remaining = length - offset;
        CC_LONG chunk = remaining > UINT32_MAX ? UINT32_MAX : (CC_LONG)remaining;
        if (CC_SHA256_Update(&context, data + offset, chunk) != 1) {
            return 0;
        }
        offset += chunk;
    }
    if (CC_SHA256_Final(digest, &context) != 1) {
        return 0;
    }
    static const char digits[] = "0123456789abcdef";
    for (size_t index = 0; index < sizeof(digest); index++) {
        output[index * 2] = digits[digest[index] >> 4];
        output[index * 2 + 1] = digits[digest[index] & 0x0f];
    }
    output[64] = '\0';
    return 1;
}

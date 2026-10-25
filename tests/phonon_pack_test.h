#ifndef PHONON_PACK_TEST_H
#define PHONON_PACK_TEST_H

#include "phonon_pack.h"

/* Test-only route through the production parser without the pinned index check. */
int phonon_pack_open_for_tests(const char *path, PhononPack **pack);

#endif

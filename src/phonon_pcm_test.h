#ifndef PHONON_PCM_TEST_H
#define PHONON_PCM_TEST_H
#include "phonon_conv.h"
#include "phonon_pcm.h"
/* Test-only synthetic graph injection. On success takes ownership of all eleven
 * convolution models, borrows Mimi tensor backing; same production state path. */
int phonon_pcm_test_model(const PhononMimiModel *mimi, PhononConvModel *conv[11],
                          PhononPCMModel **model);
#endif

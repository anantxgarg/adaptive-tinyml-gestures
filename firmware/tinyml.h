#ifndef TINYML_H
#define TINYML_H

#include <stddef.h>
#include <stdint.h>
#include "features.h"

#define TML_EMB_DIM 16

/* Reset the adaptive head and replay buffer to factory state. */
void tml_init(void);

/* Classify a window. Returns class id, writes softmax confidence to *conf (may be NULL). */
int tml_predict(const float win[TML_WIN][3], float *conf);

/* Frozen int8 backbone output (TML_EMB_DIM values). */
void tml_embed(const float win[TML_WIN][3], int8_t emb[TML_EMB_DIM]);

/* On-device learning: teach the device that `win` is gesture `label`. Returns 0 on success. */
int tml_learn(const float win[TML_WIN][3], int label);

/* Allocate a slot for a brand-new gesture. Returns its class id, or -1 if full. */
int tml_add_class(void);

int tml_num_classes(void);
size_t tml_flash_bytes(void);   /* constant model data */
size_t tml_ram_bytes(void);     /* mutable learning state */

#endif

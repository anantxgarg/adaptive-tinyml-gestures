/*
 * Adaptive TinyML engine.
 *   features -> int8 backbone (frozen, flash) -> float linear head (trainable, RAM)
 * On-device learning: SGD on the head over a small replay buffer of int8 embeddings,
 * seeded with factory exemplars so untouched classes are not forgotten.
 * No malloc, no external dependencies.
 */
#include <math.h>
#include <string.h>
#include <stdint.h>
#include "tinyml.h"
#include "model_data.h"

_Static_assert(TML_EMB_DIM == TML_H2, "embedding size mismatch");

#ifndef REPLAY_PER_CLASS
#define REPLAY_PER_CLASS 8
#endif
#ifndef LEARN_EPOCHS
#define LEARN_EPOCHS 30
#endif
#ifndef LEARN_RATE
#define LEARN_RATE 0.05f
#endif
#ifndef L2_TO_FACTORY
#define L2_TO_FACTORY 0.001f
#endif

static float head_w[TML_MAX_CLASSES][TML_H2];
static float head_b[TML_MAX_CLASSES];
static int8_t replay[TML_MAX_CLASSES][REPLAY_PER_CLASS][TML_H2];
static uint8_t replay_n[TML_MAX_CLASSES];     /* valid slots */
static uint8_t replay_next[TML_MAX_CLASSES];  /* ring write position */
static int n_classes;

void tml_init(void)
{
    memset(head_w, 0, sizeof head_w);
    memset(head_b, 0, sizeof head_b);
    memset(replay_n, 0, sizeof replay_n);
    memcpy(head_w, tml_head_w0, sizeof tml_head_w0);
    memcpy(head_b, tml_head_b0, sizeof tml_head_b0);
    for (int c = 0; c < TML_N_BASE_CLASSES; c++) {
        memcpy(replay[c], tml_exemplars[c], sizeof tml_exemplars[c]);
        replay_n[c] = TML_N_EXEMPLARS;
        replay_next[c] = TML_N_EXEMPLARS;   /* fill free slots first, then overwrite factory */
    }
    n_classes = TML_N_BASE_CLASSES;
}

static int8_t sat_relu(int32_t acc, float m)
{
    float v = (float)acc * m;
    if (v <= 0.f) return 0;
    int32_t q = (int32_t)(v + 0.5f);
    return (int8_t)(q > 127 ? 127 : q);
}

void tml_embed(const float win[TML_WIN][3], int8_t emb[TML_EMB_DIM])
{
    float f[TML_N_FEAT];
    int8_t x[TML_N_FEAT], h1[TML_H1];

    tml_features(win, f);
    for (int i = 0; i < TML_N_FEAT; i++) {
        float v = roundf((f[i] - tml_feat_mean[i]) * tml_feat_inv_std[i] * TML_IN_INV_SCALE);
        x[i] = (int8_t)(v > 127.f ? 127 : (v < -127.f ? -127 : v));
    }
    for (int o = 0; o < TML_H1; o++) {
        int32_t acc = tml_b1[o];
        for (int i = 0; i < TML_N_FEAT; i++) acc += (int32_t)tml_w1[o][i] * x[i];
        h1[o] = sat_relu(acc, TML_M1);
    }
    for (int o = 0; o < TML_H2; o++) {
        int32_t acc = tml_b2[o];
        for (int i = 0; i < TML_H1; i++) acc += (int32_t)tml_w2[o][i] * h1[i];
        emb[o] = sat_relu(acc, TML_M2);
    }
}

/* Softmax over active classes; returns argmax. */
static int head_forward(const int8_t emb[TML_H2], float e[TML_H2], float p[TML_MAX_CLASSES])
{
    float mx = -1e30f, sum = 0.f;
    int best = 0;
    for (int i = 0; i < TML_H2; i++) e[i] = emb[i] * TML_EMB_SCALE;
    for (int c = 0; c < n_classes; c++) {
        float z = head_b[c];
        for (int i = 0; i < TML_H2; i++) z += head_w[c][i] * e[i];
        p[c] = z;
        if (z > mx) { mx = z; best = c; }
    }
    for (int c = 0; c < n_classes; c++) sum += (p[c] = expf(p[c] - mx));
    for (int c = 0; c < n_classes; c++) p[c] /= sum;
    return best;
}

int tml_predict(const float win[TML_WIN][3], float *conf)
{
    int8_t emb[TML_H2];
    float e[TML_H2], p[TML_MAX_CLASSES];
    tml_embed(win, emb);
    int c = head_forward(emb, e, p);
    if (conf) *conf = p[c];
    return c;
}

static void sgd_step(const int8_t emb[TML_H2], int label)
{
    float e[TML_H2], p[TML_MAX_CLASSES];
    head_forward(emb, e, p);
    for (int c = 0; c < n_classes; c++) {
        float g = p[c] - (c == label ? 1.f : 0.f);
        for (int i = 0; i < TML_H2; i++) {
            float w0 = c < TML_N_BASE_CLASSES ? tml_head_w0[c][i] : 0.f;
            head_w[c][i] -= LEARN_RATE * (g * e[i] + L2_TO_FACTORY * (head_w[c][i] - w0));
        }
        head_b[c] -= LEARN_RATE * g;
    }
}

int tml_learn(const float win[TML_WIN][3], int label)
{
    if (label < 0 || label >= n_classes) return -1;

    uint8_t *slot = &replay_next[label];
    tml_embed(win, replay[label][*slot]);
    *slot = (uint8_t)((*slot + 1) % REPLAY_PER_CLASS);
    if (replay_n[label] < REPLAY_PER_CLASS) replay_n[label]++;

    /* class-interleaved passes over the replay buffer */
    for (int ep = 0; ep < LEARN_EPOCHS; ep++)
        for (int s = 0; s < REPLAY_PER_CLASS; s++)
            for (int c = 0; c < n_classes; c++)
                if (s < replay_n[c]) sgd_step(replay[c][s], c);
    return 0;
}

int tml_add_class(void)
{
    if (n_classes >= TML_MAX_CLASSES) return -1;
    int c = n_classes++;
    memset(head_w[c], 0, sizeof head_w[c]);
    head_b[c] = 0.f;
    replay_n[c] = 0;
    replay_next[c] = 0;
    return c;
}

int tml_num_classes(void) { return n_classes; }

size_t tml_flash_bytes(void)
{
    return sizeof tml_feat_mean + sizeof tml_feat_inv_std + sizeof tml_w1 + sizeof tml_b1 +
           sizeof tml_w2 + sizeof tml_b2 + sizeof tml_head_w0 + sizeof tml_head_b0 +
           sizeof tml_exemplars;
}

size_t tml_ram_bytes(void)
{
    return sizeof head_w + sizeof head_b + sizeof replay + sizeof replay_n +
           sizeof replay_next + sizeof n_classes;
}

/* Feature extraction. Must stay in sync with python/features.py. */
#include <math.h>
#include "features.h"

void tml_features(const float win[TML_WIN][3], float out[TML_N_FEAT])
{
    int n = 0;
    float mean[3], sd[3];

    for (int ax = 0; ax < 3; ax++) {
        float s = 0.f, mn = win[0][ax], mx = win[0][ax], ad = 0.f, var = 0.f;
        for (int i = 0; i < TML_WIN; i++) {
            float v = win[i][ax];
            s += v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            if (i) ad += fabsf(v - win[i - 1][ax]);
        }
        float mu = s / TML_WIN;
        int zc = 0;
        for (int i = 0; i < TML_WIN; i++) {
            float c = win[i][ax] - mu;
            var += c * c;
            if (i && ((win[i - 1][ax] - mu >= 0.f) != (c >= 0.f))) zc++;
        }
        mean[ax] = mu;
        sd[ax] = sqrtf(var / TML_WIN);
        out[n++] = mu;
        out[n++] = sd[ax];
        out[n++] = mn;
        out[n++] = mx;
        out[n++] = ad / (TML_WIN - 1);
        out[n++] = (float)zc / (TML_WIN - 1);
    }

    float mag[TML_WIN], ms = 0.f, mv = 0.f;
    for (int i = 0; i < TML_WIN; i++) {
        mag[i] = sqrtf(win[i][0] * win[i][0] + win[i][1] * win[i][1] + win[i][2] * win[i][2]);
        ms += mag[i];
    }
    ms /= TML_WIN;
    for (int i = 0; i < TML_WIN; i++) mv += (mag[i] - ms) * (mag[i] - ms);
    out[n++] = ms;
    out[n++] = sqrtf(mv / TML_WIN);

    static const int pairs[3][2] = {{0, 1}, {0, 2}, {1, 2}};
    for (int p = 0; p < 3; p++) {
        int a = pairs[p][0], b = pairs[p][1];
        float cov = 0.f;
        for (int i = 0; i < TML_WIN; i++) cov += (win[i][a] - mean[a]) * (win[i][b] - mean[b]);
        out[n++] = (cov / TML_WIN) / (sd[a] * sd[b] + 1e-6f);
    }
}

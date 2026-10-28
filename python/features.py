"""Feature extraction. Must stay in sync with firmware/features.c."""
import numpy as np

N_FEAT = 23


def extract(w):
    """w: (WIN, 3) float32 window -> (N_FEAT,) float32."""
    w = w.astype(np.float32)
    f = []
    for ax in range(3):
        x = w[:, ax]
        mu = x.mean()
        c = x - mu
        f += [mu, np.sqrt((c * c).mean()), x.min(), x.max(),
              np.abs(np.diff(x)).mean(),
              np.count_nonzero((c[:-1] >= 0) != (c[1:] >= 0)) / (len(x) - 1)]
    mag = np.sqrt((w * w).sum(axis=1))
    mmu = mag.mean()
    f += [mmu, np.sqrt(((mag - mmu) ** 2).mean())]
    cen = w - w.mean(axis=0)
    sd = np.sqrt((cen * cen).mean(axis=0))
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        f.append((cen[:, i] * cen[:, j]).mean() / (sd[i] * sd[j] + 1e-6))
    return np.array(f, dtype=np.float32)


def extract_all(X):
    return np.stack([extract(w) for w in X])

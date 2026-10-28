"""Synthetic 3-axis accelerometer gesture generator.

Each user has a "style": how the device is worn (rotation), how fast and how big
they move, sensor bias and noise. Factory users are similar to each other; the
deployment user is deliberately different so the factory model degrades on them.
"""
import numpy as np

FS = 50          # Hz
WIN = 100        # samples per window (2 s)
BASE_CLASSES = ["idle", "wave", "circle", "shake", "punch"]
NEW_CLASS = "flip"   # never seen in the factory; learned on-device


def rot(yaw, pitch, roll):
    """Rotation matrix from degrees (Z-Y-X)."""
    y, p, r = np.radians([yaw, pitch, roll])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    return rz @ ry @ rx


def factory_user(rng):
    return dict(
        R=rot(rng.uniform(-15, 15), rng.uniform(-8, 8), rng.uniform(-8, 8)),
        speed=rng.uniform(0.85, 1.15),
        amp=rng.uniform(0.85, 1.2),
        bias=rng.normal(0, 0.02, 3),
        noise=rng.uniform(0.02, 0.05),
    )


def new_user():
    """The deployment user: device worn rotated, fast and small movements."""
    return dict(
        R=rot(70, 15, 25),
        speed=1.4,
        amp=0.65,
        bias=np.array([0.06, -0.05, 0.04]),
        noise=0.06,
    )


def gesture(name, user, rng):
    t = np.arange(WIN) / FS
    sp = user["speed"] * rng.uniform(0.92, 1.08)
    amp = user["amp"] * rng.uniform(0.9, 1.1)
    ph = rng.uniform(0, 2 * np.pi)
    g = np.tile([0.0, 0.0, 1.0], (WIN, 1))   # gravity, device flat
    m = np.zeros((WIN, 3))                   # linear motion

    if name == "idle":
        m[:, 0] = 0.03 * np.sin(2 * np.pi * 0.3 * t + ph)
    elif name == "wave":
        m[:, 0] = 0.8 * amp * np.sin(2 * np.pi * 2.0 * sp * t + ph)
    elif name == "circle":
        m[:, 0] = 0.6 * amp * np.sin(2 * np.pi * 1.2 * sp * t + ph)
        m[:, 1] = 0.6 * amp * np.cos(2 * np.pi * 1.2 * sp * t + ph)
    elif name == "shake":
        for ax in range(3):
            f = rng.uniform(6, 8) * sp
            m[:, ax] = 0.5 * amp * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        m += rng.normal(0, 0.15 * amp, m.shape)
    elif name == "punch":
        c = rng.uniform(0.5, 1.5)
        w = 0.06 / sp
        m[:, 0] = 2.0 * amp * np.exp(-((t - c) / w) ** 2) \
            - 0.8 * amp * np.exp(-((t - c - 2.5 * w) / (1.5 * w)) ** 2)
    elif name == "flip":
        start = rng.uniform(0.3, 0.9)
        dur = 0.6 / sp
        ang = np.pi * np.clip((t - start) / dur, 0, 1)    # rotate about x by 180 deg
        g = np.stack([np.zeros(WIN), np.sin(ang), np.cos(ang)], axis=1)
    else:
        raise ValueError(name)

    a = (g + m) @ user["R"].T + user["bias"]
    a += rng.normal(0, user["noise"], a.shape)
    return a.astype(np.float32)


def make_set(users, classes, per_class, rng):
    X, y = [], []
    for u in users:
        for ci, c in enumerate(classes):
            for _ in range(per_class):
                X.append(gesture(c, u, rng))
                y.append(ci)
    return np.array(X), np.array(y)

"""End-to-end demo: drives the compiled C firmware (firmware/device_sim) like a real device.

1. parity check   C int8 vs Python int8 reference
2. factory model  on factory-like users vs. a new, very different user
3. personalize    few-shot on-device learning, accuracy after each round of shots
4. new gesture    add a gesture the factory never saw, learned on-device
"""
import os
import subprocess
import sys

import numpy as np

from data import BASE_CLASSES, NEW_CLASS, factory_user, make_set, new_user
from features import extract_all
from train import NPZ, ROOT, head_logits, main as train_main, q_embed

FW = os.path.join(ROOT, "firmware")


class Device:
    def __init__(self):
        self.p = subprocess.Popen([os.path.join(FW, "device_sim")], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True)

    def cmd(self, c, win=None):
        if win is not None:
            c += " " + " ".join(f"{v:.9g}" for v in win.ravel())
        self.p.stdin.write(c + "\n")
        self.p.stdin.flush()
        out = self.p.stdout.readline().strip()
        if out == "ERR":
            raise RuntimeError(f"device rejected command {c[:20]!r}")
        return out

    def predict(self, X):
        r = [self.cmd("P", w).split() for w in X]
        return np.array([int(a) for a, _ in r]), np.array([float(b) for _, b in r])

    def accuracy(self, X, y):
        p, conf = self.predict(X)
        return (p == y).mean(), conf.mean()

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def main():
    print("== factory training ==")
    train_main()
    subprocess.run(["make", "-s", "-C", FW], check=True)
    q = dict(np.load(NPZ))

    rng = np.random.default_rng(7)
    ref_users = [factory_user(rng) for _ in range(2)]
    Xf, yf = make_set(ref_users, BASE_CLASSES, 30, rng)          # factory-like users
    user = new_user()
    Xa, ya = make_set([user], BASE_CLASSES, 8, rng)               # user's teaching pool
    Xt, yt = make_set([user], BASE_CLASSES, 60, rng)              # user's test set
    Xa_new, _ = make_set([user], [NEW_CLASS], 8, rng)
    Xt_new, _ = make_set([user], [NEW_CLASS], 60, rng)

    dev = Device()
    print("\n== device ==")
    print(dev.cmd("I"))

    # 1. parity
    c_emb = np.array([[int(v) for v in dev.cmd("E", w).split()] for w in Xt])
    py_emb = q_embed(extract_all(Xt), q)
    c_pred, _ = dev.predict(Xt)
    py_pred = head_logits(py_emb, q).argmax(1)
    print("\n== parity (C firmware vs Python int8 reference) ==")
    print(f"embeddings bit-exact : {(c_emb == py_emb).all(1).mean() * 100:.1f}%  "
          f"(max |diff| {np.abs(c_emb - py_emb).max()} LSB)")
    print(f"predictions agree    : {(c_pred == py_pred).mean() * 100:.1f}%")

    # 2. factory model
    print("\n== factory model ==")
    acc_f, _ = dev.accuracy(Xf, yf)
    acc_u, conf_u = dev.accuracy(Xt, yt)
    print(f"factory-like users : {acc_f * 100:5.1f}%")
    print(f"new user           : {acc_u * 100:5.1f}%   (mean confidence {conf_u:.2f})")

    # 3. few-shot personalization, one shot per class per round
    print("\n== on-device personalization (new user) ==")
    print(f"{'shots/class':>11} | {'new user':>8} | {'confidence':>10} | {'factory users':>13}")
    print(f"{0:>11} | {acc_u * 100:7.1f}% | {conf_u:10.2f} | {acc_f * 100:12.1f}%")
    idx = {c: np.flatnonzero(ya == c) for c in range(len(BASE_CLASSES))}
    for shot in range(1, 9):
        for c in range(len(BASE_CLASSES)):
            dev.cmd(f"L {c}", Xa[idx[c][shot - 1]])
        if shot in (1, 2, 3, 5, 8):
            a, cf = dev.accuracy(Xt, yt)
            af, _ = dev.accuracy(Xf, yf)
            print(f"{shot:>11} | {a * 100:7.1f}% | {cf:10.2f} | {af * 100:12.1f}%")

    # 4. learn a brand-new gesture
    new_id = int(dev.cmd("A"))
    print(f"\n== new gesture '{NEW_CLASS}' added on-device as class {new_id} ==")
    X6 = np.concatenate([Xt, Xt_new])
    y6 = np.concatenate([yt, np.full(len(Xt_new), new_id)])
    print(f"{'shots':>5} | {NEW_CLASS + ' acc':>8} | {'old classes':>11} | {'all 6':>6}")
    for shot in range(1, 6):
        dev.cmd(f"L {new_id}", Xa_new[shot - 1])
        if shot in (1, 3, 5):
            p, _ = dev.predict(X6)
            new_mask = y6 == new_id
            print(f"{shot:>5} | {(p[new_mask] == new_id).mean() * 100:7.1f}% | "
                  f"{(p[~new_mask] == y6[~new_mask]).mean() * 100:10.1f}% | "
                  f"{(p == y6).mean() * 100:5.1f}%")

    print("\n== final confusion matrix (rows = true, cols = predicted) ==")
    names = BASE_CLASSES + [NEW_CLASS]
    p, _ = dev.predict(X6)
    print(" " * 8 + "".join(f"{n[:6]:>7}" for n in names))
    for c, n in enumerate(names):
        print(f"{n:>8}" + "".join(f"{v:>7}" for v in np.bincount(p[y6 == c], minlength=len(names))))

    print("\n" + dev.cmd("I"))
    macs = q["w1"].size + q["w2"].size + q["head_w"].shape[1] * new_id
    print(f"inference cost: ~{macs} MACs/window (int8 backbone + float head), 1 window / 2 s")
    dev.close()


if __name__ == "__main__":
    sys.exit(main())

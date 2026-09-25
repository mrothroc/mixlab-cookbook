#!/usr/bin/env python3
"""CIFAR-10 -> normalized 4x4 patch sequences, with no torchvision dependency.

Reads the canonical pickle batches directly so it runs inside the mixlab RunPod
image, which ships numpy but not torch. Output is byte-identical to
killgate/prep_cifar.py --norm reference.
"""
import argparse, json, os, pickle, sys, tarfile
import numpy as np

P, G = 4, 8
T, F = G * G, P * P * 3
SHUFFLE_SEED = 1234
REFERENCE_MEAN = (0.4914, 0.4822, 0.4465)
REFERENCE_STD = (0.2023, 0.1994, 0.2010)


def patchify(x):
    n = x.shape[0]
    x = x.reshape(n, G, P, G, P, 3).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(n, T, F)


def unpatchify(p):
    n = p.shape[0]
    x = p.reshape(n, G, G, P, P, 3).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(n, 32, 32, 3)


def load_raw(tar_path, work):
    with tarfile.open(tar_path) as tf:
        tf.extractall(work)
    root = os.path.join(work, "cifar-10-batches-py")
    def rd(name):
        with open(os.path.join(root, name), "rb") as f:
            d = pickle.load(f, encoding="bytes")
        # stored as [N, 3072] with channel-major planes; restore HWC
        x = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        return x, np.asarray(d[b"labels"], dtype=np.int64)
    xs, ys = zip(*[rd(f"data_batch_{i}") for i in range(1, 6)])
    xte, yte = rd("test_batch")
    return np.concatenate(xs), np.concatenate(ys), xte, yte


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    xtr, ytr, xte, yte = load_raw(a.tar, a.out)
    assert xtr.shape == (50000, 32, 32, 3) and xte.shape == (10000, 32, 32, 3)

    mean = np.asarray(REFERENCE_MEAN, dtype=np.float64)
    std = np.asarray(REFERENCE_STD, dtype=np.float64)
    norm = lambda u8: ((u8.astype(np.float64) / 255.0 - mean) / std).astype(np.float32)

    assert np.array_equal(unpatchify(patchify(xtr[:64])), xtr[:64]), "round-trip failed"
    ptr, pte = patchify(norm(xtr)), patchify(norm(xte))
    assert np.isfinite(ptr).all() and np.isfinite(pte).all()

    rng = np.random.default_rng(SHUFFLE_SEED)
    perm = rng.permutation(50000); ptr, ytr = ptr[perm], ytr[perm]
    permt = rng.permutation(10000); pte, yte = pte[permt], yte[permt]
    assert (np.bincount(ytr, minlength=10) == 5000).all()
    assert len(set(ytr[:512].tolist())) == 10

    np.save(os.path.join(a.out, "train_features.npy"), ptr)
    np.save(os.path.join(a.out, "test_features.npy"), pte)
    for name, y in (("train_labels.tsv", ytr), ("test_labels.tsv", yte)):
        with open(os.path.join(a.out, name), "w") as f:
            for i, lab in enumerate(y):
                f.write(f"{i}\t{int(lab)}\n")
    json.dump({"mean": mean.tolist(), "std": std.tolist(), "source": "reference"},
              open(os.path.join(a.out, "norm_stats.json"), "w"), indent=2)
    print(f"PREP_OK train={ptr.shape} test={pte.shape}")


if __name__ == "__main__":
    sys.exit(main())

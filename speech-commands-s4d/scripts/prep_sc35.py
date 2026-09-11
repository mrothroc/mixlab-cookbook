#!/usr/bin/env python3
"""Speech Commands v0.02 -> normalized [N,16000,1] float32 arrays for mixlab.

Matches state-spaces/s4 `src/dataloaders/datasets/sc.py`:
  * right-pad zeros to 16000
  * official validation_list.txt / testing_list.txt splits
  * z-score with TRAIN statistics only, (std + 1e-5), applied to every split

The reference's /2**15 is omitted: z-scoring is scale invariant, so it is a no-op.

Three things here are not cosmetic. Each one cost us real time to find:

1. GLOBAL PERMUTATION. `mixlab -mode prepare` preserves input record order. If you write
   records grouped by class, every shard ends up nearly single-label, batches contain
   about one distinct class, and accuracy collapses by roughly 87 points. We permute all
   three splits with a fixed seed before writing.

2. FLOAT64 STATISTICS. Accumulating the mean over 1.36e9 float32 values shifts it in the
   4th significant digit, which quietly changes the normalization. Both accumulators are
   float64 here.

3. LABELS FOLLOW THE PERMUTATION. Labels are permuted with the same index array, so row i
   of the array still matches row i of the label file.

Usage:
    python prep_sc35.py /path/to/SpeechCommands/speech_commands_v0.02 out_dir [seed]

Expected on v0.02 (verify these before training):
    train 84,843   val 9,981   test 11,005
    TRAIN mean -2.791959   std 2818.166625
"""
import os, sys, json, wave
import numpy as np

ALL_CLASSES = ["bed","cat","down","five","forward","go","house","left","marvin","no","on",
               "right","sheila","tree","up","visual","yes","backward","bird","dog","eight",
               "follow","four","happy","learn","nine","off","one","seven","six","stop",
               "three","two","wow","zero"]
L = 16000

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
OUT  = sys.argv[2] if len(sys.argv) > 2 else "prep"
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 2222
os.makedirs(OUT, exist_ok=True)
assert len(ALL_CLASSES) == 35, f"expected 35 classes, got {len(ALL_CLASSES)}"

for req in ("validation_list.txt", "testing_list.txt"):
    p = os.path.join(ROOT, req)
    if not os.path.exists(p):
        sys.exit(f"missing {p}. Point ROOT at the directory containing the 35 class folders "
                 f"and the official split lists.")

val_list  = set(open(os.path.join(ROOT, "validation_list.txt")).read().split())
test_list = set(open(os.path.join(ROOT, "testing_list.txt")).read().split())
overlap = val_list & test_list
if overlap:
    sys.exit(f"{len(overlap)} files appear in BOTH validation_list.txt and testing_list.txt "
             f"(e.g. {sorted(overlap)[0]}). Splits must be disjoint.")

# Pass 1: enumerate deterministically, assign splits from the official lists.
items = {"train": [], "val": [], "test": []}
for yi, cls in enumerate(ALL_CLASSES):
    d = os.path.join(ROOT, cls)
    if not os.path.isdir(d):
        sys.exit(f"missing class directory {d}")
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".wav"):
            continue
        rel = f"{cls}/{fn}"
        split = "val" if rel in val_list else ("test" if rel in test_list else "train")
        items[split].append((os.path.join(d, fn), yi, rel))

# FIX 1: global permutation. Without this, shards are class ordered and training collapses.
rng = np.random.default_rng(SEED)
for split in items:
    order = rng.permutation(len(items[split]))
    items[split] = [items[split][i] for i in order]
    print(f"  {split:6} {len(items[split]):>7,}  permuted with seed {SEED}", flush=True)

def read_wav(path):
    with wave.open(path, "rb") as w:
        # Sample rate is checked because getting it wrong is SILENT: an 8 kHz file
        # produces bit-identical arrays and identical mean/std, so the printed
        # statistics cannot catch it, and the model then scores at chance.
        assert w.getframerate() == 16000, f"{path}: expected 16000 Hz, got {w.getframerate()}"
        assert w.getsampwidth() == 2 and w.getnchannels() == 1, path
        raw = w.readframes(w.getnframes())
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    if a.shape[0] >= L:
        return a[:L]
    out = np.zeros(L, dtype=np.float32)   # right-pad, matching the reference
    out[:a.shape[0]] = a
    return out

arrays = {s: np.lib.format.open_memmap(os.path.join(OUT, f"{s}_X.npy"), mode="w+",
                                       dtype=np.float32, shape=(len(v), L, 1))
          for s, v in items.items()}

# FIX 2: float64 accumulation for BOTH moments.
s_acc = ss_acc = 0.0
n_acc = 0
for split, lst in items.items():
    A = arrays[split]
    for i, (p, yi, rel) in enumerate(lst):
        x = read_wav(p)
        A[i, :, 0] = x
        if split == "train":
            x64 = x.astype(np.float64)
            s_acc  += float(x64.sum())
            ss_acc += float((x64 ** 2).sum())
            n_acc  += L
        if i and i % 10000 == 0:
            print(f"  {split} {i}/{len(lst)}", flush=True)

mean = s_acc / n_acc
std  = float(np.sqrt(max(ss_acc / n_acc - mean * mean, 0.0)))
print(f"  TRAIN mean={mean:.6f} std={std:.6f}  (n={n_acc:,})", flush=True)
print( "  expected on v0.02: mean=-2.791959 std=2818.166625", flush=True)

for split in items:
    A = arrays[split]
    for i in range(0, A.shape[0], 4096):
        A[i:i+4096] = (A[i:i+4096] - mean) / (std + 1e-5)
    A.flush()
    # FIX 3: labels carry the same permutation as the rows.
    labels = np.array([yi for _, yi, _ in items[split]], dtype=np.int64)
    np.savetxt(os.path.join(OUT, f"{split}_labels.tsv"),
               np.stack([np.arange(len(labels)), labels], 1), fmt="%d", delimiter="\t")

json.dump({"classes": ALL_CLASSES, "seq_len": L, "seed": SEED,
           "counts": {k: len(v) for k, v in items.items()},
           "train_mean": mean, "train_std": std,
           "normalization": "z-score with TRAIN stats, (std + 1e-5), per state-spaces/s4 sc.py",
           "padding": "right-pad zeros to 16000",
           "order": f"globally permuted, numpy default_rng({SEED})",
           "splits": "official validation_list.txt / testing_list.txt"},
          open(os.path.join(OUT, "manifest.json"), "w"), indent=2)
print("  DONE", flush=True)

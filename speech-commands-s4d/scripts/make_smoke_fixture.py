#!/usr/bin/env python3
"""Build a tiny synthetic Speech Commands tree, for exercising the pipeline.

This produces the same LAYOUT as the real v0.02 archive (35 class folders plus
validation_list.txt and testing_list.txt), with a handful of generated clips per
class instead of 105,829 real ones. It exists so you can run steps 1 through 5
end to end in a couple of minutes before committing a GPU day to a real run.

The audio is noise. Nothing trained on it will classify anything. The point is
to prove the plumbing works: the prep script, the shard counts, the checkpoint
interval, eval, and resume.

Usage:
    python3 scripts/make_smoke_fixture.py smoke_data [clips_per_class_per_split]
"""
import os, sys, wave, struct, random

# Same order as prep_sc35.py. Class identity does not matter for a smoke test,
# but the folder names must match or prep_sc35.py will not find them.
ALL_CLASSES = ["bed","cat","down","five","forward","go","house","left","marvin","no","on",
               "right","sheila","tree","up","visual","yes","backward","bird","dog","eight",
               "follow","four","happy","learn","nine","off","one","seven","six","stop",
               "three","two","wow","zero"]
L = 16000

ROOT = sys.argv[1] if len(sys.argv) > 1 else "smoke_data"
PER  = int(sys.argv[2]) if len(sys.argv) > 2 else 2
rng  = random.Random(2222)

os.makedirs(ROOT, exist_ok=True)
val_lines, test_lines = [], []

for cls in ALL_CLASSES:
    d = os.path.join(ROOT, cls)
    os.makedirs(d, exist_ok=True)
    for split in ("train", "val", "test"):
        for i in range(PER):
            name = "%s_%s_%d.wav" % (split, cls, i)
            # 16 kHz, mono, PCM16: what prep_sc35.py asserts on.
            with wave.open(os.path.join(d, name), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(struct.pack(
                    "<%dh" % L, *(rng.randint(-3000, 3000) for _ in range(L))))
            if split == "val":
                val_lines.append("%s/%s" % (cls, name))
            elif split == "test":
                test_lines.append("%s/%s" % (cls, name))

for fn, lines in (("validation_list.txt", val_lines), ("testing_list.txt", test_lines)):
    with open(os.path.join(ROOT, fn), "w") as f:
        f.write("\n".join(lines) + "\n")

n = len(ALL_CLASSES) * PER
print("wrote %s: %d clips total, %d train / %d val / %d test across %d classes"
      % (ROOT, n * 3, n, n, n, len(ALL_CLASSES)))
print("next: python3 scripts/prep_sc35.py %s smoke_prep" % ROOT)

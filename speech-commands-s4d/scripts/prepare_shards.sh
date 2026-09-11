#!/usr/bin/env bash
# Convert the normalized arrays from prep_sc35.py into mixlab shards.
#
# Each split gets its own directory, and -val-split MUST be 0.
#
# mixlab's -val-split defaults to 0.1. Omitting the flag does NOT disable it: it
# holds back 10% of whatever you hand it. Speech Commands ships official
# validation_list.txt / testing_list.txt splits, so any extra holdout both
# shrinks the training set and makes "the full devel split" unreachable.
set -euo pipefail

PREP=${1:?usage: prepare_shards.sh <prep_dir> <out_dir>}
OUT=${2:?usage: prepare_shards.sh <prep_dir> <out_dir>}
# Set SC35_SKIP_COUNT_CHECK=1 for a toy/synthetic smoke test. Never for a real run:
# the check is what catches a -val-split holdout silently shrinking your training set.
SKIP=${SC35_SKIP_COUNT_CHECK:-0}
MIXLAB=${MIXLAB:-mixlab}

command -v "$MIXLAB" >/dev/null || { echo "mixlab not found. See the README Install section." >&2; exit 1; }

for split in train val test; do
  echo "== ${split}"
  "$MIXLAB" -mode prepare \
    -input "${PREP}/${split}_X.npy" \
    -input-format continuous \
    -label-file "${PREP}/${split}_labels.tsv" \
    -continuous-modality waveform \
    -val-split 0 \
    -prepare-output-dir "${OUT}/${split}"
done

echo
if [ "$SKIP" = "1" ]; then
  echo "SC35_SKIP_COUNT_CHECK=1: skipping the official-count check (toy data)."
  exit 0
fi

echo "Check the record counts. They must match the official splits exactly:"
echo "  train 84,843   val 9,981   test 11,005"
python3 - "$OUT" <<'PY'
import json, sys, os
ok = True
want = {"train": 84843, "val": 9981, "test": 11005}
for name in ("train", "val", "test"):
    path = os.path.join(sys.argv[1], name, "mixlab.dataset.json")
    if not os.path.exists(path):
        print("  %-5s MISSING %s" % (name, path)); ok = False; continue
    splits = json.load(open(path)).get("splits", {})
    def total(sp):
        return sum(int(v) for v in (splits.get(sp, {}).get("class_counts") or {}).values())
    consumed = total("train")
    held = sum(total(sp) for sp in splits if sp != "train")
    ncls = len(splits.get("train", {}).get("class_counts") or {})
    if consumed != want[name]:
        print("  %-5s %7d records  <-- EXPECTED %d" % (name, consumed, want[name])); ok = False
    elif held:
        print("  %-5s %7d records but %d held back in %s" % (name, consumed, held, sorted(k for k in splits if k != "train"))); ok = False
    elif ncls != 35:
        print("  %-5s %7d records, %d classes  <-- EXPECTED 35" % (name, consumed, ncls)); ok = False
    else:
        print("  %-5s %7d records, 35 classes, nothing held back" % (name, consumed))
if not ok:
    print("")
    print("  Wrong. Usual cause: a -val-split holdout. It defaults to 0.1 and must be 0.")
    print("  This check reads the 'train' split INSIDE each directory, which is what")
    print("  training and eval actually glob. A holdout hides in a sibling split.")
    sys.exit(1)
print("")
print("  OK")
PY

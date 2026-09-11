#!/usr/bin/env bash
# Train one arm.
#
#   ./train.sh configs/s4d_10ep.json shards runs/s4d_10ep
#
# Re-running the same command RESUMES from the newest checkpoint if there is
# one. That matters: long mamba3-canonical runs on this workload hang
# intermittently (see the README), and resuming is the documented recovery.
set -euo pipefail

CFG=${1:?usage: train.sh <config.json> <shard_dir> <run_dir>}
SHARDS=${2:?usage: train.sh <config.json> <shard_dir> <run_dir>}
RUN=${3:?usage: train.sh <config.json> <shard_dir> <run_dir>}
MIXLAB=${MIXLAB:-mixlab}

command -v "$MIXLAB" >/dev/null || { echo "mixlab not found. See the README Install section." >&2; exit 1; }
[ -f "$CFG" ] || { echo "no config at $CFG" >&2; exit 1; }
[ -d "${SHARDS}/train" ] || { echo "no shards at ${SHARDS}/train. Run prepare_shards.sh first." >&2; exit 1; }
[ -f "${SHARDS}/train/mixlab.dataset.json" ] || { echo "no mixlab.dataset.json in ${SHARDS}/train; the epoch length is read from it. Re-run prepare_shards.sh." >&2; exit 1; }
mkdir -p "${RUN}"

# Checkpoint once per epoch, derived from the config and the prepared shards
# rather than guessed:
#   records per step = batch_tokens / seq_len
#   steps per epoch  = floor(train_records / records per step)
# train_records comes from the shards mixlab actually globs, so this is correct
# for the full dataset and for a small smoke fixture alike.
# floor, NOT ceil: every shipped config's `steps` is an exact multiple of the
# floor value, so the final epoch lands exactly on a checkpoint. With ceil the
# last boundary overshoots `steps`, mixlab writes no end-of-run checkpoint, and
# the final-epoch result the README advertises can never be evaluated.
EPOCH_STEPS=$(python3 - "$CFG" "${SHARDS}/train/mixlab.dataset.json" <<'PYEOF'
import json, sys
c = json.load(open(sys.argv[1]))
t = c["training"]
meta = json.load(open(sys.argv[2]))
counts = (meta.get("splits", {}).get("train", {}).get("class_counts") or {})
records = sum(int(v) for v in counts.values())
if not records:
    sys.exit("no train records in %s; re-run prepare_shards.sh" % sys.argv[2])
rec_per_step = t["batch_tokens"] / c["seq_len"]
n = max(1, int(records // rec_per_step))
if t["steps"] % n:
    sys.stderr.write("warning: steps=%d is not a multiple of %d (one epoch over "
                     "%d records); the final step will not be checkpointed\n"
                     % (t["steps"], n, records))
print(n)
PYEOF
)
echo "checkpointing every ${EPOCH_STEPS} steps (one epoch)"

# macOS ships bash 3.2, where "${ARR[@]}" on an EMPTY array trips `set -u`.
# The ${ARR[@]+...} guard is what keeps this working on a stock Mac.
RESUME=()
if compgen -G "${RUN}/ckpt/*.resume.json" > /dev/null; then
  RESUME=(-resume "${RUN}/ckpt")
  echo "resuming from the newest bundle in ${RUN}/ckpt"
fi

# append, never truncate: a resumed run must not destroy the earlier log
"$MIXLAB" -mode arch -config "$CFG" \
  -train "${SHARDS}/train/train_*.bin" \
  -val   "${SHARDS}/val/train_*.bin" \
  -checkpoint-dir "${RUN}/ckpt" -checkpoint-every "$EPOCH_STEPS" \
  -val-every "$EPOCH_STEPS" \
  ${RESUME[@]+"${RESUME[@]}"} \
  2>&1 | tee -a "${RUN}/train.log"

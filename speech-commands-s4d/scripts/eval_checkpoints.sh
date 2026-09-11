#!/usr/bin/env bash
# Full-split evaluation of every checkpoint in a run directory.
#
# Use this, not the acc= printed during training. That number is a small sampled
# metric; over class-ordered data it reads near zero in a way indistinguishable
# from a collapsed model.
set -euo pipefail

CFG=${1:?usage: eval_checkpoints.sh <config.json> <run_dir> <shard_dir> [split]}
RUN=${2:?usage: eval_checkpoints.sh <config.json> <run_dir> <shard_dir> [split]}
SHARDS=${3:?usage: eval_checkpoints.sh <config.json> <run_dir> <shard_dir> [split]}
SPLIT=${4:-val}
MIXLAB=${MIXLAB:-mixlab}

[ -d "${RUN}/ckpt" ]     || { echo "no checkpoint dir at ${RUN}/ckpt" >&2; exit 1; }
[ -d "${SHARDS}/${SPLIT}" ] || { echo "no shard dir at ${SHARDS}/${SPLIT}" >&2; exit 1; }
shopt -s nullglob
CKPTS=("${RUN}"/ckpt/*.st)
[ ${#CKPTS[@]} -gt 0 ] || { echo "no .st checkpoints in ${RUN}/ckpt" >&2; exit 1; }

fail=0
for ck in "${CKPTS[@]}"; do
  echo "== $(basename "$ck")"
  out=$("$MIXLAB" -mode eval -config "$CFG" \
          -safetensors-load "$ck" \
          -val "${SHARDS}/${SPLIT}/train_*.bin" \
          -val-batches 0 2>&1) || { echo "$out" | tail -3 >&2; fail=1; continue; }
  line=$(echo "$out" | grep -E "classification validation" || true)
  if [ -z "$line" ]; then
    echo "  no validation line in output. Last 3 lines:" >&2
    echo "$out" | tail -3 >&2
    fail=1
  else
    echo "$line"
  fi
done
exit $fail

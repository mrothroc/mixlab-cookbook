#!/usr/bin/env bash
# Evaluate every fine-tune checkpoint on the held-out VALIDATION split and on TEST,
# so you can pick the best-validation checkpoint (the Mamba classifier overfits fast,
# so this early-stopping-by-selection step is essential). mixlab's `-mode eval` for a
# classification checkpoint prints loss/acc/MCC/macro-F1/AUROC directly, so there is
# nothing to parse.
#
# Usage:
#   scripts/select_checkpoint.sh <config.json> <ckpt_dir> <cls_train_shards_dir> <test_shards_dir>
# Example:
#   scripts/select_checkpoint.sh configs/dna_mamba3_cls_mean.json \
#       checkpoints/mamba3_cls_ckpts data/cohn_cls/train_shards data/cohn_cls/test_shards
#
# Note on the glob: `mixlab -mode eval` derives the val shards from the -train glob by
# replacing the first "train" in the path with "val". To read a dir's val_*.bin we pass
# "<dir>/train_*.bin" where <dir> has no "train" in its name. The fine-tune's
# train_shards *does* contain "train", so we copy it to a neutral "heldout" dir first.
set -euo pipefail
CONFIG=$1; CKPT_DIR=$2; TRAIN_DIR=$3; TEST_DIR=$4
MX=${MIXLAB:-mixlab}
export MIXLAB_MLX_CACHE_LIMIT_MB=${MIXLAB_MLX_CACHE_LIMIT_MB:-8192}

# neutral copy so the eval val-glob resolves to the held-out validation split
HELDOUT="$(dirname "$TRAIN_DIR")/heldout"
rm -rf "$HELDOUT"; cp -r "$TRAIN_DIR" "$HELDOUT"

echo "checkpoint | VALIDATION (held-out) | TEST"
for ck in "$CKPT_DIR"/step_*.st; do
  [ -f "$ck" ] || continue
  v=$("$MX" -mode eval -config "$CONFIG" -safetensors-load "$ck" \
        -train "$HELDOUT/train_*.bin" -classification-out /tmp/_val.jsonl 2>&1 \
        | grep -oE "acc=[0-9.]+ .*auroc=[0-9.]+" || echo "eval-failed")
  t=$("$MX" -mode eval -config "$CONFIG" -safetensors-load "$ck" \
        -train "$TEST_DIR/train_*.bin" -classification-out /tmp/_test.jsonl 2>&1 \
        | grep -oE "acc=[0-9.]+ .*auroc=[0-9.]+" || echo "eval-failed")
  echo "$(basename "$ck") | VAL $v | TEST $t"
done
echo "Pick the checkpoint with the highest VALIDATION accuracy; report its TEST number."

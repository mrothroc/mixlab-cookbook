# Enhance the baseline: swap attention for canonical Mamba

With the attention baseline from [REPRODUCE_BASELINE.md](REPRODUCE_BASELINE.md) at 0.717 accuracy, the next step is to swap the mixer and compare. State-space models like Mamba are strong on long, structured sequences, so DNA is a natural fit. In mixlab that swap is a config edit, not a rewrite.

## The change

Take the pretrain config and replace every attention block with a canonical Mamba-3 block. Attention:

```json
{ "type": "plain", "heads": 8, "attention_mask": "causal" }
```

becomes

```json
{ "type": "mamba3-canonical", "inner_dim": 384, "state_size": 16, "n_groups": 4, "conv_kernel": 4, "use_conv": true }
```

The model dimension (384), sequence length (512), step count (20,000), learning rate (6e-4), and the genome data are all the same. The substantive change is the mixer, while the size and budget stay the same. (Two minor schedule details differ between the shipped configs: the Mamba config uses `warmup_ratio` 0.1 and an `example_framing` block that the recurrent mixer needs, and the attention config carries a couple of weight-decay group overrides. Neither changes the parameter count or the comparison.) The finished config is `configs/dna_mamba3_15m.json`.

Then run the same pipeline: pretrain on the genome stream, fine-tune on the benchmark.

```bash
export MIXLAB_MLX_CACHE_LIMIT_MB=8192
# pretrain (same command as the baseline, different config)
mixlab -mode arch -config configs/dna_mamba3_15m.json \
  -train 'data/shards/train_*.bin' -val-every 1000 \
  -checkpoint-dir checkpoints/mamba3_ckpts -checkpoint-every 1000 \
  -safetensors checkpoints/dna_mamba3.safetensors
```

About 20,000 steps in ~6 hours on Apple Silicon (since mixlab v0.81 the native Metal kernel for canonical Mamba is roughly 5x faster than the earlier fallback, which is what makes this practical on a laptop). It trains stably at learning rate 6e-4, and the pretrain validation loss lands at 1.13, close to but slightly higher than the attention model's 1.08 (lower is better, so attention is the better pure language model here,  but that does not decide the downstream task).

## Fine-tune, natively

For the Mamba fine-tune we use mixlab's built-in classification training (`objective: classification`), which fine-tunes a labeled classifier on the backbone with no external code. The config is `configs/dna_mamba3_cls_mean.json`. (As of mixlab v0.84 the canonical Mamba backbone also exports to Hugging Face, so the trained model can be published and loaded with `transformers`, see the models linked from the [README](README.md). Native classification is simply the shortest path for the fine-tune itself.)

First prepare the benchmark as labeled shards (the attention path skipped this; the native classifier needs it). `scripts/prep_cohn_classification.py` downloads `human_enhancers_cohn` and writes it as FASTA plus a label TSV. **It shuffles the training records**, which matters: the raw dataset is class-ordered (all negatives, then all positives), and mixlab does not reshuffle labeled records across epochs, so unshuffled data makes the model see one class at a time and never learn.

```bash
python scripts/prep_cohn_classification.py    # -> data/cohn_cls/{train_shuf,test}.{fa,labels.tsv}
# train shards (with a small held-out validation split for checkpoint selection)
mixlab -mode prepare -input data/cohn_cls/train_shuf.fa -input-format fasta \
  -label-file data/cohn_cls/train_shuf.labels.tsv -prepare-output-dir data/cohn_cls/train_shards \
  -nucleotide-alphabet dna -val-split 0.05 \
  -record-seq-len 512 -record-pad-id 0 -record-bos-id 1 -record-eos-id 2 -record-overflow drop
# test shards (put ~all test records in the val split so eval scores the full test set)
mixlab -mode prepare -input data/cohn_cls/test.fa -input-format fasta \
  -label-file data/cohn_cls/test.labels.tsv -prepare-output-dir data/cohn_cls/test_shards \
  -nucleotide-alphabet dna -val-split 0.999 \
  -record-seq-len 512 -record-pad-id 0 -record-bos-id 1 -record-eos-id 2 -record-overflow drop
```

Now fine-tune, checkpointing every 500 steps:

```bash
mixlab -mode arch -config configs/dna_mamba3_cls_mean.json \
  -train 'data/cohn_cls/train_shards/train_*.bin' \
  -safetensors-load checkpoints/dna_mamba3.safetensors \
  -safetensors checkpoints/dna_mamba3_cls.safetensors \
  -checkpoint-dir checkpoints/mamba3_cls_ckpts -checkpoint-every 500
```

## Score and select the checkpoint

`mixlab -mode eval` on a classification checkpoint prints loss/accuracy/MCC/macro-F1/AUROC directly (nothing to parse). Because the model overfits, evaluate every checkpoint on the held-out validation split, choose the one with the best validation accuracy, and report its test number. `scripts/select_checkpoint.sh` runs the full sweep:

```bash
scripts/select_checkpoint.sh configs/dna_mamba3_cls_mean.json \
  checkpoints/mamba3_cls_ckpts data/cohn_cls/train_shards data/cohn_cls/test_shards
# prints, per checkpoint: VALIDATION acc/auroc and TEST acc/auroc.
# Pick the highest-validation row (usually ~step 500); its TEST number is your result.
```

Important: the evaluation config must use the same pooling as the checkpoint was trained with (`configs/dna_mamba3_cls_mean.json` is mean-pooled; evaluating a mean-trained checkpoint with a last-pooled config silently produces incorrect results).

## Two things you have to get right

### 1. Use mean pooling, not last-token pooling
A classifier needs to pool the sequence into one vector. For attention, taking the last token works fine. For a state-space model it does not: the recurrent state decays, so early motifs are faint by the time you reach the end. On this task, last-token pooling scored 0.598 accuracy (barely above chance) while mean pooling scored 0.73. The config sets `"pooling": "mean"`. If you fine-tune with mean pooling you must also **evaluate with mean pooling** (a mismatched eval config silently produces incorrect results).

### 2. Early-stop, or select on validation
The canonical Mamba backbone is expressive and overfits the ~19.8k fine-tuning examples fast. Left to run 2,500 steps, its training loss reaches 0.04 while test loss climbs to 2.5. Early stopping or checkpoint selection on a held-out validation split is therefore important. We checkpoint every 500 steps and choose the one with the best validation accuracy (the fine-tune shards carry a small held-out split for this). The best checkpoint is consistently very early, around step 500.

The general pattern: a more expressive mixer offers a higher ceiling but needs more careful regularization to reach it.

## Result

Three seeds, checkpoint selected on validation, reported on the full test set:

| model (mixer) | accuracy | AUROC | MCC |
|---|---|---|---|
| **`mamba3-canonical`** | **0.728 ± 0.007** | **0.811** | **0.459** |
| `plain` attention | 0.717 | 0.799 | 0.436 |

In these runs, canonical Mamba has higher metrics at the same size and compute. The step-500 test accuracy across the three seeds is close: 0.734 / 0.732 / 0.733. The small ±0.007 spread reflects how far each run drifts into overfitting before you stop it, rather than the architecture. Numbers: [`results/mamba3_3seed.json`](results/mamba3_3seed.json).

## One trap worth knowing about

mixlab has a block named `legacy_mamba`. It is a simple per-channel exponential moving average, not a selective-scan state-space model, and it is not a fair test of Mamba. To use the canonical Mamba implementation, choose `mamba3-canonical`. (For context, the legacy EMA
block, with mean pooling, also happened to reach 0.722 on this task, between attention and canonical Mamba, but it is not Mamba.)

## The takeaway

This example improves a real benchmark result by editing one field in a JSON config, on a laptop, with no new model code. In mixlab, the architecture is a swept axis. If you would like to keep experimenting, try `rwkv`, `gated_deltanet`, `retnet`, or `mlstm` the same way, or change the optimizer from `adamw` to `muon` in one line. The block library is there for you to explore.

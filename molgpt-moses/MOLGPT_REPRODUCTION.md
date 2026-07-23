# Train the base MolGPT model yourself

This tutorial trains a capacity-matched MolGPT in mixlab, one molecule per sequence. After about 5h on Apple Silicon (M1 Max, newer chips are massively faster) and 33,000 steps (about 10 epochs), the 6,342,656-parameter model closely matches MolGPT's published MOSES metrics.

Prefer to skip training? The finished model is on Hugging Face: [mrothroc/molgpt-moses-smiles-mixlab](https://huggingface.co/mrothroc/molgpt-moses-smiles-mixlab).

## What you will reproduce

The model uses 8 layers, 256 dimensions, 8 heads, a GELU MLP, tied embeddings, and learned absolute positions. It has 6,342,656 parameters, just 0.05% fewer than MolGPT's 6,345,728. Training uses 1,426,197 MOSES rows, a vocab-30 character tokenizer, and drops or truncates 0 records.

| Metric | mixlab (this repro) | MolGPT (published, firm) | Verdict |
|---|---|---|---|
| Valid | **0.988** | 0.994 | within 0.006 |
| Unique@10k | **0.991** | ~1.000 | ok |
| Novelty | **0.783** | 0.797 | within 0.014 |
| IntDiv | **0.857** | 0.857 | exact |
| IntDiv2 | 0.849 | 0.851 | ok |
| FCD/Test (directional) | 2.83 | ~0.07 (soft reimpl anchor) | higher, see note |
| Frag/Test | 0.984 | - | (high, good) |
| Filters | 0.996 | - | (high, good) |

The original 6.3M devalab MolGPT does not have a directly comparable published FCD/Test, so treat FCD as directional here. Full metrics are in `results/molgpt_record_30k.metrics.json`.

Training one molecule per sequence, not a packed stream, is what made this reproduction work. Each example is framed as `[BOS] mol [EOS] [PAD]*`, with PAD-masked causal loss. At 4k steps, this changed validity from **0.50 to 0.97**.

## Training recipe

The matched recipe uses lr 6e-4, betas 0.9/0.95, weight_decay 0.1 (selective), grad_clip 1.0, about 10% warmup, seed 42, and a final train loss of 0.426. Checkpoints are resumable.

Run from this recipe directory. You need mixlab >= v0.73.0 on PATH and a Python environment with `rdkit`, `molsets` (MOSES), `fcd_torch`, and `tokenizers`.

```bash
export MIXLAB_MLX_CACHE_LIMIT_MB=4096
export MIXLAB_SCRIPTS=/path/to/mixlab/scripts      # mixlab prepare shells out here
export PATH="$PWD/.venv/bin:$PATH"                  # venv (rdkit/moses) on PATH

# 1. content-only char tokenizer (vocab 30; loader adds BOS/EOS/PAD) + per-record shards
python scripts/build_tokenizer.py --no-template --out data/tokenizer_content.json
python scripts/make_jsonl.py --split train --out data/train_full.jsonl --no-canon
mixlab -mode prepare -input data/train_full.jsonl -output data/record_shards \
  -tokenizer-path data/tokenizer_content.json -text-field text \
  -frame-per-record -record-seq-len 64 -record-pad-id 0 -record-bos-id 1 -record-eos-id 2

# 2. train the matched GPT, one molecule per sequence (~10 epochs)
mixlab -mode arch -config configs/molgpt_record.json -train 'data/record_shards/train_*.bin' \
  -checkpoint-dir checkpoints/molgpt_record_ckpts -checkpoint-every 3000 \
  -safetensors checkpoints/molgpt_record.safetensors

# 3. sample 30k + score with official moses.metrics
python scripts/sample_and_score.py --config configs/molgpt_record.json \
  --ckpt checkpoints/molgpt_record.safetensors --tokenizer data/tokenizer_content.json \
  --max-tokens 64 --n-jobs 8 --out samples/molgpt_record_30k.smi
#    -> expect Valid~0.99, Novelty~0.78, IntDiv~0.857
```

The helper scripts handle tokenization, MOSES conversion, sampling, and scoring. The key mixlab inputs are `-frame-per-record`, `-mode arch`, and `configs/molgpt_record.json`.

## Useful files

- `configs/molgpt_record.json`: the 8L/256d/8h training configuration.
- `results/molgpt_record_30k.metrics.json`: the full metric dictionary.
- `CONSTRAINED_DECODE.md`, `SELFIES_LEG.md`, and `GOAL_DIRECTED.md`: next experiments with validity and property steering.

# Make every sample a valid molecule with SELFIES

SMILES asks the model to learn chemical validity. [SELFIES](https://github.com/aspuru-guzik-group/selfies) moves that constraint into the representation, so every generated symbol string decodes to a valid molecule. This tutorial retrains the same-size model on SELFIES and reaches literal 1.000 validity.

Prefer to skip training? The finished model is on Hugging Face: [mrothroc/molgpt-moses-selfies-mixlab](https://huggingface.co/mrothroc/molgpt-moses-selfies-mixlab).

## What changes

Only the representation and sequence length change. The model remains capacity-matched at 8L/256d/8h, with `vocab_size 30` (4 specials + 26 symbols) and `seq_len 72`. The recipe still uses 33000 steps, lr 6e-4, betas 0.9/0.95, wd 0.1, grad_clip 1.0, warmup 0.1, seed 42, and batch_tokens 27648. Training takes about 5h8m on Apple Silicon.

| Metric | SELFIES (this leg) | SMILES char repro | MolGPT |
|---|---|---|---|
| **Valid** | **1.0000** (by construction) | 0.988 | 0.994 |
| Unique@10k | 0.9984 | 0.991 | ~1.0 |
| **Novelty** | **0.8588** | 0.783 | 0.797 |
| IntDiv | 0.8568 | 0.857 | 0.857 |
| **FCD/Test** | **0.4259** | 2.83 | - |
| Frag/Test | 1.0000 | 0.984 | - |
| Filters | 0.9757 | 0.996 | - |

On 30k MOSES samples, validity is 1.0000, Novelty is 0.8588, IntDiv is 0.8568, and FCD/Test is 0.4259. Full metrics are in `results/selfies_record_30k.metrics.json`. The final logged `val=8.29` is a framing measurement artifact; the generation metrics are the useful evaluation here. Train loss moves from 3.44 to 0.51, compared with the about 3.4-nat uniform-random floor.

## Why the guarantee works

`sf.decoder` turns every SELFIES symbol string into syntactically valid, valence-valid SMILES. Validity therefore comes from the representation, not from a perfectly trained model. Even a 500-step smoke model shows the guarantee:

| | value |
|---|---|
| smoke train | 500 steps on the full SELFIES shards |
| decoded samples | 10,000 |
| non-empty decodes | 10,000 / 10,000 |
| **RDKit validity** | **1.0000** (10000/10000) |

For comparison, the character model gives 0.9875 validity unconstrained and about 0.994 with the structural DFA (deterministic finite automaton) described in [CONSTRAINED_DECODE.md](CONSTRAINED_DECODE.md). SELFIES also covers the remaining valence and kekulization gap.

## Data pipeline

- `scripts/build_selfies_tokenizer.py` uses `sf.split_selfies`, not a regex, to make one token per SELFIES symbol. Specials are `[PAD]=0 [BOS]=1 [EOS]=2 [UNK]=3`; outputs are `data/selfies_tokenizer_content.json` and `data/selfies_alphabet.json` (26 symbols). It checks round trips on 2000 samples.
- `scripts/prep_selfies.py` encodes the full MOSES train with `sf.encoder`: 1,584,663 molecules, about 100% success. It writes `data/selfies_train.jsonl`, then prepares `data/selfies_record_shards/` with record-seq-len 72 and pad/bos/eos 0/1/2. The result has 50 train + 6 val bins, 0 dropped / 0 truncated.
- `configs/molgpt_selfies_record.json` holds the matched training recipe.
- `scripts/sample_and_score.py --selfies` decodes model ids to SELFIES symbols, calls `sf.decoder`, and scores SMILES with official `moses.metrics`. Without the flag, the character path is byte-for-byte unchanged.

This is the representation choice discussed in *Verification Surfaces in Language-Model Systems: Token, Schema, and Structured-Output Reliability*, 2026, [doi:10.5281/zenodo.20331399](https://doi.org/10.5281/zenodo.20331399), section 8 and section 2: choose a representation that makes the required property automatic.

## Run it

You need mixlab >= v0.73.0 on PATH.

```bash
export MIXLAB_MLX_CACHE_LIMIT_MB=4096
export PATH="$PWD/.venv/bin:$PATH"

# train the SELFIES model (same recipe as the SMILES model, over SELFIES symbols)
mixlab -mode arch -config configs/molgpt_selfies_record.json \
  -train 'data/selfies_record_shards/train_*.bin' \
  -checkpoint-dir checkpoints/selfies_record_ckpts -checkpoint-every 1000 \
  -safetensors checkpoints/molgpt_selfies_record.safetensors

# sample 30k and score
python scripts/sample_and_score.py --selfies \
  --config configs/molgpt_selfies_record.json \
  --ckpt checkpoints/molgpt_selfies_record.safetensors \
  --tokenizer data/selfies_tokenizer_content.json \
  --n 30000 --n-jobs 8 --max-tokens 72 --out samples/selfies_record_30k.smi
# -> Valid 1.000, Unique@10k 0.998, Novelty 0.859, IntDiv 0.857, FCD/Test 0.43
```

# Reproduce an attention DNA language model on human_enhancers_cohn

This trains a small causal DNA language model on human genome, then fine-tunes it into an enhancer classifier: the baseline that [ENHANCE_MAMBA3.md](ENHANCE_MAMBA3.md) builds on. It runs on a Mac in a few hours.

## The plan

DNA is a sequence over a tiny alphabet (A, C, G, T, plus N for unknown). Here, the DNA language model is a small causal transformer over a vocab of 9 tokens (4 bases, N, and BOS/EOS/PAD/MASK). We pretrain it to predict the next base on real genome, then fine-tune a classifier head on the labeled benchmark.

## Model

`configs/dna_causal_15m_s512.json`: 14.6M params, 8 `plain` causal attention layers, model_dim 384, seq_len 512, vocab 9. A standard GPT-style stack. The only genomics-specific setting is
`reverse_complement_prob: 0.5`, which randomly feeds the reverse complement during training (DNA is double-stranded, so a sequence and its reverse complement mean the same thing).

## Data

### Genome (for pretraining)
UCSC hg38 chromosomes 1-3, about 669 Mbp after cleaning (the stream prepare below then splits this into roughly 562M training tokens plus a held-out validation split). `scripts/prep_fasta.py` uppercases the soft-masked (lowercase) regions and splits on runs of N into clean ACGT contigs.

```bash
mkdir -p data/raw && cd data/raw
for c in 1 2 3; do curl -sSLO "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr$c.fa.gz"; done
gunzip -kf chr1.fa.gz chr2.fa.gz chr3.fa.gz && cd ../..
python scripts/prep_fasta.py data/raw/chr1.fa data/raw/chr2.fa data/raw/chr3.fa \
  --out data/clean/pretrain.fa --min-len 1000
```

Then prepare it as a **continuous stream** (not per-record). Stream framing concatenates the genome and cuts fixed windows, a common way to train genomic language models. It is also required for the recurrent Mamba mixer in the next writeup. Attention works with it too, so we can use one data path for both:

```bash
mixlab -mode prepare -input data/clean/pretrain.fa -input-format fasta \
  -prepare-output-dir data/shards -nucleotide-alphabet dna \
  -nucleotide-framing stream -nucleotide-stream-separator eos
```

### Benchmark (for fine-tuning)
For the attention baseline you do not prepare any benchmark shards by hand: `scripts/finetune_gate.py` downloads `human_enhancers_cohn` itself (via the `genomic-benchmarks` package) and tokenizes it in memory. The labeled-shard prep is only needed for the Mamba native-classification path in [ENHANCE_MAMBA3.md](ENHANCE_MAMBA3.md).

## Pretrain

```bash
export MIXLAB_MLX_CACHE_LIMIT_MB=8192
mixlab -mode arch -config configs/dna_causal_15m_s512.json \
  -train 'data/shards/train_*.bin' -val-every 1000 \
  -checkpoint-dir checkpoints/attn_ckpts -checkpoint-every 1000 \
  -safetensors checkpoints/dna_attn.safetensors
```

20,000 steps, about 1.2 epochs over the genome stream, roughly 6-7 hours on Apple Silicon. Final validation loss lands around 1.08.

## Fine-tune and score

For attention, this recipe exports to Hugging Face and fine-tunes with `transformers`. (The Mamba path in the next writeup uses mixlab's native classifier instead, which needs no external code; both model backbones export to HF, so either can be published, see the models linked from the [README](README.md).) `scripts/finetune_gate.py` does the fine-tune: it loads an exported checkpoint as `AutoModelForSequenceClassification`, fine-tunes three arms (pretrained / from-scratch / frozen) across three seeds, and reports accuracy, MCC, and AUROC. We use the pretrained-e2e arm as the baseline:

```bash
# export-hf needs a tokenizer.json; build one from the vocab mixlab wrote during prepare
python scripts/build_nucleotide_tokenizer.py \
  --vocab data/shards/nucleotide_vocab.json --out data/tokenizer.json
mixlab -mode export-hf -config configs/dna_causal_15m_s512.json \
  -safetensors-load checkpoints/dna_attn.safetensors \
  -tokenizer-path data/tokenizer.json -export-dir hf/dna_attn
# fine-tune 3 arms x 3 seeds; downloads the benchmark itself, writes results/attention_baseline.json
python scripts/finetune_gate.py --hf-dir hf/dna_attn --arms pretrained scratch --seeds 42 123 2024 \
  --out results/attention_baseline.json
```

## Result

Pretrained-e2e, full 6,948-record test set, three seeds:

| metric | value |
|---|---|
| accuracy | 0.717 |
| AUROC | 0.799 |
| MCC | 0.436 |

Full numbers in [`results/attention_baseline.json`](results/attention_baseline.json). These results are at or
above the short-context baselines reported for this dataset. In these runs, pretraining helps:
pretrained-e2e beats a from-scratch control by a non-overlapping margin across seeds. That is the baseline; [ENHANCE_MAMBA3.md](ENHANCE_MAMBA3.md) changes one block and compares.

## A note on leakage

The Cohn enhancers are genomic loci, and we pretrain on chr1-3 without excluding benchmark intervals, so some test sequences could appear in pretraining. This is symmetric across every model here (attention and Mamba both pretrain on the same genome), so the comparison between them is fair. For an absolute claim against the literature you would pretrain on chromosomes disjoint from the benchmark.

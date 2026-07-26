# A DNA model in mixlab: reproduce an attention baseline, then improve it by swapping the mixer

This recipe trains a small **DNA language model** on a Mac, reproduces an attention baseline on a real genomics benchmark, then improves on it by changing **one block in a JSON config**: swap the attention mixer for a canonical **Mamba** state-space mixer. Same size, same training budget, so the mixer is the only real variable. It also doubles as a worked example of trying mixlab's built-in blocks by editing JSON.

The task is [Genomic Benchmarks](https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks) `human_enhancers_cohn`: given a 500 bp DNA sequence, predict whether it is an enhancer. It is balanced (20,843 train / 6,948 test) and small enough to try on an Apple Silicon laptop. We picked it because it is standard and comparable: it is one of the curated tasks in Genomic Benchmarks (Gresova et al. 2023, [BMC Genomic Data](https://doi.org/10.1186/s12863-023-01123-8)), a collection built to make genomic sequence classification reproducible across models, with a published baseline to reference.

## The two-step arc

1. **Reproduce.** Pretrain a small causal DNA language model (attention, ~14.6M params) on human genome, then fine-tune a classifier on the benchmark. This is the baseline.
2. **Enhance.** Take the exact same recipe and change the mixer blocks in the config from `plain` (attention) to `mamba3-canonical` (a canonical selective-scan state-space model). Retrain, then compare the results with the baseline.

## Results

Both models are the same size (~14.5M params), pretrained on the same GRCh38 chr1-3 stream for the same 20,000 steps (about 6 hours on an M4 Max), then fine-tuned on `human_enhancers_cohn`. Test set, full 6,948 records:

| model (mixer) | accuracy | AUROC | MCC |
|---|---|---|---|
| **`mamba3-canonical`** (3 seeds) | **0.728 ± 0.007** | **0.811** | **0.459** |
| `plain` attention (3 seeds) | 0.717 | 0.799 | 0.436 |

The arc: the benchmark's own [baseline CNN](https://pmc.ncbi.nlm.nih.gov/articles/PMC10150520/) reaches about 0.69-0.70 accuracy on this task (same split, same metric). We pretrain a small DNA language model and fine-tune a classifier, and both variants clear that baseline: the attention model at 0.717, and canonical Mamba, reached by swapping one block in the config, at 0.728. Because both mixers are built into mixlab, moving between them is a config edit, not new model code. Numbers: [`results/mamba3_3seed.json`](results/mamba3_3seed.json), [`results/attention_baseline.json`](results/attention_baseline.json).

For wider context, the stronger published results on this dataset (HyenaDNA, Caduceus, and similar) sit around 0.72-0.75, but they are pretrained on the whole human genome. Our variants see only three chromosomes and train on a single laptop, so landing in that band is a good showing. Cross-paper genomic comparisons are notoriously inconsistent, so we anchor only on the benchmark's own baseline; accuracy is the metric the benchmark reports, and the AUROC above is our own, with no published comparator.

Note that the canonical Mamba model is expressive and **overfits fast** on ~19.8k fine-tuning examples, so early stopping or checkpoint selection on a validation split matters. The best checkpoint is consistently very early (~step 500), and its test accuracy is close across seeds (0.734 / 0.732 / 0.733). More on that in [ENHANCE_MAMBA3.md](ENHANCE_MAMBA3.md).

## Load the trained models

Both pretrained DNA backbones are on Hugging Face. Load one and fine-tune it on your own genomic task, or reproduce the enhancer classifier from the writeups. Token ids are BOS=1, EOS=2, PAD=0, and A/C/G/T/N = 4/5/6/7/8.

- **Attention baseline** (native `GPT2LMHeadModel`, no `trust_remote_code`): [mrothroc/dna-enhancers-attention-mixlab](https://huggingface.co/mrothroc/dna-enhancers-attention-mixlab)
- **Canonical Mamba** (custom code, load with `trust_remote_code=True`): [mrothroc/dna-enhancers-mamba3-mixlab](https://huggingface.co/mrothroc/dna-enhancers-mamba3-mixlab)

```python
import torch
from transformers import AutoModel

mamba = AutoModel.from_pretrained("mrothroc/dna-enhancers-mamba3-mixlab", trust_remote_code=True).eval()
hidden = mamba(input_ids=torch.tensor([[1, 4, 5, 6, 7, 2]])).last_hidden_state   # BOS A C G T EOS -> [1, 6, 384]
```

The attention model is a `GPT2LMHeadModel`, so it also generates DNA directly (`AutoModelForCausalLM` +
`model.generate`). For classification, load either with `AutoModelForSequenceClassification` and fine-tune.

## The one-line change

The change that matters between the two models is the mixer. The attention baseline stacks `plain` self-attention blocks (each with a built-in feed-forward), so a block looks like:

```json
{ "type": "plain", "heads": 8, "attention_mask": "causal", "ffn_activation": "gelu_new" }
```

The Mamba model replaces the token mixer with a canonical Mamba block, paired with a SwiGLU block for channel mixing:

```json
{ "type": "mamba3-canonical", "inner_dim": 384, "state_size": 16, "n_groups": 4 }, { "type": "swiglu" }
```

The model dimension, sequence length, step count, learning rate, data, and tokenizer stay the same. You only swap the mixer family, with no need to write a new model. (See the two configs in `configs/` for the exact block lists; [ENHANCE_MAMBA3.md](ENHANCE_MAMBA3.md) walks through the swap.)

## Setup

You will need the [mixlab](https://github.com/mrothroc/mixlab) binary, **v0.83.0 or newer**, on your PATH (install it per the mixlab README). v0.79-0.81 added the native Metal kernel for canonical Mamba that makes this fast on an Apple laptop, and v0.83 makes `mixlab -mode prepare` self-contained. `prepare` still needs `python3` with numpy on PATH. The Python environment below provides both and is also used for the attention-baseline harness:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PATH="$PWD/.venv/bin:$PATH"    # so mixlab's prepare finds python3 + numpy
```

The benchmark (`human_enhancers_cohn`) downloads itself on first use via the `genomic-benchmarks` package. The genome is a one-line `curl` of the UCSC chromosome FASTA (uses the standard `curl` and `gunzip` that ship with macOS); the exact command is in [REPRODUCE_BASELINE.md](REPRODUCE_BASELINE.md).

## Reproduce it

The two writeups below contain the full, runnable commands. Here is the overall flow:

```bash
# genome: download + clean chr1-3, prepare a continuous nucleotide stream (shared by both models)
python scripts/prep_fasta.py data/raw/chr1.fa data/raw/chr2.fa data/raw/chr3.fa --out data/clean/pretrain.fa
mixlab -mode prepare -input data/clean/pretrain.fa -input-format fasta \
  -prepare-output-dir data/shards -nucleotide-alphabet dna \
  -nucleotide-framing stream -nucleotide-stream-separator eos

# then, per model: pretrain the DNA LM on the stream, fine-tune the classifier, score it
#   - attention  -> REPRODUCE_BASELINE.md  (export to HF + scripts/finetune_gate.py)
#   - Mamba      -> ENHANCE_MAMBA3.md       (native classification + scripts/select_checkpoint.sh)
```

## Where to go next

- **Reproduce the attention baseline** -> [REPRODUCE_BASELINE.md](REPRODUCE_BASELINE.md): follow the DNA LM end to end, from genome download to the 0.717 classifier.
- **Try canonical Mamba** -> [ENHANCE_MAMBA3.md](ENHANCE_MAMBA3.md): swap the mixer, retrain, compare it with the baseline, and learn the two details to get right (pooling and early stopping).

## Exercising and enhancing mixlab

This recipe leans on several mixlab features: `-nucleotide-framing stream` (continuous genome windows for recurrent mixers), the native `objective: classification` training mode (fine-tune a labeled classifier without any external code), and the `mamba3-canonical` block with its native Metal kernel. The configs in `configs/` can be used as starting points for your own mixlab experiments; `results/` holds the metric JSONs behind the tables above.
mixlab: https://github.com/mrothroc/mixlab

## Citing this work

This recipe reproduces a benchmark and builds on prior work, so please cite what it uses:

- **Genomic Benchmarks** (the task): Gresova, K.; Martinek, V.; Cechak, D.; Simecek, P.; Alexiou, P.
  "Genomic benchmarks: a collection of datasets for genomic sequence classification." *BMC Genomic Data*
  2023, 24, 25. [doi:10.1186/s12863-023-01123-8](https://doi.org/10.1186/s12863-023-01123-8)
- **Mamba** (the foundational selective state-space model): Gu, A.; Dao, T. "Mamba: Linear-Time Sequence
  Modeling with Selective State Spaces." 2023. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
- **Mamba-3** (the specific architecture the `mamba3-canonical` block implements): Lahoti, A.; Li, K. Y.;
  Chen, B.; Wang, C.; Bick, A.; Kolter, J. Z.; Dao, T.; Gu, A. "Mamba-3: Improved Sequence Modeling using
  State Space Principles." ICLR 2026. [arXiv:2603.15569](https://arxiv.org/abs/2603.15569)
- **GRCh38 / hg38** human reference genome, produced by the Genome Reference Consortium (assembly
  GCA_000001405.15) and accessed via the UCSC Genome Browser: Kent, W. J.; Sugnet, C. W.; Furey, T. S.;
  Roskin, K. M.; Pringle, T. H.; Zahler, A. M.; Haussler, D. "The Human Genome Browser at UCSC."
  *Genome Research* 2002, 12 (6), 996-1006. [doi:10.1101/gr.229102](https://doi.org/10.1101/gr.229102)

If you want to point at this specific recipe or the trainer:

```bibtex
@software{genomics_mamba_mixlab,
  author = {Rothrock, Michael},
  title  = {A DNA model in mixlab: reproduce an attention baseline, then beat it with canonical Mamba},
  year   = {2026},
  url    = {https://github.com/mrothroc/mixlab-cookbook/tree/main/genomics-mamba}
}
@software{mixlab,
  author = {Rothrock, Michael},
  title  = {mixlab: a compact ML architecture lab},
  url    = {https://github.com/mrothroc/mixlab}
}
```

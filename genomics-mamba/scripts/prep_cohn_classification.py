#!/usr/bin/env python3
"""Prepare human_enhancers_cohn as labeled FASTA+TSV for mixlab classification.

Reflects the pipeline used for the Mamba classification fine-tune:
  1. dump train/test sequences to FASTA (one record per enhancer) + sibling label TSV,
  2. SHUFFLE the train records on disk (the dataset is class-ordered: all 0s then all 1s;
     mixlab does not shuffle labeled records across the epoch, so unshuffled data made the
     model see 10421 zeros then 10422 ones -> erratic loss),
  3. mixlab -mode prepare -input-format fasta -label-file ... -> mixlab_labeled_sequence_shard_v1.

Test set is NOT shuffled (order is irrelevant to eval metrics). Run this, then the mixlab
prepare commands in REPRODUCE_BASELINE.md and ENHANCE_MAMBA3.md.
"""
import random


def dump_split(split, shuffle, seed=42):
    from genomic_benchmarks.dataset_getters.pytorch_datasets import HumanEnhancersCohn
    ds = HumanEnhancersCohn(split=split, version=0)
    pairs = [(f"{split}_{i}", seq.upper(), int(lab)) for i, (seq, lab) in enumerate(ds)]
    if shuffle:
        random.seed(seed)
        random.shuffle(pairs)
    fa = f"data/cohn_cls/{split}{'_shuf' if shuffle else ''}.fa"
    tsv = f"data/cohn_cls/{split}{'_shuf' if shuffle else ''}.labels.tsv"
    with open(fa, "w") as f, open(tsv, "w") as t:
        for rid, seq, lab in pairs:
            f.write(f">{rid}\n{seq}\n")
            t.write(f"{rid}\t{lab}\n")
    n0 = sum(1 for _, _, l in pairs if l == 0)
    print(f"{split}: {len(pairs)} records ({n0} class0 / {len(pairs)-n0} class1) -> {fa}")


if __name__ == "__main__":
    import os
    os.makedirs("data/cohn_cls", exist_ok=True)
    dump_split("train", shuffle=True)   # shuffled: training order matters
    dump_split("test", shuffle=False)   # order irrelevant to eval

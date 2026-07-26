#!/usr/bin/env python3
"""Build a Hugging Face tokenizer.json for the fixed nucleotide vocab.

`mixlab -mode export-hf` needs a tokenizer.json, but FASTA `prepare` only writes
`nucleotide_vocab.json`. This converts that vocab into a HF WordLevel tokenizer so
the exported attention model loads with stock `transformers`. (You feed raw token
ids at fine-tune/inference time, so this tokenizer is only there to satisfy the
exporter and record the vocab.)

Usage:
  python scripts/build_nucleotide_tokenizer.py \
      --vocab data/shards/nucleotide_vocab.json --out data/tokenizer.json
"""
import argparse
import json

from tokenizers import Tokenizer, models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", required=True, help="nucleotide_vocab.json from `mixlab prepare`")
    ap.add_argument("--out", required=True, help="output tokenizer.json path")
    args = ap.parse_args()

    vocab = json.load(open(args.vocab))["tokens"]  # e.g. {"<PAD>":0,...,"A":4,"C":5,"G":6,"T":7,"N":8}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<MASK>"))
    tok.save(args.out)
    print(f"wrote {args.out} (vocab size {len(vocab)})")


if __name__ == "__main__":
    main()

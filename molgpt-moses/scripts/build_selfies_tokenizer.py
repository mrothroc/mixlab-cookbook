#!/usr/bin/env python3
"""Build a SELFIES-symbol tokenizer.json for mixlab per-record prepare.

Each SELFIES symbol (for example '[C]', '[=O]', '[Branch1]', or '[Ring1]') is
one token. Training text is space-joined by prep_selfies.py, so WhitespaceSplit
preserves bracketed symbols intact. The tokenizer is content-only: mixlab's
-frame-per-record loader adds [BOS]/[EOS]/[PAD] to each row.

Special ids are fixed: [PAD]=0, [BOS]=1, [EOS]=2, [UNK]=3, then sorted symbols.
Prefer --alphabet-json from prep_selfies.py to avoid encoding the full training
set twice; without it, the alphabet is derived directly from MOSES train.
"""
import argparse, json, sys
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers
import moses
import selfies as sf

SPECIALS = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]


def build(symbols):
    vocab = {tok: i for i, tok in enumerate(SPECIALS)}
    for symbol in sorted(symbols):
        vocab[symbol] = len(vocab)

    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    # Split only on whitespace: generic Whitespace would split SELFIES punctuation.
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return tok, vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/selfies_tokenizer_content.json",
                    help="output tokenizer.json path")
    ap.add_argument("--alphabet-json",
                    help="precomputed JSON symbol list from prep_selfies.py (preferred)")
    ap.add_argument("--max-train", type=int, default=0,
                    help="cap train molecules used to derive the alphabet (0=all)")
    args = ap.parse_args()

    train = moses.get_dataset("train")
    if args.max_train:
        train = train[: args.max_train]
    if args.alphabet_json:
        symbols = json.load(open(args.alphabet_json))
    else:
        encoded = [sf.encoder(s) for s in train]
        symbols = sf.get_alphabet_from_selfies(encoded)
    tok, vocab = build(symbols)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(outp))
    print(f"vocab_size={len(vocab)}  ({len(SPECIALS)} specials + {len(symbols)} SELFIES symbols)")
    print(f"saved {outp}")

    # Round-trip check against SELFIES strings (content-only: no BOS/EOS to strip).
    id2tok = {i: t for t, i in vocab.items()}
    fails = 0
    sample = train[:2000]
    for s in sample:
        selfies = sf.encoder(s)
        text = " ".join(sf.split_selfies(selfies))
        ids = tok.encode(text).ids
        recon = "".join(id2tok[i] for i in ids)
        if recon != selfies:
            fails += 1
            if fails <= 3:
                print(f"  MISMATCH: {selfies!r} -> {recon!r}", file=sys.stderr)
    print(f"round-trip: {len(sample)-fails}/{len(sample)} exact")
    if fails:
        sys.exit(1)


if __name__ == "__main__":
    main()

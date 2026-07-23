#!/usr/bin/env python3
"""Build a character-level SMILES tokenizer.json for mixlab prepare.

MOSES CharRNN is a genuine character-level model: every SMILES character
(including two-letter atoms like 'Cl'/'Br', which split into 'C','l'/'B','r')
is one token. We emit a HuggingFace WordLevel tokenizer whose pre-tokenizer
isolates every character, plus a TemplateProcessing post-processor that wraps
each encoded molecule in [BOS] ... [EOS]. Because mixlab's prepare concatenates
per-record encodings into a flat token stream, this yields

    [BOS] m1_chars [EOS] [BOS] m2_chars [EOS] ...

and training uses attention_segment_mask=boundary_token on [BOS] so molecules
never attend across each other.

Special ids are fixed: [PAD]=0, [BOS]=1, [EOS]=2, [UNK]=3, then sorted chars.
"""
import argparse, json, sys
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, processors, Regex
import moses

SPECIALS = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]


def build(train_smiles, add_template=True):
    chars = sorted({c for s in train_smiles for c in s})
    vocab = {tok: i for i, tok in enumerate(SPECIALS)}
    for c in chars:
        vocab[c] = len(vocab)

    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    # Isolate every single character as its own pre-token.
    tok.pre_tokenizer = pre_tokenizers.Split(pattern=Regex("."), behavior="isolated")
    if add_template:
        # Packed training (attention_segment_mask): the tokenizer wraps each molecule [BOS] ... [EOS].
        tok.post_processor = processors.TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[("[BOS]", vocab["[BOS]"]), ("[EOS]", vocab["[EOS]"])],
        )
    # else CONTENT-ONLY (per-record framing, mixlab -frame-per-record): emit bare content tokens;
    # the loader adds [BOS]/[EOS]/[PAD] per row. This is the tokenizer the reproduction uses
    # (data/tokenizer_content.json). See MOLGPT_REPRODUCTION.md.
    return tok, vocab, chars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output tokenizer.json path")
    ap.add_argument("--max-train", type=int, default=0,
                    help="cap train molecules used to derive the alphabet (0=all)")
    ap.add_argument("--no-template", action="store_true",
                    help="emit a CONTENT-ONLY tokenizer (no [BOS]/[EOS] wrapping) for per-record "
                         "framing (mixlab -frame-per-record); the loader adds BOS/EOS/PAD per row")
    args = ap.parse_args()

    add_template = not args.no_template
    train = moses.get_dataset("train")
    if args.max_train:
        train = train[: args.max_train]
    tok, vocab, chars = build(train, add_template=add_template)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(outp))
    print(f"vocab_size={len(vocab)}  ({len(SPECIALS)} specials + {len(chars)} chars)")
    print(f"chars: {''.join(chars)!r}")
    print(f"saved {outp}")

    # Round-trip check against the raw SMILES (strip specials, join chars).
    id2tok = {i: t for t, i in vocab.items()}
    fails = 0
    sample = train[:2000]
    for s in sample:
        ids = tok.encode(s).ids
        if add_template:
            assert ids[0] == vocab["[BOS]"] and ids[-1] == vocab["[EOS]"], (s, ids[:3])
            body = ids[1:-1]
        else:
            body = ids  # content-only: no BOS/EOS to strip
        recon = "".join(id2tok[i] for i in body)
        if recon != s:
            fails += 1
            if fails <= 3:
                print(f"  MISMATCH: {s!r} -> {recon!r}", file=sys.stderr)
    print(f"round-trip: {len(sample)-fails}/{len(sample)} exact")
    if fails:
        sys.exit(1)


if __name__ == "__main__":
    main()

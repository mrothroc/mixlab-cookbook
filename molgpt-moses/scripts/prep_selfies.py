#!/usr/bin/env python3
"""Encode a MOSES split as SELFIES-symbol JSONL and prepare record shards.

Each JSONL row contains space-joined SELFIES symbols. The script always writes
the JSONL and sorted alphabet, then (unless --skip-prepare) invokes mixlab with
per-record framing. The tokenizer must exist before prepare.

Two-pass flow: first run with --skip-prepare, build the tokenizer with
build_selfies_tokenizer.py using the emitted --alphabet-out, then run this
script again to re-encode and prepare. --jsonl-exists-ok is accepted as an
explicit note that the JSONL is replaced; replacement is always allowed.
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

from rdkit import RDLogger
import moses
import selfies as sf

RDLogger.DisableLog("rdApp.*")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test", "test_scaffolds"])
    ap.add_argument("--out-jsonl", default="data/selfies_train.jsonl")
    ap.add_argument("--shards-out", default="data/selfies_record_shards")
    ap.add_argument("--tokenizer", default="data/selfies_tokenizer_content.json")
    ap.add_argument("--alphabet-out", default="data/selfies_alphabet.json")
    ap.add_argument("--record-seq-len", type=int, default=72)
    ap.add_argument("--limit", type=int, default=0, help="0=all")
    ap.add_argument("--mixlab", default="mixlab")
    ap.add_argument("--skip-prepare", action="store_true",
                    help="only write JSONL and alphabet; skip mixlab prepare")
    ap.add_argument("--jsonl-exists-ok", action="store_true",
                    help="document that the existing JSONL will be replaced (replacement is automatic)")
    args = ap.parse_args()

    data = moses.get_dataset(args.split)
    if args.limit:
        data = data[: args.limit]

    outp = Path(args.out_jsonl)
    outp.parent.mkdir(parents=True, exist_ok=True)
    alphabet, encoded, fails, max_symbol_len = set(), 0, 0, 0
    with open(outp, "w") as f:
        for smiles in data:
            try:
                selfies = sf.encoder(smiles)
                symbols = list(sf.split_selfies(selfies))
            except Exception:
                fails += 1
                continue
            alphabet.update(symbols)
            max_symbol_len = max(max_symbol_len, len(symbols))
            f.write(json.dumps({"text": " ".join(symbols)}) + "\n")
            encoded += 1

    total = len(data)
    rate = encoded / total if total else 0.0
    print(f"encoded {encoded:,} / {total:,}")
    print(f"fails: {fails:,}")
    print(f"encode success rate: {rate:.6%}")
    print(f"max symbol length: {max_symbol_len}")
    print(f"alphabet size: {len(alphabet)}")

    alphabetp = Path(args.alphabet_out)
    alphabetp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(sorted(alphabet), open(alphabetp, "w"), indent=2)
    print(f"wrote alphabet to {alphabetp}")

    if max_symbol_len + 2 > args.record_seq_len:
        print(f"error: max symbol length {max_symbol_len} + BOS/EOS exceeds "
              f"record_seq_len {args.record_seq_len}", file=sys.stderr)
        sys.exit(1)
    if args.skip_prepare:
        return
    tokenizerp = Path(args.tokenizer)
    if not tokenizerp.exists():
        print(f"error: tokenizer {tokenizerp} does not exist; run "
              "build_selfies_tokenizer.py with --alphabet-json first", file=sys.stderr)
        sys.exit(1)

    env = dict(os.environ, MIXLAB_MLX_CACHE_LIMIT_MB="4096")
    subprocess.run(
        [args.mixlab, "-mode", "prepare", "-input", str(outp),
         "-output", args.shards_out, "-tokenizer-path", str(tokenizerp),
         "-text-field", "text", "-frame-per-record",
         "-record-seq-len", str(args.record_seq_len), "-record-pad-id", "0",
         "-record-bos-id", "1", "-record-eos-id", "2"],
        check=True, env=env)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Emit a MOSES split as JSONL ({"text": "<SMILES>"}) for mixlab prepare.

RDKit-canonicalizes each SMILES (MOSES is already canonical; this makes the
canonicalization explicit and drops anything RDKit can't parse). One molecule
per line -> mixlab's prepare wraps each in [BOS]...[EOS] via the tokenizer and
concatenates into the flat training stream.
"""
import argparse, json, sys
from pathlib import Path
from rdkit import Chem, RDLogger
import moses

RDLogger.DisableLog("rdApp.*")


def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m is not None else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test", "test_scaffolds"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0=all")
    ap.add_argument("--no-canon", action="store_true", help="skip RDKit canonicalization")
    args = ap.parse_args()

    data = moses.get_dataset(args.split)
    if args.limit:
        data = data[: args.limit]

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    n, dropped = 0, 0
    with open(outp, "w") as f:
        for s in data:
            smi = s if args.no_canon else canon(s)
            if smi is None:
                dropped += 1
                continue
            f.write(json.dumps({"text": smi}) + "\n")
            n += 1
    print(f"wrote {n:,} molecules to {outp} (dropped {dropped})")


if __name__ == "__main__":
    main()

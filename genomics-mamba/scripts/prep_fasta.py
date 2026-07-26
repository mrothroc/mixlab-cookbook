#!/usr/bin/env python3
"""Clean a UCSC chromosome FASTA into mixlab-ready contigs.

- Uppercases (drops soft-masking; we do not use repeat masks for the smoke).
- Splits on runs of N (assembly gaps) into ACGT-only contigs.
- Emits contigs >= --min-len as separate FASTA records (mixlab treats each
  record as an independent example: no attention/target bleed across contigs).
"""
import argparse, os, re, sys

def contigs(path):
    seq = []
    name = None
    for line in open(path):
        if line.startswith('>'):
            if name is not None:
                yield name, ''.join(seq)
            name = line[1:].strip().split()[0]
            seq = []
        else:
            seq.append(line.strip())
    if name is not None:
        yield name, ''.join(seq)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs', nargs='+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--min-len', type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    n_records = 0
    total_bp = 0
    with open(args.out, 'w') as out:
        for path in args.inputs:
            for name, s in contigs(path):
                s = s.upper()
                for i, seg in enumerate(re.split(r'N+', s)):
                    if len(seg) < args.min_len:
                        continue
                    assert set(seg) <= set('ACGT'), f'non-ACGT in {name}: {set(seg)-set("ACGT")}'
                    out.write(f'>{name}_seg{i} len={len(seg)}\n')
                    for j in range(0, len(seg), 80):
                        out.write(seg[j:j+80] + '\n')
                    n_records += 1
                    total_bp += len(seg)
    print(f'wrote {n_records:,} contigs, {total_bp:,} bp -> {args.out}', file=sys.stderr)

if __name__ == '__main__':
    main()

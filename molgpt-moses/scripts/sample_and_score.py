#!/usr/bin/env python3
"""Sample molecules from a trained mixlab char-LM and score with moses.metrics.

Sampling: mixlab's native `-num-samples`/`-gen-seed` batch sampler emits a
diverse, deterministic set of sequences in one process.

Decoding: each generated stream starts at [BOS](1); we take tokens up to the
first [EOS](2) and map ids->chars via tokenizer.json. Scoring: the official
moses.metrics.get_all_metrics harness (never reimplemented).
"""
import argparse, json, subprocess, tempfile, os
from pathlib import Path
import moses
import selfies as sf
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

BOS, EOS = 1, 2


def load_id2tok(tok_path):
    vocab = json.load(open(tok_path))["model"]["vocab"]
    id2 = {i: t for t, i in vocab.items()}
    return id2


def decode(token_ids, id2tok):
    # strip leading BOS, cut at first EOS, map remaining ids to chars
    ids = token_ids
    if ids and ids[0] == BOS:
        ids = ids[1:]
    out = []
    for i in ids:
        if i == EOS:
            break
        out.append(id2tok.get(i, ""))
    return "".join(out)


def decode_selfies(token_ids, id2tok):
    # strip leading BOS, cut at first EOS, filter specials, map ids to SELFIES symbols
    ids = token_ids
    if ids and ids[0] == BOS:
        ids = ids[1:]
    out = []
    for i in ids:
        if i == EOS:
            break
        if i in (0, BOS, EOS, 3):
            continue
        out.append(id2tok.get(i, ""))
    try:
        return sf.decoder("".join(out))
    except Exception:
        return ""


def sample_batch(args, id2tok):
    with tempfile.NamedTemporaryFile(suffix=".smi", delete=False) as tf:
        tmp = tf.name
    try:
        env = dict(os.environ, MIXLAB_MLX_CACHE_LIMIT_MB="4096", MIXLAB_MLX_MEM_LOG_EVERY="0")
        r = subprocess.run(
            [args.mixlab, "-mode", "generate", "-config", args.config,
             "-safetensors-load", args.ckpt, "-num-samples", str(args.n),
             "-gen-seed", str(args.gen_seed), "-eos-token-id", str(EOS),
             "-generate-out", tmp,
             "-prompt", f"token_ids:{BOS}", "-max-tokens", str(args.max_tokens),
             "-temperature", str(args.temperature), "-top-k", str(args.top_k),
             "-gen-batch", str(args.gen_batch)],
            stderr=subprocess.PIPE, text=True, env=env)
        if r.returncode != 0:
            raise RuntimeError(f"mixlab generate failed ({r.returncode}):\n{r.stderr}")
        smiles = []
        with open(tmp) as f:
            for line in f:
                ids = [int(x) for x in line.strip().split(",") if x != ""]
                smiles.append(decode_selfies(ids, id2tok) if args.selfies else decode(ids, id2tok))
        return smiles
    finally:
        os.unlink(tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixlab", default="mixlab")
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="data/tokenizer_content.json",
                    help="vocab-30 char tokenizer (ids match the trained model; content-only is fine "
                         "for decoding since it shares the same vocab as the template variant)")
    ap.add_argument("--n", type=int, default=30000, help="MOSES reproduction protocol = 30000")
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--gen-seed", type=int, default=0)
    ap.add_argument("--gen-batch", type=int, default=128, help="sequences per batched forward (mixlab v0.70+)")
    ap.add_argument("--test-subset", type=int, default=0,
                    help="0=full test/test_scaffolds reference (default). >0 subsets ONLY test/tsf "
                         "(directional FCD/SNN); the train reference for Novelty is ALWAYS full.")
    ap.add_argument("--n-jobs", type=int, default=1, help="moses.metrics parallelism")
    ap.add_argument("--out", required=True, help="samples .smi output path")
    ap.add_argument("--selfies", action="store_true",
                    help="decode generated token ids as SELFIES symbols, then convert to SMILES")
    args = ap.parse_args()

    id2tok = load_id2tok(args.tokenizer)
    smiles = sample_batch(args, id2tok)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(smiles) + "\n")
    print(f"wrote {len(smiles)} raw samples to {args.out}")
    if args.selfies:
        nonempty = [s for s in smiles if s]
        valid_nonempty = sum(Chem.MolFromSmiles(s) is not None for s in nonempty)
        validity_nonempty = valid_nonempty / len(nonempty) if nonempty else 0.0
        print(f"empty decodes: {len(smiles) - len(nonempty)}/{len(smiles)}")
        print(f"non-empty decodes: {len(nonempty)}/{len(smiles)}")
        print(f"validity among non-empty: {validity_nonempty:.4f}")

    # moses.get_dataset returns numpy ndarrays; pass Python lists so moses.metrics'
    # internal `train = train or get_dataset('train')` guard (metrics.py:77) short-circuits
    # on truthiness instead of raising "ambiguous truth value of an array".
    test = list(moses.get_dataset("test"))
    tsf = list(moses.get_dataset("test_scaffolds"))
    train = list(moses.get_dataset("train"))  # ALWAYS full: Novelty checks membership in the train set,
                                              # so a subset would inflate Novelty with false "novel" hits.
    if args.test_subset:
        # Subset only the test/test_scaffolds reference (directional FCD/SNN); never the train set.
        test = test[: args.test_subset]
        tsf = tsf[: args.test_subset]

    m = moses.metrics.get_all_metrics(
        smiles, n_jobs=args.n_jobs, device="cpu", batch_size=512,
        test=test, test_scaffolds=tsf, train=train)
    print("\n=== moses.metrics ===")
    for k in sorted(m):
        print(f"  {k:16s} {float(m[k]):.4f}")
    json.dump({k: float(v) for k, v in m.items()},
              open(Path(args.out).with_suffix(".metrics.json"), "w"), indent=2)


if __name__ == "__main__":
    main()

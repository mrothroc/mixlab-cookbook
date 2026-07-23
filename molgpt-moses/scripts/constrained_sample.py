#!/usr/bin/env python3
"""Sample MOSES SMILES with a token-DFA and report two validity surfaces."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import moses
from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")

BOS, EOS = 1, 2
BASELINE_VALID = 0.9875
NOVELTY_BAND = (0.747, 0.847)
INTDIV_BAND = (0.837, 0.877)


def load_id2tok(tok_path):
    with open(tok_path) as f:
        vocab = json.load(f)["model"]["vocab"]
    return {i: token for token, i in vocab.items()}


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


def _run_chunk(args, seed, count, tmp):
    env = dict(os.environ, MIXLAB_MLX_CACHE_LIMIT_MB="4096", MIXLAB_MLX_MEM_LOG_EVERY="0")
    command = [
        args.mixlab, "-mode", "generate", "-config", args.config,
        "-safetensors-load", args.ckpt, "-num-samples", str(count),
        "-gen-seed", str(seed), "-eos-token-id", str(EOS),
        "-generate-out", tmp, "-prompt", f"token_ids:{BOS}",
        "-max-tokens", str(args.max_tokens), "-temperature", str(args.temperature),
        "-top-k", str(args.top_k), "-gen-batch", str(args.gen_batch),
        "-grammar-table", args.grammar,
    ]
    return subprocess.run(command, stderr=subprocess.PIPE, text=True, env=env)


def sample_batch(args, id2tok):
    """Resilient constrained generation.

    mixlab v0.73 ABORTS the entire run when any one sample cannot complete the grammar
    within -max-tokens (e.g. rings still open) -- it writes the good samples first, then
    exits non-zero. We read the partial output, advance -gen-seed, and keep generating
    until we have >= args.n samples. (Upstream fix specced: a -grammar-on-incomplete=skip
    flag so mixlab drops a stuck sample instead of failing the run.)
    """
    smiles = []
    seed = args.gen_seed
    aborts = attempts = 0
    chunk = max(2000, args.gen_batch)
    max_attempts = 200
    while len(smiles) < args.n and attempts < max_attempts:
        attempts += 1
        with tempfile.NamedTemporaryFile(suffix=".smi", delete=False) as tf:
            tmp = tf.name
        try:
            result = _run_chunk(args, seed, chunk, tmp)
            with open(tmp) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    ids = [int(x) for x in line.split(",") if x != ""]
                    smiles.append(decode(ids, id2tok))
            if result.returncode != 0:
                if "does not complete the grammar" not in (result.stderr or ""):
                    raise RuntimeError(f"mixlab generate failed ({result.returncode}):\n{result.stderr}")
                aborts += 1
        finally:
            os.unlink(tmp)
        seed += 1
    print(f"[constrained] collected {len(smiles)} samples over {attempts} chunk(s), "
          f"{aborts} grammar-incomplete abort(s) (mixlab robustness gap)")
    if len(smiles) < args.n:
        print(f"[constrained] WARNING: only {len(smiles)} < requested {args.n} after {attempts} attempts")
    return smiles[: args.n]


def structural_valid(smiles):
    """Check the promised structural subset independently of the compiled DFA."""
    if not smiles or smiles[-1] in "-=#" or "()" in smiles:
        return False
    depth = 0
    ringmask = 0
    bracket = None
    for char in smiles:
        if bracket is not None:
            if char == "]":
                if bracket not in ("nH", "H"):
                    return False
                bracket = None
            elif char == "[" or len(bracket) >= 2:
                return False
            else:
                bracket += char
            continue
        if char == "[":
            bracket = ""
        elif char == "]":
            return False
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return False
        elif char in "123456":
            ringmask ^= 1 << (int(char) - 1)
    return depth == 0 and ringmask == 0 and bracket is None


def residual_class(smiles):
    molecule = Chem.MolFromSmiles(smiles, sanitize=False)
    if molecule is None:
        return "parse"
    try:
        Chem.SanitizeMol(molecule)
        return None
    except Exception as error:
        message = str(error).lower()
        if "kekulize" in message or "aromatic" in message:
            return "kekulization"
        if "valence" in message:
            return "valence"
        return "other"


def metric(metrics, *names):
    for name in names:
        if name in metrics:
            return float(metrics[name])
    raise KeyError(f"none of the metric keys are present: {names}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixlab", default="mixlab")
    parser.add_argument("--config", default="configs/molgpt_record.json")
    parser.add_argument("--ckpt", default="checkpoints/molgpt_record.safetensors")
    parser.add_argument("--tokenizer", default="data/tokenizer_content.json")
    parser.add_argument("--grammar", default="grammars/smiles_structural.token_dfa.json")
    parser.add_argument("--n", type=int, default=30000)
    parser.add_argument("--gen-seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--gen-batch", type=int, default=128)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--test-subset", type=int, default=0,
                        help="0=full test references; train is always full")
    parser.add_argument("--out", default="samples/constrained_30k.smi")
    args = parser.parse_args()

    smiles = sample_batch(args, load_id2tok(args.tokenizer))
    if len(smiles) != args.n:
        raise RuntimeError(f"mixlab emitted {len(smiles)} samples; expected {args.n}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write("\n".join(smiles) + "\n")

    structural_rate = sum(structural_valid(s) for s in smiles) / len(smiles)
    valid_mask = [Chem.MolFromSmiles(s) is not None for s in smiles]
    delivered_count = sum(valid_mask)
    residual = Counter({"parse": 0, "kekulization": 0, "valence": 0, "other": 0})
    for smi, valid in zip(smiles, valid_mask):
        if not valid:
            category = residual_class(smi)
            residual[category or "other"] += 1

    # Current MOSES returns NumPy arrays, but get_all_metrics truth-tests these
    # arguments internally; lists preserve the full datasets and avoid ambiguity.
    test = list(moses.get_dataset("test"))
    test_scaffolds = list(moses.get_dataset("test_scaffolds"))
    train = list(moses.get_dataset("train"))
    if args.test_subset:
        test = test[:args.test_subset]
        test_scaffolds = test_scaffolds[:args.test_subset]
    metrics = moses.metrics.get_all_metrics(
        smiles, n_jobs=args.n_jobs, device="cpu", batch_size=512,
        test=test, test_scaffolds=test_scaffolds, train=train)
    metrics_json = {key: float(value) for key, value in metrics.items()}
    novelty = metric(metrics_json, "Novelty")
    intdiv = metric(metrics_json, "IntDiv")
    results = {
        "n_samples": len(smiles),
        "structural_valid_rate": structural_rate,
        "constrained_only_valid": metric(metrics_json, "valid"),
        "baseline_unconstrained_valid": BASELINE_VALID,
        "constrained_plus_filter_valid": 1.0,
        "delivered_count": delivered_count,
        "dropped_count": len(smiles) - delivered_count,
        "residual_error_breakdown": dict(residual),
        "novelty": novelty,
        "intdiv": intdiv,
        "intdiv2": metric(metrics_json, "IntDiv2"),
        "unique_at_1000": metric(metrics_json, "unique@1000", "unique_at_1000"),
        "unique_at_10000": metric(metrics_json, "unique@10000", "unique_at_10000"),
        "coverage_guard": {
            "novelty_band": list(NOVELTY_BAND),
            "intdiv_band": list(INTDIV_BAND),
            "novelty_ok": NOVELTY_BAND[0] <= novelty <= NOVELTY_BAND[1],
            "intdiv_ok": INTDIV_BAND[0] <= intdiv <= INTDIV_BAND[1],
        },
    }
    metrics_path = out_path.with_suffix(".metrics.json")
    results_path = out_path.with_suffix(".results.json")
    with metrics_path.open("w") as f:
        json.dump(metrics_json, f, indent=2)
        f.write("\n")
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")

    guard = results["coverage_guard"]
    print(f"wrote {len(smiles)} raw samples to {out_path}")
    print("\n=== constrained decoding summary ===")
    print(f"{'surface':30s} {'value':>10s}")
    print(f"{'structural-valid':30s} {structural_rate:10.4f}")
    print(f"{'constrained-only RDKit valid':30s} {results['constrained_only_valid']:10.4f}")
    print(f"{'unconstrained baseline':30s} {BASELINE_VALID:10.4f}")
    print(f"{'+ RDKit filter (delivered)':30s} {1.0:10.4f}")
    print(f"delivered={delivered_count} dropped={len(smiles) - delivered_count}")
    print(f"residual={dict(residual)}")
    print(f"novelty={novelty:.4f} band={NOVELTY_BAND} ok={guard['novelty_ok']}")
    print(f"intdiv={intdiv:.4f} band={INTDIV_BAND} ok={guard['intdiv_ok']}")
    print(f"wrote {metrics_path}")
    print(f"wrote {results_path}")


if __name__ == "__main__":
    main()

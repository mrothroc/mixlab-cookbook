#!/usr/bin/env python3
"""Goal-directed QED optimization of the reproduced mixlab MolGPT."""

import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from rdkit import Chem, RDLogger
from tdc import Oracle
from moses.metrics import internal_diversity as moses_internal_diversity

from sample_and_score import load_id2tok, sample_batch


RDLogger.DisableLog("rdApp.*")
TOKENIZER = "data/tokenizer_content.json"
FT_CONFIG = "configs/molgpt_ft.json"


def sampling_args(args, ckpt, n, seed):
    """Build the namespace expected by sample_and_score.sample_batch."""
    return SimpleNamespace(
        mixlab=args.mixlab,
        config=args.config,
        ckpt=str(ckpt),
        n=n,
        gen_seed=seed,
        max_tokens=80,
        temperature=1.0,
        top_k=0,
        gen_batch=128,
        selfies=False,
    )


def sample_valid(args, id2tok, ckpt, n, seed):
    raw = sample_batch(sampling_args(args, ckpt, n, seed), id2tok)
    valid = [smiles for smiles in raw if smiles and Chem.MolFromSmiles(smiles) is not None]
    return raw, valid


def score_smiles(smiles, oracle):
    return [float(oracle(smile)) for smile in smiles]


def internal_diversity(smiles):
    """Compute the official MOSES IntDiv metric used by this target."""
    return float(moses_internal_diversity(smiles, n_jobs=1, device="cpu"))


def run_checked(command):
    env = dict(os.environ, MIXLAB_MLX_CACHE_LIMIT_MB="4096")
    subprocess.run(command, check=True, env=env)


def write_round_config(path, ft_steps):
    config = json.loads(Path(FT_CONFIG).read_text())
    config["training"]["steps"] = ft_steps
    path.write_text(json.dumps(config, indent=2) + "\n")


def fine_tune(args, elites, current_ckpt, round_index):
    round_dir = Path(args.workdir) / f"round_{round_index:02d}"
    shards_dir = round_dir / "shards"
    round_dir.mkdir(parents=True, exist_ok=True)
    shards_dir.mkdir(parents=True, exist_ok=True)

    elite_path = round_dir / "elite.jsonl"
    with elite_path.open("w") as output:
        for smiles in elites:
            output.write(json.dumps({"text": smiles}) + "\n")

    run_checked([
        args.mixlab, "-mode", "prepare", "-input", str(elite_path),
        "-output", str(shards_dir), "-tokenizer-path", TOKENIZER,
        "-text-field", "text", "-frame-per-record", "-record-seq-len", "64",
        "-record-pad-id", "0", "-record-bos-id", "1", "-record-eos-id", "2",
    ])

    round_config = round_dir / "molgpt_ft.json"
    write_round_config(round_config, args.ft_steps)
    next_ckpt = round_dir / "model.safetensors"
    run_checked([
        args.mixlab, "-mode", "arch", "-config", str(round_config),
        "-train", str(shards_dir / "train_*.bin"),
        "-safetensors-load", str(current_ckpt), "-safetensors", str(next_ckpt),
    ])
    if not next_ckpt.is_file():
        raise RuntimeError(f"fine-tuning did not create {next_ckpt}")
    return next_ckpt


def evaluate(args, id2tok, oracle, ckpt):
    raw, valid = sample_valid(args, id2tok, ckpt, args.eval_n, args.eval_seed)
    if not valid:
        raise RuntimeError(f"evaluation produced no valid molecules from {ckpt}")
    scores = score_smiles(valid, oracle)
    return {
        "mean": float(np.mean(scores)),
        "validity": len(valid) / len(raw) if raw else 0.0,
        "max": float(max(scores)),
        "valid_smiles": valid,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Elitist hill-climb optimization of MolGPT on QED")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--pool", type=int, default=2000)
    parser.add_argument("--elite-frac", type=float, default=0.125)
    parser.add_argument("--ft-steps", type=int, default=150)
    parser.add_argument("--eval-n", type=int, default=5000)
    parser.add_argument("--eval-seed", type=int, default=20260721)
    parser.add_argument("--oracle", default="QED")
    parser.add_argument("--mixlab", default="mixlab")
    parser.add_argument("--config", default="configs/molgpt_record.json")
    parser.add_argument("--base-ckpt", default="checkpoints/molgpt_record.safetensors")
    parser.add_argument("--workdir", default="checkpoints/goal_directed")
    parser.add_argument("--out", default="results/goal_directed_qed.json")
    args = parser.parse_args()
    if args.rounds < 1 or args.pool < 1 or args.ft_steps < 1 or args.eval_n < 1:
        parser.error("--rounds, --pool, --ft-steps, and --eval-n must be positive")
    if not 0.0 < args.elite_frac <= 1.0:
        parser.error("--elite-frac must be in (0, 1]")
    return args


def main():
    args = parse_args()
    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    id2tok = load_id2tok(TOKENIZER)
    oracle = Oracle(name=args.oracle)

    print(f"Evaluating baseline checkpoint: {args.base_ckpt}", flush=True)
    baseline = evaluate(args, id2tok, oracle, args.base_ckpt)
    current_ckpt = Path(args.base_ckpt)
    trajectory = []

    for round_index in range(1, args.rounds + 1):
        pool_seed = args.eval_seed + 10_000 + round_index
        raw, valid = sample_valid(args, id2tok, current_ckpt, args.pool, pool_seed)
        if not valid:
            raise RuntimeError(f"round {round_index} produced no valid molecules")
        scores = score_smiles(valid, oracle)
        elite_count = max(1, math.ceil(len(valid) * args.elite_frac))
        ranked = sorted(zip(scores, valid), key=lambda item: item[0], reverse=True)
        elite_scores = [item[0] for item in ranked[:elite_count]]
        elites = [item[1] for item in ranked[:elite_count]]
        entry = {
            "round": round_index,
            "checkpoint_in": str(current_ckpt),
            "pool_seed": pool_seed,
            "pool_validity": len(valid) / len(raw) if raw else 0.0,
            "pool_valid_count": len(valid),
            "pool_mean_qed": float(np.mean(scores)),
            "elite_count": elite_count,
            "elite_mean_qed": float(np.mean(elite_scores)),
            "elite_max_qed": float(max(elite_scores)),
        }
        print(f"round {round_index}: pool mean={entry['pool_mean_qed']:.4f}, "
              f"elite mean={entry['elite_mean_qed']:.4f}", flush=True)
        current_ckpt = fine_tune(args, elites, current_ckpt, round_index)
        entry["checkpoint_out"] = str(current_ckpt)
        trajectory.append(entry)

    print(f"Evaluating optimized checkpoint: {current_ckpt}", flush=True)
    optimized = evaluate(args, id2tok, oracle, current_ckpt)
    result = {
        "oracle": args.oracle,
        "method": "elitist_hill_climb",
        "rounds": args.rounds,
        "pool": args.pool,
        "elite_frac": args.elite_frac,
        "ft_steps": args.ft_steps,
        "eval_n": args.eval_n,
        "eval_seed": args.eval_seed,
        "baseline_mean_qed": baseline["mean"],
        "optimized_mean_qed": optimized["mean"],
        "lift": optimized["mean"] - baseline["mean"],
        "baseline_validity": baseline["validity"],
        "optimized_validity": optimized["validity"],
        "optimized_intdiv": internal_diversity(optimized["valid_smiles"]),
        "per_round_trajectory": trajectory,
        "bestofN_max_qed": optimized["max"],
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

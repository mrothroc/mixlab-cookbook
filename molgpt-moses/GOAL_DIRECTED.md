# Steer the model toward a molecular property with fine tuning

This tutorial nudges the reproduced 6.34M MolGPT checkpoint toward higher drug-likeness, measured with TDC's deterministic `QED` oracle. It fine-tunes the model you already trained rather than training a new one from scratch. It uses a REINVENT/GuacaMol-style elitist hill climb: sample a fresh pool, score the RDKit-valid molecules, retain the top 12.5%, then briefly fine-tune on those elites with per-record framing.

The important guardrail is diversity. Measure IntDiv next to QED so a model cannot appear to improve by repeating one molecule.

## What the gentle schedule does

One round with 40 fine-tune steps shifts fresh unconditional samples from the original and final checkpoints:

| Metric | Baseline | Optimized | |
|---|---|---|---|
| **mean QED** (official TDC oracle, 5000 fresh unconditional samples) | 0.8034 | **0.8931** | **lift +0.0897** |
| validity | 0.9888 | 0.9802 | preserved |
| **IntDiv** (MOSES internal diversity) | ~0.856 | **0.8424** | preserved, no mode collapse |
| best-of-N QED | - | 0.9482 | (context) |

The unconditional mean QED rises by **+0.090**, above the target of >=+0.03, while validity stays about 0.98 and IntDiv stays about 0.84. The comparison uses 5000 fresh, identically seeded samples rather than selecting the best molecules from one pool. See `results/goal_directed_qed.json`.

## Why the diversity guardrail matters

An aggressive schedule of **5 rounds / 150 steps** gives a larger nominal lift of +0.143 and mean QED of 0.9467, but **IntDiv = 0.0**. By rounds 4 to 5, the model emits a single molecule with QED 0.9466523221135084 to 15 decimals.

That is mode collapse, not useful steering. Elitist self-distillation should be gentle, with few rounds and few steps, and every objective result should be reported alongside diversity.

## Run it

Start from `checkpoints/molgpt_record.safetensors`, the model trained in [MOLGPT_REPRODUCTION.md](MOLGPT_REPRODUCTION.md); run that leg first. You need mixlab >= v0.73.0 on PATH.

```bash
export MIXLAB_MLX_CACHE_LIMIT_MB=4096
export MIXLAB_SCRIPTS=/path/to/mixlab/scripts
export PATH="$PWD/.venv/bin:$PATH"
# Gentle schedule (the headline result: lift ~+0.09, IntDiv preserved):
python scripts/goal_directed.py --rounds 1 --pool 2000 --elite-frac 0.125 --ft-steps 40 \
  --eval-n 5000 --eval-seed 20260721 --out results/goal_directed_qed.json
# (Aggressive --rounds 5 --ft-steps 150 reproduces the mode-collapse artifact: bigger lift, IntDiv 0.)
```

You can swap the oracle with `--oracle DRD2`, `GSK3B`, or another supported oracle. The loop itself does not depend on QED. Round checkpoints, elite JSONL, prepared shards, and effective fine-tune configurations are stored under `checkpoints/goal_directed/`. Fine-tuning uses `configs/molgpt_ft.json`, a low-learning-rate, short-schedule warm start from the reproduction.

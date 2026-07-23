# BabyLM 2025 GPT-BERT masked-focus baseline — reproduced in mixlab

> **Canonical repo → https://github.com/mrothroc/mixlab-babylm-gptbert**
> (public). Full eval reports, per-task breakdowns, footnotes, and the architecture write-up (read from
> the weights, not the card) live there — this is a discovery summary for the cookbook index.

A faithful, from-scratch reproduction of the BabyLM 2025 GPT-BERT *masked-focus* Strict-Small baseline
([`BabyLM-community/babylm-baseline-10m-gpt-bert-masked-focus`](https://huggingface.co/BabyLM-community/babylm-baseline-10m-gpt-bert-masked-focus)),
trained on a **single Apple-silicon GPU** with [mixlab](https://github.com/mrothroc/mixlab). Not a
leaderboard entry — a faithfulness demonstration. GPT-BERT masked-focus is a real test of the trainer:
a **masked+causal hybrid** (~33M) with disentangled relative attention, a gated attention output, dense
layer aggregation, and a BERT-style MLM head.

## Per-component parity (one identical eval harness)

| Component | Reference baseline | mixlab reproduction | Δ |
|---|---:|---:|---:|
| BLiMP | 70.36 | 70.64 | +0.28 |
| BLiMP-supplement | 63.71 | 61.79 | −1.92 |
| EWoK | 51.63 | 50.88 | −0.75 |
| entity tracking | 40.14 | 40.33 | +0.19 |
| COMPS | 53.55 | 52.85 | −0.70 |
| reading (eye+SPR) | 6.39 | 7.30 | +0.91 |
| GLUE † | 66.20 | 64.18 | −2.02 |

Within ≈2 pts on every component, slightly above the reference on BLiMP and entity tracking. The six
zero-shot rows are **firm, within-harness** parity — the reference column is *our* re-evaluation of the
official 2025 model, tracking the published baselines-paper Table 2. **†** GLUE's reference is the official
*published* baseline (not re-fine-tuned here). AoA is forfeited (per-checkpoint trajectory not in these
artifacts; plus the upstream scorer bug) — reported per-component, not as a macro. Details in the
canonical repo.

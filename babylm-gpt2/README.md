# BabyLM 2026 GPT-2 Strict-Small baseline — reproduced in mixlab

> **Canonical repo & submission → https://github.com/mrothroc/mixlab-babylm-gpt2**
> (public). Full eval reports, per-task breakdowns, footnotes, and the submission artifacts live there —
> this is a discovery summary for the cookbook index.

A faithful, from-scratch, **data-identical** reproduction of the official BabyLM 2026 GPT-2 *Strict-Small*
baseline ([`BabyLM-community/babylm-baseline-10m-gpt2`](https://huggingface.co/BabyLM-community/babylm-baseline-10m-gpt2)),
trained end-to-end on a **single Apple M1 Max** with [mixlab](https://github.com/mrothroc/mixlab) (native
GPT-2 — zero new features needed). Trained on the same 10M-word Strict-Small corpus the organizers use.

## Per-component parity (one identical eval harness — `babylm-eval` @ `3bf5142`, causal)

| Component | Reference baseline | mixlab reproduction | Δ |
|---|---:|---:|---:|
| BLiMP | 66.35 | 65.86 | −0.49 |
| BLiMP-supplement | 57.07 | 58.04 | +0.97 |
| EWoK | 49.23 | 49.30 | +0.07 |
| COMPS | 51.72 | 52.05 | +0.33 |
| reading (eye+SPR) | 6.50 | 7.15 | +0.65 |
| GLUE (macro) † | 63.62 | 64.15 | +0.53 |

Within ≈1 pt on every stable zero-shot component. The six zero-shot rows are **firm, within-harness**
parity — the reference column is *our* re-evaluation of the official model through the same harness (it
reproduces the official published numbers, e.g. BLiMP 66.4). **†** GLUE's reference is the official
*published* baseline (we did not re-fine-tune it), so that row is a published-reference comparison, not a
within-harness re-eval. Entity-tracking (a length-bias artifact) and AoA (forfeited; plus a confirmed
upstream scorer bug) are excluded from any aggregate — details + reports in the canonical repo.

# mixlab cookbook

Reproductions, demos, and example recipes for **[mixlab](https://github.com/mrothroc/mixlab)**, a compact
ML architecture lab for training and experimenting on modest compute. It is accelerated on both Apple Silicon and CUDA so you can iterate locally then scale up on a big server.

Each recipe is a subdirectory: what it reproduces, the mixlab config(s), reproduce commands, and the resulting numbers, compared to references when available. An entry is either **self-contained** or a **summary + pointer** to a canonical standalone repo (used when that reproduction has its own submission or DOI); either way it shows up here so the whole set is discoverable in one place.

Metrics are typically **firm parity by default**, which is an apples-to-apples claim against an authoritative reference. The **directional** label is the *rare exception*, reserved for a metric whose true reference doesn't exist (e.g. an unpublished number, or an anchor from a different-scale reimplementation). We report the number for direction but do **not** claim parity.

## Recipes

| Recipe | Reproduces | Notes |
|---|---|---|
| [molgpt-moses](molgpt-moses/) | MolGPT (SMILES transformer) on MOSES | firm-metric parity (FCD directional) + grammar-constrained decoding, a SELFIES variant, goal-directed QED · self-contained |
| [genomics-mamba](genomics-mamba/) | DNA enhancer classifier (Genomic Benchmarks) | reproduce an attention baseline (0.717), then beat it (0.728) by swapping the mixer to canonical Mamba in the JSON · self-contained |
| [babylm-gpt2](babylm-gpt2/) | BabyLM 2026 GPT-2 Strict-Small baseline | per-component parity on a single Mac · → [standalone repo](https://github.com/mrothroc/mixlab-babylm-gpt2) |
| [babylm-gptbert](babylm-gptbert/) | BabyLM 2025 GPT-BERT masked-focus baseline | per-component parity of a masked+causal hybrid · → [standalone repo](https://github.com/mrothroc/mixlab-babylm-gptbert) |

## What a recipe subdirectory contains

```
<recipe>/
├── README.md      # what it reproduces, the parity table (firm vs directional), reproduce commands
├── configs/       # mixlab JSON config(s), forkable
└── results/       # small result JSONs (metrics); NO model weights or large data
```

## Relationship to mixlab

These recipes exercise mixlab and double as its example library. When useful, the bare configs also live under `mixlab/examples/` in that repo. The cookbook adds the narrative + reproduce numbers around them.

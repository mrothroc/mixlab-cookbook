# A vision transformer in mixlab: reproduce a ViT on CIFAR-10, then change the optimizer

The baseline is [`kentaroy47/vision-transformers-cifar10`](https://github.com/kentaroy47/vision-transformers-cifar10)
at commit `79fa30c` — the [lucidrains ViT](https://github.com/lucidrains/vit-pytorch) at patch 4 with CLS
pooling, 9.5M parameters, trained from scratch on CIFAR-10 with no pretraining and no distillation. Its
own checked-in log for this architecture ([`log/log_vit_patch4.csv`](https://github.com/kentaroy47/vision-transformers-cifar10/blob/79fa30c/log/log_vit_patch4.csv))
ends at **79.08%** after 200 epochs.

This recipe rebuilds that model in [mixlab](https://github.com/mrothroc/mixlab) from a JSON config and
reaches **79.25% ± 0.29** over three seeds, against **79.16% ± 0.25** from our own three-seed run of the
reference on the same hardware. That second number is the like-for-like comparison: we run the reference
with RandAugment off (see Results), and upstream's log does not record which flags produced its 79.08.

Changing the optimizer block in the config, and retuning that group's learning rate, brings the same
architecture and 200-epoch schedule to **85.88% ± 0.28**.

There are two models here: the reference recipe, and the same recipe with the matrix group on
[Muon](https://github.com/KellerJordan/Muon). We chose this reference because you can run it — exact
code at a pinned commit, and a checked-in training log to check against.

Use the recipe as-is, or as a starting point for your own experiments.

## The two-step arc

1. **Reproduce.** Match the reference recipe in a mixlab JSON config and hit its accuracy.
2. **Enhance.** Keep everything else fixed and route the `matrix` group to Muon instead of AdamW. Sweep
   its learning rate, and give AdamW the same number of tries.

## Results

Same architecture (9,523,722 params), same data, same 200-epoch / 19,600-step budget, same three seeds
(42 / 1234 / 2024). Final-epoch accuracy on the 10,000-image test set. The `±` values are sample
standard deviations across those three seeds, not confidence intervals — the 95% t half-width for the
paired Muon−AdamW difference is 0.22, and `results/extend.json` records it that way.

| recipe | test accuracy |
|---|---|
| **`matrix` group on Muon @ `matrix_lr` 0.01** | **85.88 ± 0.28** |
| best AdamW we found (`lr` 3e-4) | 82.56 ± 0.19 |
| the published recipe (`lr` 1e-4) | 79.25 ± 0.29 |
| the PyTorch reference, run by us | 79.16 ± 0.25 |

Of the 6.6-point improvement over the published recipe, raising AdamW's learning rate accounts for 3.3
points. Muon adds a further **3.32 ± 0.09** over the best AdamW setting we found, paired across the three
seeds.

Both learning rates were picked on the test set. Read 85.88 and 82.56 as the best of a sweep, not a
held-out number. The reproduction config was locked before the first run.

Three seeds per arm, so this is a pilot rather than a powered comparison. We also ran a control giving AdamW its own `matrix_lr`
with the other groups held at 1e-4, in case the gap was just the auxiliary rates: it reached 82.40 at
seed 42, no better than the arm above. Both are in [`results/extend.json`](results/extend.json).

What we reproduce is the repository at this exact invocation — quoted for provenance; the setup it needs
is under [Optional: the PyTorch reference arm](#optional-the-pytorch-reference-arm), and it will not run
on an unpatched clone:

```text
python train_cifar10.py --noaug --nowandb --noamp --n_epochs 200 --seed <42|1234|2024>
```

`--lr 1e-4`, `--bs 512` and `--n_epochs 200` are upstream's defaults. `--noaug` is **not** a default:
upstream declares it `action='store_false'` and assigns `aug = args.noaug`, so RandAugment is ON unless
you pass `--noaug`. We turn it off, leaving crop+flip, because RandAugment is a second variable and the
mixlab side has no equivalent. So this reproduces upstream's architecture and training budget with
augmentation deliberately reduced — not upstream out of the box.

The learning-rate *trajectory* is close but not identical. Upstream calls `scheduler.step(epoch-1)` once
per epoch, so its cosine is quantised to epochs and the off-by-one briefly raises the rate between epochs
1 and 2; mixlab steps its cosine every step. At the start of epoch 100 upstream is at 5.157e-05 against
mixlab's 5.000e-05. Same starting rate and epoch budget, near-zero at the end on both sides
(upstream lands at ~5.6e-08 rather than exactly 0), marginally different path in between.

The same repo's ViT appears in [arXiv:2302.03751](https://arxiv.org/abs/2302.03751) at 81.36, but that
is 500 epochs and a different recipe.

### About the published checkpoints

To have weights to publish we repeated both seed-42 recipes. AdamW scored 79.35% in both of its RTX 4090
runs. Muon scored 86.13% on an L4 and 86.01% on the 4090 re-run — 0.12 points apart.

That difference is in the training, not the scoring. Evaluating the *same* checkpoint on both GPU models
returns identical accuracy and loss to six decimal places, so inference is reproducible across cards and
the drift accumulated over 19,600 training steps. Which card caused it, or whether the card caused it at
all, these runs do not say — AdamW's two runs never crossed GPU models.

The published checkpoints score 79.35% and 86.01% on CUDA. On Apple Silicon the Muon checkpoint scores
86.00% — one image — and the reproduction is identical at 79.35%.

Numbers: [`results/reproduce.json`](results/reproduce.json), [`results/extend.json`](results/extend.json).

## Load the trained models

Both models are on Hugging Face.

- **Reproduction** (published recipe, 79.25 ± 0.29): [mrothroc/vit-cifar10-reproduce-mixlab](https://huggingface.co/mrothroc/vit-cifar10-reproduce-mixlab)
- **Muon variant** (85.88 ± 0.28): [mrothroc/vit-cifar10-muon-mixlab](https://huggingface.co/mrothroc/vit-cifar10-muon-mixlab)

Published checkpoints are seed 42: 79.35% and 86.01%. Inputs are CIFAR-10 images as 4×4 patch
sequences — `[batch, 64, 48]`, per-channel normalized with the reference constants, which
`scripts/prep_cifar.py` produces.

## Change the optimizer

The reproduction:

```json
"optimizer": "adamw",
"lr": 0.0001
```

The Muon variant:

```json
"optimizer": "muon",
"lr": 0.0001,
"matrix_lr": 0.01
```

mixlab partitions weights into four groups — `matrix`, `scalar`, `head`, `embed` — and each takes its
own optimizer and learning rate. `optimizer: "muon"` puts the `matrix` group on Muon and leaves the
rest on AdamW. A hybrid Muon/AdamW split is what Muon's own README recommends, though not this exact
partition — upstream suggests keeping embeddings on AdamW, and mixlab's routing puts
`position_embeddings` in `matrix`. `matrix_lr` then sets that group's rate independently.

Setting `lr` alone would move **all four groups**. Use `matrix_lr` to move only the matrix group. It works for
AdamW and LAMB too.

Run `mixlab -mode optimizer-report -config configs/extend_muon.json` to inspect each tensor's group,
optimizer and learning rate without a GPU. On this model `position_embeddings`, `cls_token` and
`input_adapter_proj` all belong to `matrix`, and `embed` is empty — so `embed_lr` has no effect here,
and Muon updates the position embeddings:

```
matrix  muon   lr=0.01     9,495,552 params
scalar  adamw  lr=0.0001      23,040 params
head    adamw  lr=0.0001       5,130 params
embed   -      lr=-                 0 params
```

## Setup

You will need the [mixlab](https://github.com/mrothroc/mixlab) binary, **v0.117.0 or newer**, on your
PATH (install it per the mixlab README). v0.116.0 added `weight_init: "pytorch_linear_all"` and the two
embedding init overrides this recipe needs; v0.117.0 added `-mode optimizer-report`.

Data prep is numpy-only — no torch, no torchvision:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PATH="$PWD/.venv/bin:$PATH"    # so mixlab's prepare finds python3 + numpy
```

CIFAR-10 downloads in step 1 below. Verify its md5 — a truncated download produces a file that looks
fine and trains to nonsense.

### Optional: the PyTorch reference arm

Only needed if you want to re-run the comparison rather than take our numbers for it. Skip to
**Reproduce it** for the mixlab path.

Upstream needs seven changes to `train_cifar10.py` before it will run this recipe at all — it has no
`--seed`, calls `.cuda()` unconditionally, and has a bug that makes `--nowandb` ineffective. Rather than
have you transcribe them, the entry ships the diff; it applies cleanly to the pinned commit and fails
loudly if it does not.

Run this from *outside* this recipe directory, so the upstream checkout lands beside it rather than
inside it — the paths below assume that layout:

```bash
RECIPE=$PWD/vit-cifar10-muon          # adjust if you are elsewhere
git clone https://github.com/kentaroy47/vision-transformers-cifar10
cd vision-transformers-cifar10
git checkout 79fa30c
git apply "$RECIPE/scripts/reference.patch"
pip install -r "$RECIPE/scripts/reference-requirements.txt"
python train_cifar10.py --noaug --nowandb --noamp --n_epochs 200 --seed 42
cd ..                                 # back out before continuing with the mixlab path
```

One epoch takes about 2 minutes on an M1 Max, so use `--n_epochs 1` first to confirm it starts. Outputs
land in `apollo_log/` and `checkpoint/`. Note that neither filename carries the seed and the CSV is
opened in write mode, so **run seeds in separate checkouts or copy the outputs between runs.**

What [`scripts/reference.patch`](scripts/reference.patch) changes, and why each one is needed:

| change | why |
|---|---|
| add `--seed`, seed `random`/`numpy`/`torch` | upstream seeds nothing, so its runs are not repeatable |
| `usewandb = ~args.nowandb` → `not args.nowandb` | upstream bug: `~True` is `-2`, truthy, so `--nowandb` never disabled W&B |
| add MPS to device selection; `net.cuda()` → `net.to(device)` | upstream calls `.cuda()` unconditionally and cannot run on a Mac |
| `num_workers` 8 → 0 on both loaders | avoids worker-count as a variable across machines |
| `log/` → `apollo_log/` in all three logging lines | purely ours: keeps our run's output in its own directory. It does **not** prevent a collision — upstream writes `log_vit_cifar10_patch4.csv` while the file we cite is `log_vit_patch4.csv`, so the names never clashed. All three lines must move together: upstream's `os.makedirs("log")` means renaming only one path gives `FileNotFoundError`. |

Only the last is cosmetic. The rest are required for the run to happen at all, or to be repeatable.

## Reproduce it

```bash
# 1. get the canonical archive and check it
curl -L -o cifar-10-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
echo 'c58f30108f718f92721af3b95e74349a  cifar-10-python.tar.gz' | md5sum -c -

# 2. CIFAR-10 -> normalized 4x4 patch sequences (.npy + label .tsv)
python scripts/prep_cifar.py --tar cifar-10-python.tar.gz --out data/

# 3. .npy -> mixlab shards. -val-split 0 is required: the default holds back 10%,
#    which would break the official 50k/10k split.
mixlab -mode prepare -input data/train_features.npy -prepare-output-dir data/shards_train \
       -input-format continuous -continuous-modality image \
       -label-file data/train_labels.tsv -val-split 0
mixlab -mode prepare -input data/test_features.npy  -prepare-output-dir data/shards_test \
       -input-format continuous -continuous-modality image \
       -label-file data/test_labels.tsv  -val-split 0

# 4. check the config resolves the way you expect (seconds, no GPU)
mixlab -mode count            -config configs/reproduce_adamw.json   # 9523722
mixlab -mode optimizer-report -config configs/reproduce_adamw.json

# 5. train (19,600 steps = 200 epochs)
mixlab -mode arch -config configs/reproduce_adamw.json -train 'data/shards_train/train_*.bin' \
       -safetensors out.st

# 6. evaluate once on the test split
mixlab -mode eval -config configs/reproduce_adamw.json -safetensors-load out.st \
       -val 'data/shards_test/train_*.bin'
```

Swap in `configs/extend_muon.json` for the Muon arm.

For a two-minute smoke instead of the full run, override just the step count:

```bash
python -c "import json;d=json.load(open('configs/extend_muon.json'));d['training']['steps']=200;\
           json.dump(d,open('smoke.json','w'),indent=2)"
mixlab -mode arch -config smoke.json -train 'data/shards_train/train_*.bin' -safetensors smoke.st
mixlab -mode eval -config smoke.json -safetensors-load smoke.st -val 'data/shards_test/train_*.bin'
```

That compresses the cosine schedule into 1% of its length, so the accuracy means nothing — it proves the
pipeline runs.

### Initialization

The shipped configs already set this correctly. The note is for when you change them.

Matching a PyTorch reference needs `weight_init: "pytorch_linear_all"` plus the two embedding standard
deviations, because the lucidrains ViT draws `pos_embedding` and `cls_token` from `torch.randn` rather
than mixlab's 0.02. The narrower `pytorch_linear` covers only tensors the architecture explicitly marks
— on this model 2 of 38 weight matrices — leaving the rest Xavier-uniform, √3 wider, which costs about
four points.

Every run prints what its init policy actually covered. If you change the architecture, read it:

```
weight_init: pytorch_linear_all applied to 38/38 ordinary affine matrices and 20/20 paired biases;
0 matrices retain Xavier, 28 tensors retain separate policies (2 explicit normal overrides)
```

## Files

```
configs/reproduce_adamw.json      the reproduction: AdamW everywhere, lr 1e-4
configs/extend_muon.json          the Muon variant: matrix group on Muon, matrix_lr 0.01
scripts/prep_cifar.py             CIFAR-10 -> 4x4 patch sequences, numpy only
results/reproduce.json            3 seeds vs the reference
results/extend.json               all arms, both LR sweeps, and the matched-auxiliary control
results/optimizer_report_*.json   resolved group assignment per arm, from mixlab itself
```

## Known rough edges

**Pin the GPU model, not just the memory size.** We allowed two RunPod pools (`ADA_24,AMPERE_24`) and
got both RTX 4090s and L4s. The same Muon recipe took 49.8 minutes on a 4090 and 131.8 minutes on an
L4 — 2.65× — and we lost a correct run to a 90-minute timeout before noticing. Pin one model, or give
the timeout room for the slower one.

**`matrix_lr` has a narrow usable band.** At seed 42: 0.003 → 83.25, 0.01 → 86.13, 0.03 → 54.91. The failure at the
top is not divergence — no NaNs — it simply never settles, and final training loss stays near 1.3
instead of 0.001. If a Muon run looks flat, suspect the rate before the architecture.

**Apple Silicon works, but every timing here is NVIDIA.** The same configs run unchanged on a Mac. A
200-step smoke run takes **108 s (AdamW) and 113 s (Muon)** on an M1 Max at ~60k tok/s — a cheap way to
confirm the whole pipeline (train, checkpoint, reload, evaluate) before renting a GPU. Both published
checkpoints also score correctly there: 79.35% and 86.00%, with mixlab's MLX path agreeing with PyTorch
MPS.

Those smoke runs are reproducible across machines. Running the identical configs and seed on an M1 Max
and an M4 Max, from byte-identical shards, gives the same final loss to four decimals on both:

| 200-step smoke | M1 Max | M4 Max |
|---|---|---|
| AdamW | 1.7524 | 1.7524 |
| Muon | 1.3063 | 1.3063 |

Don't read the gap between the two optimizers there as a preview of the headline — 200 steps compresses
the cosine schedule into 1% of its length, so these are pipeline checks.

What we have *not* done is a full 19,600-step run on Apple Silicon. At the smoke-run rate that
extrapolates to roughly three hours, but we have not run it, and sustained thermal behaviour over that
long is untested — so the entry quotes no Mac wall-clock for the headline results.


## Where to go next

- **Point it at your own images.** `scripts/prep_cifar.py` writes `[N, 64, 48]` patch sequences; any
  32×32 dataset with labels drops into the same config by changing `num_labels`.
- **Mixer swaps** live in [genomics-mamba](../genomics-mamba/) and
  [speech-commands-s4d](../speech-commands-s4d/).

## Exercising and enhancing mixlab

This recipe leans on the `linear_patches` image input adapter, native `objective: classification`,
per-group optimizer assignment (`optimizer` + `matrix_lr`), and `weight_init: "pytorch_linear_all"` with
`position_embedding_init_std` / `cls_token_init_std` for matching a PyTorch reference's initialization.

Building it added `pytorch_linear_all`, the embedding init overrides, and `-mode optimizer-report`.
The failed learning rates are in [`results/extend.json`](results/extend.json).

mixlab: https://github.com/mrothroc/mixlab

## Citing this work

This reproduces a published recipe and uses Muon, so cite both:

- **The reference implementation**: <https://github.com/kentaroy47/vision-transformers-cifar10> @ `79fa30c`
- **The ViT implementation** it wraps: Phil Wang, `vit-pytorch`, <https://github.com/lucidrains/vit-pytorch>
- **Vision Transformer**: Dosovitskiy, A.; Beyer, L.; Kolesnikov, A.; Weissenborn, D.; Zhai, X.;
  Unterthiner, T.; Dehghani, M.; Minderer, M.; Heigold, G.; Gelly, S.; Uszkoreit, J.; Houlsby, N.
  "An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR 2021.
  [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- **Muon**: Keller Jordan et al., <https://github.com/KellerJordan/Muon>
- **CIFAR-10**: Krizhevsky, A. "Learning Multiple Layers of Features from Tiny Images." Technical
  report, University of Toronto, 2009.
- A paper that trains this repo's ViT on CIFAR-10: Zhu, H.; Chen, B.; Yang, C. "Understanding Why
  ViT Trains Badly on Small Datasets: An Intuitive Perspective." 2023.
  [arXiv:2302.03751](https://arxiv.org/abs/2302.03751)

```bibtex
@software{vit_cifar10_muon_mixlab,
  author = {Rothrock, Michael},
  title  = {A vision transformer in mixlab: reproduce a ViT on CIFAR-10, then change the optimizer},
  year   = {2026},
  url    = {https://github.com/mrothroc/mixlab-cookbook/tree/main/vit-cifar10-muon}
}
@software{mixlab,
  author = {Rothrock, Michael},
  title  = {mixlab: a compact ML architecture lab},
  url    = {https://github.com/mrothroc/mixlab}
}
```

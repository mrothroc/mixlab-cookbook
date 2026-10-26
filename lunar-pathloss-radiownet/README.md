# Lunar radio maps in mixlab: reproduce the challenge baseline, then add terrain geometry

The [First Lunar Pathloss Radio Map Prediction Challenge](https://lunarradiomapchallenge.github.io/) (an IEEE
ICASSP 2027 Grand Challenge) asks for a model that predicts how much a radio signal weakens across a 256 × 256 patch
of simulated lunar terrain, given the heightmap and the transmitter's position. There are two frequency bands,
415 MHz and 5.8 GHz, and the score is RMSE in dB over the pixels the ray tracer covered.

The organizers' baseline is RadioWNet, the two-stage U-Net from
[RadioUNet](https://github.com/RonLevie/RadioUNet), trained once per band. Their
[repository](https://github.com/anderspearson206/lunar-radiomap-challenge) at commit `fa3ef9b` has the model code,
data loader and scorer, and the [challenge site](https://lunarradiomapchallenge.github.io/) reports **4.66 dB**
(415 MHz) and **6.28 dB** (5.8 GHz) on the test set.

This recipe rebuilds that model in [mixlab](https://github.com/mrothroc/mixlab) from a JSON config. It is the same
network, not an approximation: it starts from the reference's own PyTorch initialization, and on a trained checkpoint
we checked, its predictions stay within 0.005 dB of a float64 run of the organizers' code at every pixel. Trained for
150 + 50 epochs, it reaches **4.88 dB** and **6.78 dB** on the official validation split, averaged over three seeds.
The published numbers are on the test set from an unstated number of epochs, so they're a reference point rather than
a like-for-like target. The recipe was worked out on a Mac and then scaled up on rented GPUs, and the same configs run
on both.

Then it gives the same U-Net a few extra input channels computed from the terrain (distance, a line-of-sight estimate,
clearance). That brings it to **3.95 dB** and **5.24 dB**, with a shorter training run.

## The two-step arc

1. **Reproduce.** Express RadioWNet in a mixlab config, start from the reference's own initialization, and train
   with the reference recipe: AdamW at 1e-4 with zero weight decay (that is, Adam) and a cosine schedule, batch 16,
   flips and rotations, then a second stage that refines the first.
2. **Extend.** Keep the network and the optimizer. Add geometry channels to the input and widen the layers that
   read it. Train stage 1 only, for 50 epochs.

## Results

Official validation split, 2,330 maps per band, scored with the organizers' unmodified `score_submission.py`. Three
weight seeds per model; `±` is the sample standard deviation across seeds.

| model | 415 MHz | 5.8 GHz |
|---|---|---|
| **geometry variant:** stage 1 + geometry channels, 50 epochs | **3.95 ± 0.01** | **5.24 ± 0.01** |
| **reproduction:** RadioWNet, 150 + 50 epochs | 4.88 ± 0.26 | 6.78 ± 0.34 |
| the organizers' baseline, as published (test set) | 4.66 | 6.28 |

### Is it the same model?

Before comparing scores with anyone else's, we checked that mixlab computes the same function as the organizers' code.
We loaded one trained checkpoint (415 MHz, seed 0) into their PyTorch module, ran it in float64 on the CPU as the
reference, and compared every pixel of 80 validation maps. Largest difference from the reference at any pixel:

| runtime | largest per-pixel difference |
|---|---|
| mixlab, Apple Silicon | 0.0036 dB |
| mixlab, CUDA, strict FP32 | 0.0047 dB |
| PyTorch, CPU, FP32 | 0.0029 dB |
| PyTorch, CUDA, strict FP32 | 0.0898 dB |

mixlab is about as close to the reference as PyTorch itself on the CPU. The same weights also score the same on both of
mixlab's backends over the full validation split: 4.673 dB on Apple Silicon, 4.674 dB on CUDA. And the starting point
is the reference's own: `make_init.py` builds the organizers' PyTorch initialization and converts it tensor for
tensor. So the model and its arithmetic match. What the published numbers can't pin down is the training run: how
long, which seeds, and which split. Both checks are in [`results/reproduce.json`](results/reproduce.json).

### How close to the published numbers?

We fixed what would count as a match before training: the three-seed average had to land within 0.5 dB of the
published number in each band. At 415 MHz it lands 0.22 dB above. At 5.8 GHz it lands 0.497 dB above, only 0.003 dB
inside that threshold. The splits differ, and the seed spread is wide:

- **Different splits.** The published numbers are on the test set, whose answers arrive with the challenge's
  evaluation release. Ours are on the validation split. We also picked each run's best epoch on that same split,
  which flatters our numbers slightly.
- **Seed spread.** In both bands one of the three seeds lands above the limit on its own, and the spread across
  seeds is wide (0.26 and 0.34 dB). Most of it comes from how long stage 1 sits on a plateau before it starts
  learning, which varies from seed to seed.

### How it was built: Mac first, then CUDA

The published recipe doesn't say how many epochs produced its numbers, and that turned out to matter. The first pass
ran on an Apple M4 Max: three seeds per band at the repo's default 50 + 50 epochs. It came in high, at
5.18 and 7.63 dB, and the training curves showed why: stage 1 was still improving at epoch 50. A longer run on the
same Mac, from the same init, took one 5.8 GHz model from 7.17 to 6.53 dB at 150 + 50. With the schedule settled, the
three-seed runs in the table above went to rented RTX 4090s, with no change to the configs. The Mac runs are in
[`results/reproduce.json`](results/reproduce.json) under `mac_first_pass` and `schedule_probe`.

### Where the geometry variant's gain comes from

The geometry variant changes several things at once compared with the reproduction: the inputs, the schedule (50
epochs, not 150 + 50), the stage count and the mixlab version. So "0.93 dB and 1.54 dB better" is a recipe-to-recipe
comparison. To isolate the channels, we ran a matched pair on a held-out tenth of the training terrains: the same
50-epoch stage-1 run with 2 or with 5 input channels, otherwise identical (two seeds each, mixlab v0.125.0).

| band | 2 channels | 5 channels | gain (95% CI) |
|---|---|---|---|
| 415 MHz | 5.15 / 5.18 | 4.01 / 4.05 | 1.13 dB (1.06–1.21) |
| 5.8 GHz | 7.28 / 7.41 | 5.03 / 5.06 | 2.30 dB (2.17–2.44) |

At 415 MHz, two more channels (seven in all) add another 0.13 dB (0.11–0.15). At 5.8 GHz they made no difference in
a shorter screen (+0.006 dB, −0.007 to 0.019), so that band keeps five.

Every number here, with its source, is in [`results/reproduce.json`](results/reproduce.json) and
[`results/extend.json`](results/extend.json).

## About the trained models

We haven't published weights yet. The challenge's data terms are still being settled, and models trained on that data
will follow once they are. Meanwhile you can train your own with the commands below.

## The change, in config

The network's structure doesn't change except for its input width. The reproduction reads two channels:

```json
"input_adapter": { "kind": "grid", "channels": 2, "height": 256, "width": 256 }
```

The geometry variant reads seven:

```json
"input_adapter": { "kind": "grid", "channels": 7, "height": 256, "width": 256 }
```

Every convolution that reads the raw input also has to grow to match: the
first layer, and the two places where the U-Net concatenates the input back in.
`configs/extend_58.json` reads the same seven-channel data and keeps the first five with a `slice` op at the top of the
graph:

```json
{ "op": "slice", "inputs": ["x"], "output": "x_in", "params": { "start": 0, "end": 5, "step": 1, "axis": 3 } }
```

The new input weights start at zero (`scripts/make_init.py --channels`), so at step 0 the widened model computes
exactly what the two-channel reference does, and training decides what the new channels are worth.

The channels come from `scripts/physics_features.py`. All are computed from the heightmap and the transmitter pixel,
with nothing fitted to data:

| channel | what it is |
|---|---|
| heightmap, TX | the reference's two inputs |
| log-distance | normalized 3-D distance from the transmitter, in log space |
| LOS | 1 where the straight path from the transmitter clears the terrain |
| clearance | how far that path passes above the terrain (negative when blocked) |
| horizon margin | 415 MHz only: how far below the steepest terrain angle the receiver sits |
| blocked fraction | 415 MHz only: how much of the path runs below the terrain |

The challenge rules invite this kind of feature ("participants can make use of the input images to extract
additional features").

## Setup

You'll need the [mixlab](https://github.com/mrothroc/mixlab) binary, **v0.129.0 or newer**, on your PATH (install it
per the mixlab README). The reproduction was trained on v0.129.0; the geometry variant on v0.127.0, and its configs run
unchanged on newer versions. We tested the commands below on v0.131.0.

Data prep, the reference init and scoring use Python. Use a version PyTorch publishes wheels for (3.10–3.12 is safe;
a brand-new Python may not have them yet). We tested with Python 3.12, NumPy 2.5, PyTorch 2.14 and torchvision 0.29.
Run everything from this recipe's directory, `lunar-pathloss-radiownet/`:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

mixlab's `prepare` calls `python3` and needs NumPy, which the activated venv provides, so re-activate it in each new
shell.

Then get the organizers' code at the pinned commit. The recipe uses their data loader and scorer as they are, and their
model code to build the starting weights, so it links to them rather than copying them:

```bash
git clone https://github.com/anderspearson206/lunar-radiomap-challenge
git -C lunar-radiomap-challenge checkout fa3ef9b
```

**Data.** The dataset is on Kaggle. Join the competition from the
[challenge's dataset page](https://lunarradiomapchallenge.github.io/dataset.html), download the archives for the
bands you want, check them against the `SHA256SUMS.txt` published with them, and unpack them into `data/LunarRM/`.
The recipe expects the flat layout they unpack to: `train_hm.npy`, `train_pl415.npy`, `val_index.csv` and so on, plus
`lunar_dataset.py` and `metadata.json`.

**Disk.** One band's float32 records take about 24 GB and its mixlab shards about 20 GB; you can delete the records
once the shards are written. The geometry variant's seven-channel shards are stored as float16, about 31 GB per band.

## Reproduce it

First time? Do the [small run](#try-it-small-first) below before the full one: it checks every step in about 10
minutes. Everything the commands write (shards, runs, predictions, scores) lands in this directory.

One band, one weight seed. Set `B=58` for 5.8 GHz and `S=1` or `S=2` for the other seeds; the paths include both, so
runs don't overwrite each other.

There are two seeds in play. `seed: 42` in the configs fixes the data order and augmentation and stays as it is. The
weight seed is `S`, which picks the reference initialization. Our runs used weight seeds 0, 1 and 2.

First the settings:

```bash
B=415; S=0            # band (415 or 58) and weight seed (0, 1 or 2)
STAGE1=configs/reproduce_stage1.json
STAGE2=configs/reproduce_stage2.json
LIMIT=                # the full data; the small run below sets this
```

Then the steps, which run as-is for both the full and the small run:

```bash
# 1. the organizers' arrays -> grid records (via their own loader) -> mixlab shards
python scripts/build_grid_source.py --root data/LunarRM --out src_$B --band $B --dtype float32 $LIMIT
mixlab -mode prepare -input-format grid -input src_$B/source.json -prepare-output-dir shards_$B
rm -rf src_$B

# 2. the reference initialization: torch.manual_seed(S); RadioWNet(inputs=2), converted to mixlab's layout
#    (it runs a one-step mixlab job to read that layout, so it needs the GPU too)
python scripts/make_init.py --config $STAGE1 --dataset shards_$B/mixlab.dataset.json \
  --reference-repo lunar-radiomap-challenge --seed $S --out inits/seed${S}_2ch.safetensors

# 3. stage 1: 150 epochs. best.safetensors is the best epoch on the val split
mixlab -mode arch -config $STAGE1 -train shards_$B/mixlab.dataset.json \
  -safetensors-load inits/seed${S}_2ch.safetensors -checkpoint-dir runs/stage1_${B}_seed$S

# 4. stage 2: 50 epochs, starting from stage 1 with the first U-Net frozen
mixlab -mode arch -config $STAGE2 -train shards_$B/mixlab.dataset.json \
  -safetensors-load runs/stage1_${B}_seed$S/best.safetensors -checkpoint-dir runs/stage2_${B}_seed$S

# 5. predict the val split, convert to the challenge's submission format, score with the organizers' scorer
mixlab -mode predict-grid -config $STAGE2 -safetensors-load runs/stage2_${B}_seed$S/best.safetensors \
  -grid-in shards_$B/mixlab.dataset.json -grid-split val -grid-out pred_${B}_seed$S
python scripts/to_submission.py val-gt --data-root data/LunarRM --gt-root gt_$B --band $B    # once per band
python scripts/to_submission.py submission --pred-dir pred_${B}_seed$S --out sub_${B}_seed$S --band $B \
  --meta data/LunarRM/metadata.json
python lunar-radiomap-challenge/score_submission.py --submission sub_${B}_seed$S --gt-root gt_$B \
  --data-root data/LunarRM --band $B --out score_${B}_seed$S.json
```

On an RTX 4090, stage 1 took 6.5–10 hours and stage 2 3–4 hours per model. Scoring takes 5–10 minutes and prints
nothing until it finishes. The configs
run unchanged on Apple Silicon, just more slowly: as a check, our 415 MHz seed-0 model scored 4.673 dB through step 5
on an M1 Max, against 4.674 from the same weights on CUDA.

`val-gt` lays out the validation answers the way the scorer expects test answers, so the scorer runs unmodified. It
refuses a partial submission, so step 5 needs predictions for all 2,330 maps. Its warning about a missing
`runtime.json` is about the challenge's timing report and doesn't matter for scoring validation.

When stage 1 starts, mixlab lists 38 trainable and 38 frozen-or-unreachable weights, and the second list names the
`W*` layers. That's expected: stage 1 trains the first U-Net, and the second one waits for stage 2.

### Try it small first

To check the whole chain before committing hours, train on a small slice with short configs. On an M1 Max this takes
about 10 minutes and about 4 GB of disk, most of it spent predicting and scoring the full val split, which the
scorer needs.

Use these settings instead of the ones above, then run the steps block unchanged:

```bash
python scripts/make_smoke_configs.py                  # writes configs/*_smoke.json: 4 training steps each
B=415; S=0
STAGE1=configs/reproduce_stage1_smoke.json
STAGE2=configs/reproduce_stage2_smoke.json
LIMIT=--limit=train=32                                # a small training slice; val stays whole for the scorer
```

The score will be terrible (a 4-step model scores in the high 40s of dB), but every command runs. Before a full run,
clear the small run's outputs, since the full run writes to the same places:

```bash
rm -rf shards_$B inits runs/stage1_${B}_seed$S runs/stage2_${B}_seed$S pred_${B}_seed$S sub_${B}_seed$S \
  score_${B}_seed$S.json
```

## Extend it

The same flow, with the geometry builder and the geometry configs. The settings:

```bash
B=415; C=7; S=0       # 5.8 GHz: B=58; C=5 (its config reads the first five of the seven channels)
EXT=configs/extend_$B.json
LIMIT=
```

For a small run instead, run `python scripts/make_smoke_configs.py` if you haven't, and use
`EXT=configs/extend_${B}_smoke.json` and `LIMIT=--limit=train=32`; afterwards, clear its outputs before a full run with
`rm -rf gshards_$B inits runs/extend_${B}_seed$S epred_${B}_seed$S esub_${B}_seed$S escore_${B}_seed$S.json`. Then the
steps:

```bash
python scripts/build_geometry_source.py --root data/LunarRM --out gsrc_$B --band $B \
  --layout official --inputs geometry2 --workers 8 $LIMIT
mixlab -mode prepare -input-format grid -input gsrc_$B/source.json -prepare-output-dir gshards_$B
rm -rf gsrc_$B

python scripts/make_init.py --config $EXT --dataset gshards_$B/mixlab.dataset.json \
  --reference-repo lunar-radiomap-challenge --seed $S --channels $C --out inits/seed${S}_${C}ch.safetensors

mixlab -mode arch -config $EXT -train gshards_$B/mixlab.dataset.json \
  -safetensors-load inits/seed${S}_${C}ch.safetensors -checkpoint-dir runs/extend_${B}_seed$S

mixlab -mode predict-grid -config $EXT -safetensors-load runs/extend_${B}_seed$S/best.safetensors \
  -grid-in gshards_$B/mixlab.dataset.json -grid-split val -grid-out epred_${B}_seed$S
python scripts/to_submission.py val-gt --data-root data/LunarRM --gt-root gt_$B --band $B    # skip if gt_$B exists
python scripts/to_submission.py submission --pred-dir epred_${B}_seed$S --out esub_${B}_seed$S --band $B \
  --meta data/LunarRM/metadata.json
python lunar-radiomap-challenge/score_submission.py --submission esub_${B}_seed$S --gt-root gt_$B \
  --data-root data/LunarRM --band $B --out escore_${B}_seed$S.json
```

A 50-epoch run took 2.8–4.5 hours on an RTX 4090. A smoke-run geometry model barely moves from its starting point
(around 52 dB at 415 MHz, 77 dB at 5.8 GHz), which is expected after four steps.

The geometry features are computed on the CPU, and that's the slow part: 6–25 minutes for the val split alone with 8
workers on an M1 Max, so allow several hours for a full band. A smoke run still builds the whole val split, so this
step is most of its time. Our runs built the same training records in a
different order, so your data order, and your exact numbers, will differ slightly.

## Files

```
configs/reproduce_stage1.json     RadioWNet stage 1, 150 one-epoch cosine phases, 2 input channels
configs/reproduce_stage2.json     stage 2, 50 epochs, first U-Net frozen
configs/extend_415.json           stage 1, 50 epochs, 7 input channels
configs/extend_58.json            stage 1, 50 epochs, reads 5 of 7 channels
configs/export_map_2ch.json       mixlab <-> PyTorch tensor map for RadioWNet (from mixlab's examples)
scripts/build_grid_source.py      organizers' arrays -> grid records, via their LunarRadioMapDataset
scripts/build_geometry_source.py  the same records plus geometry channels
scripts/physics_features.py       the geometry channels
scripts/make_init.py              the reference PyTorch init -> mixlab checkpoint (optionally widened)
scripts/make_smoke_configs.py     short copies of the configs for a quick end-to-end check
scripts/to_submission.py          mixlab predictions -> challenge submission files; val answers -> scorer layout
results/reproduce.json            per-seed scores for both stages, timings, and the cross-platform checks
results/extend.json               per-seed extend scores, timings, and the matched channel comparison
```

## Known rough edges

- **The schedule is 150 one-epoch phases.** That reproduces the reference's per-epoch cosine exactly. It also means
  a finished run can't be extended by resuming: for a different length, start a fresh run.
- **The init matters.** On this data, mixlab's default initialization can leave every output pixel at zero after
  the final ReLU and never recover. The reference init doesn't, which is why the recipe converts it.
- **Precision.** mixlab allows reduced precision by default on CUDA (TF32, as our runs used) and on recent Apple
  Silicon, and its startup banner says which. `MLX_ENABLE_TF32=0` gives strict FP32; it moved our three-seed
  averages by less than 0.001 dB.
- **Comparing with PyTorch.** If you load the weights into the reference module
  (`mixlab -mode export-torch-state` writes them) to compare outputs, do it on the CPU or in float64. On an RTX 4090,
  PyTorch's own FP32 convolutions were the larger source of disagreement (see the table above).

## Where to go next

- **Try your own channels.** Add one in `physics_features.py`, widen the config and the init with `--channels`, and
  run a matched 2-vs-N comparison before trusting a gain.
- **Bigger receptive fields.** Dilated convolutions (`conv2d` with `dilation`, mixlab v0.129+) are the classic next
  step for radio maps: PMNet, which won the ICASSP 2023 pathloss challenge, is built on them.
- **The challenge itself** runs until December 2026. Details and rules are on the
  [challenge site](https://lunarradiomapchallenge.github.io/).

## Exercising and enhancing mixlab

You can reuse these pieces for other image-shaped regression problems. `prepare -input-format grid` handles image
records; the custom graph uses `conv2d`, `conv_transpose2d`, `max_pool2d`, `concat` and `slice`. Training uses
`dense_regression` with masked MSE, joint dihedral augmentation, and `freeze` for the two stages. `predict-grid` writes
predictions, and `export-torch-state` loads the weights back into PyTorch.
mixlab: https://github.com/mrothroc/mixlab

## Citing this work

Please cite what this recipe builds on:

- **The dataset and challenge**: Torrado, P.; Pearson, A.; Klein, J.; Moscibroda, A.; Smith, J. "RadioLunaDiff:
  Estimation of Wireless Network Signal Strength in Lunar Terrain." *ICASSP 2026*, pp. 21231–21235.
- **The baseline model**: Levie, R.; Yapar, Ç.; Kutyniok, G.; Caire, G. "RadioUNet: Fast Radio Map Estimation with
  Convolutional Neural Networks." *IEEE Transactions on Wireless Communications* 2021, 20 (6), 4001–4015.
  [doi:10.1109/TWC.2021.3054977](https://doi.org/10.1109/TWC.2021.3054977)
- **The ray tracer behind the ground truth**: Hoydis, J. et al. "Sionna." 2022.

If you want to point at this specific recipe or the trainer:

```bibtex
@software{lunar_pathloss_mixlab,
  author = {Rothrock, Michael},
  title  = {Lunar radio maps in mixlab: reproduce the challenge baseline, then add terrain geometry},
  year   = {2026},
  url    = {https://github.com/mrothroc/mixlab-cookbook/tree/main/lunar-pathloss-radiownet}
}
@software{mixlab,
  author = {Rothrock, Michael},
  title  = {mixlab: a compact ML architecture lab},
  url    = {https://github.com/mrothroc/mixlab}
}
```

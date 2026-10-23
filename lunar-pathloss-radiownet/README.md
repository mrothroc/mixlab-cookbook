# Lunar radio maps in mixlab: reproduce the challenge baseline, then add terrain geometry

The [First Lunar Pathloss Radio Map Prediction Challenge](https://lunarradiomapchallenge.github.io/) (an IEEE
ICASSP 2027 Grand Challenge) asks for a model that predicts how much a radio signal weakens across a 256 × 256 patch
of simulated lunar terrain, given the heightmap and the transmitter's position. There are two frequency bands,
415 MHz and 5.8 GHz, and the score is RMSE in dB over the pixels the ray tracer covered.

The organizers' baseline is RadioWNet, the two-stage U-Net from
[RadioUNet](https://github.com/RonLevie/RadioUNet), trained once per band. Their
[repository](https://github.com/anderspearson206/lunar-radiomap-challenge) at commit `fa3ef9b` has the model code,
data loader and scorer, and the challenge site reports **4.66 dB** (415 MHz) and **6.28 dB** (5.8 GHz) on the
test set.

This recipe rebuilds that model in [mixlab](https://github.com/mrothroc/mixlab) from a JSON config and lands at
**4.88 dB** and **6.78 dB** on the official validation split, averaged over three seeds. Then it gives the same U-Net
a few extra input channels computed from the terrain (distance, a line-of-sight estimate, clearance). That brings it
to **3.95 dB** and **5.24 dB**, with a shorter training run.

## The two-step arc

1. **Reproduce.** Express RadioWNet in a mixlab config, start from the reference's own initialization, and train
   with the reference recipe: Adam at 1e-4 with a cosine schedule, batch 16, flips and rotations, then a second
   stage that refines the first.
2. **Extend.** Keep the network and the optimizer. Add geometry channels to the input and widen the first
   layers to read them. Train stage 1 only, for 50 epochs.

## Results

Official validation split, 2,330 maps per band, scored with the organizers' unmodified `score_submission.py`. Three
weight seeds per model; `±` is the sample standard deviation across seeds.

| model | 415 MHz | 5.8 GHz |
|---|---|---|
| **extend:** stage 1 + geometry channels, 50 epochs | **3.95 ± 0.01** | **5.24 ± 0.01** |
| **reproduce:** RadioWNet, 150 + 50 epochs | 4.88 ± 0.26 | 6.78 ± 0.34 |
| the organizers' baseline, as published (test set) | 4.66 | 6.28 |

### How close is the reproduction?

We fixed what would count as a match before training: the three-seed average had to come in within 0.5 dB of the
published number, in each band. The 415 MHz model lands 0.22 dB above it. The 5.8 GHz model lands 0.497 dB above it,
which is only just inside the limit, so read that band as a near miss as much as a match. Two more caveats:

- **Different splits.** The published numbers are on the test set, whose answers are still withheld. Ours are on
  the validation split. We also picked each run's best epoch on that same split, which flatters our numbers
  slightly.
- **Seed spread.** In both bands one of the three seeds lands above the limit on its own, and the spread across
  seeds is wide (0.26 and 0.34 dB). Most of it comes from how long stage 1 sits on a plateau before it starts
  learning, which varies from seed to seed.

The published recipe doesn't say how many epochs produced its numbers. The repo defaults to 50 + 50, but stage 1 is
still improving at epoch 50. Trained on the same machine from the same init, one 5.8 GHz model went from 7.17 dB at
50 + 50 to 6.53 dB at 150 + 50. 150 + 50 is what this recipe uses.

### Where the extend's gain comes from

The extend changes several things at once compared with the reproduction: the inputs, the schedule (50 epochs, not
150 + 50), the stage count and the mixlab version. So "0.93 dB and 1.54 dB better" is a recipe-to-recipe comparison.
To isolate the channels, we ran a matched pair on a held-out tenth of the training terrains: the same 50-epoch
stage-1 run, with 2 or with 5 input channels, otherwise identical.

| band | 2 channels | 5 channels | gain (95% CI) |
|---|---|---|---|
| 415 MHz | 5.15 / 5.18 | 4.01 / 4.05 | 1.13 dB (1.06–1.21) |
| 5.8 GHz | 7.28 / 7.41 | 5.03 / 5.06 | 2.30 dB (2.17–2.44) |

At 415 MHz, two more channels (seven in all) add another 0.13 dB (0.11–0.15). At 5.8 GHz they add nothing, so that
band keeps five.

Numbers: [`results/reproduce.json`](results/reproduce.json), [`results/extend.json`](results/extend.json).

## About the trained models

We haven't published weights. The challenge's data terms are not settled yet, and a model trained on that data
waits until they are. You can train your own with the commands below.

## The change, in config

Nothing about the network's structure changes except its input width. The reproduction reads two channels:

```json
"input_adapter": { "kind": "grid", "channels": 2, "height": 256, "width": 256 }
```

The extend reads seven:

```json
"input_adapter": { "kind": "grid", "channels": 7, "height": 256, "width": 256 }
```

That is not quite a one-line change, because every convolution that reads the raw input has to grow to match: the
first layer, and the places where each U-Net concatenates the input back in. `configs/extend_415.json` has those six
kernels widened from 2 to 7 input channels. `configs/extend_58.json` reads the same seven-channel data and keeps only
the first five with a `slice` op at the top of the graph:

```json
{ "op": "slice", "inputs": ["x"], "output": "x_in", "params": { "start": 0, "end": 5, "step": 1, "axis": 3 } }
```

The new input weights start at zero (`scripts/make_init.py --channels`), so at step 0 the widened model computes
exactly what the two-channel reference does, and training decides what the new channels are worth.

The channels themselves come from `scripts/physics_features.py`. All are computed from the heightmap and the
transmitter pixel, with nothing fitted to data:

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

You will need the [mixlab](https://github.com/mrothroc/mixlab) binary, **v0.129.0 or newer**, on your PATH (install
it per the mixlab README). The results above were trained on v0.129.0 (reproduce) and v0.127.0 (extend). Data prep,
the reference init and scoring use Python:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PATH="$PWD/.venv/bin:$PATH"    # so mixlab's prepare finds python3 + numpy
```

Then get the organizers' code at the pinned commit. The recipe uses their data loader, model code and scorer as they
are, so it links to them rather than copying them:

```bash
git clone https://github.com/anderspearson206/lunar-radiomap-challenge
git -C lunar-radiomap-challenge checkout fa3ef9b
```

**Data.** The dataset is on Kaggle. Join the competition from the
[challenge's dataset page](https://lunarradiomapchallenge.github.io/dataset.html), download the archives for the
bands you want into `data/LunarRM/`, and check them against the published `SHA256SUMS.txt`. The recipe expects the
flat layout the archives unpack to (`train_hm.npy`, `train_pl415.npy`, `val_index.csv`, … plus `lunar_dataset.py`
and `metadata.json`).

**Disk.** Building one band's float32 records takes about 23 GB, and its mixlab shards about 21 GB. You can delete
the records once the shards are written. The extend's seven-channel shards are stored as float16, about 29 GB per band.

## Reproduce it

One band, one seed. Repeat with `--seed 1` and `2`, and with `B=58` for 5.8 GHz.

```bash
B=415    # or 58

# 1. the organizers' arrays -> grid records (via their own loader) -> mixlab shards
python scripts/build_grid_source.py --root data/LunarRM --out src_$B --band $B --dtype float32
mixlab -mode prepare -input-format grid -input src_$B/source.json -prepare-output-dir shards_$B
rm -rf src_$B

# 2. the reference initialization: torch.manual_seed(0); RadioWNet(inputs=2), converted to mixlab's layout
python scripts/make_init.py --config configs/reproduce_stage1.json --dataset shards_$B/mixlab.dataset.json \
  --reference-repo lunar-radiomap-challenge --seed 0 --out inits/seed0_2ch.safetensors

# 3. stage 1: 150 epochs. best.safetensors is the best epoch on the val split
mixlab -mode arch -config configs/reproduce_stage1.json -train shards_$B/mixlab.dataset.json \
  -safetensors-load inits/seed0_2ch.safetensors -checkpoint-dir runs/stage1_${B}_seed0

# 4. stage 2: 50 epochs, starting from stage 1 with the first U-Net frozen
mixlab -mode arch -config configs/reproduce_stage2.json -train shards_$B/mixlab.dataset.json \
  -safetensors-load runs/stage1_${B}_seed0/best.safetensors -checkpoint-dir runs/stage2_${B}_seed0

# 5. predict the val split, convert to the challenge's submission format, score with the organizers' scorer
mixlab -mode predict-grid -config configs/reproduce_stage2.json -safetensors-load runs/stage2_${B}_seed0/best.safetensors \
  -grid-in shards_$B/mixlab.dataset.json -grid-split val -grid-out pred_$B
python scripts/to_submission.py val-gt --data-root data/LunarRM --gt-root gt_$B --band $B
python scripts/to_submission.py submission --pred-dir pred_$B --out sub_$B --band $B --meta data/LunarRM/metadata.json
python lunar-radiomap-challenge/score_submission.py --submission sub_$B --gt-root gt_$B \
  --data-root data/LunarRM --band $B --out score_$B.json
```

On an RTX 4090, stage 1 took 6.7–10.3 hours and stage 2 3.2–4.0 hours per model. Scoring takes a few minutes.
The configs run unchanged on Apple Silicon, just more slowly. As a check, we pushed our 415 MHz seed-0 model through
step 5 on an M1 Max and got 4.673 dB, against 4.674 from the same weights on CUDA.

`val-gt` lays out the validation answers the way the scorer expects test answers, so the scorer runs unmodified. It
refuses a partial submission, so step 5 needs predictions for all 2,330 maps.

**The two seeds are different things.** `seed` in the configs (42) fixes the data order and augmentation. The weight
seed is the one you pass to `make_init.py`. Our runs used weight seeds 0, 1 and 2.

### Try it small first

To check the whole chain in a few minutes before committing hours, build a slice with `--limit train=32` (the val
split stays whole so the scorer accepts it), and train a copy of each config with `phases` cut to two short phases:

```python
import json
for name in ("reproduce_stage1", "reproduce_stage2"):
    c = json.load(open(f"configs/{name}.json"))
    c["name"] += "_smoke"
    c["training"]["phases"] = [dict(p, steps=2) for p in c["training"]["phases"][:2]]
    c["training"]["val_every_steps"] = 2
    json.dump(c, open(f"configs/{name}_smoke.json", "w"), indent=1)
```

The score will be terrible (a 4-step model scores in the high 40s), but every command runs.

## Extend it

The same flow, with the geometry builder and the extend configs:

```bash
B=415; C=7    # 5.8 GHz: B=58; C=5 (its config reads the first five of the seven channels)

python scripts/build_geometry_source.py --root data/LunarRM --out gsrc_$B --band $B \
  --layout official --inputs geometry2 --workers 8
mixlab -mode prepare -input-format grid -input gsrc_$B/source.json -prepare-output-dir gshards_$B
rm -rf gsrc_$B

python scripts/make_init.py --config configs/extend_$B.json --dataset gshards_$B/mixlab.dataset.json \
  --reference-repo lunar-radiomap-challenge --seed 0 --channels $C --out inits/seed0_${C}ch.safetensors

mixlab -mode arch -config configs/extend_$B.json -train gshards_$B/mixlab.dataset.json \
  -safetensors-load inits/seed0_${C}ch.safetensors -checkpoint-dir runs/extend_${B}_seed0
```

Then predict, convert and score exactly as in step 5, with `configs/extend_$B.json`, the `gshards_$B` dataset and
the extend run's `best.safetensors`. A 50-epoch run took 2.8–4.5 hours on an RTX 4090.

The geometry features take a while to compute on CPU; `--workers` spreads them across cores. Our runs built the
same training records in a different order, so your data order (and so your exact numbers) will differ slightly.

## Files

```
configs/reproduce_stage1.json   RadioWNet stage 1, 150 one-epoch cosine phases, 2 input channels
configs/reproduce_stage2.json   stage 2, 50 epochs, first U-Net frozen
configs/extend_415.json         stage 1, 50 epochs, 7 input channels
configs/extend_58.json          stage 1, 50 epochs, reads 5 of 7 channels
configs/export_map_2ch.json     mixlab <-> PyTorch tensor map for RadioWNet (from mixlab's examples)
scripts/build_grid_source.py    organizers' arrays -> grid records, via their LunarRadioMapDataset
scripts/build_geometry_source.py  the same records plus geometry channels
scripts/physics_features.py     the geometry channels
scripts/make_init.py            the reference PyTorch init -> mixlab checkpoint (optionally widened)
scripts/to_submission.py        mixlab predictions -> challenge submission files; val answers -> scorer layout
results/reproduce.json          per-seed scores for both stages, both bands
results/extend.json             per-seed extend scores and the matched channel comparison
```

## Known rough edges

- **The schedule is 150 one-epoch phases.** That reproduces the reference's per-epoch cosine exactly. It also means
  a finished run can't be extended by resuming: for a different length, start a fresh run.
- **The init matters.** mixlab's default initialization can leave every output pixel at zero after the final ReLU
  and never recover on this data. The reference init escapes in about 40 steps, which is why the recipe converts it.
- **Precision on CUDA.** mixlab uses TF32 for convolutions on CUDA by default, as our runs did. Strict FP32
  (`MLX_ENABLE_TF32=0`) moves the three-seed averages by less than 0.001 dB. Recent mixlab on Apple Silicon may also
  use reduced precision by default; the startup banner says so, and `MLX_ENABLE_TF32=0` turns it off.
- **Checking against PyTorch on a GPU.** If you load the weights into the reference PyTorch module to compare
  outputs (`mixlab -mode export-torch-state` writes them), compare on the CPU or in float64. On an RTX 4090, PyTorch's
  own strict-FP32 output differed from float64 by up to 0.09 dB at single pixels on one of these models, while
  mixlab's differed by 0.005 dB. A GPU-to-GPU comparison mostly measures PyTorch's convolution choice.
- **Validation, not test.** Test answers arrive with the challenge's evaluation release. Until then every number
  here is on the validation split.

## Where to go next

- **Try your own channels.** `physics_features.py` is the place to add one; widen the config and the init with
  `--channels`, and run the matched 2-vs-N comparison before trusting a gain.
- **Bigger receptive fields.** Dilated convolutions (`conv2d` with `dilation`, mixlab v0.129+) are the classic next
  step for radio maps: PMNet, which won the ICASSP 2023 pathloss challenge, is built on them.
- **The challenge itself** runs until December 2026. Details and rules are on the
  [challenge site](https://lunarradiomapchallenge.github.io/).

## Exercising and enhancing mixlab

This recipe leans on mixlab's dense-grid support: `prepare -input-format grid` for image-shaped records, a custom
graph of `conv2d`, `conv_transpose2d`, `max_pool2d`, `concat` and `slice` ops, the `dense_regression` objective with
masked MSE, joint dihedral augmentation, two-stage training with `freeze`, `predict-grid`, and
`export-torch-state` for loading the weights back into PyTorch. Building it drove several mixlab releases, from
the grid adapter and U-Net ops through explicit TF32 control, CUDA memory recovery and the precision banner.
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

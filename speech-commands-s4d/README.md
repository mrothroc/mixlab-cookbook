# S4D in mixlab: reproduce a raw-waveform audio model

[S4D](https://arxiv.org/abs/2206.11893) (Gu et al., *On the Parameterization and Initialization of Diagonal State Space Models*, NeurIPS 2022) is a diagonal state-space model that classifies spoken keywords **directly from the 16 kHz waveform**, with no spectrogram or MFCC front end. This recipe rebuilds its S4D-Lin variant in [mixlab](https://github.com/mrothroc/mixlab) and trains it on [Speech Commands](https://arxiv.org/abs/1804.03209) on a single Apple M4, landing at 96.14% against the 96.25% the paper reports in Table 11. A 1 second clip is a 16,000 step sequence, and the model that handles it has 306K parameters; for comparison, a 26M-parameter ConvNet scores lower on the same task.

The architecture is defined with a JSON config, so swapping the sequence mixer is an edit rather than a rewrite. In addition to the reproduction, this entry runs the experiment of trading S4D for Mamba-3 at a matched budget. The same file runs on an Apple Silicon laptop and on rented CUDA with no changes: we used these configs unmodified on an M4 Max and on RTX 4090 and L4 GPUs.

Use the recipe as-is, or as a starting point for your own experiments.

## The reproduction

Official `testing_list.txt` split, 11,005 utterances:

| | test accuracy | params |
|---|---|---|
| **this recipe** | **96.14%** | 306,083 |
| S4D-Lin, published ([Table 11](https://arxiv.org/abs/2206.11893)) | 96.25% (+/-0.03) | "306K" |

Our single-seed result is 0.11 points below the published mean and outside the reported +/-0.03 interval. That interval is a multi-seed spread and this is one seed, so the comparison is indicative rather than a like-for-like miss. Call it a single-seed best-effort reconstruction: close enough to show the architecture is faithfully rebuilt, not a claim to have matched 96.25 +/- 0.03. `results/s4d_anchor_test.json` records the claim in those terms.

The paper does not specify two details used here: `n_ssm: 2` parameter sharing, and the checkpoint selection rule. We report the **best devel checkpoint** (epoch 36 of 40, step 190,872), scored once on test, which is the selection rule the reference uses.

`results/s4d_anchor_test.json` records the test result, the two assumptions, and the comparison with the published interval. `results/anchor_test_audit.json` carries the audit trail: the checkpoint's sha256 (byte-identical to the published weights), the verbatim eval output, per-class accuracy, and the devel curve for epochs 33 to 40 showing epoch 36 is the peak. `results/anchor_test_predictions.tsv` has all 11,005 predictions, including the 10,580 correct ones, so the headline number can be recounted from source.

Epoch 36 beats epochs 39 and 40 by **0.0001 devel**, about one utterance in 9,981. The selection was made without looking at test, but it did not find a meaningfully better checkpoint.

## Try it

The weights from the 40 epoch run are published ready to try, at the epoch 36 checkpoint the table above reports:
[mrothroc/sc35-s4d-lin-mixlab](https://huggingface.co/mrothroc/sc35-s4d-lin-mixlab). It ships a standalone PyTorch loader, the label map, and a script that classifies real clips and checks the predictions against their folder names. CPU is fine.

## Install

mixlab is a standalone binary, not a Python package. It installs through Homebrew, which you need already. The formula pulls in MLX, so this path wants **Apple Silicon and macOS 14 (Sonoma) or newer**. MLX itself arrives as a bottle on those versions, but mixlab has no bottle and always compiles, so expect Homebrew to pull Go and build it, and expect to need working Apple command line tools. On Linux or an NVIDIA box, skip this and use the container image described under **Same file on a laptop, then on CUDA**.

```bash
brew install mrothroc/tap/mixlab
```

The scripts here were last exercised against **0.111.0**. The results predate it: the 96.14% anchor was produced on **0.106.2**, and the mamba3 `state_lr` experiment in `results/mamba3_state_lr_smoke.json` needs **0.110.0 or newer** for that field to be accepted on a `mamba3-canonical` block. None of the shipped configs set `state_lr` on mamba3, so 0.106.2 is enough to run everything in `configs/`.

Regardless, you should probably use the latest version. There were bugs fixed along the way.

You also need **Python 3.10 or newer, with numpy**, on the `python3` that is on your **PATH**. `mixlab -mode prepare` shells out to it, so this is needed for step 2 as well as step 1, and a venv only works if it is activated.

3.10 is a hard floor, and a stock Mac does not meet it: `/usr/bin/python3` is 3.9, and mixlab's own preparation script uses `X | None` type syntax that 3.9 evaluates at import time. Step 1 succeeds on 3.9 and then step 2 dies with `TypeError: unsupported operand type(s) for |`.

On macOS `python3` is usually that `/usr/bin/python3`, while `pip` belongs to a different interpreter, so `pip install numpy` can report success and change nothing that `python3` sees. Check the version and the import together:

```bash
python3 -c "import sys, numpy; print(sys.version); print(sys.executable, numpy.__version__)"
```

If that prints 3.9, or fails on numpy, install a newer Python and work inside a venv. Homebrew keeps versioned formulae off the unversioned name, so `brew install python@3.12` alone will not change what `python3` resolves to; the venv is what does that:

```bash
brew install python@3.12
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Re-run the check above inside the activated venv. It must report 3.10 or newer and import numpy. Every step below assumes that venv is active, `mixlab -mode prepare` included, since it shells out to whatever `python3` it finds.

## Training your own model

Steps 1 and 2 are shared by every run; then choose a config:

| goal | config | result |
|---|---|---|
| a shorter first run | `configs/s4d_10ep.json` | 95.06% devel, ~11 h on one GPU |
| **the 96.14% test headline** | **`configs/s4d_anchor_40ep.json`** | 96.14% test, ~32 h |
| the mixer comparison | `configs/s4d_6ep.json` and `configs/mamba3_6ep_lr1e3.json` | 94.98% vs 90.74% devel |

Run every command from the entry directory. The commands below use the 10 epoch config; swap the config path for any of the others.

### Smoke test first

Before downloading anything, run all five steps on synthetic audio. It takes about a minute:

```bash
python3 scripts/make_smoke_fixture.py smoke_data   # 210 clips of noise, 70 per split
python3 scripts/prep_sc35.py smoke_data smoke_prep
SC35_SKIP_COUNT_CHECK=1 ./scripts/prepare_shards.sh smoke_prep smoke_shards
./scripts/train.sh configs/smoke.json smoke_shards runs/smoke
./scripts/eval_checkpoints.sh configs/smoke.json runs/smoke smoke_shards val
mixlab -mode eval -config configs/smoke.json \
  -safetensors-load runs/smoke/ckpt/step_000034.st \
  -val 'smoke_shards/test/train_*.bin' -val-batches 0
```

The audio is noise, so accuracy stays at chance and the printed statistics will not match the real-data values below. What it proves is the plumbing: prep runs, the shard counts come out right, checkpoints land on epoch boundaries, and eval reads the full split rather than a sample. `configs/smoke.json` is sized to the fixture at 34 steps, two epochs over 70 training records.

`SC35_SKIP_COUNT_CHECK=1` bypasses the official-count gate, which synthetic data cannot satisfy. Never set it for a real run: that check is what catches a `-val-split` holdout quietly shrinking your training set.

### The real dataset

Download the raw Speech Commands v0.02 archive (2.3 GB). Use the tarball, not the TFDS-prepared dataset: the TFDS catalog page serves a 12 class variant with different splits, which is a different task from the 35 way one here.

```bash
curl -O http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
mkdir -p speech_commands_v0.02 && tar xzf speech_commands_v0.02.tar.gz -C speech_commands_v0.02
```

That gives you the 35 class folders plus `validation_list.txt` and `testing_list.txt`, which is the layout `prep_sc35.py` expects. (There is a separate `speech_commands_test_set_v0.02.tar.gz`; you do not need it; the official test split is defined by `testing_list.txt` inside the main archive.)

**Budget about 20 GB of disk** before checkpoints: 2.4 GB archive, ~3.4 GB of extracted wavs (PCM16 audio decompresses to more than the archive), 6.8 GB of normalized arrays, ~6.8 GB of shards. Each checkpoint adds ~3.7 MB: 1.2 MB of weights plus optimizer state and a resume manifest. You can delete the archive and the arrays once the shards exist. The CUDA container in the last section needs its own space on top of this.

```bash
# 1. wavs -> normalized [N,16000,1] arrays. Prints stats you must check.
python3 scripts/prep_sc35.py speech_commands_v0.02 prep

# 2. arrays -> mixlab shards. Fails loudly if the split counts are wrong.
./scripts/prepare_shards.sh prep shards

# 3. train. 10 epochs, 10h54m measured on an RTX 4090.
./scripts/train.sh configs/s4d_10ep.json shards runs/s4d_10ep

# 4. score every checkpoint on the full devel split
./scripts/eval_checkpoints.sh configs/s4d_10ep.json runs/s4d_10ep shards val

# 5. pick the best checkpoint from step 4, then score THAT ONE on test.
#    Do not loop over checkpoints on test: that is selecting on the test set.
#    Copy the filename step 4 printed. mixlab zero-pads the step to six
#    digits, so short runs look like step_000017.st and long ones do not pad.
BEST=runs/s4d_10ep/ckpt/step_212100.st
mixlab -mode eval -config configs/s4d_10ep.json \
  -safetensors-load "$BEST" \
  -val 'shards/test/train_*.bin' -val-batches 0
```

### Checking a real prep run

After step 1 on the real dataset, check the printed statistics against these:

```
train 84,843   val 9,981   test 11,005
TRAIN mean=-2.791959 std=2818.166625
```

If they differ, something is wrong with the data; fix it before starting a long run. (A smoke fixture will differ, by design.)

Re-running step 3 with the same run directory **resumes** from the newest checkpoint. That is the recovery path for an interrupted run, and it matters for the mamba3 arm; see **Known rough edges**. It only applies to a run that has not finished: once the newest checkpoint is at `training.steps`, mixlab exits with `checkpoint is at step N, at or after configured training steps N` rather than repeating work.

## Config files

| config | what it is | cost |
|---|---|---|
| `s4d_10ep.json` | the shorter run, 95.06% devel | 10h54m measured, RTX 4090 |
| `s4d_anchor_40ep.json` | the 40 epoch reconstruction, 96.14% test | 32h01m of compute on an M4 Max (a reboot cost ~9.5 h more) |
| `s4d_6ep.json` | the s4d side of the mixer comparison, 94.98% devel | CUDA; wall clock not recorded |
| `mamba3_6ep_lr1e3.json` | the mamba3 side, 90.74% devel | ~34 h *projected* from throughput; the run hung and was resumed three times |
| `smoke.json` | the synthetic-fixture smoke test, no result | ~20 s |

**The GPUs differ between rows, so the times are not comparable to each other**, and only the first was measured end to end on one machine. Use them for budgeting only.

## Same file on a laptop, then on CUDA

**No backend-specific config edits are needed**, and we relied on that: we developed on the Mac and ran the long jobs on rented CUDA with the same configs and the same CLI.

| where | what ran there | why                                          |
|---|---|----------------------------------------------|
| M4 Max, Apple Silicon | the 40 epoch reproduction, 32h01m | iteration is cheap on hardware we already own |
| RTX 4090 / L4, rented CUDA | the 10 epoch run and both mixer arms | the mixer comparison is ~2 arms x 20-34 h    |

mixlab selects the native Metal S4D kernel on the Mac and the CUDA path on NVIDIA.

On the CUDA side we used mixlab's container image rather than a local build. The scripts here call `$MIXLAB` as a single executable, so a bare `MIXLAB="docker run ..."` will not work. Either run the scripts inside the container, or put a wrapper on your PATH:

```bash
sudo tee /usr/local/bin/mixlab >/dev/null <<'SH'
#!/bin/sh
exec docker run --gpus all --rm \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/w" -w /w michaelrothrock/mixlab "$@"
SH
sudo chmod +x /usr/local/bin/mixlab
```

Needs the NVIDIA container toolkit installed for `--gpus all`. The `--user` flag keeps `shards/`, `runs/` and `train.log` owned by you instead of root.

Under the wrapper, `mixlab -mode prepare` runs inside the container, so it shells out to the *container's* `python3`, not yours. **The image has `python3` but no numpy**, so step 2 fails there as shipped. Two ways around it: add numpy to the image (`pip3 install numpy` on top of it, rebuilt once), or run steps 1 and 2 on a machine that has numpy and copy `shards/` up, which is what we did. Step 1 (`prep_sc35.py`) always runs on your host and always needs host numpy.

Note the container has to include the CUDA tools, so it is massive. Allow time to pull the image. Also, the provided container supports many but not all chips, so verify it supports your target. (A base CUDA image is provided so you can add your architecture and rebuild if needed.)

The command above pulls `:latest`, which moves. Record the digest you actually ran (`docker image inspect --format '{{index .RepoDigests 0}}' michaelrothrock/mixlab`) if you need the run to be reproducible later; the image labels do not carry the mixlab version. Check the [mixlab repo](https://github.com/mrothroc/mixlab) for the current image name and tags.

## Swapping the mixer

`configs/s4d_6ep.json` and `configs/mamba3_6ep_lr1e3.json` are identical in every training field except the learning rate: same 127,260 steps, same schedule, same `batch_tokens`, same seed. The edit is one block definition (repeated for all six layers) plus one learning rate:

```diff
-  { "type": "s4d", "state_size": 64, "n_ssm": 2, "bidirectional": true,
-    "output_transform": "glu", "discretization": "bilinear",
-    "measure": "diag-lin", "state_lr": 0.001, "trainable_b": true }
+  { "type": "mamba3-canonical", "state_size": 64, "bidirectional": true,
+    "inner_dim": 48, "n_groups": 4, "scan_chunk_size": 64 }
...
-  "lr": 0.01
+  "lr": 0.001
```

At that matched 6 epoch budget, on the full 9,981 record devel split:

| | s4d | mamba3-canonical |
|---|---|---|
| final devel accuracy | **94.98%** | **90.74%** |
| parameters | 306,083 | 320,579 (+4.7%) |
| learning rate | 0.01 (the reference recipe) | 0.001 (screened for stability, not tuned) |

**With 4.7% more parameters, mamba3 scored 4.24 points lower.**

s4d uses the authors' learning rate, tuned on this task. The mamba3 rate is simply one that trains: we screened 3e-4, 1e-3 and 3e-3 for stability and did not tune to convergence, so 90.74 is not necessarily mamba3's ceiling. Each arm ran with a single seed. A time invariant model may simply suit raw audio better, but this comparison is too loose to show it.

**Note that mamba3 will not train at s4d's `lr` 0.01.** It goes non-finite at steps 263 through 1011, five times out of six during the 1,000 step warmup. That is not an artifact of s4d getting a separate `state_lr` for its SSM parameters: we gave mamba3 the matching treatment on `A_log` and `dt_bias` and it still diverged. 3e-4, 1e-3 and 3e-3 all survived **3,000 steps**, which covers the window where every 0.01 divergence
happened, so 1e-3 is one workable choice rather than a required one. That is a short-run stability screen, not evidence any of them is stable over a full run (`results/mamba3_lr_screen.json`, and the matched-`state_lr` test in `results/mamba3_state_lr_smoke.json`). If your own mamba3 run diverges early, lower the global learning rate first.

## Learning curves

Per epoch on the full devel split. The two 6 epoch arms share a schedule and are directly comparable to each other. **Do not compare either of them epoch-by-epoch against the 10 epoch run**: a cosine schedule spends most of its gain in the final epochs, so the longer run reads *lower* at epoch 5 than a 6 epoch run that is already annealing.

s4d, 6 epoch schedule (`results/s4d_6ep_devel_curve.json`):

| epoch | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| devel | 75.33 | 82.31 | 86.95 | 90.71 | 93.72 | **94.98** |

mamba3, 6 epoch schedule (`results/mamba3_6ep_tuned_devel_curve.json`):

| epoch | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| devel | 50.48 | 75.55 | 81.18 | 86.95 | 89.86 | **90.74** |

s4d, 10 epoch schedule (`results/s4d_10ep_devel_curve.json`):

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| devel | 74.16 | 80.60 | 82.54 | 81.90 | 86.13 | 88.34 | 90.99 | 92.38 | 94.42 | **95.06** |

## Data preparation pitfalls

**`-val-split` defaults to 0.1 and omitting it does not turn it off.** Hand `prepare` a split and it quietly holds back 10% of it, so you train on less data than you think and "the full devel split" becomes unreachable. `prepare_shards.sh` passes `-val-split 0` and then verifies the counts, failing if anything was held back.

**Shuffle before you shard.** `mixlab -mode prepare` preserves input record order. Feed it class ordered data and every shard is nearly single label: batches carry about 1 distinct class instead of 13. We ran the same config and seed both ways: 3.52% devel at epoch 1 class-ordered against 92.50% shuffled, and the class-ordered arm was still at 4.26% by epoch 18, barely above the 2.86% chance floor. Numbers in `results/shard_order_ablation.json`. `prep_sc35.py` permutes all three splits before writing, so following the steps above avoids this.

```
class-ordered   distinct classes per 16-record batch: mean  1.01
permuted        distinct classes per 16-record batch: mean 12.99
```

**The `acc=` in the training log is not full-split accuracy.** It is a small sampled metric. Over class ordered data it reads near zero, indistinguishable from a collapsed model. Score checkpoints with `-mode eval -val-batches 0` on the full split.

**Accumulate statistics in float64.** Summing 1.36e9 float32 values moves the computed training mean from -2.791959 to -2.791973. The shift is far too small to affect training, but it keeps the printed statistics from matching the reference values above.

## What it took in mixlab

The work went into reference details invisible in the paper's config: `n_ssm: 2` parameter sharing, a separate `state_lr` for the SSM parameters, and GELU before GLU inside the block. We found those by reading the reference implementation, not its YAML. Matching the parameter count to 0.2% did not mean the mapping was complete.

A separate PyTorch port of the block matched mixlab's forward outputs on 35 inputs to 6.251e-05 max absolute logit difference with 35/35 argmax agreement, measured on a one step checkpoint. That shows the layer maths and weight layout line up. The port ships with the model as `modeling_s4d.py`, but the comparison was run against a one step checkpoint that is not published, so the figure is a recorded result rather than something you can rerun from these files. The [model card](https://huggingface.co/mrothroc/sc35-s4d-lin-mixlab) documents separate checks on the trained weights, including a label-mapping check against real audio.

## Known rough edges

**Long `mamba3-canonical` runs occasionally hang.** mixlab has telemetry options for catching it, but the cause is still open. Training stops with no error, the process stays alive, the GPU goes idle. The 6 epoch mamba3 run hung three times before finishing, at different step counts and on different hardware; we resumed from a checkpoint each time. The s4d arms have never done it. The recovery is to stop the hung process first, confirm it has exited, then rerun the same `train.sh` command to resume from the latest checkpoint, which it writes every epoch. Do not start the rerun alongside the hung one: nothing locks the run directory, so two trainers would write the same checkpoints and log. Reported to mixlab; the behavior was present through 0.111.0.

**mixlab writes no end-of-run checkpoint.** Checkpoints appear only on `-checkpoint-every` boundaries, so that interval must divide `steps` exactly or the final epoch is never saved and its accuracy cannot be evaluated. `train.sh` computes one epoch as `floor(train_records / (batch_tokens / seq_len))`, reading `train_records` from the prepared shards rather than assuming the full dataset, so it is also correct for the smoke fixture. That divides every config here exactly, and it warns if you point it at one where it does not.

**Config provenance.** mixlab records a `config_hash` in every resume bundle, so a config can be matched to the run that used it. `configs/s4d_6ep.json` reproduces the hash its run recorded; the value is in `results/s4d_6ep_devel_curve.json` if you want to check.

## Citing this work

This rebuilds someone else's architecture, so cite what it builds on:

- S4D: Gu, A.; Gupta, A.; Goel, K.; Re, C. "On the Parameterization and Initialization of Diagonal
  State Space Models." *Advances in Neural Information Processing Systems* 2022. arXiv:2206.11893
- Speech Commands: Warden, P. "Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition."
  2018. arXiv:1804.03209
- Mamba-3, for the comparison arm: Lahoti, M. et al. "Mamba-3." *ICLR* 2026. arXiv:2603.15569

Optionally, the recipe itself:

```bibtex
@software{speech_commands_s4d_mixlab,
  title  = {Speech Commands raw waveform with S4D, reproduced in mixlab},
  author = {Rothrock, Michael},
  year   = {2026},
  url    = {https://github.com/mrothroc/mixlab-cookbook/tree/main/speech-commands-s4d}
}
```

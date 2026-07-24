# MolGPT in mixlab: reproduce it on your Mac, then tinker

[MolGPT](https://github.com/devalab/molgpt) is a small SMILES transformer that generates drug-like molecules. This recipe rebuilds it in [mixlab](https://github.com/mrothroc/mixlab) so you can train, sample, and experiment with a molecular generator on a single Apple-silicon Mac (about 5 hours on an M1 Max), no GPU cluster needed. If you want to poke at how a small generative chemistry model behaves, this is a friendly place to start.

There are two models here: a faithful reproduction of MolGPT, and a variant that comes out better once you change how the molecules are represented.

Prefer the story first? There's a short narrative writeup of this recipe: [You Don't Code the Model, You Edit the JSON](https://michael.roth.rocks/blog/edit-the-json/).

## Verification against the benchmark

Same architecture as [MolGPT](https://github.com/devalab/molgpt) (6,342,656 params: 8 layers, 256-dim, 8 heads), trained one molecule per sequence on MOSES and scored on 30k samples with the official `moses.metrics`:

| Metric | mixlab | MolGPT (published) |
|---|---|---|
| Valid | 0.988 | 0.994 |
| Unique@10k | 0.991 | ~1.000 |
| Novelty | 0.783 | 0.797 |
| IntDiv | 0.857 | 0.857 |

Within a point on every metric, so you're working from a faithful starting point, not an approximation. (FCD/Test is 2.83 here; MolGPT never published an FCD at this size, so there's no firm number to match. The SELFIES model below brings it down to 0.43.) Full numbers: [`results/molgpt_record_30k.metrics.json`](results/molgpt_record_30k.metrics.json).

## Improvement over the benchmark via representation

A plain SMILES model has to learn chemical validity from the data, and it never quite reaches 100%. The verification-surfaces work (Michael Rothrock, *Verification Surfaces in Language-Model Systems*, 2026, [doi:10.5281/zenodo.20331399](https://doi.org/10.5281/zenodo.20331399) section 8) makes a point that is relevant here: a molecular generator is a structured-output task with a formal validity rule, and the representation you pick decides what you get for free. So instead of a bigger model, we retrained the same one on [SELFIES](https://github.com/aspuru-guzik-group/selfies), where every string decodes to a valid molecule by construction.

Same size, same budget, and it improves across the board:

| Metric | SMILES | SELFIES |
|---|---|---|
| Valid | 0.988 | 1.000 |
| Novelty | 0.783 | 0.859 |
| FCD/Test | 2.83 | 0.43 |
| Filters | 0.996 | 0.976 |

Validity is now free, novelty and distribution fit both go up, and the MOSES medchem `Filters` rate slips a little (0.996 to 0.976). SELFIES isn't ours ([Krenn et al. 2020](https://github.com/aspuru-guzik-group/selfies)); the nice part is that the framing told us to reach for a better representation instead of more compute. Details: [SELFIES_LEG.md](SELFIES_LEG.md).

## Load the trained models

Don't want to wait for training? Both models are on Hugging Face and load with stock `transformers` (no `trust_remote_code`):

- SMILES reproduction: [mrothroc/molgpt-moses-smiles-mixlab](https://huggingface.co/mrothroc/molgpt-moses-smiles-mixlab)
- SELFIES variant: [mrothroc/molgpt-moses-selfies-mixlab](https://huggingface.co/mrothroc/molgpt-moses-selfies-mixlab)

The cards have full details; here is the whole thing for the SMILES model (the SELFIES model is identical, just decode with `selfies.decoder`):

```python
# pip install torch transformers huggingface_hub tokenizers rdkit
import torch
from transformers import AutoModelForCausalLM
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download

repo = "mrothroc/molgpt-moses-smiles-mixlab"
model = AutoModelForCausalLM.from_pretrained(repo).eval()   # GPT2LMHeadModel, no trust_remote_code
tok = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
BOS, EOS, PAD = 1, 2, 0

out = model.generate(torch.full((8, 1), BOS), do_sample=True, temperature=1.0, max_length=65)
for row in out.tolist():
    seq = [t for t in row[1:] if t not in (BOS, PAD)]
    if EOS in seq: seq = seq[:seq.index(EOS)]
    print("".join(tok.id_to_token(t) for t in seq))         # SMILES strings
```

To train the models yourself instead, read on.

## Reproduce it

### Setup

You need [mixlab](https://github.com/mrothroc/mixlab) v0.73.0 or newer on your PATH (see the mixlab README for the `brew` install) and a Python environment for data prep and scoring. From this directory:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps molsets==0.3.1     # MOSES; its own deps conflict with the modern pins
# goal-directed leg only:  pip install PyTDC==1.0.0
```

The MOSES dataset downloads automatically on first use (via `molsets`); nothing to fetch by hand. Then run the pipeline from this directory:

```bash
export MIXLAB_MLX_CACHE_LIMIT_MB=4096
export MIXLAB_SCRIPTS=/path/to/mixlab/scripts       # mixlab prepare shells out here
export PATH="$PWD/.venv/bin:$PATH"                   # venv (rdkit/moses) on PATH

# 1. data prep: content-only char tokenizer (vocab 30) + per-record shards from MOSES train
python scripts/build_tokenizer.py --no-template --out data/tokenizer_content.json
python scripts/make_jsonl.py --split train --out data/train_full.jsonl --no-canon
mixlab -mode prepare -input data/train_full.jsonl -output data/record_shards \
  -tokenizer-path data/tokenizer_content.json -text-field text \
  -frame-per-record -record-seq-len 64 -record-pad-id 0 -record-bos-id 1 -record-eos-id 2

# 2. train: matched 6.34M GPT, one molecule per sequence, ~10 epochs (33k steps, ~5h on Apple Silicon)
mixlab -mode arch -config configs/molgpt_record.json -train 'data/record_shards/train_*.bin' \
  -checkpoint-dir checkpoints/molgpt_record_ckpts -checkpoint-every 3000 \
  -safetensors checkpoints/molgpt_record.safetensors

# 3. sample 30k + score with the official moses.metrics harness
python scripts/sample_and_score.py --config configs/molgpt_record.json \
  --ckpt checkpoints/molgpt_record.safetensors --tokenizer data/tokenizer_content.json \
  --max-tokens 64 --n-jobs 8 --out samples/molgpt_record_30k.smi
#    -> Valid ~0.99, Novelty ~0.78, IntDiv ~0.857   (matches results/molgpt_record_30k.metrics.json)
```

The SELFIES model is the same pipeline over SELFIES symbols with `configs/molgpt_selfies_record.json`; see [SELFIES_LEG.md](SELFIES_LEG.md).

## Where to go next

Each part of the recipe has a short writeup. Pick the one that matches what you want to do:

- **Train the base model yourself** -> [MOLGPT_REPRODUCTION.md](MOLGPT_REPRODUCTION.md): the reproduction end to end, including the one choice (one molecule per sequence) that makes it work.
- **Make every sample a valid molecule** -> [SELFIES_LEG.md](SELFIES_LEG.md): retrain on SELFIES so invalid molecules are impossible by construction.
- **Get validity without retraining** -> [CONSTRAINED_DECODE.md](CONSTRAINED_DECODE.md): keep the SMILES model as-is, constrain generation to a molecule grammar, then filter the rest.
- **Steer toward a property** -> [GOAL_DIRECTED.md](GOAL_DIRECTED.md): nudge the model toward drug-likeness (QED) while keeping the outputs diverse.

## Exercising and enhancing mixlab

The two mixlab features this recipe leans on are `-frame-per-record` (one complete molecule per training sequence) and `-grammar-table` (grammar-constrained decoding). The bare configs in `configs/` are forkable mixlab example recipes, and `results/` holds the small metric JSONs behind the tables above. mixlab: https://github.com/mrothroc/mixlab

## Citing this work

This is a reproduction, so if it helps your work, please cite what it builds on:

- **MolGPT**: Bagal, V.; Aggarwal, R.; Vinod, P. K.; Priyakumar, U. D. "MolGPT: Molecular Generation Using a Transformer-Decoder Model." *J. Chem. Inf. Model.* 2022, 62 (9), 2064-2076. [doi:10.1021/acs.jcim.1c00600](https://doi.org/10.1021/acs.jcim.1c00600)
- **MOSES benchmark**: Polykovskiy, D. et al. "Molecular Sets (MOSES): A Benchmarking Platform for Molecular Generation Models." *Front. Pharmacol.* 2020, 11, 565644. [doi:10.3389/fphar.2020.565644](https://doi.org/10.3389/fphar.2020.565644)
- **SELFIES**: Krenn, M.; Häse, F.; Nigam, A.; Friederich, P.; Aspuru-Guzik, A. "Self-Referencing Embedded Strings (SELFIES): A 100% Robust Molecular String Representation." *Mach. Learn.: Sci. Technol.* 2020, 1 (4), 045024. [doi:10.1088/2632-2153/aba947](https://doi.org/10.1088/2632-2153/aba947)
- **Verification surfaces** (the representation lesson used here; see section 8, with section 2 for the representation argument and section 4 for the grammar/verifier split): Rothrock, M. "Verification Surfaces in Language-Model Systems: Token, Schema, and Structured-Output Reliability." 2026. [doi:10.5281/zenodo.20331399](https://doi.org/10.5281/zenodo.20331399)

If you want to point at this specific reproduction or the trainer:

```bibtex
@software{molgpt_mixlab,
  author = {Rothrock, Michael},
  title  = {MolGPT reproduced in mixlab (MOSES)},
  year   = {2026},
  url    = {https://github.com/mrothroc/mixlab-cookbook/tree/main/molgpt-moses}
}
@software{mixlab,
  author = {Rothrock, Michael},
  title  = {mixlab: a compact ML architecture lab},
  url    = {https://github.com/mrothroc/mixlab}
}
```

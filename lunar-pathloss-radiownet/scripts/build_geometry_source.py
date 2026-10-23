"""Build a mixlab grid-prepare source for the extend: the reproduction's records plus geometric input channels.

Channels 0-1 are exactly the reproduction's records [heightmap / 496 m, TX one-hot], taken from the organizers'
`LunarRadioMapDataset` as in `build_grid_source.py`. Channels 2-4 come
from `physics_features.channels`: normalised log-distance, the LOS proxy and signed clearance. They are computed from the
de-normalised heightmap and the TX pixel the network sees, with fixed constants, so nothing is fitted on
any split. Targets and masks are unchanged.

Layouts:
- `dev` (screening): the official train split, divided by terrain. A seeded fraction of terrains per
  `group` (multi/single TX) becomes `val`; the rest is `train`. The official val split is not
  used, so it stays clean for confirming finalists.
- `official`: official train and val, as in the reproduction. This is the layout the recipe uses.

float16 storage is the default: five float32 channels would need ~45 GB per band. The source targets
are float16 already, and the build report records the rounding of the normalised values.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import build_grid_source as b1
import physics_features as pf

CHANNELS = ["heightmap", "tx", "log_distance", "los", "clearance"]
INPUTS = {"geometry": CHANNELS, "base": CHANNELS[:2],  # base = the 2-channel control: same records and storage
          "geometry2": CHANNELS + ["horizon_margin", "blocked_fraction"]}  # the 7-channel set used at 415 MHz
LAYOUTS = ("dev", "official")


def _features(args):
    hm, tx, band, extended = args
    import torch
    torch.set_num_threads(1)
    return pf.channels(hm, tx, band, extended=extended)


def dev_terrains(rows: list, fraction: float, seed: int) -> set:
    """Seeded terrain-level holdout, stratified by `group`, at least one terrain per group."""
    by_group = {}
    for r in rows:
        by_group.setdefault(r["group"], set()).add(r["terrain_id"])
    rng = np.random.default_rng(seed)
    held = set()
    for g in sorted(by_group):
        terr = sorted(by_group[g])
        k = max(1, round(fraction * len(terr)))
        held.update(terr[i] for i in rng.permutation(len(terr))[:k])
    return held


def build(root, out, *, band, dtype="float16", layout="dev", dev_fraction=0.1, dev_seed=2027, workers=None,
          limits=None, inputs="geometry", records_per_shard=64, splits=("train", "val")) -> dict:
    """`limits` keeps only the first N records of an output split (for cross-machine spot checks). `splits` builds
    only those output splits (e.g. the official val split alone, when the official train records are already
    built as the dev layout's two splits)."""
    root, out = Path(root), Path(out)
    if band not in b1.BANDS:
        raise ValueError(f"band must be one of {b1.BANDS}")
    if dtype not in ("float32", "float16"):
        raise ValueError("dtype must be float32 or float16")
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}")
    if inputs not in INPUTS:
        raise ValueError(f"inputs must be one of {sorted(INPUTS)}")
    n_ch = len(INPUTS[inputs])
    if layout == "dev" and not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be in (0, 1)")
    limits = limits or {}
    if set(limits) - {"train", "val"} or any(n <= 0 for n in limits.values()):
        raise ValueError(f"limits must map train/val to positive counts, got {limits}")
    if not splits or set(splits) - {"train", "val"}:
        raise ValueError(f"splits must be a nonempty subset of train/val, got {splits}")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output directory not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    ld = b1.load_loader(root)
    meta = json.loads((root / "metadata.json").read_text())
    hm_lo, hm_hi = meta["heightmap_range_m"]
    pl_lo, pl_hi = meta["pathloss_range_db"]
    np_dtype = np.dtype(dtype)
    h, w = ld.GRID

    # (output split, source split, predicate on the loader row)
    if layout == "official":
        plan = [("train", "train", None), ("val", "val", None)]
        held = set()
    else:
        rows = list(csv.DictReader(open(root / "train_index.csv")))
        held = dev_terrains(rows, dev_fraction, dev_seed)
        plan = [("train", "train", lambda r: r["terrain_id"] not in held),
                ("val", "train", lambda r: r["terrain_id"] in held)]
    plan = [step for step in plan if step[0] in splits]

    spec = {"dtype": dtype, "records_per_shard": records_per_shard, "splits": {},
            "normalization": {"fit_split": "train",
                              "input_offset": [float(hm_lo)] + [0.0] * (n_ch - 1),
                              "input_scale": [float(hm_hi - hm_lo)] + [1.0] * (n_ch - 1),
                              "target_offset": [float(pl_lo)], "target_scale": [float(pl_hi - pl_lo)]}}
    report = {"band": band, "dtype": dtype, "layout": layout, "channels": INPUTS[inputs], "records": {}, "rounding": {},
              "physics": {"tx_agl_m": pf.TX_AGL_M, "rx_agl_m": pf.RX_AGL_M, "d_range_m": [pf.D_MIN_M, pf.D_MAX_M],
                          "clearance_cap_m": pf.CLEARANCE_CAP_M,
                          "module_sha256": hashlib.sha256(Path(pf.__file__).read_bytes()).hexdigest()}}
    if layout == "dev":
        report.update(dev_fraction=dev_fraction, dev_seed=dev_seed, dev_terrains=len(held))

    datasets = {}
    n_workers = workers or os.cpu_count()
    pool = ProcessPoolExecutor(max_workers=n_workers) if n_workers > 1 else None
    try:
        for split, source, keep in plan:
            if source not in datasets:
                datasets[source] = ld.LunarRadioMapDataset(str(root), split=source, band=band, normalize=True,
                                                           augment=False, return_mask=True)
            ds = datasets[source]
            idx = [i for i in range(len(ds)) if keep is None or keep(ds.rows[ds.samples[i][0]])]
            idx = idx[:limits.get(split, len(idx))]
            if not idx:
                raise ValueError(f"split {split} is empty")
            files = {k: f"{split}_{k}.npy" for k in ("inputs", "targets", "masks")}
            x = np.lib.format.open_memmap(out / files["inputs"], mode="w+", dtype=np_dtype, shape=(len(idx), n_ch, h, w))
            y = np.lib.format.open_memmap(out / files["targets"], mode="w+", dtype=np_dtype, shape=(len(idx), 1, h, w))
            m = np.lib.format.open_memmap(out / files["masks"], mode="w+", dtype=np.uint8, shape=(len(idx), 1, h, w))
            ids, groups, tasks, base = [], [], [], []
            in_err = tg_err = 0.0
            for j, i in enumerate(idx):
                xi, yi, mi = (t.numpy() for t in ds[i])
                if xi.shape != (2, h, w):
                    raise ValueError(f"{source}[{i}]: input shape {xi.shape}")
                row = ds.rows[ds.samples[i][0]]
                tx = tuple(int(v) for v in np.argwhere(xi[1] == 1)[0])
                tasks.append((xi[0] * (hm_hi - hm_lo) + hm_lo, tx, band, inputs == "geometry2"))
                base.append(xi)
                sy = yi.astype(np_dtype)
                y[j], m[j] = sy, mi.astype(np.uint8)
                tg_err = max(tg_err, float(np.abs(sy.astype(np.float32) - yi).max()))
                ids.append(f"{row['sample_id']}_{band}")
                groups.append(row["terrain_id"])
                if len(tasks) == 256 or j == len(idx) - 1:
                    if inputs == "base":
                        feats = [np.zeros((0, h, w), np.float32)] * len(tasks)
                    else:
                        feats = pool.map(_features, tasks, chunksize=8) if pool else map(_features, tasks)
                    for k, (xb, f) in enumerate(zip(base, feats)):
                        full = np.concatenate([xb, f]).astype(np.float32)
                        sx = full.astype(np_dtype)
                        x[j - len(tasks) + 1 + k] = sx
                        in_err = max(in_err, float(np.abs(sx.astype(np.float32) - full).max()))
                    tasks, base = [], []
            for arr in (x, y, m):
                arr.flush()
            del x, y, m
            (out / f"{split}_ids.json").write_text(json.dumps(ids))
            (out / f"{split}_groups.json").write_text(json.dumps(groups))
            spec["splits"][split] = {**files, "ids": f"{split}_ids.json", "groups": f"{split}_groups.json"}
            report["records"][split] = len(idx)
            report["rounding"][split] = {"inputs_max_abs": in_err, "targets_max_abs": tg_err}
    finally:
        if pool:
            pool.shutdown()

    (out / "source.json").write_text(json.dumps(spec, indent=2) + "\n")
    (out / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--band", choices=b1.BANDS, required=True)
    p.add_argument("--dtype", choices=["float32", "float16"], default="float16")
    p.add_argument("--layout", choices=LAYOUTS, required=True)
    p.add_argument("--inputs", choices=sorted(INPUTS), default="geometry")
    p.add_argument("--dev-fraction", type=float, default=0.1)
    p.add_argument("--dev-seed", type=int, default=2027)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--limit", action="append", default=[], metavar="SPLIT=N")
    p.add_argument("--split", action="append", choices=["train", "val"], help="build only these output splits")
    a = p.parse_args()
    limits = {k: int(v) for k, _, v in (item.partition("=") for item in a.limit)}
    print(json.dumps(build(a.root, a.out, band=a.band, dtype=a.dtype, layout=a.layout, dev_fraction=a.dev_fraction,
                           dev_seed=a.dev_seed, workers=a.workers, limits=limits, inputs=a.inputs,
                           splits=tuple(a.split or ("train", "val"))), indent=2))


if __name__ == "__main__":
    main()

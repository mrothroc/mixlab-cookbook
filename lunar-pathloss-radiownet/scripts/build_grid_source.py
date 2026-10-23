"""Build a mixlab grid-prepare source for one band of the lunar challenge data.

Records are taken verbatim from the organizers' own `LunarRadioMapDataset` (the copy bundled with the
data, which is logic-identical to the audited repo loader). Settings are `normalize=True`,
`augment=False` and `return_mask=True`, iterated in loader order. Each record is the two-channel input
[heightmap / 496 m, TX one-hot], the target (pl - 20) / 208, and the validity mask. Joint D4
augmentation is left to mixlab (`training.grid_augmentation.dihedral`). Groups are terrain IDs, so the
terrain-level split can be audited.

The two-model baseline trains one model per band, so this builds one band at a time; `both` is
rejected. Output: `<out>/source.json` plus .npy arrays, ids and groups for
`mixlab -mode prepare -input-format grid`.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

BANDS = ("415", "58")
SPLITS = ("train", "val")


def load_loader(root):
    """Import the `lunar_dataset.py` shipped with the data, as the reference `train.py` prefers."""
    path = Path(root) / "lunar_dataset.py"
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(f"lunar_dataset_{abs(hash(str(path.resolve())))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _records(ds, n: int, band: str, np_dtype, shape: tuple):
    """The stored form of the loader's first n records: (input, target, mask, id, group, input and target rounding)."""
    h, w = shape
    for i in range(n):
        xi, yi, mi = (t.numpy() for t in ds[i])
        if xi.shape != (2, h, w):
            raise ValueError(f"record {i}: input shape {xi.shape}")
        sx, sy = xi.astype(np_dtype), yi.astype(np_dtype)
        row = ds.rows[ds.samples[i][0]]
        yield (sx, sy, mi.astype(np.uint8), f"{row['sample_id']}_{band}", row["terrain_id"],
               float(np.abs(sx.astype(np.float32) - xi).max()), float(np.abs(sy.astype(np.float32) - yi).max()))


def stream_digest(root, *, band, dtype, splits=SPLITS) -> dict:
    """`source_digest.full_digest` of the source `build` would write, computed without writing any array."""
    import source_digest as sd
    root = Path(root)
    ld = load_loader(root)
    per = {}
    for split in splits:
        ds = ld.LunarRadioMapDataset(str(root), split=split, band=band, normalize=True, augment=False,
                                     return_mask=True)
        recs, ids, groups = [], [], []
        for sx, sy, mi, rid, grp, _ie, _te in _records(ds, len(ds), band, np.dtype(dtype), ld.GRID):
            recs.append(sd.record_sha(sx, sy, mi))
            ids.append(rid)
            groups.append(grp)
        per[split] = (recs, ids, groups)
    return sd.combine(per)


def build(root, out, *, band, dtype, splits=SPLITS, limits=None, records_per_shard=64) -> dict:
    root, out = Path(root), Path(out)
    if band not in BANDS:
        raise ValueError(f"band must be one of {BANDS} (one model per band), got {band!r}")
    if dtype not in ("float32", "float16"):
        raise ValueError("dtype must be float32 or float16")
    limits = limits or {}
    for name, n in limits.items():
        if name not in splits:
            raise ValueError(f"limit for unknown split {name!r}")
        if n <= 0:
            raise ValueError(f"limit for {name!r} must be positive")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output directory not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    ld = load_loader(root)
    np_dtype = np.dtype(dtype)
    meta = json.loads((root / "metadata.json").read_text())
    hm_lo, hm_hi = meta["heightmap_range_m"]
    pl_lo, pl_hi = meta["pathloss_range_db"]
    spec = {"dtype": dtype, "records_per_shard": records_per_shard, "splits": {},
            "normalization": {"fit_split": "train",
                              "input_offset": [float(hm_lo), 0.0], "input_scale": [float(hm_hi - hm_lo), 1.0],
                              "target_offset": [float(pl_lo)], "target_scale": [float(pl_hi - pl_lo)]}}
    report = {"band": band, "dtype": dtype, "records": {}, "rounding": {}}

    for split in splits:
        ds = ld.LunarRadioMapDataset(str(root), split=split, band=band, normalize=True, augment=False,
                                     return_mask=True)
        n = min(len(ds), limits.get(split, len(ds)))
        if n == 0:
            raise ValueError(f"split {split} is empty")
        h, w = ld.GRID
        files = {k: f"{split}_{k}.npy" for k in ("inputs", "targets", "masks")}
        x = np.lib.format.open_memmap(out / files["inputs"], mode="w+", dtype=np_dtype, shape=(n, 2, h, w))
        y = np.lib.format.open_memmap(out / files["targets"], mode="w+", dtype=np_dtype, shape=(n, 1, h, w))
        m = np.lib.format.open_memmap(out / files["masks"], mode="w+", dtype=np.uint8, shape=(n, 1, h, w))
        ids, groups = [], []
        in_err = tg_err = 0.0
        for i, (sx, sy, mi, rid, grp, ie, te) in enumerate(_records(ds, n, band, np_dtype, (h, w))):
            x[i], y[i], m[i] = sx, sy, mi
            in_err, tg_err = max(in_err, ie), max(tg_err, te)
            ids.append(rid)
            groups.append(grp)
        for arr in (x, y, m):
            arr.flush()
        del x, y, m
        (out / f"{split}_ids.json").write_text(json.dumps(ids))
        (out / f"{split}_groups.json").write_text(json.dumps(groups))
        spec["splits"][split] = {**files, "ids": f"{split}_ids.json", "groups": f"{split}_groups.json"}
        report["records"][split] = n
        report["rounding"][split] = {"inputs_max_abs": in_err, "targets_max_abs": tg_err}

    (out / "source.json").write_text(json.dumps(spec, indent=2) + "\n")
    (out / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", required=True, help="unpacked LunarRM directory (flat arrays + lunar_dataset.py)")
    p.add_argument("--out", required=True, help="new, empty output directory")
    p.add_argument("--band", choices=BANDS, required=True)
    p.add_argument("--dtype", choices=["float32", "float16"], required=True)
    p.add_argument("--limit", action="append", default=[], metavar="SPLIT=N")
    a = p.parse_args()
    limits = {}
    for item in a.limit:
        k, sep, v = item.partition("=")
        if not sep:
            raise SystemExit(f"--limit must be SPLIT=N, got {item!r}")
        limits[k] = int(v)
    print(json.dumps(build(a.root, a.out, band=a.band, dtype=a.dtype, limits=limits), indent=2))


if __name__ == "__main__":
    main()

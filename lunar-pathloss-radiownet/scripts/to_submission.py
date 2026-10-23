"""Challenge submission adapter, plus a val-as-test ground-truth tree for the organizers' scorer.

`write_submission`: mixlab `predict-grid` output → `pl<band>_<sample_id>.npy`, float32 [256,256], in dB.
- mixlab writes numbered `[Ct,H,W]` float32 files in normalized model units, plus `predictions.json`
  of the form `{"format": "mixlab.grid_predictions.v1", "units": "model", "records": [{"id", "file"}]}`
  (observed from v0.123.0 output; the format and units are asserted). Our record IDs are `<sample_id>_<band>`
  (`build_grid_source.py`).
- dB = pl_min + (pl_max - pl_min) * p, with the range from the data's `metadata.json`. No clipping:
  the organizers' `score_submission.py` scores unclipped predictions, and clipping would change the
  evaluated model.

`write_val_gt_tree`: lays out the shipped val split in the per-file `test/` layout that
`score_submission.py` reads: `index.csv` with `pl<band>` paths, and `mask_index.csv` with `mask<band>`
paths. This lets the organizers' unmodified scorer score our val predictions.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

GRID = (256, 256)


def sample_id_from_record(record_id: str, band: str) -> str:
    suffix = f"_{band}"
    if not record_id.endswith(suffix):
        raise ValueError(f"record {record_id!r} is not band {band}")
    return record_id[: -len(suffix)]


def write_submission(pred_dir, out, *, band: str, pl_range) -> int:
    pred_dir, out = Path(pred_dir), Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"submission directory not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    lo, hi = (float(v) for v in pl_range)
    manifest = json.loads((pred_dir / "predictions.json").read_text())
    if manifest.get("format") != "mixlab.grid_predictions.v1" or manifest.get("units") != "model":
        raise ValueError(f"unexpected predictions manifest: format={manifest.get('format')!r} "
                         f"units={manifest.get('units')!r}")
    records = manifest.get("records") or []
    if not records:
        raise ValueError("predictions.json has no records")
    seen = set()
    for rec in records:
        rid, fn = rec["id"], rec["file"]
        sid = sample_id_from_record(rid, band)
        if sid in seen:
            raise ValueError(f"duplicate sample {sid}")
        seen.add(sid)
        p = np.load(pred_dir / fn)
        if p.shape != (1, *GRID) or p.dtype != np.float32:
            raise ValueError(f"{fn}: expected float32 (1,256,256), got {p.dtype} {p.shape}")
        db = (lo + (hi - lo) * p[0].astype(np.float64)).astype(np.float32)
        if not np.isfinite(db).all():
            raise ValueError(f"{fn}: non-finite prediction")
        np.save(out / f"pl{band}_{sid}.npy", db)
    return len(seen)


def write_val_gt_tree(data_root, gt_root, *, band: str, split: str = "val") -> int:
    data_root, gt = Path(data_root), Path(gt_root) / "test"
    gt.mkdir(parents=True, exist_ok=False)
    rows = list(csv.DictReader(open(data_root / f"{split}_index.csv", newline="")))
    pl = np.load(data_root / f"{split}_pl{band}.npy", mmap_mode="r")
    mk = np.load(data_root / f"{split}_mask{band}.npy", mmap_mode="r")
    if len(rows) != pl.shape[0] or len(rows) != mk.shape[0]:
        raise ValueError("index / array row counts disagree")
    (gt / "pl").mkdir()
    (gt / "mask").mkdir()
    idx, midx = [], []
    for i, r in enumerate(rows):
        sid = r["sample_id"]
        tp, mp = f"pl/pl{band}_{sid}.npy", f"mask/mask{band}_{sid}.npy"
        np.save(gt / tp, np.asarray(pl[i], dtype=np.float32))
        np.save(gt / mp, np.unpackbits(np.asarray(mk[i])).reshape(GRID).astype(np.uint8))
        idx.append({"sample_id": sid, f"pl{band}": tp})
        midx.append({"sample_id": sid, f"mask{band}": mp})
    for name, data in (("index.csv", idx), ("mask_index.csv", midx)):
        with open(gt / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    return len(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("submission")
    a.add_argument("--pred-dir", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--band", choices=["415", "58"], required=True)
    a.add_argument("--meta", required=True, help="metadata.json with pathloss_range_db")
    g = sub.add_parser("val-gt")
    g.add_argument("--data-root", required=True)
    g.add_argument("--gt-root", required=True)
    g.add_argument("--band", choices=["415", "58"], required=True)
    args = p.parse_args()
    if args.cmd == "submission":
        rng = json.loads(Path(args.meta).read_text())["pathloss_range_db"]
        print(write_submission(args.pred_dir, args.out, band=args.band, pl_range=rng), "files written")
    else:
        print(write_val_gt_tree(args.data_root, args.gt_root, band=args.band), "val samples laid out")


if __name__ == "__main__":
    main()

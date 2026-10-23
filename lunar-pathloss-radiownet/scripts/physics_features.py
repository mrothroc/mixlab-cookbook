"""Geometric input channels for the lunar pathloss task, computed from a heightmap and a TX position.

These are an input-derived **LOS proxy**, not the simulator's geometry. The simulator samples receivers on
the triangles of a terrain mesh and interpolates them onto the raster (metadata.json). Here the terrain is
bilinear between pixel centres.

Simulation geometry (metadata.json): 1 m/px, TX antenna 3 m above its pixel, RX 1 m above each pixel.
The simulator models line of sight, specular and diffuse reflection and refraction, with no diffraction.
On val, NLOS pixels of this proxy sit 17 dB (415 MHz) / 39 dB (5.8 GHz) above LOS pixels at similar
distance (`los_separation_db`).

Per RX pixel:
- `clearance_m`: the smallest height of the TX->RX segment above the terrain over the segment's
  interior. It is negative when terrain rises above the segment, and capped at `CLEARANCE_CAP_M` when
  the segment has no interior grid crossing (the TX pixel and its neighbours);
- `los`: clearance > 0;
- `fspl_db`: free-space path loss 20·log10(4π d f / c) over the 3-D distance. Normalised for a
  per-band model it is just log-distance, since the frequency term is a constant;
- `horizon_margin_deg` (G2): the steepest terrain elevation angle seen from the TX antenna along the
  segment's interior, minus the angle to the RX. Positive means shadowed; -90 where the segment has no
  interior crossing;
- `blocked_fraction` (G2): the fraction of the segment's interior row and column crossings where terrain is
  above the segment; positive exactly when the LOS proxy is blocked.

Along a straight segment, the bilinear surface is exact at every crossing of an integer row or column,
where it reduces to linear interpolation between two pixel centres. The segment is evaluated at all such
crossings, so the evaluation commutes with grid rotations and flips. Maxima inside a cell, where the
surface is quadratic along the line, are not sampled. On 8,192 real rays that changed no LOS label and
moved clearance by at most 0.018 m (checked during development).
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import torch

C = 299_792_458.0
TX_AGL_M, RX_AGL_M = 3.0, 1.0
BANDS_HZ = {"415": 415e6, "58": 5.775e9}
D_MIN_M, D_MAX_M = 1.0, 1000.0  # fixed FSPL normalisation range: covers 256 m tiles with 500 m relief
CLEARANCE_CAP_M = 500.0
CHANNEL_CHUNK = 8192  # pixels per evaluation chunk in channels(): bounds builder-worker memory, same values  # |clearance| encoding range; also the value for segments with no interior


def _crossing_stats(g: torch.Tensor, r0: int, c0: int, z0: float, idx: torch.Tensor, axis: int):
    """For the TX->pixel segments of flat pixel indices `idx`, over every crossing of an integer column (axis=1)
    or row (axis=0) strictly inside the segment:
    - the largest (terrain - line height), -inf if none;
    - the largest terrain elevation angle seen from the TX antenna, in radians, -inf if none;
    - the number of crossings where terrain is above the line, and the number of crossings.
    Elementwise in each crossing, so evaluating any subset of pixels gives the same values."""
    H, W = g.shape
    dev = g.device
    r1, c1 = (idx // W).to(torch.float64).reshape(-1, 1), (idx % W).to(torch.float64).reshape(-1, 1)
    dr, dc = r1 - r0, c1 - c0
    z1 = g.reshape(-1)[idx].reshape(-1, 1) + RX_AGL_M
    along, across = (dc, dr) if axis == 1 else (dr, dc)
    n_along = along.abs()
    k_max = int(n_along.max().item()) if idx.numel() else 0
    ninf = torch.full((idx.numel(),), -math.inf, dtype=torch.float64, device=dev)
    zero = torch.zeros(idx.numel(), dtype=torch.float64, device=dev)
    if k_max <= 1:
        return ninf, ninf.clone(), zero, zero.clone()
    k = torch.arange(1, k_max, dtype=torch.float64, device=dev).reshape(1, -1)
    valid = k < n_along
    t = torch.where(valid, k / n_along.clamp(min=1), torch.zeros((), dtype=torch.float64, device=dev))
    pos_along = (c0 if axis == 1 else r0) + torch.sign(along) * k          # integer grid line
    pos_across = (r0 if axis == 1 else c0) + t * across                    # fractional
    n_across, n_al = (H, W) if axis == 1 else (W, H)
    lo = pos_across.floor().clamp(0, n_across - 1)
    hi = (lo + 1).clamp(max=n_across - 1)
    w = pos_across - lo
    a = pos_along.clamp(0, n_al - 1).expand_as(lo).long()
    if axis == 1:
        h = g[lo.long(), a] * (1 - w) + g[hi.long(), a] * w
    else:
        h = g[a, lo.long()] * (1 - w) + g[a, hi.long()] * w
    excess = h - (z0 + t * (z1 - z0))
    reach = torch.sqrt(dr * dr + dc * dc)                                   # horizontal TX->RX distance (m)
    angle = torch.atan2(h - z0, t * reach)
    m = torch.full((), -math.inf, dtype=torch.float64, device=dev)
    return (torch.where(valid, excess, m).amax(1), torch.where(valid, angle, m).amax(1),
            (valid & (excess > 0)).sum(1).to(torch.float64), valid.sum(1).to(torch.float64))


def features(hm: np.ndarray, tx: tuple, freq_hz: float, device: str = "cpu", chunk: int | None = None) -> dict:
    """`chunk` bounds memory by evaluating that many pixels at a time; results are identical. Geometry runs in
    float64 on `device` (CPU or CUDA; MPS has no float64)."""
    hm = np.array(hm, np.float32)  # a writable copy: torch.from_numpy needs one
    H, W = hm.shape
    r0, c0 = int(tx[0]), int(tx[1])
    if not (0 <= r0 < H and 0 <= c0 < W):
        raise ValueError(f"TX {tx} outside a {H}x{W} grid")
    g = torch.from_numpy(hm).to(device=device, dtype=torch.float64)
    z0 = float(hm[r0, c0]) + TX_AGL_M
    P = H * W
    step = chunk or P
    ex, ang, frac = [], [], []
    for s in range(0, P, step):
        idx = torch.arange(s, min(s + step, P), device=device)
        e1, a1, b1, n1 = _crossing_stats(g, r0, c0, z0, idx, 1)
        e0, a0, b0, n0 = _crossing_stats(g, r0, c0, z0, idx, 0)
        ex.append(torch.maximum(e1, e0))
        ang.append(torch.maximum(a1, a0))
        # over both axes' crossings: any blocked crossing gives a positive fraction (consistent with LOS). A path
        # through a lattice point samples it on both axes, in numerator and denominator alike.
        nb, nv = b1 + b0, n1 + n0
        frac.append(torch.where(nv > 0, nb / nv.clamp(min=1), torch.zeros_like(nv)))
    excess = torch.cat(ex).cpu()
    clearance = (-excess).clamp(max=CLEARANCE_CAP_M).reshape(H, W).numpy()   # no crossings: -(-inf) -> cap
    rr, cc = np.mgrid[0:H, 0:W]
    reach = np.hypot(rr - r0, cc - c0)
    rx_angle = np.arctan2(hm.astype(np.float64) + RX_AGL_M - z0, reach)
    margin = np.degrees(torch.cat(ang).cpu().reshape(H, W).numpy() - rx_angle)   # no crossings: -inf
    margin = np.clip(np.nan_to_num(margin, neginf=-90.0), -90.0, 90.0)
    d = np.sqrt(reach ** 2 + (hm.astype(np.float64) + RX_AGL_M - z0) ** 2)
    fspl = 20 * np.log10(4 * math.pi * np.maximum(d, D_MIN_M) * freq_hz / C)
    return {"los": clearance > 0, "clearance_m": clearance.astype(np.float32), "fspl_db": fspl.astype(np.float32),
            "horizon_margin_deg": margin.astype(np.float32),
            "blocked_fraction": torch.cat(frac).cpu().reshape(H, W).numpy().astype(np.float32)}


def channels(hm: np.ndarray, tx: tuple, band: str, device: str = "cpu", extended: bool = False) -> np.ndarray:
    """[3, H, W] float32: normalised log-distance in [0, 1], LOS in {0, 1}, and signed clearance
    asinh(c)/asinh(cap) in [-1, 1]. Fixed normalisation, so nothing is fitted on any split."""
    if band not in BANDS_HZ:
        raise ValueError(f"band must be one of {sorted(BANDS_HZ)}, got {band!r}")
    f = BANDS_HZ[band]
    fs = features(hm, tx, f, device=device, chunk=CHANNEL_CHUNK)
    lo, hi = (20 * math.log10(4 * math.pi * d * f / C) for d in (D_MIN_M, D_MAX_M))
    fspl = np.clip((fs["fspl_db"] - lo) / (hi - lo), 0, 1)
    clr = np.arcsinh(np.clip(fs["clearance_m"], -CLEARANCE_CAP_M, CLEARANCE_CAP_M)) / math.asinh(CLEARANCE_CAP_M)
    chans = [fspl, fs["los"].astype(np.float32), clr]
    if extended:  # 7-channel set: horizon margin / 90 in [-1, 1], blocked-path fraction in [0, 1]
        chans += [fs["horizon_margin_deg"] / 90.0, fs["blocked_fraction"]]
    return np.stack(chans).astype(np.float32)


def los_separation_db(data_root: Path, band: str, n: int = 12, flip_hm: bool = False) -> float:
    """Mean over distance bins of (mean NLOS pathloss - mean LOS pathloss) on valid pixels of `n` evenly
    spaced val records. A diagnostic for feature/target alignment."""
    data_root = Path(data_root)
    rows = list(csv.DictReader(open(data_root / "val_index.csv")))
    hms = np.load(data_root / "val_hm.npy", mmap_mode="r")
    pl = np.load(data_root / f"val_pl{band}.npy", mmap_mode="r")
    masks = np.load(data_root / f"val_mask{band}.npy", mmap_mode="r")
    diffs = []
    for i in np.linspace(0, len(rows) - 1, n).round().astype(int):
        row = rows[i]
        hm = np.asarray(hms[int(row["hm_row"])], np.float32)
        if flip_hm:
            hm = np.ascontiguousarray(np.flipud(hm))
        tx = (int(row["tx_row"]), int(row["tx_col"]))
        f = features(hm, tx, BANDS_HZ[band])
        y = np.asarray(pl[i], np.float32)
        m = np.unpackbits(masks[i]).reshape(hm.shape).astype(bool)
        rr, cc = np.mgrid[0:hm.shape[0], 0:hm.shape[1]]
        dist = np.hypot(rr - tx[0], cc - tx[1])
        for a, b in ((20, 60), (60, 120), (120, 400)):
            sel = m & (dist >= a) & (dist < b)
            los, nlos = sel & f["los"], sel & ~f["los"]
            if los.sum() >= 50 and nlos.sum() >= 50:
                diffs.append(y[nlos].mean() - y[los].mean())
    if not diffs:
        raise ValueError("no distance bin had both LOS and NLOS pixels")
    return float(np.mean(diffs))

"""Write the organizers' RadioWNet initialization as a mixlab checkpoint, so mixlab starts from the same weights.

The reference `train.py` seeds torch, then builds `RadioWNet(inputs=2)`. This script does the same with the
organizers' own `RadioUNet/modules.py` (from their repo, not copied here) and converts the weights to mixlab's
layout with mixlab's export map (`configs/export_map_2ch.json`, applied in reverse).

Why bother: on this data, mixlab's own default init starts with almost every output pixel at zero after the final
ReLU and can stay stuck there. The reference init has a few live pixels and escapes within ~40 steps.

For the extend, `--channels 5` or `7` widens the input to the extra geometry channels. Only the convolutions that
read the raw input grow, by zero weights on the new input slices, so the widened model starts out computing exactly
what the 2-channel reference does.

mixlab names its checkpoint tensors internally, so the script first asks mixlab for a one-step checkpoint of the
same config (`--dataset`) and copies its names and metadata. Those weights are then replaced entirely.

Usage:
  python scripts/make_init.py --config configs/reproduce_stage1.json --dataset shards_415/mixlab.dataset.json \
      --reference-repo lunar-radiomap-challenge --seed 0 --out inits/seed0_2ch.safetensors
"""
from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXPORT_MAP = HERE.parent / "configs" / "export_map_2ch.json"
KEEP_META = ("format", "logical_weights", "model_dim", "name", "seq_len", "task", "vocab_size")
# The convolutions that read the raw input: its first layer and every place it is concatenated back in, in both U-Nets.
WIDENED = ("layer00.0", "conv_up00.0", "conv_up000.0", "Wlayer00.0", "Wconv_up00.0", "Wconv_up000.0")


def read_safetensors(path: Path):
    b = Path(path).read_bytes()
    n = struct.unpack("<Q", b[:8])[0]
    header = json.loads(b[8:8 + n])
    meta = header.pop("__metadata__", {})
    tensors = {}
    for k, v in header.items():
        if v["dtype"] != "F32":
            raise ValueError(f"{k}: unsupported dtype {v['dtype']}")
        s, e = v["data_offsets"]
        tensors[k] = np.frombuffer(b[8 + n + s:8 + n + e], dtype="<f4").reshape(v["shape"])
    return tensors, meta


def write_safetensors(path: Path, tensors: dict, meta: dict):
    header, blobs, off = {}, [], 0
    for k in sorted(tensors):
        a = np.ascontiguousarray(tensors[k], dtype="<f4")
        header[k] = {"dtype": "F32", "shape": list(a.shape), "data_offsets": [off, off + a.nbytes]}
        blobs.append(a.tobytes())
        off += a.nbytes
    header["__metadata__"] = {k: str(v) for k, v in meta.items()}
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    Path(path).write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def reference_state(seed: int, channels: int, reference_repo: Path) -> dict:
    """torch.manual_seed(seed); RadioWNet(inputs=2): the reference init, widened with zero input slices."""
    import torch
    sys.path.insert(0, str(reference_repo / "RadioUNet"))
    from modules import RadioWNet
    torch.manual_seed(seed)
    state = {k: v.detach().cpu().numpy() for k, v in RadioWNet(inputs=2, phase="firstU").state_dict().items()}
    for name in WIDENED:
        w = state[f"{name}.weight"]
        pad = np.zeros((w.shape[0], channels - 2) + w.shape[2:], dtype=w.dtype)
        state[f"{name}.weight"] = np.concatenate([w, pad], axis=1)
    return state


def export_map(channels: int) -> dict:
    """mixlab's 2-channel export map, with the input-reading convolutions' PyTorch shapes widened to C inputs."""
    emap = json.loads(EXPORT_MAP.read_text())
    if emap.get("format") != "mixlab.torch_state_map.v1":
        raise ValueError("unexpected export map format")
    targets = {f"{n}.weight" for n in WIDENED}
    hit = [m for m in emap["mappings"] if m["target"] in targets]
    if len(hit) != len(targets):
        raise ValueError(f"export map has {len(hit)} of {len(targets)} input-reading convolutions")
    for m in hit:
        m["shape"][1] += channels - 2
    return emap


def template(config: Path, dataset: Path, workdir: Path) -> Path:
    """A one-step mixlab checkpoint of `config`: its tensor names and metadata are all we use."""
    c = json.loads(config.read_text())
    t = c["training"]
    t["phases"] = [dict(t["phases"][0], steps=1)] if t.get("phases") else t.get("phases")
    t["val_every_steps"] = 10 ** 9
    t.pop("freeze", None)
    t.pop("allow_tf32", None)  # one step; precision does not matter and an explicit value can clash with the env
    cfg = workdir / "template.json"
    cfg.write_text(json.dumps(c))
    out = workdir / "template.safetensors"
    r = subprocess.run(["mixlab", "-mode", "arch", "-config", str(cfg), "-train", str(dataset),
                        "-safetensors", str(out)], capture_output=True, text=True)
    if r.returncode or not out.exists():
        raise SystemExit(f"mixlab could not write a template checkpoint:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return out


def convert(state: dict, template_path: Path, emap: dict) -> tuple:
    tmpl, meta = read_safetensors(template_path)
    logical = json.loads(meta["logical_weights"])
    tensors = {}
    for m in emap["mappings"]:
        t = state[m["target"]]
        if list(t.shape) != m["shape"]:
            raise ValueError(f"{m['target']}: shape {t.shape} != map {m['shape']}")
        if m["transform"] == "transpose":
            t = np.transpose(t, np.argsort(m["axes"]))
        elif m["transform"] != "identity":
            raise ValueError(f"unknown transform {m['transform']}")
        key = logical[m["source"]]
        if tuple(tmpl[key].shape) != t.shape:
            raise ValueError(f"{key}: converted shape {t.shape} != mixlab's {tmpl[key].shape}; check --channels")
        tensors[key] = t
    if set(tmpl) != set(tensors) or len(state) != len(tensors):
        raise ValueError(f"incomplete conversion: mixlab has {len(tmpl)} tensors, converted {len(tensors)}")
    return tensors, {k: meta[k] for k in KEEP_META if k in meta}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True, help="the stage-1 config you will train")
    p.add_argument("--dataset", required=True, help="its prepared mixlab.dataset.json")
    p.add_argument("--reference-repo", required=True, help="a checkout of the organizers' repo (it reads RadioUNet/modules.py)")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--channels", type=int, default=2, help="input channels the weights read: 2, 5 or 7")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if shutil.which("mixlab") is None:
        raise SystemExit("mixlab is not on PATH")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tpl = template(Path(a.config), Path(a.dataset), Path(tmp))
        tensors, meta = convert(reference_state(a.seed, a.channels, Path(a.reference_repo)), tpl, export_map(a.channels))
    write_safetensors(out, tensors, meta)
    print(out)


if __name__ == "__main__":
    main()

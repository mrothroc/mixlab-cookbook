"""Write short `*_smoke.json` copies of the recipe's configs, for checking the whole chain in a few minutes.

Each copy keeps the model and optimizer and cuts training to two phases of two steps, with one validation at the
end, so `best.safetensors` is written. Scores from these runs are meaningless (high 40s to 50s of dB); the point is
that every command runs.

Usage: python scripts/make_smoke_configs.py      # writes configs/<name>_smoke.json for all four configs
"""
import json
from pathlib import Path

CONFIGS = Path(__file__).resolve().parent.parent / "configs"

for name in ("reproduce_stage1", "reproduce_stage2", "extend_415", "extend_58"):
    c = json.loads((CONFIGS / f"{name}.json").read_text())
    c["name"] += "_smoke"
    c["training"]["phases"] = [dict(p, steps=2) for p in c["training"]["phases"][:2]]
    c["training"]["val_every_steps"] = 4
    (CONFIGS / f"{name}_smoke.json").write_text(json.dumps(c, indent=1) + "\n")
    print(CONFIGS / f"{name}_smoke.json")

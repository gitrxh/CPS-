"""Freeze task-family stratified pilot splits before observing policy outcomes."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gnrl.alfworld_env import list_games

ROOT = Path(__file__).resolve().parents[1]
DATA = "/home/jsy/zhangranxu-adrs-week-20260929/data/alfworld/json_2.1.1"
OUT = ROOT / "data/manifests"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261012

def stratified(split, per_family=None):
    groups = defaultdict(list)
    for p in list_games(DATA, split):
        kind = json.loads((p.parent / "traj_data.json").read_text())["task_type"]
        groups[kind].append(str(p))
    for kind, values in sorted(groups.items()):
        random.Random(f"{SEED}:{split}:{kind}").shuffle(values)
    chosen = []
    for kind, values in sorted(groups.items()):
        chosen.extend(values if per_family is None else values[:per_family])
    return chosen, {kind: len(values) for kind, values in groups.items()}

for name, split, n in [("train120", "train", 20), ("h1_pilot24", "train", 4),
                       ("dev24", "valid_seen", 4), ("unseen_locked", "valid_unseen", None)]:
    games, available = stratified(split, n)
    manifest = {"version": 1, "name": name, "split": split, "selection_seed": SEED,
                "per_family": n, "available_by_family": available, "games": games,
                "game_sha256": {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in games}}
    manifest["hash"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    path = OUT / f"{name}.json"
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite frozen manifest {path}")
    path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"name": name, "count": len(games), "hash": manifest["hash"]}))

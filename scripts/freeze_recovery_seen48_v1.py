"""Freeze a new research split from metadata only; never constructs an environment."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gnrl.alfworld_env import TASK_TYPES, list_games


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def relative_game(game):
    return str(game).split("/json_2.1.1/", 1)[1]


def metadata(game):
    game = Path(game).resolve()
    meta = game.parent / "traj_data.json"
    x = read(meta)
    params = x["pddl_params"]
    # Original metadata identities, never natural-language task matching.
    recipe = [x["task_type"], params.get("object_target", ""),
              params.get("parent_target", ""), params.get("mrecep_target", ""),
              params.get("toggle_target", ""), bool(params.get("object_sliced", False)),
              x["scene"]["floor_plan"]]
    return {"gamefile": str(game), "relative_game": relative_game(game),
            "family": x["task_type"], "recipe_key": recipe,
            "scene": x["scene"]["floor_plan"],
            "game_sha256": sha(game), "metadata_sha256": sha(meta)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    candidate = read(a.candidate)
    if candidate["new_research_evaluation"]["tasks_per_model"] != 48:
        raise ValueError("Only the fixed seen48 protocol is supported")
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    # Inspect only game membership in prior episode records; no outcome field is used.
    registries, old_games = [], set()
    for path in sorted((ROOT / "outputs").rglob("episodes.jsonl")):
        games = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                game = str(Path(json.loads(line)["gamefile"]).resolve())
                if "/valid_unseen/" in game:
                    raise ValueError("Prior unseen evaluation found: exposure assumption invalid")
                games.append(game)
        registries.append({"path": path.relative_to(ROOT).as_posix(),
                           "sha256": sha(path), "ordered_gamefiles": games})
        old_games.update(games)
    dev = ROOT / "data/manifests/dev24.json"
    train = ROOT / "data/manifests/train120.json"
    old_games.update(str(Path(g).resolve()) for g in read(dev)["games"])
    train_games = {str(Path(g).resolve()) for g in read(train)["games"]}
    old_meta = [metadata(g) for g in sorted(old_games)]
    old_keys = {tuple(x["recipe_key"]) for x in old_meta}
    old_scenes = {x["scene"] for x in old_meta}
    train_scenes = {metadata(g)["scene"] for g in train_games}
    inventory, by_family = [], defaultdict(list)
    for game in list_games(a.data_root, "valid_seen"):
        x = metadata(game)
        reasons = []
        if x["gamefile"] in old_games:
            reasons.append("previous_evaluation_identity")
        if tuple(x["recipe_key"]) in old_keys:
            reasons.append("previous_evaluation_recipe")
        if x["gamefile"] in train_games:
            reasons.append("training_identity")
        x["exclusion_reasons"] = reasons
        x["selection_key"] = hashlib.sha256(
            ("recovery_projection_v1_research_seen48|" + x["relative_game"]).encode()).hexdigest()
        inventory.append(x)
        if not reasons:
            by_family[x["family"]].append(x)
    selected = []
    for family in sorted(TASK_TYPES):
        selected.extend(sorted(by_family[family], key=lambda x: (x["selection_key"], x["relative_game"]))[:8])
    available = {f: len(by_family[f]) for f in sorted(TASK_TYPES)}
    complete = all(n >= 8 for n in available.values())
    report = {
        "schema_version": 1, "kind": "metadata_only_seen48_freeze",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete": complete, "stop_reason": "complete" if complete else "insufficient_fixed_family_quota",
        "candidate_sha256": sha(a.candidate), "producer_sha256": sha(__file__),
        "filter_source_sha256": sha(ROOT / "gnrl/alfworld_env.py"),
        "train_manifest_sha256": sha(train), "dev_manifest_sha256": sha(dev),
        "previous_evaluation_registries": registries,
        "previous_unique_task_count": len(old_games), "previous_task_metadata": old_meta,
        "supported_solvable_universe": inventory, "available_by_family": available,
        "recipe_key_fields": ["task_type", "object_target", "parent_target", "mrecep_target",
                              "toggle_target", "object_sliced", "scene.floor_plan"],
        "selected": selected if complete else [],
        "selected_scene_counts": dict(Counter(x["scene"] for x in selected)) if complete else {},
        "scene_overlap_with_train": sorted({x["scene"] for x in selected} & train_scenes) if complete else [],
        "scene_overlap_with_previous_eval": sorted({x["scene"] for x in selected} & old_scenes) if complete else [],
        "new_environment_calls": 0, "new_model_calls": 0, "unseen_metadata_or_outcomes_read": False,
        "performance_fields_accessed": False,
        "no_outcome_claim": "Selection used only identities and original metadata, with old file bytes hashed for provenance.",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if complete:
        manifest = {"version": 1, "name": "recovery_research_seen48_v1", "split": "valid_seen",
                    "per_family": 8, "games": [x["gamefile"] for x in selected],
                    "freeze_report_sha256": sha(out / "report.json")}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"complete": complete, "available_by_family": available,
                      "selected_count": len(selected) if complete else 0,
                      "previous_unique_task_count": len(old_games), "output": str(out)}))
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())

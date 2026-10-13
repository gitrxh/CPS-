"""Fixed60 canonical parser/witness preflight; resets only, no candidate search."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "": raise RuntimeError("CPU only")
    if args.output.exists(): raise RuntimeError("Audit output must be new")
    from gnrl.alfworld_env import AlfworldReplayEnv
    from gnrl import successful_path_pools_v2 as m
    config = json.loads((ROOT / "configs/successful_path_selection_candidate_v1.json").read_text())
    native = ROOT / "data/planner_train120_raw.jsonl"
    manifest = json.loads((ROOT / "data/manifests/train120.json").read_text())
    source = {json.loads(line)["gamefile"]: line for line in native.read_bytes().splitlines(keepends=True)}
    games = config["fixed_training_cohort"]["frozen_gamefiles"]
    records = []; steps = 0; resets = 0; started = time.monotonic()
    for i, gamefile in enumerate(games):
        env = None
        record = {"index": i, "gamefile": gamefile, "complete": False}
        try:
            line = source[gamefile]; row = json.loads(line)
            if not row["success"] or not row["replay_verified"]: raise RuntimeError("Invalid canonical source")
            game_raw = Path(gamefile).read_bytes()
            if hashlib.sha256(game_raw).hexdigest() != manifest["game_sha256"][gamefile]: raise RuntimeError("Game changed")
            goal, costs, hashes = m.audit_game(json.loads(game_raw), json.loads(Path(gamefile).with_name("traj_data.json").read_text()))
            stratum = m._check_trace(row["transitions"], costs)
            env = AlfworldReplayEnv(gamefile, seed=row["seed"], max_steps=100, expert_type=None)
            resets += 1
            state = env.reset()
            if state.canonical_state != row["transitions"][0]["state_before"] or state.observation != row["initial_observation"]:
                raise RuntimeError("Fresh initial state/observation differs")
            names = m.entity_names(env)
            assignment = m.derive_canonical_assignment(row["transitions"], goal, names)
            if not m.witness(row["transitions"][-1]["state_after"], goal, assignment): raise RuntimeError("Original goal witness missing")
            record.update(complete=True, family=goal.family, assignment=assignment, canonical_length_cost=list(stratum),
                actions=[t["action"] for t in row["transitions"]], source_line_sha256=hashlib.sha256(line).hexdigest(),
                hashes=hashes, terminal_semantics_audit=m.terminal_semantics_audit(row["transitions"], goal, assignment),
                entity_names=names, entity_names_digest=m.digest(names))
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
        finally:
            if env is not None:
                steps += env.interactions
                try: env.close()
                except Exception as error: record.update(complete=False, close_error=repr(error))
        records.append(record)
        print(json.dumps({k: record[k] for k in ("index", "complete", "error") if k in record}), flush=True)
    result = {"kind": "canonical_assignment_preflight_v2", "complete": len(records) == 60 and all(r["complete"] for r in records),
        "records": records, "actual_environment_steps": steps, "actual_reset_calls_started": resets,
        "candidate_search_attempts": 0, "gpu_jobs": 0, "wall_seconds": time.monotonic() - started,
        "source120_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
        "module_sha256": hashlib.sha256((ROOT / "gnrl/successful_path_pools_v2.py").read_bytes()).hexdigest(),
        "interpretation": "Existing successful canonical traces were audited; fresh resets provide names/initial states only. No candidate search or new full replay."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream: json.dump(result, stream, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)
    return 0 if result["complete"] else 1


if __name__ == "__main__": raise SystemExit(main())

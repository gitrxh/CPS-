"""Read actual rollout evidence and paired pilot effects without selecting checkpoints."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random


def read_arm(folder):
    folder = Path(folder)
    rows = [json.loads(line) for line in (folder / "episodes.jsonl").read_text().splitlines() if line.strip()]
    keys = [r["gamefile"] for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate evaluation tasks")
    meta = json.loads((folder / "summary.json").read_text())
    if not meta["complete"] or meta["technical_errors"]:
        raise ValueError("Evaluation incomplete or technically failed: " + str(folder))
    if (len(rows) != meta["planned_episodes"] or len(rows) != meta["attempted_episodes"]
            or any("error" in r for r in rows)
            or sum(bool(r["success"]) for r in rows) != meta["successes"]):
        raise ValueError("Evaluation records do not match the complete summary")
    actions = [t for r in rows for t in r["transitions"]]
    invalid = sum(not t["policy"].get("is_admissible", t["action"] in t["admissible_commands"]) for t in actions)
    families = {}
    for r in rows:
        family = Path(r["gamefile"]).parents[1].name.split("-", 1)[0]
        families.setdefault(family, Counter())
        families[family]["tasks"] += 1
        families[family]["successes"] += int(r["success"])
    summary = {"tasks": len(rows), "successes": sum(r["success"] for r in rows),
               "success_rate": sum(r["success"] for r in rows)/len(rows),
               "environment_steps": len(actions), "invalid_actions": invalid,
               "invalid_action_rate": invalid/len(actions) if actions else 0,
               "by_family": {k:dict(v) for k,v in families.items()}}
    return {r["gamefile"]:r for r in rows}, summary


def compare(left, right):
    if set(left) != set(right):
        raise ValueError("Arms evaluated different task identities")
    if any(left[k]["seed"] != right[k]["seed"] for k in left):
        raise ValueError("Arms evaluated different task seeds")
    deltas = [int(right[k]["success"])-int(left[k]["success"]) for k in sorted(left)]
    gains, losses = deltas.count(1), deltas.count(-1)
    rng = random.Random(20261012)
    samples = sorted(sum(rng.choices(deltas, k=len(deltas)))/len(deltas) for _ in range(10000))
    n_discordant = gains+losses
    p_exact = min(1.0, 2*sum(math.comb(n_discordant,i) for i in range(min(gains,losses)+1))/(2**n_discordant)) if n_discordant else 1.0
    return {"right_minus_left_pp": 100*sum(deltas)/len(deltas), "gained_tasks": gains, "lost_tasks": losses,
            "paired_task_bootstrap95_pp": [100*samples[249],100*samples[9749]],
            "mcnemar_exact_two_sided_p":p_exact,
            "scope":"Single-seed development-set pilot. No confirmatory, multi-seed, or unseen-generalization claim."}


p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--arm", nargs=2, action="append", required=True, metavar=("NAME","FOLDER"))
p.add_argument("--output", required=True)
a=p.parse_args()
arms={}
report={"arms":{},"comparisons":{},"effect_confirmed":False}
for name,folder in a.arm:
    arms[name],report["arms"][name]=read_arm(folder)
for left,right in [("base","expert"),("base","canonical"),("expert","canonical"),
                   ("canonical","uniform"),("canonical","brp"),("uniform","brp")]:
    if left in arms and right in arms:
        report["comparisons"][right+"_vs_"+left]=compare(arms[left],arms[right])
Path(a.output).write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))

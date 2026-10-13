"""Construct identical-task canonical baseline and branch pools for phase1."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gnrl.macro_project import candidate_id, validate_pool


def rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--planner", required=True)
p.add_argument("--blocks", required=True)
p.add_argument("--manifest", required=True)
p.add_argument("--output", required=True)
a=p.parse_args()
manifest=json.loads(Path(a.manifest).read_text())
expected=manifest["games"]
if len(expected)!=len(set(expected)):
    raise ValueError("Duplicate tasks in frozen manifest")
planner_rows=[r for r in rows(a.planner) if r.get("kind")=="trajectory"]
planner={r["gamefile"]:r for r in planner_rows}
if len(planner)!=len(planner_rows):
    raise ValueError("Duplicate planner task rows")
blocks_list=rows(a.blocks)
blocks={r["gamefile"]:r for r in blocks_list}
if len(blocks)!=len(blocks_list):
    raise ValueError("Duplicate block task")
if set(planner)!=set(expected):
    raise ValueError("Native planner must cover the entire frozen120-task manifest before phase1 assembly")
if not set(blocks)<=set(expected):
    raise ValueError("Unexpected block tasks outside fixed training manifest")
out=Path(a.output)
out.mkdir(parents=True,exist_ok=False)
counts=Counter()
with (out/"candidate_pools.jsonl").open("w") as pool_stream,(out/"canonical_baseline.jsonl").open("w") as baseline:
    for game in expected:
        source=planner[game]
        if not source.get("success") or not source.get("replay_verified") or source["split"]!="train":
            raise ValueError("Planner source is not replay-verified successful train data")
        is_two=Path(game).parents[1].name.startswith("pick_two_obj_and_place-")
        if is_two and game not in blocks:
            raise ValueError("Two-object task missing canonical block pool")
        if game in blocks:
            pool=blocks[game]
            if pool["seed"]!=source["seed"]:
                raise ValueError("Seed mismatch across source/blocks")
            pool_original=validate_pool(pool)
            if (pool["initial_observation"]!=source["initial_observation"] or
                pool_original["transitions"][0]["state_before"]!=source["transitions"][0]["state_before"]):
                raise ValueError("Initial observation or initial canonical state differs across source/blocks")
            counts["two_object_tasks"]+=1
            counts["two_object_tasks_with_alternatives"]+=int(len(pool["candidates"])>1)
        else:
            cid=candidate_id([t["action"] for t in source["transitions"]])
            pool={"kind":"macro_candidate_pool","schema_version":1,"gamefile":game,"seed":source["seed"],
                  "split":"train","task":source["task"],"initial_observation":source["initial_observation"],
                  "success":True,"original_candidate_id":cid,
                  "candidates":[{"candidate_id":cid,"origin":"native_planner_identity",
                                 "transitions":source["transitions"],"evidence":{"full_replay_verified":True,"terminal_success":True}}]}
        original=validate_pool(pool)
        digest=hashlib.sha256(json.dumps(pool,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        row={"kind":"trajectory","gamefile":game,"seed":source["seed"],"split":"train",
             "task":pool["task"],"initial_observation":pool["initial_observation"],
             "transitions":original["transitions"],"success":True,"reset_replay_verified":True,"replay_verified":True,
             "selection":{"arm":"canonical_baseline","pool_sha256":digest,"candidate_id":pool["original_candidate_id"],
                          "pool_size":len(pool["candidates"]),"method":"fixed native planner/canonical recipe order"}}
        pool_stream.write(json.dumps(pool)+"\n")
        baseline.write(json.dumps(row)+"\n")
        counts["tasks"]+=1
        counts["actions"]+=len(original["transitions"])
summary={"counts":dict(counts),"source_sha256":{name:hashlib.sha256(Path(path).read_bytes()).hexdigest() for name,path in [("planner",a.planner),("blocks",a.blocks),("manifest",a.manifest)]},
         "gate_pass":counts["two_object_tasks_with_alternatives"]>=10,
         "scope":"Phase1 matches canonical baseline, uniform, and learner selection task-by-task. Raw-handcoded versus canonical gains are not BRP gains."}
(out/"assembly_summary.json").write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))

"""Explicitly adopted recovery stages: collect, oracle, or replay.

No controller, scheduler, training, scoring, diagnostic continuation or held-out
evaluation is implemented here. Workers never create adoption receipts. Imports
are inert. A future controller must issue the exact stage contract below and
verify runtime/model inventories before/after each stage.

Stage adoption JSON:
 schema_version=1, kind=recovery_projection_stage_adoption, adopted=true,
 adopted_by=root_after_independent_GPT_6_Astra_review, stage=collect|oracle|replay,
 candidate=<relative JSON>, candidate_sha256, physical_gpu=2,
 source_inventory=<all current gnrl/scripts/tests .py paths>,
 file_sha256={<relative source/input/review/checkpoint paths>:SHA256},
 original_file_sha256={<absolute selected game/metadata paths>:SHA256},
 implementation_review=<relative JSON with recovery_collection_review_passed>,
 output=<new outputs/recovery_projection_... directory>,
 previous_cost_json=null for collect or <hash-bound previous cost_progress.json>,
 input_records=null for collect or <hash-bound prior stage records.jsonl>,
 input_summary=null for collect or <hash-bound complete prior summary.json>,
 global_deadline_unix=<controller's fixed absolute deadline>,
 max_wall_seconds=<no more than stage ceiling>.

Each output contains config.json, records.jsonl, events.jsonl,
cost_progress.json and summary.json. Successful summary has complete=true and
record_count60 (collect) or180 (oracle/replay). Costs are cumulative from the
previous hash-bound ledger; the controller must serialize this ledger lineage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gnrl import recovery_projection_v1 as m

MODEL = "/home/jsy/zhangranxu-adrs-week-20260929/models/Qwen2.5-3B-Instruct"
ADAPTER = "checkpoints/successful_path_canonical_seed42_v1/final"
MODULE = "gnrl/recovery_projection_v1.py"
RUNNER = "scripts/collect_recovery_projection_v1.py"
TEST = "tests/test_recovery_projection_v1.py"
STAGE_LIMITS = {
    "collect": {"environment_steps_started": 6000, "resets_started": 60, "generated_tokens": 384000},
    "oracle": {"environment_steps_started": 144000, "resets_started": 1440, "generated_tokens": 0},
    "replay": {"environment_steps_started": 24480, "resets_started": 1440, "generated_tokens": 0},
}
STAGE_WALL = {"collect": 7200, "oracle": 43200, "replay": 7200}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_path(name, root=ROOT):
    p = Path(name)
    m.require(isinstance(name, str) and not p.is_absolute() and ".." not in p.parts
              and p.as_posix() == name, "Expected normalized relative artifact path")
    p = (root / name).resolve()
    m.require(p.is_relative_to(root.resolve()), "Artifact escapes workspace")
    return p


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def source_inventory(root=ROOT):
    return sorted(p.relative_to(root).as_posix() for sub in ("gnrl", "scripts", "tests")
                  for p in (root / sub).rglob("*.py"))


def validate_request(path, root=ROOT):
    request = read_json(path)
    for k, v in {"schema_version": 1, "kind": "recovery_projection_stage_adoption", "adopted": True,
                 "adopted_by": "root_after_independent_GPT_6_Astra_review", "physical_gpu": 2}.items():
        m.require(type(request.get(k)) is type(v) and request[k] == v, "Stage adoption refused: " + k)
    stage = request["stage"]
    m.require(stage in STAGE_LIMITS, "Unknown recovery stage")
    m.require(request["source_inventory"] == source_inventory(root), "Stage source inventory differs")
    hashes = request["file_sha256"]
    required = set(request["source_inventory"]) | {request["candidate"], request["implementation_review"],
        MODULE, RUNNER, TEST, "gnrl/alfworld_env.py", "gnrl/policy.py",
        "data/manifests/train120.json", "configs/successful_path_selection_candidate_v1.json"}
    required |= {ADAPTER + "/adapter_config.json", ADAPTER + "/training_state.json"}
    m.require(any(ADAPTER + "/" + p in hashes for p in ("adapter_model.safetensors", "adapter_model.bin")),
              "Original canonical adapter weights must be bound")
    if stage == "collect":
        m.require(all(request.get(k) is None for k in ("previous_cost_json", "input_records", "input_summary")),
                  "Collection cannot inherit another collection or resume")
    else:
        required |= {request["previous_cost_json"], request["input_records"], request["input_summary"]}
    m.require(required <= hashes.keys(), "Missing stage input/source/checkpoint bindings")
    for name, value in hashes.items():
        m.require(sha(safe_path(name, root)) == value, "Changed stage-bound file: " + name)
    m.require(request["candidate_sha256"] == hashes[request["candidate"]], "Candidate hash differs")
    review = read_json(safe_path(request["implementation_review"], root))
    m.require(review.get("recovery_collection_review_passed") is True and
              review.get("module_sha256") == hashes[MODULE] and review.get("runner_sha256") == hashes[RUNNER],
              "Missing independent review for exact collection implementation")
    candidate = read_json(safe_path(request["candidate"], root))
    games = candidate["training_cohort"]["frozen_gamefiles"]
    train = read_json(root / "data/manifests/train120.json")["games"]
    old = read_json(root / "configs/successful_path_selection_candidate_v1.json")["fixed_training_cohort"]["frozen_gamefiles"]
    m.require(games == [g for g in train if g not in set(old)] and len(games) == len(set(games)) == 60,
              "Recovery cohort differs from exact complementary60")
    originals = request["original_file_sha256"]
    for game in games:
        m.relative_game(game)
        for original in (game, str(Path(game).with_name("traj_data.json"))):
            m.require(original in originals and sha(original) == originals[original], "Changed/unbound original game or metadata")
    m.require(candidate["frozen_start_policy"]["adapter"] == ADAPTER and
              candidate["frozen_start_policy"]["base"] == MODEL and
              candidate["finite_budgets"]["physical_gpu"] == 2, "Candidate model/GPU differs")
    budgets = candidate["finite_budgets"]
    limits = {"environment_steps_started": budgets["environment_step_upper"]["hard_global_step_calls_started"],
              "resets_started": budgets["max_total_reset_calls"],
              "generated_tokens": budgets["max_total_generated_tokens"]}
    m.require(limits == m.GLOBAL_LIMITS and
              budgets["global_wall_seconds_including_waits_verification_cleanup"] == 171840,
              "Candidate main-study cumulative budgets differ")
    m.require(type(request["max_wall_seconds"]) in (int, float) and
              math.isfinite(request["max_wall_seconds"]) and 0 < request["max_wall_seconds"] <= STAGE_WALL[stage],
              "Stage wall ceiling changed")
    remaining = request["global_deadline_unix"] - time.time()
    m.require(math.isfinite(remaining) and 30 < remaining <= 171840, "Invalid/exhausted global deadline")
    output = safe_path(request["output"], root)
    m.require(request["output"].startswith("outputs/recovery_projection_") and not output.exists(),
              "Stage output exists or wrong namespace")
    initial = None
    if stage != "collect":
        previous = read_json(safe_path(request["previous_cost_json"], root))
        m.require(previous["kind"] == "recovery_cost_ledger" and previous["active_call"] is None and
                  previous["unknown_generation_after_error"] is False, "Cannot continue an incomplete cost ledger")
        initial = previous["totals"]
        summary = read_json(safe_path(request["input_summary"], root))
        expected_input = "collect" if stage == "oracle" else "oracle"
        m.require(summary.get("complete") is True and summary.get("stage") == expected_input and
                  summary.get("records_sha256") == hashes[request["input_records"]], "Incomplete/wrong prior artifact")
        m.require(summary["costs"]["totals"] == initial and previous["global_limits"] == limits,
                  "Prior complete artifact and inherited cumulative costs differ")
    return request, games, output, initial


def input_rows(request, games):
    if request["stage"] == "collect":
        return []
    path = safe_path(request["input_records"])
    raw = path.read_bytes()
    m.require(raw.endswith(b"\n"), "Truncated prior JSONL")
    rows = [json.loads(line) for line in raw.splitlines()]
    if request["stage"] == "oracle":
        m.require(len(rows) == 60 and [r["gamefile"] for r in rows] == games, "Collection identity/order differs")
        anchors = []
        for row in rows:
            m.validate_episode(row)
            anchors.extend(row["anchors"])
        return anchors
    m.require(len(rows) == 180 and [(r["anchor"]["gamefile"], r["anchor"]["after_actions"]) for r in rows]
              == [(g, k) for g in games for k in m.ANCHOR_STEPS], "Pool anchor identities/order differ")
    for row in rows:
        m.validate_pool(row)
    return rows


def run_stage(request_path):
    request_hash = sha(request_path)
    request, games, output, initial_costs = validate_request(request_path)
    items = input_rows(request, games)
    stage = request["stage"]
    if stage != "collect":
        m.require(not os.environ.get("CUDA_VISIBLE_DEVICES"), "CPU oracle/replay stage must hide CUDA")
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "config.json", {"request_sha256": request_hash, "request": request,
                "worker_executable": sys.executable, "model": MODEL, "adapter": ADAPTER,
                "protocol": "alfworld-causal-v1", "environment_seed": m.SEED})
    deadline = time.monotonic() + min(request["max_wall_seconds"], request["global_deadline_unix"] - time.time() - 30)
    ledger = m.BudgetLedger(stage_limits=STAGE_LIMITS[stage], deadline=deadline, initial_totals=initial_costs,
                           global_limits=m.GLOBAL_LIMITS,
                           persist=lambda row: atomic_json(output / "cost_progress.json", row))
    ledger.flush()
    count = 0
    summary = {"schema_version": 1, "kind": "recovery_stage_summary", "stage": stage, "complete": False,
               "planned_records": 60 if stage == "collect" else 180, "record_count": 0,
               "request_sha256": request_hash, "retry_authorized": False}
    with (output / "events.jsonl").open("x", encoding="utf-8", buffering=1) as events, \
         (output / "records.jsonl").open("x", encoding="utf-8", buffering=1) as records:
        def write(stream, row):
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()

        def emit(row):
            write(events, row)

        def factory(gamefile, expert_type):
            from gnrl.alfworld_env import AlfworldReplayEnv
            return AlfworldReplayEnv(gamefile, seed=m.SEED, max_steps=m.MAX_STEPS, expert_type=expert_type)

        try:
            ledger.check()
            if stage == "collect":
                from gnrl.policy import HFCausalPolicy
                policy = HFCausalPolicy(MODEL, str(ROOT / ADAPTER), max_context_tokens=2048, max_action_tokens=64)
                ledger.check()
                for game in games:
                    env = factory(game, None)
                    try:
                        row = m.collect_learner_task(env, policy, game, ledger, emit)
                    finally:
                        env.close()
                    write(records, row)
                    count += 1
                    print(json.dumps({"stage": stage, "completed_records": count}), flush=True)
            else:
                operation = m.certify_anchor if stage == "oracle" else m.replay_certificates
                for item in items:
                    ledger.check()
                    row = operation(factory, item, ledger, emit)
                    write(records, row)
                    count += 1
                    print(json.dumps({"stage": stage, "completed_records": count}), flush=True)
            m.require(count == summary["planned_records"], "Incomplete stage record count")
            m.require(sha(request_path) == request_hash, "Stage request changed during execution")
            for name, value in request["file_sha256"].items():
                ledger.check()
                m.require(sha(safe_path(name)) == value, "Stage-bound file changed: " + name)
            summary.update(complete=True, records_sha256=sha(output / "records.jsonl"))
        except BaseException as error:
            summary.update(error=f"{type(error).__name__}: {error}", technical_incomplete=True)
            raise
        finally:
            summary.update(record_count=count, costs=ledger.snapshot())
            ledger.flush()
            atomic_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adoption-json", required=True, type=Path)
    args = parser.parse_args(argv)
    m.require(sys.platform.startswith("linux") and ROOT.resolve() == Path("/home/jsy/GNRL_20261012").resolve(),
              "Recovery workers run only in chemistry workspace after adoption")
    os.chdir(ROOT)
    run_stage(args.adoption_json)


if __name__ == "__main__":
    main()

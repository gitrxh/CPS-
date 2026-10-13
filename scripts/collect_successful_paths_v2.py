"""Adoption-gated, CPU-only frozen60 successful-path feasibility runner.

This file does not create an adoption or start any work when imported. Required
future adoption JSON (not supplied here): schema_version=1,
kind='successful_path_cpu_feasibility_adoption', adopted=true, cpu_only=true,
scope='frozen60_successful_path_feasibility_only', phase1_complete=true,
no_directional_benefit=true, phase1_successes={canonical,uniform,brp}, exact
candidate_config_sha256/module_sha256/runner_sha256, and file_sha256 containing
every path returned by required_adoption_hash_paths(). The decision is checked
against all three COMPLETE phase1 training/evaluation artifacts, never inferred
from a technical failure. No model, training, new evaluation, or GPU is invoked.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG = "configs/successful_path_selection_candidate_v1.json"
MODULE = "gnrl/successful_path_pools_v2.py"
RUNNER = "scripts/collect_successful_paths_v2.py"
TRAIN_MANIFEST = "data/manifests/train120.json"
DEV_MANIFEST = "data/manifests/dev24.json"
SOURCE = "data/planner_train120_raw.jsonl"
SOURCE_SUMMARY = "data/planner_train120_raw.summary.json"
PRIOR_FAILURE = "outputs/successful_paths_cpu_v1/summary.json"
PREFLIGHT = "outputs/successful_path_canonical_preflight_v2.json"
ARMS = ("canonical", "uniform", "brp")
ARM_DATA = {"canonical": "outputs/phase1_assembly_v1/canonical_baseline.jsonl",
            "uniform": "outputs/phase1_selection_v1/uniform_valid.jsonl",
            "brp": "outputs/phase1_selection_v1/learner_selected.jsonl"}
MODEL = "/home/jsy/zhangranxu-adrs-week-20260929/models/Qwen2.5-3B-Instruct"
TRAIN_PROTOCOL = {"model": MODEL, "max_steps": 100, "batch_size": 1,
                  "gradient_accumulation": 8, "learning_rate": 2e-4,
                  "lora_rank": 16, "lora_alpha": 32, "lora_dropout": 0.05,
                  "seed": 42, "max_context_tokens": 2048, "max_action_tokens": 64}
EVAL_PROTOCOL = {"model": MODEL, "selection": "generate", "max_steps": 100,
                 "seed": 20261012, "max_context_tokens": 2048,
                 "max_action_tokens": 64, "protocol": "alfworld-causal-v1"}
SOURCE_FILES = (CONFIG, MODULE, RUNNER, "scripts/audit_successful_path_canonical_v2.py", "gnrl/alfworld_env.py", "gnrl/alfworld_probe.py",
                "gnrl/two_object_blocks.py", "gnrl/macro_project.py", "gnrl/policy.py")
FIXED_BUDGETS = {"max_tasks": 60, "max_pool_members_per_task": 8,
                 "max_new_recipe_attempts_total": 420,
                 "max_environment_steps_including_all_replay": 100000,
                 "max_cpu_wall_seconds": 3600, "max_recipe_actions": 24,
                 "episode_max_steps": 100, "gpu_jobs": 0}


def require(value, reason):
    if not value:
        raise RuntimeError(reason)


def utc():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): result.update(chunk)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def dump_line(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    stream.flush()


def atomic_json(path, row):
    path = Path(path)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
    pending.replace(path)


def arm_paths(arm):
    return {"training_summary": f"checkpoints/phase1_{arm}_seed42_v1/summary.json",
            "training_config": f"checkpoints/phase1_{arm}_seed42_v1/config.json",
            "evaluation_summary": f"outputs/eval_phase1_{arm}_dev24_v1/summary.json",
            "evaluation_config": f"outputs/eval_phase1_{arm}_dev24_v1/config.json",
            "evaluation_episodes": f"outputs/eval_phase1_{arm}_dev24_v1/episodes.jsonl"}


def required_adoption_hash_paths():
    return list(SOURCE_FILES) + [TRAIN_MANIFEST, DEV_MANIFEST, SOURCE, SOURCE_SUMMARY, PRIOR_FAILURE, PREFLIGHT] + list(ARM_DATA.values()) + [
        p for arm in ARMS for p in arm_paths(arm).values()]


def validate_adoption(path, root=ROOT):
    """Read phase1 outcomes ONLY after the explicit CPU adoption exists."""
    path, root = Path(path), Path(root)
    require(path.is_file(), "Explicit --adoption-json file is required; candidate config is not authorization")
    adopted = read_json(path)
    for key, expected in {"schema_version": 1, "kind": "successful_path_cpu_feasibility_adoption",
                          "adopted": True, "cpu_only": True,
                          "scope": "frozen60_successful_path_feasibility_only",
                          "phase1_complete": True, "no_directional_benefit": True}.items():
        require(type(adopted.get(key)) is type(expected) and adopted[key] == expected,
                "Adoption gate missing/incorrect: " + key)
    for field, name in (("candidate_config_sha256", CONFIG), ("module_sha256", MODULE), ("runner_sha256", RUNNER)):
        require(adopted.get(field) == file_hash(root / name), "Adoption source binding differs: " + field)
    hashes = adopted.get("file_sha256", {})
    for name in required_adoption_hash_paths():
        require(name in hashes and hashes[name] == file_hash(root / name), "Missing/changed adopted input: " + name)
    prior = read_json(root / PRIOR_FAILURE); preflight = read_json(root / PREFLIGHT)
    require(prior.get("complete") is False and prior.get("coverage_gate_passed") is False
            and prior.get("environment_step_calls_started") == 507, "Missing immutable v1 technical-failure cost")
    require(preflight.get("complete") is True and len(preflight.get("records", [])) == 60
            and all(r.get("complete") is True for r in preflight["records"])
            and preflight.get("module_sha256") == hashes[MODULE]
            and preflight.get("actual_environment_steps") == 0
            and preflight.get("candidate_search_attempts") == 0, "Canonical60 v2 preflight incomplete")
    config = read_json(root / CONFIG)
    require([r.get("gamefile") for r in preflight["records"]] == config["fixed_training_cohort"]["frozen_gamefiles"],
            "Preflight differs from frozen60 order")
    for key, value in FIXED_BUDGETS.items():
        require(config["feasibility_budget_and_gate"].get(key) == value, "Frozen budget differs: " + key)
    enumeration = config["finite_assignment_enumeration"]
    require(enumeration["max_new_recipe_attempts_per_task"] == 7 and
            enumeration["max_retained_candidates_per_task_including_original"] == 8 and
            enumeration["max_assignment_tuples_considered_per_task"] == 4096, "Proposal limits differ")
    games = read_json(root / DEV_MANIFEST)
    require(games.get("split") == "valid_seen", "Phase1 evaluation manifest is not frozen valid_seen")
    expected = games["games"]
    require(len(expected) == len(set(expected)) == 24, "Phase1 dev cohort is not complete24")
    counts = {}
    for arm in ARMS:
        files = arm_paths(arm)
        training = read_json(root / files["training_summary"])
        train_config = read_json(root / files["training_config"])
        summary = read_json(root / files["evaluation_summary"])
        evaluation = read_json(root / files["evaluation_config"])
        episodes = [json.loads(s) for s in (root / files["evaluation_episodes"]).read_text().splitlines() if s.strip()]
        require(training.get("matched_step_budget_complete") is True and training.get("optimizer_steps") == 100
                and training.get("requested_steps") == 100 and training.get("stop_reason") == "max_steps",
                "Phase1 training incomplete: " + arm)
        require(all(train_config.get(k) == v for k, v in TRAIN_PROTOCOL.items()), "Phase1 training protocol differs: " + arm)
        require(Path(train_config.get("data", "")).as_posix().endswith(ARM_DATA[arm])
                and train_config.get("data_sha256") == hashes[ARM_DATA[arm]], "Phase1 training data differs: " + arm)
        require(summary.get("complete") is True and summary.get("technical_errors") == 0
                and summary.get("planned_episodes") == 24 and summary.get("attempted_episodes") == 24,
                "Technical/incomplete phase1 evaluation cannot activate candidate: " + arm)
        require([r["gamefile"] for r in episodes] == expected and len(episodes) == 24
                and all(r.get("seed") == 20261012 and "error" not in r and type(r.get("success")) is bool
                        and r.get("stop_reason") in {"environment_done", "max_steps", "no_admissible_commands"}
                        for r in episodes), "Phase1 episode identities/results incomplete: " + arm)
        successes = sum(r["success"] for r in episodes)
        require(successes == summary.get("successes"), "Phase1 successes differ from actual episode records: " + arm)
        require(evaluation.get("manifest_sha256") == hashes[DEV_MANIFEST]
                and all(evaluation.get(k) == v for k, v in EVAL_PROTOCOL.items()),
                "Phase1 evaluation protocol differs: " + arm)
        require(Path(evaluation.get("adapter", "")).as_posix().endswith(f"checkpoints/phase1_{arm}_seed42_v1/final"),
                "Phase1 evaluation adapter does not identify its arm")
        counts[arm] = successes
    require(counts == adopted.get("phase1_successes"), "Adoption interpretation differs from verified phase1 results")
    require(not (counts["brp"] > counts["canonical"] and counts["brp"] > counts["uniform"]),
            "Learner strictly exceeds both controls; registered fallback gate is not met")
    return config, {"adoption_path": str(path.resolve()), "adoption_sha256": file_hash(path),
                    "phase1_successes": counts, "phase1_complete": True,
                    "activation_criterion": "learner_not_strictly_greater_than_both_controls",
                    "file_sha256": {name: hashes[name] for name in required_adoption_hash_paths()}}


def cumulative_costs(adoption):
    prior = read_json(ROOT / PRIOR_FAILURE); preflight = read_json(ROOT / PREFLIGHT)
    return {"prior_failed_acquisition_environment_steps": prior["environment_step_calls_started"],
            "prior_failed_acquisition_resets": prior["reset_calls_started"],
            "canonical_preflight_environment_steps": preflight["actual_environment_steps"],
            "canonical_preflight_resets": preflight["actual_reset_calls_started"],
            "prior_failure_sha256": adoption["file_sha256"][PRIOR_FAILURE],
            "preflight_sha256": adoption["file_sha256"][PREFLIGHT]}


def select_native_rows(raw, expected120, selected60, seed):
    """Preserve every source line byte, including native expert annotations."""
    records = []
    for line in raw.splitlines(keepends=True):
        if line.strip(): records.append((json.loads(line), line))
    require(len(records) == 120 and [r["gamefile"] for r, _ in records] == expected120,
            "Native120 canonical identities/order are incomplete")
    require(len(set(expected120)) == 120 and len(selected60) == len(set(selected60)) == 60,
            "Frozen cohorts contain duplicates or incorrect counts")
    index = {}
    for row, line in records:
        require(row.get("kind") == "trajectory" and row.get("success") is True
                and row.get("replay_verified") is True and row.get("reset_replay_verified") is True
                and row.get("split") == "train" and row.get("seed") == seed,
                "Native120 contains failed/unverified/nontraining canonical source")
        require(line.endswith(b"\n"), "Canonical JSONL line lacks newline; cannot concatenate unchanged")
        index[row["gamefile"]] = (row, line)
    require(set(selected60) <= set(index), "Frozen60 not contained in canonical120")
    return [index[g] for g in selected60]


def load_inputs(config, root=ROOT):
    root = Path(root); fixed = config["fixed_training_cohort"]; games = fixed["frozen_gamefiles"]
    manifest = read_json(root / TRAIN_MANIFEST); source_summary = read_json(root / SOURCE_SUMMARY)
    require(file_hash(root / TRAIN_MANIFEST) == config["read_only_evidence"]["existing_train_manifest_sha256"],
            "Training manifest changed from candidate registration")
    require(hashlib.sha256(json.dumps(games, separators=(",", ":")).encode()).hexdigest()
            == fixed["ordered_gamefiles_sha256_compact_json"], "Frozen60 order/hash differs")
    require(manifest["split"] == "train" and source_summary["split"] == "train"
            and source_summary["expert_type"] == "planner" and source_summary["games"] == manifest["games"]
            and source_summary["seed"] == fixed["reset_seed"], "Canonical native planner source identity differs")
    for key in ("tasks_attempted", "expert_successes", "expert_replays_verified"):
        require(source_summary["counts"].get(key) == 120, "Native120 acquisition incomplete: " + key)
    require(not any(v for k, v in source_summary["counts"].items() if "error" in k), "Native120 has technical errors")
    by_family = Counter(); reconstructed = []
    for game in manifest["games"]:
        family = Path(game).parent.parent.name.split("-")[0]
        if by_family[family] < 10: reconstructed.append(game)
        by_family[family] += 1
    require(reconstructed == games and len(by_family) == 6 and set(by_family.values()) == {20},
            "Frozen60 are not first10 per original family")
    pairs = select_native_rows((root / SOURCE).read_bytes(), manifest["games"], games, fixed["reset_seed"])
    return pairs, manifest, source_summary


def process_start_ticks(pid):
    """Linux start-time identity; zombie leaders retain it until wait reaps them."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return None
    return stat[stat.rfind(")") + 2:].split()[19]


def terminate_owned_group(process):
    """Only the session created by this runner is eligible for termination."""
    pgid = process.pid
    original_start = getattr(process, "owned_start_ticks", None)
    require(original_start is not None, "Cannot clean process group without recorded launch identity")
    def still_owned():
        current_start = process_start_ticks(pgid)
        # A reaped leader can leave members in its old group. A different live
        # leader at the same PID indicates reuse and must never be signaled.
        return current_start is None or current_start == original_start
    if not still_owned(): return
    try: os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    until = time.monotonic() + 10
    while time.monotonic() < until:
        process.poll()
        if not still_owned(): return
        try: os.killpg(pgid, 0)
        except ProcessLookupError:
            process.wait()
            return
        time.sleep(0.1)
    if not still_owned(): return
    try: os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError: pass
    process.wait()


def worker(args):
    # Importing these helpers does not construct a simulator or import torch.
    from gnrl.alfworld_env import AlfworldReplayEnv
    from gnrl.macro_project import BudgetExceeded, SearchBudget
    from gnrl.successful_path_pools_v2 import build_successful_pool, summarize_acquisition
    out = Path(args.output); controller = read_json(out / "controller.json")
    require(os.getppid() == controller["controller_pid"], "Worker lacks its owned wall-time controller")
    token = os.environ.get("GNRL_SUCCESSFUL_PATH_WORKER_TOKEN", "")
    require(token and hashlib.sha256(token.encode()).hexdigest() == controller["worker_token_sha256"], "Invalid worker launch identity")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Worker must have GPUs disabled")
    config, adoption = validate_adoption(args.adoption_json)
    require(adoption["adoption_sha256"] == controller["adoption_sha256"], "Adoption changed after launch")
    rows, manifest, source_summary = load_inputs(config)
    pools, errors, stop = [], [], "complete"

    class DurableBudget(SearchBudget):
        def __init__(self):
            super().__init__(100000, 3600)
            self.started = controller["started_monotonic"]
            self.completed_steps = 0; self.completed_resets = 0; self.context = {}
            self.save()

        def save(self, **extra):
            atomic_json(out / "cost_progress.json", {"updated_utc": utc(), "environment_step_calls_started": self.steps,
                "environment_step_calls_completed": self.completed_steps, "reset_calls_started": self.resets,
                "reset_calls_completed": self.completed_resets, "wall_seconds": time.monotonic() - self.started,
                **self.context, **extra})

        def reset(self, env):
            self.check(); self.resets += 1; self.save(pending_call="reset")
            state = env.reset(); self.completed_resets += 1; self.save(pending_call=None)
            return state

        def step(self, env, action):
            self.check()
            if self.steps >= self.max_steps: raise BudgetExceeded("environment_steps")
            self.steps += 1; self.save(pending_call="step", pending_action=action)
            state = env.step(action); self.completed_steps += 1; self.save(pending_call=None)
            return state

    budget = DurableBudget()
    with (out / "canonical_baseline.jsonl").open("xb") as stream:
        for _, line in rows: stream.write(line)
    canonical_index = [{"gamefile": r["gamefile"], "seed": r["seed"], "source_line_sha256": hashlib.sha256(line).hexdigest()}
                       for r, line in rows]
    atomic_json(out / "canonical_source_index.json", {"all60_source_lines_preserved": True,
                "new_independent_replay_status": "Tracked separately in candidate_pools and final summary",
                "records": canonical_index})
    with (out / "candidate_pools.jsonl").open("x", encoding="utf-8") as pool_stream, \
         (out / "events.jsonl").open("x", encoding="utf-8") as events:
        for index, (row, line) in enumerate(rows):
            env = None; began = budget.steps
            event = {"task_index": index, "gamefile": row["gamefile"], "started_utc": utc()}
            budget.context = {"task_index": index, "gamefile": row["gamefile"]}; budget.save()
            dump_line(events, {**event, "event": "task_started"})
            try:
                budget.check()
                for name in SOURCE_FILES:
                    require(file_hash(ROOT / name) == adoption["file_sha256"][name], "Bound source changed: " + name)
                require(file_hash(args.adoption_json) == adoption["adoption_sha256"], "Adoption changed during acquisition")
                path = Path(row["gamefile"]); game_raw = path.read_bytes(); meta_raw = path.with_name("traj_data.json").read_bytes()
                game_hash = hashlib.sha256(game_raw).hexdigest()
                require(game_hash == manifest["game_sha256"][row["gamefile"]]
                        == source_summary["game_sha256"][row["gamefile"]], "Original game bytes changed")
                game_data, metadata = json.loads(game_raw), json.loads(meta_raw)
                # This is the only point at which a live original environment is constructed.
                env = AlfworldReplayEnv(row["gamefile"], seed=row["seed"], max_steps=100, expert_type=None)
                pool = build_successful_pool(row, game_data, metadata, env, budget, source_line_bytes=line,
                                            game_bytes=game_raw, metadata_bytes=meta_raw)
                require(len(pool["attempts"]) <= 7 and len(pool["candidates"]) <= 8, "Per-task proposal bound exceeded")
                pool["acquisition_cost"] = {"environment_steps_including_all_replay": budget.steps - began,
                                             "source_canonical_line_sha256": hashlib.sha256(line).hexdigest()}
                dump_line(pool_stream, pool); pools.append(pool)
                event.update(event="task_complete", pool_size=len(pool["candidates"]), recipe_attempts=len(pool["attempts"]),
                             rejections=[a for a in pool["attempts"] if not a["accepted"]])
            except BudgetExceeded as error:
                stop = "budget_exceeded"; event.update(event="task_incomplete", error=str(error)); errors.append(event.copy())
            except Exception as error:
                stop = "technical_error"; event.update(event="task_failed", error=f"{type(error).__name__}: {error}"); errors.append(event.copy())
            finally:
                if env is not None:
                    try: env.close()
                    except Exception as error:
                        stop = "technical_error"; event["close_error"] = repr(error); errors.append(event.copy())
            event["environment_step_calls_started"] = budget.steps - began
            event["finished_utc"] = utc(); dump_line(events, event)
            print(json.dumps(event, ensure_ascii=False), flush=True)
            if stop != "complete": break
    summary = summarize_acquisition(pools, config["fixed_training_cohort"]["frozen_gamefiles"], errors, stop)
    summary.update(adoption=adoption, budgets=FIXED_BUDGETS, environment_step_calls_started=budget.steps,
                   environment_step_calls_completed=budget.completed_steps, reset_calls_started=budget.resets,
                   reset_calls_completed=budget.completed_resets, recipe_attempts_in_completed_pools=sum(len(p["attempts"]) for p in pools),
                   retained_members=sum(len(p["candidates"]) for p in pools), wall_seconds=time.monotonic() - budget.started,
                   cpu_only=True, gpu_jobs=0, environment="unmodified_original_game", finished_utc=utc())
    summary["historical_costs"] = cumulative_costs(adoption)
    summary["cumulative_environment_step_calls_started"] = budget.steps + summary["historical_costs"]["prior_failed_acquisition_environment_steps"]
    summary["cumulative_reset_calls_started"] = budget.resets + summary["historical_costs"]["prior_failed_acquisition_resets"] + summary["historical_costs"]["canonical_preflight_resets"]
    for name, expected_hash in adoption["file_sha256"].items():
        if file_hash(ROOT / name) != expected_hash:
            summary["technical_errors"].append("Bound input changed before completion: " + name)
            summary.update(complete=False, coverage_gate_passed=False, stop_reason="input_changed")
    budget.check()
    require(summary["recipe_attempts_in_completed_pools"] <= 420 and summary["retained_members"] <= 480, "Global pool limits exceeded")
    atomic_json(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if summary["complete"] else 1


def write_controller_failure(out, error, code=None):
    out = Path(out)
    progress = read_json(out / "cost_progress.json") if (out / "cost_progress.json").exists() else {}
    event = {"event": "controller_failure", "error": str(error), "returncode": code, "created_utc": utc()}
    with (out / "controller_events.jsonl").open("a", encoding="utf-8") as stream: dump_line(stream, event)
    summary_path = out / "summary.json"
    summary = read_json(summary_path) if summary_path.exists() else {}
    previous_stop_reason = summary.get("stop_reason")
    summary.update(complete=False, coverage_gate_passed=False, stop_reason="controller_failure",
                   controller_failure=event, last_durable_cost_progress=progress,
                   cpu_only=True, gpu_jobs=0, budgets=FIXED_BUDGETS)
    controller_path = out / "controller.json"
    if controller_path.exists():
        history = read_json(controller_path).get("historical_costs")
        if history:
            summary["historical_costs"] = history
            summary["cumulative_environment_step_calls_started"] = progress.get("environment_step_calls_started", 0) + history["prior_failed_acquisition_environment_steps"]
            summary["cumulative_reset_calls_started"] = progress.get("reset_calls_started", 0) + history["prior_failed_acquisition_resets"] + history["canonical_preflight_resets"]
    if previous_stop_reason and previous_stop_reason != "complete":
        summary["worker_stop_reason"] = previous_stop_reason
    atomic_json(summary_path, summary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adoption-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.adoption_json = args.adoption_json.resolve(); args.output = args.output.resolve()
    if args.worker: return worker(args)
    require(sys.platform.startswith("linux"), "Live acquisition controller requires Linux process groups")
    require(args.output != ROOT / "outputs" and args.output.is_relative_to(ROOT / "outputs"), "New outputs must remain under this workspace outputs directory")
    started = time.monotonic()
    _, adoption = validate_adoption(args.adoption_json)
    args.output.mkdir(parents=True, exist_ok=False)
    token = secrets.token_hex(24)
    controller = {"controller_pid": os.getpid(), "started_utc": utc(), "started_monotonic": started,
                  "adoption_sha256": adoption["adoption_sha256"], "adoption_receipt": adoption,
                  "worker_token_sha256": hashlib.sha256(token.encode()).hexdigest(),
                  "budgets": FIXED_BUDGETS, "outer_deadline": "3600 seconds including setup; TERM then KILL own child group after 10-second grace"}
    controller["historical_costs"] = cumulative_costs(adoption)
    atomic_json(args.output / "controller.json", controller)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               GNRL_SUCCESSFUL_PATH_WORKER_TOKEN=token)
    env.pop("GNRL_GPU_GUARD", None)
    process = None
    def termination(signum, frame):
        raise RuntimeError("CPU controller received signal " + str(signum))
    signal.signal(signal.SIGTERM, termination)
    try:
        with (args.output / "worker.stdout.log").open("xb") as out, (args.output / "worker.stderr.log").open("xb") as err:
            command = [sys.executable, str(Path(__file__).resolve()), "--adoption-json", str(args.adoption_json), "--output", str(args.output), "--worker"]
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
            process.owned_start_ticks = process_start_ticks(process.pid)
            require(process.owned_start_ticks is not None, "Cannot establish owned worker process identity")
            controller.update(worker_pid=process.pid, owned_process_group=process.pid, command=command)
            controller["worker_start_ticks"] = process.owned_start_ticks
            atomic_json(args.output / "controller.json", controller)
            code = process.wait(timeout=max(0.1, 3600 - (time.monotonic() - started)))
        summary_path = args.output / "summary.json"
        if code == 1 and summary_path.is_file():
            summary = read_json(summary_path)
            if summary.get("complete") is False and summary.get("coverage_gate_passed") is False:
                # Preserve an orderly incomplete worker's specific budget/error diagnosis.
                summary["controller_returncode"] = code
                atomic_json(summary_path, summary)
                return 1
        if code != 0 or not summary_path.is_file():
            write_controller_failure(args.output, "Worker failed or did not write final summary", code)
            return 1
        require(file_hash(args.adoption_json) == adoption["adoption_sha256"], "Adoption changed during worker execution")
        return 0
    except BaseException as error:
        write_controller_failure(args.output, f"{type(error).__name__}: {error}", process.returncode if process else None)
        return 1
    finally:
        # Also clean orphaned native children after an orderly incomplete or
        # successful exit: reaping the Python leader does not reap its group.
        if process is not None:
            try: terminate_owned_group(process)
            except BaseException as error:
                write_controller_failure(args.output, "Owned group cleanup failed: " + repr(error), process.returncode)
                return 1


if __name__ == "__main__":
    raise SystemExit(main())

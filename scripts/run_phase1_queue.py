"""Finite phase-one matched pilot queue, exclusively on physical GPU 2.

Run from the project environment: python scripts/run_phase1_queue.py
No resume/overwrite: existing stage outputs require human review and a new version.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MODEL = "/home/jsy/zhangranxu-adrs-week-20260929/models/Qwen2.5-3B-Instruct"
GPU = 2
PLAN = "configs/phase1_proposal_candidates.json"
PLANNER = "data/planner_train120_raw.jsonl"
PLANNER_SUMMARY = "data/planner_train120_raw.summary.json"
BLOCKS = "outputs/two_blocks_v1/candidate_pools.jsonl"
BLOCK_SUMMARY = "outputs/two_blocks_v1/summary.json"
ASSEMBLY = "outputs/phase1_assembly_v1"
SELECTION = "outputs/phase1_selection_v1"
TRAIN_MANIFEST = "data/manifests/train120.json"
DEV_MANIFEST = "data/manifests/dev24.json"
QUEUE = "outputs/phase1_queue_v1"
ARMS = ("canonical", "uniform", "brp")
ARM_DATA = {"canonical": ASSEMBLY + "/canonical_baseline.jsonl",
            "uniform": SELECTION + "/uniform_valid.jsonl",
            "brp": SELECTION + "/learner_selected.jsonl"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def identities(records):
    result = [(r["gamefile"], r["seed"]) for r in records]
    require(len(result) == len(set(result)), "Duplicate gamefile/seed identities")
    return result


def pool_digest(pool):
    return hashlib.sha256(json.dumps(pool, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def assert_pairing(pools, canonical, uniform, brp, expected):
    """Fail before any training on incomplete, reordered, or mismatched pools."""
    from gnrl.macro_project import candidate_id, validate_pool
    require(len(pools) == len(expected) == 120, "Expected exactly 120 candidate pools")
    require([p["gamefile"] for p in pools] == expected, "Pool order differs from train120")
    expected_ids = identities(pools)
    for name, arm in (("canonical", canonical), ("uniform", uniform), ("brp", brp)):
        require(identities(arm) == expected_ids, name + " identity/order differs from pools")
    for pool, base, uni, selected in zip(pools, canonical, uniform, brp):
        original = validate_pool(pool)
        candidates = {c["candidate_id"]: c for c in pool["candidates"]}
        original_id = pool["original_candidate_id"]
        fingerprint = pool_digest(pool)
        require(base["selection"]["candidate_id"] == original_id,
                "Canonical baseline is not the original candidate")
        for row in (base, uni, selected):
            require(row.get("success") is True and row.get("replay_verified") is True and
                    row.get("split") == "train", "Unverified/nontraining selected trajectory")
            selection = row["selection"]
            require(selection.get("pool_sha256") == fingerprint, "Paired pool hash differs")
            cid = selection["candidate_id"]
            require(cid in candidates and candidate_id([t["action"] for t in row["transitions"]]) == cid,
                    "Selected candidate identity is not a member of this pool")
            require(row["transitions"] == candidates[cid]["transitions"], "Selected trace differs from verified candidate")
            require(row["task"] == pool["task"] and row["initial_observation"] == pool["initial_observation"],
                    "Paired task or initial observation differs")
            trace = row["transitions"]
            reference = original["transitions"]
            require(trace[0]["state_before"] == reference[0]["state_before"], "Paired initial state differs")
            require(trace[-1]["state_after"] == reference[-1]["state_after"], "Paired endpoint differs")
            require(Counter(t["action"] for t in trace) == Counter(t["action"] for t in reference),
                    "Paired action multiset differs")
        require(uni["selection"].get("original_candidate_id") == original_id and
                selected["selection"].get("original_candidate_id") == original_id,
                "Selection arms identify a different original candidate")
        require(uni["selection"]["arm"] == "uniform_valid" and
                selected["selection"]["arm"] == "learner_selected", "Selection arm labels differ")
    return [{"gamefile": p["gamefile"], "seed": p["seed"], "pool_sha256": pool_digest(p)} for p in pools]


def group_exists(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def terminate_owned_group(process):
    """The PID is a recorded group created by our own start_new_session Popen.

    Kill remaining descendants even when the group leader exits on SIGTERM.
    Never inspect/kill unrelated processes or GPU users.
    """
    pgid = process.pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    grace_end = time.monotonic() + 15
    while group_exists(pgid) and time.monotonic() < grace_end:
        process.poll()  # Reap the direct child so it cannot keep the group alive as a zombie.
        time.sleep(0.25)
    if group_exists(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


class Queue:
    def __init__(self, args):
        self.args = args
        self.started = time.monotonic()
        self.deadline = self.started + args.global_wall_seconds
        self.folder = ROOT / QUEUE
        self.folder.mkdir(parents=True, exist_ok=True)
        # Atomic exclusive claim prevents two queues from racing the same outputs.
        self.claim = (self.folder / "queue_claim.json").open("x", encoding="utf-8")
        self.claim.write(json.dumps({"pid": os.getpid(), "started_utc": self.utc()}))
        self.claim.close()
        self.locked = {}
        self.state = {"version": 1, "started_utc": self.utc(), "physical_gpu": GPU,
                      "model": MODEL, "global_wall_seconds": args.global_wall_seconds,
                      "stage_wall_seconds": args.stage_wall_seconds, "stages": [],
                      "scope": "Matched native-planner canonical/uniform/reference-selected pilot; raw-handcoded gains are separate."}
        self.env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1",
                        HF_DATASETS_OFFLINE="1", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
        self.env.pop("GNRL_GPU_GUARD", None)
        self.python = str(ROOT / ".venv/bin/python")

    @staticmethod
    def utc():
        return datetime.now(timezone.utc).isoformat()

    def update(self, stage, **values):
        self.state.update(current_stage=stage, updated_utc=self.utc(), **values)
        pending = self.folder / "status.json.tmp"
        pending.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        pending.replace(self.folder / "status.json")
        print(json.dumps({"stage": stage, **values}), flush=True)

    def lock(self, paths):
        for path in paths:
            path = Path(path)
            name = path.relative_to(ROOT).as_posix() if path.is_absolute() else path.as_posix()
            value = digest(ROOT / name)
            require(name not in self.locked or self.locked[name] == value, "Previously locked file changed: " + name)
            self.locked[name] = value
        (self.folder / "locked_sha256.json").write_text(json.dumps(self.locked, indent=2), encoding="utf-8")

    def verify(self):
        require(time.monotonic() < self.deadline, "Global eight-hour queue deadline reached")
        for name, expected in self.locked.items():
            require((ROOT / name).is_file() and digest(ROOT / name) == expected,
                    "Locked source/data/manifest changed: " + name)

    def wait_summary(self, name):
        self.update("waiting_for_acquisition", pending_path=name)
        while time.monotonic() < self.deadline:
            self.verify()
            if (ROOT / name).is_file():
                try:
                    return read_json(ROOT / name)
                except json.JSONDecodeError:
                    pass  # The producer may still be writing its final summary.
            time.sleep(min(self.args.poll_seconds, max(0, self.deadline - time.monotonic())))
        raise TimeoutError("Queue deadline waiting for " + name)

    def wait_gpu(self):
        from launch_gpu import resolve_devices
        require(GPU == 2, "Phase one is restricted to physical GPU 2")
        self.update("waiting_for_gpu2")
        while time.monotonic() < self.deadline:
            self.verify()
            smi = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used",
                                           "--format=csv,noheader,nounits"], text=True, timeout=30)
            running = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid",
                                               "--format=csv,noheader,nounits"], text=True, timeout=30)
            try:
                resolve_devices([GPU], list(csv.reader(io.StringIO(smi))),
                                active_uuids={x.strip() for x in running.splitlines()})
                return
            except RuntimeError:
                time.sleep(min(self.args.poll_seconds, max(0, self.deadline - time.monotonic())))
        raise TimeoutError("GPU 2 did not become idle before the queue deadline")

    def run(self, name, args, gpu=False):
        self.verify()
        if gpu:
            self.wait_gpu()
        self.verify()
        command = [self.python] + args
        if gpu:
            command = [self.python, "scripts/launch_gpu.py", "--gpus", "2", "--receipt",
                       str(self.folder / (name + "_gpu.json")), "--"] + command
        self.update(name, pending_path=None)
        began = time.monotonic()
        record = {"name": name, "command": command, "started_utc": self.utc()}
        self.state["stages"].append(record)
        with (self.folder / (name + ".stdout.log")).open("xb") as out, \
             (self.folder / (name + ".stderr.log")).open("xb") as err:
            process = subprocess.Popen(command, cwd=ROOT, env=self.env, stdin=subprocess.DEVNULL,
                                       stdout=out, stderr=err, start_new_session=True)
            try:
                record["pid"] = process.pid
                record["owned_process_group"] = process.pid
                self.update(name)
                rc = process.wait(timeout=min(self.args.stage_wall_seconds, max(0.1, self.deadline - time.monotonic())))
            except BaseException:
                terminate_owned_group(process)
                record.update(returncode=process.returncode, terminated_own_group=True,
                              elapsed_seconds=time.monotonic() - began)
                raise
        record.update(returncode=rc, elapsed_seconds=time.monotonic() - began)
        self.update(name, last_returncode=rc)
        require(rc == 0, f"Stage {name} failed ({rc}); inspect its logs")
        self.verify()

    def validate_acquisition(self, train_games):
        self.wait_summary(PLANNER_SUMMARY)
        self.wait_summary(BLOCK_SUMMARY)
        self.lock([PLANNER_SUMMARY, PLANNER, BLOCK_SUMMARY, BLOCKS,
                   "outputs/two_blocks_v1/config.json"])
        planner_summary, block_summary = read_json(PLANNER_SUMMARY), read_json(BLOCK_SUMMARY)
        counts = planner_summary["counts"]
        require(planner_summary["games"] == train_games and planner_summary["expert_type"] == "planner",
                "Planner source is not the complete frozen train120 native-planner cohort")
        for field in ("tasks_attempted", "expert_successes", "expert_replays_verified"):
            require(counts.get(field) == 120, "Planner acquisition incomplete: " + field)
        require(not any(value for key, value in counts.items() if "error" in key), "Planner technical/replay errors")
        acquired = rows(PLANNER)
        require(all(r.get("kind") == "trajectory" for r in acquired), "Planner file contains diagnostics/errors")
        require([r["gamefile"] for r in acquired] == train_games, "Planner row order differs from train120")
        identities(acquired)
        require(all(r.get("success") is True and r.get("replay_verified") is True and r.get("split") == "train"
                    for r in acquired), "Planner row is unsuccessful/unverified/nontraining")
        two_games = [g for g in train_games if Path(g).parents[1].name.startswith("pick_two_obj_and_place-")]
        require(len(two_games) == 20, "Frozen cohort does not contain exactly 20 two-object tasks")
        require(block_summary.get("full_canonical_cohort_complete") is True and
                block_summary.get("coverage_gate_passed") is True and block_summary.get("stop_reason") == "complete",
                "Two-object acquisition failed its preregistered completion/coverage gate")
        require(all(block_summary.get(k) == 20 for k in ("canonical_tasks", "tasks_attempted", "planned_tasks")) and
                block_summary.get("tasks_with_nontrivial_candidates", 0) >= 10 and
                block_summary.get("minimum_covered_tasks") == 10 and
                not block_summary.get("technical_errors", 0), "Two-object counts violate the fixed gate")
        blocks = rows(BLOCKS)
        require([r["gamefile"] for r in blocks] == two_games, "Block pool identities/order differ from frozen20")
        self.verify()

    def train_eval(self, arm):
        checkpoint = f"checkpoints/phase1_{arm}_seed42_v1"
        evaluation = f"outputs/eval_phase1_{arm}_dev24_v1"
        self.run("train_" + arm, ["-m", "gnrl.train_sft", "--model", MODEL, "--data", ARM_DATA[arm],
            "--output", checkpoint, "--max-steps", "100", "--batch-size", "1", "--gradient-accumulation", "8",
            "--learning-rate", "0.0002", "--lora-rank", "16", "--lora-alpha", "32", "--lora-dropout", "0.05",
            "--seed", "42", "--max-context-tokens", "2048", "--max-action-tokens", "64",
            "--max-wall-seconds", str(self.args.stage_wall_seconds)], gpu=True)
        summary = read_json(Path(checkpoint) / "summary.json")
        require(summary.get("matched_step_budget_complete") is True and summary.get("optimizer_steps") == 100 and
                summary.get("requested_steps") == 100, "Training did not complete matched 100 optimizer steps")
        config = read_json(Path(checkpoint) / "config.json")
        require(config["data_sha256"] == self.locked[ARM_DATA[arm]], "Trainer used different training data")
        adapter_files = list((ROOT / checkpoint / "final").rglob("*"))
        require(any(p.name == "adapter_config.json" for p in adapter_files) and
                any(p.name in ("adapter_model.safetensors", "adapter_model.bin") for p in adapter_files),
                "Final adapter checkpoint missing")
        self.lock([p for p in adapter_files if p.is_file()] + [checkpoint + "/summary.json", checkpoint + "/config.json"])
        self.run("eval_" + arm, ["-m", "gnrl.evaluate", "--model", MODEL, "--adapter", checkpoint + "/final",
            "--games-json", DEV_MANIFEST, "--output", evaluation, "--selection", "generate",
            "--max-steps", "100", "--seed", "20261012", "--max-context-tokens", "2048", "--max-action-tokens", "64",
            "--max-wall-seconds", str(self.args.stage_wall_seconds)], gpu=True)
        summary = read_json(Path(evaluation) / "summary.json")
        episodes = rows(Path(evaluation) / "episodes.jsonl")
        expected = read_json(DEV_MANIFEST)["games"]
        require(summary.get("complete") is True and summary.get("technical_errors") == 0 and
                summary.get("planned_episodes") == 24 and summary.get("attempted_episodes") == 24,
                "Evaluation is incomplete or has technical errors")
        require([r["gamefile"] for r in episodes] == expected and len(episodes) == 24 and
                all(r.get("seed") == 20261012 and "error" not in r for r in episodes),
                "Evaluation records differ from the frozen dev24 identities/order/seed")
        config = read_json(Path(evaluation) / "config.json")
        require(config.get("manifest_sha256") == self.locked[DEV_MANIFEST] and config.get("selection") == "generate",
                "Evaluation used different manifest/decoding")
        self.lock([evaluation + "/summary.json", evaluation + "/episodes.jsonl", evaluation + "/config.json"])
        self.state.setdefault("completed_arms", []).append(arm)

    def main(self):
        require(os.name == "posix", "Queue must run on the Linux chemistry server")
        require(Path(self.python).is_file(), "Project .venv/bin/python missing")
        # Snapshot only existing source/config and frozen manifests, never changing output folders.
        source = [p for folder in ("gnrl", "scripts", "configs") for p in sorted((ROOT / folder).rglob("*"))
                  if p.is_file() and p.suffix in (".py", ".json")]
        source += list(sorted((ROOT / "data/manifests").glob("*.json")))
        self.lock(source)
        plan = read_json(PLAN)
        training = plan["matched_learning_experiment"]["pilot_training"]
        expected_training = {"model": "Qwen2.5-3B-Instruct", "optimizer_steps": 100, "batch_size": 1,
                             "gradient_accumulation": 8, "learning_rate": 0.0002, "lora_rank": 16,
                             "seed": 42, "max_context_tokens": 2048}
        require(training == expected_training, "Queue settings differ from the registered phase-one plan")
        train_games = read_json(TRAIN_MANIFEST)["games"]
        dev_games = read_json(DEV_MANIFEST)["games"]
        require(len(train_games) == len(set(train_games)) == 120 and len(dev_games) == len(set(dev_games)) == 24,
                "Manifest sizes/uniqueness differ from fixed train120/dev24")
        require(not set(train_games) & set(dev_games), "Training/development overlap")
        outputs = [ASSEMBLY, SELECTION] + [f"checkpoints/phase1_{arm}_seed42_v1" for arm in ARMS] + \
                  [f"outputs/eval_phase1_{arm}_dev24_v1" for arm in ARMS]
        require(not any((ROOT / p).exists() for p in outputs),
                "An output already exists: refusing duplicate scoring/training or overwrite; review manually")
        self.validate_acquisition(train_games)
        self.run("assemble", ["scripts/build_phase1_pools.py", "--planner", PLANNER, "--blocks", BLOCKS,
                              "--manifest", TRAIN_MANIFEST, "--output", ASSEMBLY])
        summary = read_json(ASSEMBLY + "/assembly_summary.json")
        require(summary.get("gate_pass") is True and summary["counts"].get("tasks") == 120 and
                summary["counts"].get("two_object_tasks") == 20 and
                summary["counts"].get("two_object_tasks_with_alternatives", 0) >= 10, "Assembly gate failed")
        self.lock([ASSEMBLY + "/assembly_summary.json", ASSEMBLY + "/candidate_pools.jsonl", ARM_DATA["canonical"]])
        budget = plan["fixed_budgets_and_gates"]["score_budget"]
        self.run("select", ["-m", "gnrl.macro_project", "--pools", ASSEMBLY + "/candidate_pools.jsonl",
            "--model", MODEL, "--output", SELECTION, "--max-games", "120", "--seed", "42",
            "--max-context-tokens", "2048", "--max-action-tokens", "64",
            "--max-score-seconds", str(min(budget["max_score_seconds"], self.args.stage_wall_seconds)),
            "--max-score-tokens", str(budget["max_scored_action_tokens"]),
            "--max-forward-tokens", str(budget["max_forward_tokens"])], gpu=True)
        selection = read_json(SELECTION + "/selection_summary.json")
        require(selection.get("stop_reason") == "complete" and selection.get("paired_trajectories_written") == 120 and
                selection.get("pools_considered") == 120, "Selection did not finish all 120 paired pools")
        require(selection.get("candidate_pools_sha256") == self.locked[ASSEMBLY + "/candidate_pools.jsonl"],
                "Selector used different pools")
        self.lock([SELECTION + "/selection_summary.json", ARM_DATA["uniform"], ARM_DATA["brp"]])
        paired = assert_pairing(rows(ASSEMBLY + "/candidate_pools.jsonl"),
                                *(rows(ARM_DATA[arm]) for arm in ARMS), train_games)
        (self.folder / "pairing_receipt.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")
        for arm in ARMS:
            self.train_eval(arm)
        self.verify()
        self.update("needs_astra_result_review", queue_finished=True,
                    note="All three matched pilot arms complete. No automatic positive-effect claim, unseen evaluation, extra seeds, or tuning. Raw-handcoded versus canonical improvements are separate from projection.")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--global-wall-seconds", type=float, default=28800.0, help="Finite whole-queue budget, at most eight hours")
    p.add_argument("--stage-wall-seconds", type=float, default=7200.0, help="Hard subprocess time limit, at most two hours")
    p.add_argument("--poll-seconds", type=float, default=20.0)
    return p


def main():
    args = parser().parse_args()
    require(0 < args.global_wall_seconds <= 28800 and 0 < args.stage_wall_seconds <= 7200 and
            0 < args.poll_seconds <= 60, "Budgets must be positive and within registered upper limits")
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    queue = Queue(args)

    def terminate_signal(signum, frame):
        raise SystemExit("Queue received termination signal " + str(signum))

    signal.signal(signal.SIGTERM, terminate_signal)
    try:
        queue.main()
    except BaseException as error:
        queue.update("failed_needs_astra_review", queue_finished=True,
                     error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()

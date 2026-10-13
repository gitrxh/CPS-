"""LOCAL CONDITIONAL CANDIDATE: no job without a future explicit adoption.

Execute only after independent review with /usr/bin/python3. All workers retain
the existing project Python. Importing this module is inert and CPU-only.
See confirmation_execution_schema_v1.json for the required adoption contract.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_successful_path_learning_v1 as old
from gnrl.successful_path_token_audit_v1 import sample_indices
from analysis.review_successful_path_complete_batch_v1 import validate_episode

PLAN = "analysis/successful_path_confirmation_candidate_v1.json"
PLAN_SHA = "19538894e51b1ee546c8461a6500100fbb6d33ae4f352c531e5d8153522083f9"
SELF = "analysis/run_successful_path_confirmation_candidate_v1.py"
TEST = "analysis/test_successful_path_confirmation_candidate_v1.py"
SCHEMA = "analysis/confirmation_execution_schema_v1.json"
PRIOR_REVIEW = "analysis/confirmation_candidate_review_v1.json"
EPISODE_REVIEW_SOURCE = "analysis/review_successful_path_complete_batch_v1.py"
UNSEEN = "data/manifests/unseen_locked.json"
SEEDS = (42, 123, 456)
ARMS = old.ARMS
TRAIN_ORDER = ("123/canonical", "123/uniform", "123/reference", "456/reference", "456/uniform", "456/canonical")
EVAL_ORDER = ("42/canonical", "42/uniform", "42/reference", "123/uniform", "123/reference", "123/canonical", "456/reference", "456/canonical", "456/uniform")
ALLOWED_STOP = {"environment_done", "max_steps", "no_admissible_commands"}
CLEANUP_RESERVE = 15


def require(value, reason):
    old.require(value, reason)


def derive_seed_audit(base, seed):
    """Reuse frozen tokenization evidence; recompute only the trainer shuffle."""
    require(seed in SEEDS and base.get("complete") is True and base.get("examples_per_arm") == 349,
            "Wrong seed/incomplete original token audit")
    require(base.get("score_sft_token_mismatches") == 0 and base.get("teacher_fields_used") is False,
            "Original token audit has token/causal mismatches")
    require(base["settings"] == {"seed": 42, "optimizer_steps": 100, "batch_size": 1,
            "gradient_accumulation": 8, "max_context_tokens": 2048, "max_action_tokens": 64}, "Original audit settings differ")
    keys = base["ordered_example_keys"]
    require(len(keys) == len({tuple(x) for x in keys}) == 349, "Incomplete/duplicate example keys")
    selected = sample_indices(349, seed=seed)
    result = copy.deepcopy(base)
    result["settings"]["seed"] = seed
    result["sampled_indices"] = selected
    for arm in ARMS:
        examples = base["examples"][arm]
        require([[r["gamefile"], r["step"]] for r in examples] == keys, "Arm example keys differ")
        for row in examples:
            require(row["input_tokens"] == row["prompt_tokens"] + row["target_tokens"] and
                    0 < row["target_tokens"] <= 64 and row["input_tokens"] <= 2048,
                    "Invalid original token counts")
        result["exposures"][arm] = {"sampled_" + field: sum(examples[i][field] for i in selected)
                                    for field in ("target_tokens", "input_tokens", "prompt_tokens")}
        require(result["exposures"][arm]["sampled_target_tokens"] <= 2000000, "Target token budget exhausted by audit")
    for other in ("canonical", "uniform"):
        changed = {i for i, (a, b) in enumerate(zip(base["examples"]["reference"], base["examples"][other]))
                   if a["prompt_ids_sha256"] != b["prompt_ids_sha256"]}
        result["comparisons"]["reference_vs_" + other]["changed_sampled_inputs"] = sum(i in changed for i in selected)
    result["derivation"] = "Byte-locked prior causal tokenization; new seed-specific exact 800-index trainer shuffle only."
    return result


def technical_records(path, expected_games, *, final=False):
    """Never reads success or policy outcomes. Partial last JSONL line is pending.

    Returns only task identities and technical classifications. The unmodified
    worker may start its next episode before the controller notices a flushed
    error; polling stops its owned session promptly without using outcomes.
    """
    path = Path(path)
    if not path.exists():
        require(not final, "Missing evaluation episode file")
        return []
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    if lines and not lines[-1].endswith(b"\n"):
        require(not final, "Truncated final episode record")
        lines.pop()
    result = []
    for index, line in enumerate(lines):
        row = json.loads(line)
        require(index < len(expected_games) and row.get("gamefile") == expected_games[index] and row.get("seed") == 20261012,
                "Evaluation identity/seed/order changed")
        error = "error" in row
        reason = row.get("stop_reason")
        result.append({"gamefile": expected_games[index], "seed": 20261012,
                       "status": "technical_error" if error or reason not in ALLOWED_STOP else "observed_blinded",
                       "stop_reason": reason})
    if final:
        require(len(result) == len(expected_games), "Missing planned evaluation episodes")
    return result


class TechnicalMonitor:
    """Read each complete flushed JSONL record once, without consulting outcomes."""
    def __init__(self, path, games):
        self.path, self.games, self.offset, self.records = Path(path), games, 0, []

    def poll(self):
        if not self.path.exists():
            return self.records
        require(self.path.stat().st_size >= self.offset, "Episode log was truncated")
        with self.path.open("rb") as stream:
            stream.seek(self.offset)
            while True:
                line = stream.readline()
                if not line or not line.endswith(b"\n"):
                    break
                row = json.loads(line)
                index = len(self.records)
                require(index < len(self.games) and row.get("gamefile") == self.games[index] and row.get("seed") == 20261012,
                        "Evaluation identity/seed/order changed")
                reason = row.get("stop_reason")
                self.records.append({"gamefile": self.games[index], "seed": 20261012,
                    "status": "technical_error" if "error" in row or reason not in ALLOWED_STOP else "observed_blinded",
                    "stop_reason": reason})
                self.offset = stream.tell()
                if self.records[-1]["status"] == "technical_error":
                    break
        return self.records


def assert_no_technical(rows):
    require(all(r["status"] == "observed_blinded" for r in rows), "Technical evaluation stop; no outcome inspection")


def validate_train(summary, config, seed, data_hash, tokens):
    expected = dict(old.TRAIN, seed=seed)
    require(summary.get("matched_step_budget_complete") is True and summary.get("optimizer_steps") ==
            summary.get("requested_steps") == 100 and summary.get("stop_reason") == "max_steps", "Incomplete training")
    require(summary.get("target_tokens") == tokens and config.get("data_sha256") == data_hash and
            config.get("protocol") == "alfworld-causal-v1", "Training exposure/data/protocol differs")
    require(all(config.get(k) == v for k, v in expected.items()), "Training hyperparameters differ")


def validate_final_state(state, target_tokens):
    require(state.get("optimizer_steps") == 100 and state.get("target_tokens") == target_tokens and
            state.get("protocol") == "alfworld-causal-v1", "Final checkpoint step/token/protocol differs")


def validate_complete_episode(episode, game):
    evidence = validate_episode(episode, game)
    require(all(t["policy"]["generated_tokens"] <= 64 for t in episode["transitions"]), "Action generation exceeds frozen64 limit")
    require(episode["stop_reason"] != "max_steps" or evidence["steps"] == 100, "Max-step episode stopped before100 actions")
    return evidence


def technical_eval_check(folder, games, adapter, manifest_hash):
    records = technical_records(folder / "episodes.jsonl", games, final=True)
    assert_no_technical(records)
    summary = old.read_json(folder / "summary.json")
    require(summary.get("complete") is True and summary.get("technical_errors") == 0 and
            summary.get("planned_episodes") == summary.get("attempted_episodes") == 134, "Incomplete evaluation summary")
    config = old.read_json(folder / "config.json")
    expected = dict(old.EVAL, max_generated_tokens=900000, adapter=adapter,
                    games_json=UNSEEN, max_wall_seconds=14400, manifest_sha256=manifest_hash,
                    protocol="alfworld-causal-v1")
    require(all(config.get(k) == v for k, v in expected.items()), "Evaluation config differs")
    return records


def check_adoption_shape(adoption):
    for key, value in {"schema_version": 1, "kind": "successful_path_confirmation_adoption", "adopted": True,
                        "adopted_by": "root_after_independent_GPT_6_Astra_review",
                        "scope": "fixed60_three_seed_single_unseen134_batch", "physical_gpu": 2}.items():
        require(type(adoption.get(key)) is type(value) and adoption[key] == value, "Confirmation adoption gate: " + key)


def validate_adoption(path):
    adopted = old.read_json(path)
    check_adoption_shape(adopted)
    require(old.sha256(ROOT / PLAN) == PLAN_SHA, "Confirmation candidate bytes changed")
    hashes = adopted["file_sha256"]
    required = {PLAN, SELF, TEST, SCHEMA, PRIOR_REVIEW, EPISODE_REVIEW_SOURCE, UNSEEN, adopted["current_learning_adoption"],
                adopted["current_learning_locks"], adopted["entry_review"], adopted["execution_review"],
                old.PROVENANCE, old.MODEL_INVENTORY}
    require(required <= hashes.keys(), "Missing confirmation adoption bindings")
    old.verify_hashes(hashes)
    entry = old.read_json(ROOT / adopted["entry_review"])
    review = old.read_json(ROOT / adopted["execution_review"])
    require(entry.get("confirmation_entry_gate_passed") is True and review.get("confirmation_execution_review_passed") is True,
            "Missing complete independent entry/execution review")
    require(review.get("candidate_sha256") == PLAN_SHA and review.get("controller_sha256") == hashes[SELF], "Execution review binds different code/plan")
    current = old.read_json(ROOT / adopted["current_learning_adoption"])
    require(current.get("kind") == "successful_path_learning_adoption" and current.get("adopted") is True, "Wrong seed42 parent adoption")
    require(old.source_inventory() == current["source_inventory"], "Current source inventory changed")
    current_locks = old.read_json(ROOT / adopted["current_learning_locks"])
    report_name = current["output_paths"]["queue"] + "/complete_batch_report.json"
    require(entry.get("current_learning_adoption_sha256") == hashes[adopted["current_learning_adoption"]] and
            entry.get("complete_batch_report_sha256") == current_locks.get(report_name), "Entry review binds another batch")
    for name, value in {**current["file_sha256"], **current_locks}.items():
        require(hashes.get(name) == value, "Missing/changed inherited provenance: " + name)
    plan = old.read_json(ROOT / PLAN)
    for group in ("data_files_and_sha256", "other_sha256"):
        for name, value in plan["frozen_evidence"][group].items():
            require(hashes.get(name) == value, "Different fixed confirmation input: " + name)
    require(plan["training"]["fixed_new_training_order"] == list(TRAIN_ORDER) and
            plan["one_locked_unseen_batch"]["fixed_order"] == list(EVAL_ORDER), "Fixed order differs")
    runtime = adopted["runtime"]
    require(set(runtime) == {"controller", "worker"}, "Runtime must bind exactly controller and worker")
    require(Path(sys.executable).resolve() == Path("/usr/bin/python3").resolve() == Path(runtime["controller"]["resolved"]), "Wrong controller interpreter")
    require(Path(ROOT / ".venv/bin/python").resolve() == Path(runtime["worker"]["resolved"]), "Wrong worker interpreter")
    for record in (runtime["controller"], runtime["worker"]):
        require(old.sha256(record["resolved"]) == record["executable_sha256"], "Runtime binary changed")
    originals = adopted["original_file_sha256"]
    for name, value in current["original_file_sha256"].items():
        require(originals.get(name) == value, "Original seed42 game/metadata lock lost")
    manifest = old.read_json(ROOT / UNSEEN)
    games = manifest["games"]
    require(manifest.get("split") == "valid_unseen" and len(games) == len(set(games)) == 134, "Wrong unseen manifest")
    require(not set(games) & (set(old.read_json(ROOT / old.TRAIN_MANIFEST)["games"]) |
                             set(old.read_json(ROOT / old.DEV_MANIFEST)["games"])), "Unseen identity overlaps training/development")
    for game in games:
        require(originals.get(game) == manifest["game_sha256"][game], "Unseen game not frozen")
        require(str(Path(game).with_name("traj_data.json")) in originals, "Unseen metadata not frozen")
    for name, value in originals.items():
        require(old.sha256(name) == value, "Original file bytes changed")
    paths = adopted["output_paths"]
    require(set(paths) == {"queue", "checkpoints", "evaluations"} and set(paths["checkpoints"]) == {str(s) for s in SEEDS}
            and set(paths["evaluations"]) == set(EVAL_ORDER), "Incomplete output-path adoption")
    require(paths["checkpoints"]["42"] == current["output_paths"]["checkpoints"], "Seed42 checkpoint reuse paths differ")
    new_paths = [paths["queue"], *paths["evaluations"].values()]
    for seed in ("123", "456"):
        require(set(paths["checkpoints"][seed]) == set(ARMS), "Missing new arm checkpoint paths")
        new_paths += list(paths["checkpoints"][seed].values())
    require(len(set(new_paths)) == 16, "Duplicate future output paths")
    resolved = [old.safe_relative(name) for name in new_paths]
    for name, folder in zip(new_paths, resolved):
        require((name.startswith("outputs/successful_path_confirmation_") or name.startswith("checkpoints/successful_path_confirmation_"))
                and not folder.exists(), "Existing/incorrect confirmation output namespace")
    require(not any(a != b and a.is_relative_to(b) for a in resolved for b in resolved), "Nested output directories")
    require(set(paths["checkpoints"]["42"]) == set(ARMS), "Seed42 arms incomplete")
    return adopted, current, plan, games


def quantile(values, p):
    values = sorted(values)
    x = (len(values) - 1) * p
    lo = int(x)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (x - lo) * (values[hi] - values[lo])


def analyze_complete(table, scene_counts, *, draws=50000):
    """Only called after the all-nine technical completion/hash barrier."""
    require(len(table) == 134 and set(scene_counts) == {"10", "219", "308", "424"}, "Analysis requires fixed134/four scenes")
    require(len({r["gamefile"] for r in table}) == 134 and dict(Counter(r["scene"] for r in table)) == scene_counts,
            "Analysis task identities/scene counts differ")
    require(all(type(r["outcomes"][str(seed)][arm]) is bool for r in table for seed in SEEDS for arm in ARMS), "Missing outcome; no subset analysis")
    contrasts, cluster_delta = {}, {}
    scenes = sorted(scene_counts)
    for control in ("uniform", "canonical"):
        per_seed = {}
        cluster_delta[control] = {}
        for seed in SEEDS:
            values = [(r["outcomes"][str(seed)]["reference"], r["outcomes"][str(seed)][control]) for r in table]
            difference = sum(int(a) - int(b) for a, b in values)
            per_seed[str(seed)] = {"difference_successes": difference, "rate_difference": difference / 134,
                "gains": sum(a and not b for a, b in values), "losses": sum(b and not a for a, b in values),
                "ties": sum(a == b for a, b in values)}
            cluster_delta[control][seed] = {scene: sum(int(r["outcomes"][str(seed)]["reference"]) - int(r["outcomes"][str(seed)][control]) for r in table if r["scene"] == scene) for scene in scenes}
        total_gain = sum(r["difference_successes"] for r in per_seed.values())
        contrasts[control] = {"per_seed": per_seed, "primary_mean_difference": total_gain / 402,
            "practical_target_met": total_gain >= 21 and all(r["difference_successes"] > 0 for r in per_seed.values()),
            "equal_scene_mean": sum(cluster_delta[control][s][c] / scene_counts[c] for s in SEEDS for c in scenes) / 12,
            "leave_one_scene_out": {omitted: sum(cluster_delta[control][s][c] for s in SEEDS for c in scenes if c != omitted) /
                 (3 * (134 - scene_counts[omitted])) for omitted in scenes}}
    rng, samples = random.Random(613207), {c: [] for c in contrasts}
    for _ in range(draws):
        seeds = [rng.choice(SEEDS) for _ in SEEDS]
        selected_scenes = [rng.choice(scenes) for _ in scenes]
        denominator = 3 * sum(scene_counts[c] for c in selected_scenes)
        for control in contrasts:
            samples[control].append(sum(cluster_delta[control][s][c] for s in seeds for c in selected_scenes) / denominator)
    for control in contrasts:
        contrasts[control]["descriptive_crossed_bootstrap"] = {"draws": draws, "rng_seed": 613207,
            "percentile_interval_95": [quantile(samples[control], .025), quantile(samples[control], .975)],
            "interpretation": "Descriptive only: three fixed seeds/four unequal scenes; no calibrated confirmatory significance."}
    return {"contrasts": contrasts, "practical_replication_target_met": all(v["practical_target_met"] for v in contrasts.values()),
            "positive_finite_benchmark_average": all(v["primary_mean_difference"] > 0 for v in contrasts.values())}


class ConfirmationQueue(old.Queue):
    def __init__(self, adoption_path):
        self.started = time.monotonic()
        self.deadline = self.started + 172800
        self.adoption_path = Path(adoption_path).resolve()
        self.adoption_hash = old.sha256(self.adoption_path)
        self.adopted, self.current, self.plan, self.games = validate_adoption(self.adoption_path)
        self.paths = self.adopted["output_paths"]
        self.locks = dict(self.adopted["file_sha256"])
        self.folder = ROOT / self.paths["queue"]
        self.folder.mkdir(parents=True, exist_ok=False)
        self.state = {"stages": [], "complete": False, "started_utc": old.utc(), "physical_gpu": 2,
            "max_queue_seconds": 172800, "cleanup_reserve_seconds": CLEANUP_RESERVE,
            "adoption_sha256": self.adoption_hash, "trained": [], "evaluated": []}
        self.env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
        self.env.pop("GNRL_GPU_GUARD", None)
        self.python = str(ROOT / ".venv/bin/python")
        self.game_locks = {Path(n): h for n, h in self.adopted["original_file_sha256"].items()}
        self.directory_locks = {}
        self.data = {arm: self.current["output_paths"]["selection"] + "/" + old.DATA_FILES[arm] for arm in ARMS}
        self.audit_path = self.current["output_paths"]["queue"] + "/token_exposure_audit.json"

    def verify(self):
        require(time.monotonic() < self.deadline - CLEANUP_RESERVE, "48-hour global budget reached; preserve cleanup reserve")
        require(old.sha256(self.adoption_path) == self.adoption_hash, "Confirmation adoption changed")
        require(old.source_inventory() == self.current["source_inventory"], "Frozen source inventory changed")
        old.verify_hashes(self.locks)
        old.verify_model(old.read_json(ROOT / old.PROVENANCE), full_inventory=old.read_json(ROOT / old.MODEL_INVENTORY))
        for p, h in self.game_locks.items():
            require(old.sha256(p) == h, "Original game/metadata bytes changed")
        for record in self.adopted["runtime"].values():
            require(old.sha256(record["resolved"]) == record["executable_sha256"], "Runtime binary changed")
        for folder, expected in self.directory_locks.items():
            require({p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()} == expected, "Checkpoint/output inventory changed")
        require(time.monotonic() < self.deadline - CLEANUP_RESERVE, "Global budget reached during verification")

    def run(self, name, arguments, wall, monitor=None):
        self.verify()
        self.wait_gpu()
        self.verify()
        command = [self.python, "scripts/launch_gpu.py", "--gpus", "2", "--receipt", str(self.folder / (name + "_gpu.json")),
                   "--", self.python, *arguments]
        began = time.monotonic()
        end = min(began + wall, self.deadline - CLEANUP_RESERVE)
        require(end > began, "No remaining execution time")
        record = {"name": name, "command": command, "started_utc": old.utc(), "stage_seconds_ceiling": wall}
        self.state["stages"].append(record)
        self.update(name)
        process, ticks = None, None
        tracker = TechnicalMonitor(monitor, self.games) if monitor is not None else None
        with (self.folder / (name + ".stdout.log")).open("xb") as out, (self.folder / (name + ".stderr.log")).open("xb") as err:
            try:
                process = subprocess.Popen(command, cwd=ROOT, env=self.env, stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
                identity = old.process_record(process.pid)
                require(identity and identity["session"] == process.pid, "Cannot establish owned stage identity")
                ticks = identity["start_ticks"]
                record.update(pid=process.pid, start_ticks=ticks, owned_session=process.pid)
                self.update(name)
                while True:
                    if monitor is not None:
                        progress = tracker.poll()
                        record["complete_episode_records"] = len(progress)
                        assert_no_technical(progress)
                    require(time.monotonic() < end, "Stage/global wall budget exhausted")
                    try:
                        rc = process.wait(timeout=min(.25, max(.001, end - time.monotonic())))
                        break
                    except subprocess.TimeoutExpired:
                        pass
                if monitor is not None:
                    assert_no_technical(technical_records(monitor, self.games, final=True))
                require(rc == 0, "Stage process failed")
            finally:
                if process is not None:
                    mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM})
                    try:
                        old.cleanup_owned(process, ticks)
                    finally:
                        signal.pthread_sigmask(signal.SIG_SETMASK, mask)
                record.update(returncode=process.returncode if process else None, elapsed_seconds=time.monotonic() - began)
                self.update(name)
        self.verify()

    def entry_gate(self):
        """Only after explicit future adoption, read the then-complete old batch."""
        paths = self.current["output_paths"]
        report_path = paths["queue"] + "/complete_batch_report.json"
        require(report_path in self.locks, "Current complete report not hash-bound")
        report = old.read_json(ROOT / report_path)
        require(report.get("complete") is True, "Current learning batch incomplete")
        counts = {}
        for arm in ARMS:
            folder = ROOT / paths["evaluations"][arm]
            summary, episodes, config = old.read_json(folder / "summary.json"), old.rows(folder / "episodes.jsonl"), old.read_json(folder / "config.json")
            old.validate_evaluation(summary, episodes, config, old.read_json(ROOT / old.DEV_MANIFEST)["games"],
                paths["checkpoints"][arm] + "/final", self.locks[old.DEV_MANIFEST])
            counts[arm] = sum(r["success"] for r in episodes)
        require(counts == report["successes_out_of_24"] and counts["reference"] > counts["uniform"] and
                counts["reference"] > counts["canonical"], "Complete current directional screen failed")
        base = old.read_json(ROOT / self.audit_path)
        require(base["dataset_bytes_sha256"] == {arm: self.locks[self.data[arm]] for arm in ARMS}, "Prior audit/data hashes differ")
        reconstructed42 = derive_seed_audit(base, 42)
        require(reconstructed42["sampled_indices"] == base["sampled_indices"] and reconstructed42["exposures"] == base["exposures"], "Prior seed42 audit is internally inconsistent")
        self.audits = {seed: derive_seed_audit(base, seed) for seed in SEEDS}
        for seed in SEEDS:
            self.audits[seed]["source_audit_sha256"] = self.locks[self.audit_path]
            p = self.folder / ("token_audit_seed" + str(seed) + ".json")
            old.atomic_json(p, self.audits[seed])
            self.lock([p])
        for arm in ARMS:
            self.checkpoint(42, arm)
        scenes = {game: {"family": Path(game).parent.parent.name.split("-")[0], "scene": Path(game).parent.parent.name.split("-")[-1]} for game in self.games}
        require(dict(Counter(r["scene"] for r in scenes.values())) == self.plan["estimands_and_analysis"]["scene_map_known_from_manifest_only"]["tasks_by_scene"], "Unseen scene composition differs")
        require(len({r["family"] for r in scenes.values()}) == 6, "Unseen family composition differs")
        self.scene_map = scenes
        old.atomic_json(self.folder / "manifest_scene_map.json", scenes)
        self.lock([self.folder / "manifest_scene_map.json"])

    def checkpoint(self, seed, arm):
        path = self.paths["checkpoints"][str(seed)][arm]
        folder = ROOT / path
        config, summary = old.read_json(folder / "config.json"), old.read_json(folder / "summary.json")
        validate_train(summary, config, seed, self.locks[self.data[arm]], self.audits[seed]["exposures"][arm]["sampled_target_tokens"])
        require(config.get("data") == self.data[arm] and config.get("output") == path, "Checkpoint data/output path differs")
        final = folder / "final"
        adapter = old.read_json(final / "adapter_config.json")
        require(adapter.get("base_model_name_or_path") == old.MODEL and adapter.get("r") == 16 and adapter.get("lora_alpha") == 32, "Checkpoint base/LoRA differs")
        require(any((final / p).is_file() for p in ("adapter_model.safetensors", "adapter_model.bin")) and
                old.read_json(final / "training_state.json").get("optimizer_steps") == 100, "Missing final step100 checkpoint")
        validate_final_state(old.read_json(final / "training_state.json"), self.audits[seed]["exposures"][arm]["sampled_target_tokens"])
        require(summary.get("training_examples") == 349, "Training example count differs")
        if seed == 42:
            require(all(p.relative_to(ROOT).as_posix() in self.locks for p in folder.rglob("*") if p.is_file()),
                    "Seed42 checkpoint acquired an unadopted file")
        self.lock_directory(folder)

    def train(self, key):
        seed_text, arm = key.split("/")
        seed = int(seed_text)
        args = ["-m", "gnrl.train_sft", "--data", self.data[arm], "--output", self.paths["checkpoints"][seed_text][arm], "--max-wall-seconds", "7200"]
        for k, value in dict(old.TRAIN, seed=seed).items():
            args += ["--" + k.replace("_", "-"), str(value)]
        self.run("train_" + key.replace("/", "_"), args, 7200)
        self.checkpoint(seed, arm)
        self.state["trained"].append(key)

    def nine_model_barrier(self):
        require(self.state["trained"] == list(TRAIN_ORDER), "Six new trainings must complete before any unseen evaluation")
        for seed in SEEDS:
            for arm in ARMS:
                self.checkpoint(seed, arm)
        self.verify()
        p = self.folder / "all_nine_models_locked.json"
        old.atomic_json(p, {"complete": True, "model_keys": list(EVAL_ORDER), "recorded_utc": old.utc(),
            "checkpoint_sha256": {name: value for name, value in self.locks.items() if name.startswith("checkpoints/")},
            "audit_sha256": {str(seed): self.locks[self.paths["queue"] + "/token_audit_seed" + str(seed) + ".json"] for seed in SEEDS}})
        self.lock([p])

    def evaluate(self, key):
        require(self.paths["queue"] + "/all_nine_models_locked.json" in self.locks, "Nine-model barrier missing")
        seed, arm = key.split("/")
        adapter = self.paths["checkpoints"][seed][arm] + "/final"
        folder = ROOT / self.paths["evaluations"][key]
        args = ["-m", "gnrl.evaluate", "--adapter", adapter, "--games-json", UNSEEN, "--output", self.paths["evaluations"][key], "--max-wall-seconds", "14400"]
        for k, value in dict(old.EVAL, max_generated_tokens=900000).items():
            args += ["--" + k.replace("_", "-"), str(value)]
        self.run("eval_" + key.replace("/", "_"), args, 14400, monitor=folder / "episodes.jsonl")
        technical_eval_check(folder, self.games, adapter, self.locks[UNSEEN])
        self.lock_directory(folder)
        self.state["evaluated"].append(key)
        self.update("evaluation_complete", completed_models=len(self.state["evaluated"]), episodes_per_model=134)

    def incomplete_report(self, reason):
        """Retain all cells and truthfully report whether unblinding had begun."""
        unblinded = self.state.get("outcome_unblinded") is True
        access = "unblinding_started_then_validation_incomplete" if unblinded else "not_unblinded"
        table = []
        for key in EVAL_ORDER:
            folder = ROOT / self.paths["evaluations"][key]
            try:
                records = technical_records(folder / "episodes.jsonl", self.games)
            except Exception:
                records = []
            lookup = {r["gamefile"]: r for r in records}
            for game in self.games:
                record = lookup.get(game, {"status": "missing", "stop_reason": None})
                status = "observed_validation_incomplete" if unblinded and record["status"] == "observed_blinded" else record["status"]
                table.append({"model": key, "gamefile": game, "status": status, "stop_reason": record["stop_reason"],
                              "outcome": None, "outcome_access": access})
        old.atomic_json(self.folder / "incomplete_confirmation_report.json", {"complete": False, "reason": reason,
            "planned_cells": 1206, "status_counts": dict(Counter(r["status"] for r in table)), "cells": table,
            "stages": self.state["stages"], "primary_contrasts": None, "practical_target_met": None,
            "outcome_unblinded": unblinded, "unblinding_started_utc": self.state.get("unblinding_started_utc"),
            "monitor_boundary": "Technical records are detected asynchronously after flush; an in-flight next episode may have started before owned-session termination. Its unflushed interaction count is unknown, not reported as zero.",
            "interpretation": ("Technical incompleteness after outcome unblinding began; full result validation did not complete. " if unblinded else
                               "Technical incompleteness before outcome unblinding; preserved observed records remain blinded. ") +
                              "No missing/technical outcome is recoded as a scientific failure, and no primary contrast or practical target is assessed."})

    def finalize(self):
        require(self.state["evaluated"] == list(EVAL_ORDER), "Cannot unblind incomplete confirmation")
        self.verify()
        # Persist before opening even the first outcome file. A later validation,
        # hash, or deadline failure must never be described as a blinded failure.
        self.update("unblinding_complete_batch", outcome_unblinded=True, unblinding_started_utc=old.utc())
        episodes = {key: old.rows(ROOT / self.paths["evaluations"][key] / "episodes.jsonl") for key in EVAL_ORDER}
        evidence = {key: [validate_complete_episode(ep, game) for ep, game in zip(records, self.games)] for key, records in episodes.items()}
        table, counts, costs = [], {}, {}
        for i, game in enumerate(self.games):
            initial = [(episodes[key][i]["transitions"][0]["observation"], episodes[key][i]["transitions"][0]["admissible_commands"]) for key in EVAL_ORDER]
            require(all(r == initial[0] for r in initial), "Unseen initial observations/admissibility differ")
            outcome = {str(seed): {arm: episodes[str(seed) + "/" + arm][i]["success"] for arm in ARMS} for seed in SEEDS}
            table.append({"gamefile": game, **self.scene_map[game], "outcomes": outcome})
        for key, records in episodes.items():
            summary = old.read_json(ROOT / self.paths["evaluations"][key] / "summary.json")
            require(all(type(r.get("success")) is bool for r in records), "Nonbinary/missing outcome")
            successes = sum(r["success"] for r in records)
            checked = evidence[key]
            require(summary["successes"] == successes and summary.get("environment_steps") == sum(r["steps"] for r in checked) and
                    summary.get("generated_tokens") == sum(r["generated_tokens"] for r in checked) and summary.get("scored_tokens") == 0,
                    "Evaluation success/steps/token summary mismatch")
            require(summary.get("success_rate_attempted") == summary.get("success_rate_planned_lower_bound") == successes / 134,
                    "Complete evaluation rates differ from verified outcomes")
            counts[key] = {"successes": successes, "denominator": 134, "rate": successes / 134}
            attempts = [t for r in records for t in r["transitions"]]
            seed, arm = key.split("/")
            costs[key] = {"evaluation": summary, "training": old.read_json(ROOT / self.paths["checkpoints"][seed][arm] / "summary.json"),
                "training_exposure": self.audits[int(seed)]["exposures"][arm], "invalid_actions": sum(r["invalid_actions"] for r in checked),
                "action_attempts": len(attempts), "episode_lengths_including_failures": [r["environment_steps"] for r in records]}
        grouped = {}
        for grouping in ("family", "scene"):
            grouped[grouping] = {value: {"tasks": sum(r[grouping] == value for r in table),
                "success_counts": {key: sum(r["outcomes"][key.split("/")[0]][key.split("/")[1]] for r in table if r[grouping] == value) for key in EVAL_ORDER}}
                for value in sorted({r[grouping] for r in table})}
        analysis = analyze_complete(table, self.plan["estimands_and_analysis"]["scene_map_known_from_manifest_only"]["tasks_by_scene"])
        old.atomic_json(self.folder / "complete_confirmation_report.json", {"complete": True, "counts": counts, "table": table,
            "grouped": grouped, "analysis": analysis, "run_costs": costs, "stage_elapsed_costs": self.state["stages"],
            "inherited_cost_evidence_sha256": {name: value for name, value in self.locks.items() if name.endswith(("summary.json", "cost_progress.json"))},
            "scope": "Observed fixed134 benchmark, three fixed seeds/four scenes/fixed60 data; descriptive intervals, no population-wide statistical confirmation.",
            "raw_initial_state_comparison_available": False})
        self.lock([self.folder / "complete_confirmation_report.json"])
        self.verify()
        self.update("complete_needs_astra_review", complete=True, automatic_followup=False)

    def main(self):
        self.verify()
        self.entry_gate()
        for key in TRAIN_ORDER:
            self.train(key)
        self.nine_model_barrier()
        for key in EVAL_ORDER:
            self.evaluate(key)
        self.finalize()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adoption-json", required=True, type=Path)
    args = parser.parse_args(argv)
    require(sys.platform.startswith("linux") and ROOT.resolve() == Path("/home/jsy/GNRL_20261012").resolve(), "Future runtime only in chemistry workspace")
    require(Path(sys.executable).resolve() == Path("/usr/bin/python3").resolve() and hasattr(os, "pidfd_open") and
            hasattr(signal, "pidfd_send_signal"), "Native pidfd system controller required")
    os.chdir(ROOT)
    def stopped(signum, frame):
        raise RuntimeError("Confirmation interrupted by signal " + str(signum))
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM):
        signal.signal(sig, stopped)
    signal.setitimer(signal.ITIMER_REAL, 172800 - CLEANUP_RESERVE)
    queue = None
    try:
        queue = ConfirmationQueue(args.adoption_json)
        queue.main()
        return 0
    except BaseException as error:
        if queue is not None:
            queue.incomplete_report(f"{type(error).__name__}: {error}")
            queue.update("incomplete_needs_astra_review", complete=False, error=f"{type(error).__name__}: {error}")
        else:
            print("Confirmation preflight refused: " + str(error), flush=True)
        return 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    raise SystemExit(main())

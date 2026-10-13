"""Pure fixtures only: no real adoption, phase1 outcomes, simulator or model."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from scripts import collect_successful_paths_v2 as runner


class GateFixture:
    def __init__(self, root):
        self.root = Path(root)
        self.path = self.root / "toy_gate_input.json"
        self.games = [f"/toy/valid_seen/task{i}/game.tw-pddl" for i in range(24)]
        for name in runner.required_adoption_hash_paths():
            self.write(name, {})
        self.write(runner.CONFIG, {
            "fixed_training_cohort": {"frozen_gamefiles": [f"/toy/train/task{i}" for i in range(60)]},
            "feasibility_budget_and_gate": runner.FIXED_BUDGETS,
            "finite_assignment_enumeration": {
                "max_new_recipe_attempts_per_task": 7,
                "max_retained_candidates_per_task_including_original": 8,
                "max_assignment_tuples_considered_per_task": 4096}})
        self.write(runner.PRIOR_FAILURE, {"complete": False, "coverage_gate_passed": False,
            "environment_step_calls_started": 507, "reset_calls_started": 100})
        self.write(runner.PREFLIGHT, {"complete": True, "records": [{"complete": True, "gamefile": f"/toy/train/task{i}"} for i in range(60)],
            "module_sha256": runner.file_hash(self.root / runner.MODULE), "actual_environment_steps": 0,
            "actual_reset_calls_started": 60, "candidate_search_attempts": 0})
        self.write(runner.DEV_MANIFEST, {"split": "valid_seen", "games": self.games})
        for arm in runner.ARMS:
            paths = runner.arm_paths(arm)
            self.write(paths["training_summary"], {
                "matched_step_budget_complete": True, "optimizer_steps": 100,
                "requested_steps": 100, "stop_reason": "max_steps"})
            self.write(paths["training_config"], {
                **runner.TRAIN_PROTOCOL, "data": runner.ARM_DATA[arm],
                "data_sha256": runner.file_hash(self.root / runner.ARM_DATA[arm])})
            self.write(paths["evaluation_config"], {
                **runner.EVAL_PROTOCOL,
                "adapter": f"/toy/checkpoints/phase1_{arm}_seed42_v1/final",
                "manifest_sha256": runner.file_hash(self.root / runner.DEV_MANIFEST)})
        self.counts = {"canonical": 4, "uniform": 5, "brp": 5}
        self.write_counts()
        self.refresh()

    def write(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def read(self, name):
        return runner.read_json(self.root / name)

    def write_counts(self):
        for arm, successes in self.counts.items():
            paths = runner.arm_paths(arm)
            self.write(paths["evaluation_summary"], {
                "complete": True, "technical_errors": 0, "planned_episodes": 24,
                "attempted_episodes": 24, "successes": successes})
            rows = [{"gamefile": game, "success": i < successes, "seed": 20261012,
                     "stop_reason": "environment_done" if i < successes else "max_steps"}
                    for i, game in enumerate(self.games)]
            (self.root / paths["evaluation_episodes"]).write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def refresh(self):
        self.adoption = {"schema_version": 1, "kind": "successful_path_cpu_feasibility_adoption",
            "adopted": True, "cpu_only": True, "scope": "frozen60_successful_path_feasibility_only",
            "phase1_complete": True, "no_directional_benefit": True, "phase1_successes": self.counts,
            "candidate_config_sha256": runner.file_hash(self.root / runner.CONFIG),
            "module_sha256": runner.file_hash(self.root / runner.MODULE),
            "runner_sha256": runner.file_hash(self.root / runner.RUNNER),
            "file_sha256": {name: runner.file_hash(self.root / name)
                            for name in runner.required_adoption_hash_paths()}}
        self.write("toy_gate_input.json", self.adoption)

    def validate(self):
        return runner.validate_adoption(self.path, self.root)


class AdoptionTests(unittest.TestCase):
    def test_missing_gate_does_not_read_outcomes(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "read_json") as read:
            with self.assertRaisesRegex(RuntimeError, "Explicit --adoption-json"):
                runner.validate_adoption(Path(directory) / "missing.json", directory)
            read.assert_not_called()

    def test_complete_tie_can_pass_without_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = GateFixture(directory)
            _, receipt = fixture.validate()
            self.assertEqual(receipt["phase1_successes"], fixture.counts)
            self.assertTrue(receipt["phase1_complete"])

    def test_directional_benefit_rejected_even_if_adoption_claims_otherwise(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = GateFixture(directory)
            fixture.counts["brp"] = 6
            fixture.write_counts(); fixture.refresh()
            with self.assertRaisesRegex(RuntimeError, "strictly exceeds both"):
                fixture.validate()

    def test_incomplete_training_evaluation_and_missing_protocol_rejected(self):
        changes = [("training_summary", "optimizer_steps", 99),
                   ("evaluation_summary", "complete", False),
                   ("evaluation_summary", "technical_errors", 1),
                   ("training_config", "lora_alpha", None),
                   ("evaluation_config", "selection", "sum_logprob")]
        for file_key, field, value in changes:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                fixture = GateFixture(directory)
                name = runner.arm_paths("brp")[file_key]
                data = fixture.read(name); data[field] = value; fixture.write(name, data)
                fixture.refresh()
                with self.assertRaises(RuntimeError): fixture.validate()

    def test_mutated_source_and_false_boolean_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = GateFixture(directory)
            fixture.write(runner.MODULE, {"mutated": True})
            with self.assertRaisesRegex(RuntimeError, "source binding"):
                fixture.validate()
            fixture.refresh()
            fixture.adoption["cpu_only"] = 1
            fixture.write("toy_gate_input.json", fixture.adoption)
            with self.assertRaisesRegex(RuntimeError, "cpu_only"):
                fixture.validate()

    def test_truncated_episodes_or_inconsistent_counts_rejected(self):
        for mode in ("truncate", "duplicate", "budget", "count"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                fixture = GateFixture(directory)
                paths = runner.arm_paths("canonical")
                path = fixture.root / paths["evaluation_episodes"]
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                if mode == "truncate": rows.pop()
                elif mode == "duplicate": rows[-1]["gamefile"] = rows[0]["gamefile"]
                elif mode == "budget": rows[-1]["stop_reason"] = "wall_time_budget"
                else:
                    data = fixture.read(paths["evaluation_summary"])
                    data["successes"] += 1; fixture.write(paths["evaluation_summary"], data)
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                fixture.refresh()
                with self.assertRaises(RuntimeError): fixture.validate()


class CanonicalAndFailureTests(unittest.TestCase):
    def source(self):
        games = [f"/toy/train/game{i}" for i in range(120)]
        rows = [{"gamefile": game, "kind": "trajectory", "success": True,
                 "replay_verified": True, "reset_replay_verified": True,
                 "split": "train", "seed": 20261012, "transitions": [], "task": "测试"}
                for game in games]
        lines = [(json.dumps(row, ensure_ascii=False, separators=(", ", ": ")) + "\r\n").encode()
                 for row in rows]
        return games, rows, lines

    def test_selected_canonical_line_bytes_unchanged(self):
        games, _, lines = self.source()
        selected = games[::2]
        pairs = runner.select_native_rows(b"".join(lines), games, selected, 20261012)
        self.assertEqual([line for _, line in pairs], lines[::2])

    def test_source_failures_wrong_order_seed_and_missing_newline_rejected(self):
        games, rows, lines = self.source()
        variants = [b"".join(lines[:-1]), b"".join(reversed(lines)), b"".join(lines).rstrip()]
        rows[0]["success"] = False
        variants.append(b"\n".join(json.dumps(row).encode() for row in rows) + b"\n")
        for raw in variants:
            with self.assertRaises(RuntimeError):
                runner.select_native_rows(raw, games, games[::2], 20261012)
        with self.assertRaises(RuntimeError):
            runner.select_native_rows(b"".join(lines), games, games[::2], 42)

    def test_controller_failure_preserves_cost_and_worker_diagnosis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner.atomic_json(root / "cost_progress.json", {"environment_step_calls_started": 19,
                               "environment_step_calls_completed": 18, "pending_call": "step"})
            runner.atomic_json(root / "summary.json", {"complete": True, "coverage_gate_passed": True,
                               "stop_reason": "budget_exceeded"})
            runner.write_controller_failure(root, "toy timeout", -15)
            result = runner.read_json(root / "summary.json")
            self.assertFalse(result["complete"])
            self.assertFalse(result["coverage_gate_passed"])
            self.assertEqual(result["worker_stop_reason"], "budget_exceeded")
            self.assertEqual(result["last_durable_cost_progress"]["environment_step_calls_started"], 19)
            self.assertIn("toy timeout", (root / "controller_events.jsonl").read_text())

    def test_all_worker_return_paths_cleanup_owned_group(self):
        # Mocked process only: never starts a simulator, subprocess or real job.
        for code, has_summary in ((0, True), (1, True), (7, False)):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); output = root / "outputs" / "toy"
                adoption = root / "toy_gate_input.json"; adoption.write_text("{}")
                def wait(timeout=None):
                    if has_summary:
                        runner.atomic_json(output / "summary.json", {
                            "complete": code == 0, "coverage_gate_passed": False,
                            "stop_reason": "complete" if code == 0 else "technical_error"})
                    return code
                process = SimpleNamespace(pid=123456, returncode=code, wait=wait)
                with patch.object(runner, "ROOT", root), \
                     patch.object(runner.sys, "platform", "linux"), \
                     patch.object(runner.signal, "signal"), \
                     patch.object(runner, "validate_adoption", return_value=({}, {"adoption_sha256": runner.file_hash(adoption)})), \
                     patch.object(runner, "cumulative_costs", return_value={}), \
                     patch.object(runner.subprocess, "Popen", return_value=process), \
                     patch.object(runner, "process_start_ticks", return_value="100"), \
                     patch.object(runner, "terminate_owned_group") as cleanup:
                    result = runner.main(["--adoption-json", str(adoption), "--output", str(output)])
                    self.assertEqual(result, 0 if code == 0 else 1)
                    cleanup.assert_called_once_with(process)
                    if code == 1:
                        self.assertEqual(runner.read_json(output / "summary.json")["stop_reason"], "technical_error")

    def test_cleanup_refuses_reused_leader_pid(self):
        process = SimpleNamespace(pid=123456, owned_start_ticks="100")
        with patch.object(runner, "process_start_ticks", return_value="200"), \
             patch.object(runner.os, "killpg", create=True) as kill:
            runner.terminate_owned_group(process)
            kill.assert_not_called()

    def test_cleanup_signals_original_group_after_leader_reaped(self):
        process = SimpleNamespace(pid=123456, owned_start_ticks="100", poll=lambda: 1, wait=lambda: 1)
        calls = []
        def kill(pgid, sig):
            calls.append((pgid, sig))
            if sig == 0: raise ProcessLookupError()
        with patch.object(runner, "process_start_ticks", return_value=None), \
             patch.object(runner.os, "killpg", kill, create=True), \
             patch.object(runner.signal, "SIGKILL", 9, create=True):
            runner.terminate_owned_group(process)
        self.assertEqual(calls[0], (process.pid, runner.signal.SIGTERM))
        self.assertTrue(all(pgid == process.pid for pgid, _ in calls))


if __name__ == "__main__":
    unittest.main()

"""Pure stored-metadata and temporary-byte fixtures; no model/env/process calls."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analysis import verify_recovery_learning_postqueue_v1 as r
from gnrl import recovery_projection_v1 as p


def fixture():
    paths = {stage: "outputs/" + stage for stage in r.STAGES}
    paths["queue"] = "outputs/queue"
    adoption = {"output_paths": paths, "diagnostic_parent_adoption": "configs/prior.json"}
    status = {"complete": True, "current_stage": "needs_astra_learning_result_review", "outcome_unblinded": True,
        "exploratory_practical_screen_passed": False,
        "stages": [{"name": name, "pid": i + 20, "start_ticks": i + 200, "owned_session": i + 20,
                    "returncode": 0, "owned_cleanup_complete": True} for i, name in enumerate(r.STAGES)]}
    locks, data = {}, {}
    audit_path = paths["joint_token_audit"]
    locks[audit_path] = "audit"
    data[audit_path] = {"kind": "recovery_joint_training_token_audit_v1", "complete": True,
        "score_sft_token_context_mismatches": 0, "schedule_sha256": "schedule",
        "arms": {arm: {"totals": {"target_tokens": 800}} for arm in r.TRAIN}}
    for arm in r.TRAIN:
        folder = paths["train_" + arm]
        for name in ("summary.json", "config.json", "metrics.jsonl", "example_exposures.jsonl", "final_adapter_inventory.json", "final/training_state.json"):
            locks[folder + "/" + name] = "hash"
        common = {"optimizer_steps": 100, "target_tokens": 800, "token_audit_sha256": "audit",
                  "schedule_sha256": "schedule", "sampled_examples": 800, "component_exposures": {"bc":400,"recovery":400}}
        data[folder + "/final/training_state.json"] = dict(common)
        data[folder + "/summary.json"] = dict(common, complete=True, matched_step_budget_complete=True,
            stop_reason="max_steps", technical_errors=[], requested_steps=100, actual_backward_target_tokens=800, expected_target_tokens=800)
    previous = dict.fromkeys(p.COUNTERS, 0)
    data[adoption["diagnostic_parent_adoption"]] = {"output_paths": {"diagnostic": "outputs/prior"}}
    data["outputs/prior/cost_progress.json"] = {"totals": dict(previous)}
    for arm in r.ARMS:
        folder = paths["eval_" + arm]
        for name in ("summary.json", "config.json", "cost_progress.json", "episodes.jsonl", "events.jsonl"):
            locks[folder + "/" + name] = "hash"
        counts = dict.fromkeys(p.COUNTERS, 0)
        counts.update(resets_started=48, resets_completed=48)
        totals = {key: previous[key] + counts[key] for key in p.COUNTERS}
        costs = {"kind": "recovery_cost_ledger", "active_call": None, "unknown_generation_after_error": False,
            "global_limits": p.GLOBAL_LIMITS, "stage_limits": r.evaluator.STAGE_LIMITS, "totals": totals, "stage_counts": counts}
        data[folder + "/cost_progress.json"] = costs
        data[folder + "/summary.json"] = {"complete": True, "stage": "evaluation", "arm": arm,
            "planned_episodes": 48, "attempted_episodes": 48, "completed_episodes": 48, "technical_errors": 0,
            "partial_performance_summary": None, "episodes_sha256": "hash", "costs": costs}
        previous = totals
    status["cumulative_costs"] = previous
    return adoption, status, locks, data


class LearningPostqueueTests(unittest.TestCase):
    def test_complete_technical_barrier_never_reads_outcomes(self):
        adoption, status, locks, data = fixture()
        trace = []
        def read(name):
            self.assertFalse(name.endswith(("episodes.jsonl", "events.jsonl", "result.json")))
            trace.append("read")
            return data[name]
        stages = r.metadata_barrier(adoption, status, locks, read, lambda *identity: trace.append("exit"))
        self.assertEqual(trace[:8], ["exit"] * 8)
        self.assertEqual([row["name"] for row in stages], list(r.STAGES))
        self.assertTrue(all(row["exited"] for row in stages))
        self.assertFalse(status["exploratory_practical_screen_passed"])

    def test_incomplete_or_missing_stage_cannot_open_metadata(self):
        adoption, status, locks, data = fixture()
        for change in ("complete", "stage", "cleanup"):
            changed = copy.deepcopy(status)
            if change == "complete": changed["complete"] = False
            if change == "stage": changed["stages"].pop()
            if change == "cleanup": changed["stages"][-1]["owned_cleanup_complete"] = False
            with patch.object(r.old, "read_json") as read:
                with self.assertRaises(RuntimeError):
                    r.metadata_barrier(adoption, changed, locks, read, lambda *args: None)
                read.assert_not_called()

    def test_live_worker_refuses_before_first_metadata_read(self):
        adoption, status, locks, data = fixture()
        def live(*identity): raise RuntimeError("live worker")
        with patch.object(r.old, "read_json") as read:
            with self.assertRaisesRegex(RuntimeError, "live worker"):
                r.metadata_barrier(adoption, status, locks, read, live)
            read.assert_not_called()

    def test_incomplete48_cost_reset_or_target_mismatch_rejected(self):
        for kind in ("episodes", "cost", "target", "missing_lock"):
            adoption, status, locks, data = fixture()
            if kind == "episodes": data[adoption["output_paths"]["eval_" + r.ARMS[-1]] + "/summary.json"]["completed_episodes"] = 47
            if kind == "cost": data[adoption["output_paths"]["eval_" + r.ARMS[-1]] + "/cost_progress.json"]["totals"]["resets_started"] -= 48
            if kind == "target": data[adoption["output_paths"]["train_" + r.TRAIN[0]] + "/final/training_state.json"]["target_tokens"] -= 1
            if kind == "missing_lock": locks.pop(adoption["output_paths"]["eval_" + r.ARMS[0]] + "/events.jsonl")
            with self.assertRaises(RuntimeError): r.metadata_barrier(adoption, status, locks, data.__getitem__, lambda *args: None)

    def adapters(self, root):
        paths, locks, items = {}, {}, {}
        for arm in r.ARMS:
            directory = r.v.ADAPTER if arm == r.ARMS[0] else "checkpoints/" + arm + "/final"
            if arm != r.ARMS[0]: paths["train_" + arm] = "checkpoints/" + arm
            target = root / directory
            target.mkdir(parents=True)
            values = {}
            for name in ("adapter_config.json", "training_state.json", "adapter_model.safetensors"):
                path = target / name
                path.write_bytes((arm + name).encode())
                values[directory + "/" + name] = hashlib.sha256(path.read_bytes()).hexdigest()
            items[arm] = {"path": directory, "file_sha256": values}
            locks.update(values)
        return {"complete": True, "arms": list(r.ARMS), "adapters": items}, paths, locks, items[r.ARMS[0]]["file_sha256"]

    def test_all_four_adapter_maps_are_measured_and_added_file_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            barrier, paths, locks, initial = self.adapters(root)
            with patch.object(r, "ROOT", root), patch.object(r.v, "relative", side_effect=lambda name: root / name):
                measured = r.measure_adapters(barrier, paths, locks, initial, lambda: None)
                self.assertEqual(measured, {arm: item["file_sha256"] for arm, item in barrier["adapters"].items()})
                (root / barrier["adapters"][r.ARMS[1]]["path"] / "added.json").write_text("extra")
                with self.assertRaisesRegex(RuntimeError, "inventory"):
                    r.measure_adapters(barrier, paths, locks, initial, lambda: None)

    def test_adapter_changed_bytes_wrong_role_or_unbound_file_rejected(self):
        for change in ("bytes", "role", "lock"):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                barrier, paths, locks, initial = self.adapters(root)
                first = next(iter(barrier["adapters"][r.ARMS[1]]["file_sha256"]))
                if change == "bytes": (root / first).write_bytes(b"changed")
                if change == "role": barrier["adapters"][r.ARMS[1]]["path"] = r.v.ADAPTER
                if change == "lock": locks.pop(first)
                with patch.object(r, "ROOT", root), patch.object(r.v, "relative", side_effect=lambda name: root / name):
                    with self.assertRaises(RuntimeError): r.measure_adapters(barrier, paths, locks, initial, lambda: None)

    def test_adapter_addition_during_measurement_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            barrier, paths, locks, initial = self.adapters(root)
            original = r.v.measure_one
            changed = []
            def measured(path, check):
                value = original(path, check)
                if not changed:
                    (root / r.v.ADAPTER / "late_loader_config.json").write_text("{}")
                    changed.append(True)
                return value
            with patch.object(r, "ROOT", root), patch.object(r.v, "relative", side_effect=lambda name: root / name), \
                 patch.object(r.v, "measure_one", side_effect=measured):
                with self.assertRaisesRegex(RuntimeError, "inventory changed during"):
                    r.measure_adapters(barrier, paths, locks, initial, lambda: None)

    def base(self, root):
        files=[]
        for i in range(28):
            name=f"file{i}.bin";data=f"fixture{i}".encode()
            (root/name).write_bytes(data)
            files.append({"name":name,"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()})
        return {"files":files}

    def test_full28_base_is_measured_and_declared_size_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);inventory=self.base(root)
            self.assertEqual(r.measure_base(root,inventory,lambda:None),{f["name"]:f["sha256"] for f in inventory["files"]})
            inventory["files"][0]["bytes"]+=1
            with self.assertRaisesRegex(RuntimeError,"declared file size"):r.measure_base(root,inventory,lambda:None)

    def test_base_addition_during_measurement_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);inventory=self.base(root);original=r.v.measure_one;changed=[]
            def measured(path,check):
                value=original(path,check)
                if not changed:
                    (root/"late_weights.safetensors").write_bytes(b"fixture")
                    changed.append(True)
                return value
            with patch.object(r.v,"measure_one",side_effect=measured):
                with self.assertRaisesRegex(RuntimeError,"Base inventory changed during"):
                    r.measure_base(root,inventory,lambda:None)

    def test_expired_model_measurement_does_not_enumerate_or_read(self):
        def expired():raise TimeoutError("deadline")
        with patch.object(Path,"rglob") as scan,patch.object(r.v,"measure_one") as read:
            with self.assertRaisesRegex(TimeoutError,"deadline"):r.measure_base(Path("missing"),{"files":[]},expired)
            scan.assert_not_called();read.assert_not_called()

    def test_outcome_or_resolved_alias_rejected_before_json_parse(self):
        with patch.object(r.old,"read_json") as parse:
            for name in ("outputs/fake/episodes.jsonl","outputs/fake/events.jsonl","outputs/fake/result.json"):
                with self.assertRaisesRegex(RuntimeError,"parsing forbidden"):r.read_metadata(name)
            with patch.object(r.v,"relative",return_value=Path("outputs/fake/result.json")):
                with self.assertRaisesRegex(RuntimeError,"aliases"):r.read_metadata("outputs/fake/summary.json")
            parse.assert_not_called()

    def test_metadata_paths_still_use_normalized_workspace_resolver(self):
        with patch.object(r.old,"read_json") as parse:
            for name in ("../escape.json","outputs/../escape.json","/absolute.json","outputs//summary.json"):
                with self.assertRaises(RuntimeError):r.read_metadata(name)
            parse.assert_not_called()

    def test_receipt_is_exclusive(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "receipt.json"
            r.write_receipt(path, {"complete": True})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError): r.write_receipt(path, {"complete": False})
            self.assertEqual(path.read_bytes(), before)

    def test_helper_bytes_exactly_bound(self):
        self.assertEqual(r.old.sha256(r.ROOT / r.HELPER), r.HELPER_SHA256)


if __name__ == "__main__":
    unittest.main()

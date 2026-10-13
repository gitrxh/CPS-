"""Pure local fixtures. No process launch, simulator, model or outcome-file reads."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from analysis import run_successful_path_confirmation_candidate_v1 as c


def base_audit():
    keys = [["game" + str(i // 6), i % 6] for i in range(349)]
    examples = {}
    for arm in c.ARMS:
        examples[arm] = [{"gamefile": game, "step": step, "prompt_tokens": 100 + i % 5,
            "target_tokens": 1 + i % 3, "input_tokens": 101 + i % 5 + i % 3,
            "prompt_ids_sha256": str(i if arm != "reference" else i + 1),
            "action_ids_sha256": "target" + str(i), "history_turns_dropped": i % 2} for i, (game, step) in enumerate(keys)]
    return {"complete": True, "examples_per_arm": 349, "score_sft_token_mismatches": 0,
        "teacher_fields_used": False, "settings": {"seed": 42, "optimizer_steps": 100, "batch_size": 1,
            "gradient_accumulation": 8, "max_context_tokens": 2048, "max_action_tokens": 64},
        "ordered_example_keys": keys, "examples": examples, "exposures": {},
        "comparisons": {"reference_vs_canonical": {}, "reference_vs_uniform": {}}}


def episode(game, **fields):
    return {"gamefile": game, "seed": 20261012, "stop_reason": "max_steps", **fields}


class ConfirmationTests(unittest.TestCase):
    def test_candidate_and_prior_learning_receipts_cannot_execute(self):
        for value in ({"status": "candidate", "execution_authorized": False},
                      {"schema_version": 1, "kind": "successful_path_learning_adoption", "adopted": True}):
            with self.assertRaisesRegex(RuntimeError, "adoption gate"):
                c.check_adoption_shape(value)

    def test_seed_audit_reuses_hashes_but_changes_exact_schedule(self):
        base = base_audit()
        a, b = c.derive_seed_audit(base, 123), c.derive_seed_audit(base, 456)
        self.assertNotEqual(a["sampled_indices"], b["sampled_indices"])
        self.assertEqual(a["sampled_indices"], c.sample_indices(349, seed=123))
        self.assertEqual(a["examples"], base["examples"])
        self.assertEqual(len(a["sampled_indices"]), 800)
        for segment in (a["sampled_indices"][:349], a["sampled_indices"][349:698]):
            self.assertEqual(set(segment), set(range(349)))
        self.assertEqual(a["exposures"]["canonical"]["sampled_target_tokens"],
            sum(base["examples"]["canonical"][i]["target_tokens"] for i in a["sampled_indices"]))
        self.assertEqual(base["settings"]["seed"], 42)

    def test_audit_rejects_changed_keys_counts_and_seed(self):
        for mutation in ("keys", "counts", "seed"):
            base = base_audit()
            if mutation == "keys": base["examples"]["uniform"][0]["gamefile"] = "other"
            if mutation == "counts": base["examples"]["uniform"][0]["input_tokens"] = 999
            if mutation == "seed": base["settings"]["seed"] = 123
            with self.assertRaises(RuntimeError): c.derive_seed_audit(base, 123)

    def test_monitor_does_not_consult_outcome_and_preserves_partial_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.jsonl"
            first = json.dumps(episode("a", success={"not": "a binary outcome"})) + "\n"
            second = json.dumps(episode("b", success=False))
            path.write_text(first + second[:20])
            tracker = c.TechnicalMonitor(path, ["a", "b"])
            records = tracker.poll()
            c.assert_no_technical(records)
            self.assertEqual(len(records), 1)
            self.assertNotIn("success", records[0])
            with path.open("a") as stream: stream.write(second[20:] + "\n")
            self.assertEqual(len(tracker.poll()), 2)
            self.assertEqual(len(tracker.poll()), 2)
            c.assert_no_technical(c.technical_records(path, ["a", "b"], final=True))

    def test_monitor_stops_for_error_unknown_budget_or_identity(self):
        for fields in ({"error": "fixture"}, {"stop_reason": "wall_time_budget"},
                       {"stop_reason": None}, {"stop_reason": "generated_token_budget"}):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "episodes.jsonl"
                path.write_text(json.dumps(episode("a", success=True, **fields)) + "\n")
                with self.assertRaisesRegex(RuntimeError, "Technical"):
                    c.assert_no_technical(c.TechnicalMonitor(path, ["a"]).poll())
                with self.assertRaisesRegex(RuntimeError, "identity"):
                    c.TechnicalMonitor(path, ["wrong"]).poll()

    def test_final_partial_or_missing_episode_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.jsonl"
            path.write_text(json.dumps(episode("a", success=False)))
            with self.assertRaisesRegex(RuntimeError, "Truncated"):
                c.technical_records(path, ["a"], final=True)
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(RuntimeError, "Missing"):
                c.technical_records(path, ["a", "b"], final=True)

    def test_six_training_and_nine_model_barriers(self):
        queue = c.ConfirmationQueue.__new__(c.ConfirmationQueue)
        queue.state = {"trained": list(c.TRAIN_ORDER[:-1])}
        with self.assertRaisesRegex(RuntimeError, "Six new"):
            queue.nine_model_barrier()
        queue.paths, queue.locks = {"queue": "outputs/fake"}, {}
        with self.assertRaisesRegex(RuntimeError, "Nine-model"):
            queue.evaluate("42/canonical")

    def test_incomplete_report_distinguishes_before_and_after_unblinding(self):
        for unblinded in (False, True):
            with self.subTest(unblinded=unblinded), tempfile.TemporaryDirectory() as directory:
                queue = c.ConfirmationQueue.__new__(c.ConfirmationQueue)
                queue.folder = Path(directory)
                queue.games = ["game" + str(i) for i in range(134)]
                queue.paths = {"evaluations": {key: str(queue.folder / key.replace("/", "_")) for key in c.EVAL_ORDER}}
                first_folder = Path(queue.paths["evaluations"][c.EVAL_ORDER[0]])
                first_folder.mkdir()
                (first_folder / "episodes.jsonl").write_text(json.dumps(episode(queue.games[0], success=True)) + "\n")
                queue.state = {"stages": [], "outcome_unblinded": unblinded}
                if unblinded: queue.state["unblinding_started_utc"] = "fixture timestamp"
                queue.incomplete_report("fixture technical failure")
                report = json.loads((queue.folder / "incomplete_confirmation_report.json").read_text())
                self.assertEqual(len(report["cells"]), 1206)
                self.assertEqual(report["outcome_unblinded"], unblinded)
                self.assertIsNone(report["primary_contrasts"])
                self.assertTrue(all(r["outcome"] is None for r in report["cells"]))
                expected = "unblinding_started_then_validation_incomplete" if unblinded else "not_unblinded"
                self.assertEqual({r["outcome_access"] for r in report["cells"]}, {expected})
                self.assertEqual(report["cells"][0]["status"], "observed_validation_incomplete" if unblinded else "observed_blinded")
                if unblinded:
                    self.assertNotIn("remain blinded", report["interpretation"])
                    self.assertEqual(report["unblinding_started_utc"], "fixture timestamp")

    def test_each_training_seed_is_checked_against_its_own_audit(self):
        config = dict(c.old.TRAIN, seed=123, data_sha256="data", protocol="alfworld-causal-v1")
        summary = {"matched_step_budget_complete": True, "optimizer_steps": 100, "requested_steps": 100,
                   "stop_reason": "max_steps", "target_tokens": 1600}
        c.validate_train(summary, config, 123, "data", 1600)
        for seed, tokens in ((42, 1600), (123, 1599)):
            with self.assertRaises(RuntimeError): c.validate_train(summary, config, seed, "data", tokens)

    def test_final_checkpoint_token_and_protocol_receipt(self):
        state = {"optimizer_steps": 100, "target_tokens": 1600, "protocol": "alfworld-causal-v1"}
        c.validate_final_state(state, 1600)
        for bad in ({"target_tokens": 1599}, {"protocol": "different"}, {"optimizer_steps": 99}):
            with self.assertRaises(RuntimeError): c.validate_final_state({**state, **bad}, 1600)

    def test_unblinded_episode_flags_are_verified_before_analysis(self):
        ep = {"gamefile": "g", "seed": 20261012, "success": True, "environment_steps": 1,
              "stop_reason": "environment_done", "terminal": True, "final_observation": "next",
              "transitions": [{"step": 0, "observation": "obs", "next_observation": "next",
                  "admissible_commands": ["take a"], "action": "take a", "won": True, "done": True,
                  "policy": {"selection": "generate", "raw_completion": " take a ", "is_admissible": True, "generated_tokens": 4}}]}
        self.assertTrue(c.validate_complete_episode(ep, "g")["success"])
        for mutation in ("success", "admissible", "tokens"):
            bad = copy.deepcopy(ep)
            if mutation == "success": bad["success"] = False
            if mutation == "admissible": bad["transitions"][0]["policy"]["is_admissible"] = False
            if mutation == "tokens": bad["transitions"][0]["policy"]["generated_tokens"] = 65
            with self.assertRaises(RuntimeError): c.validate_complete_episode(bad, "g")

    def test_owned_cleanup_on_stage_success_failure_and_technical_record(self):
        for scenario in ("success", "nonzero", "wait_failure", "technical_record"):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as directory:
                queue = c.ConfirmationQueue.__new__(c.ConfirmationQueue)
                queue.folder = Path(directory)
                queue.deadline = c.time.monotonic() + 30
                queue.python, queue.env, queue.games = "fixture-python", {}, ["g"]
                queue.state = {"stages": []}
                queue.verify = queue.wait_gpu = lambda: None
                queue.update = lambda *a, **kw: None
                path = queue.folder / "episodes.jsonl"
                row = episode("g", success=False)
                if scenario == "technical_record": row["error"] = "fixture error"
                path.write_text(json.dumps(row) + "\n")
                process = SimpleNamespace(pid=91, returncode=3 if scenario == "nonzero" else 0)
                def wait(timeout):
                    if scenario == "wait_failure": raise RuntimeError("fixture wait failure")
                    return process.returncode
                process.wait = wait
                with patch.object(c.subprocess, "Popen", return_value=process), \
                     patch.object(c.old, "process_record", return_value={"session": 91, "start_ticks": 123}), \
                     patch.object(c.old, "cleanup_owned") as cleanup, \
                     patch.object(c.signal, "pthread_sigmask", create=True), \
                     patch.object(c.signal, "SIG_BLOCK", 0, create=True), \
                     patch.object(c.signal, "SIG_SETMASK", 1, create=True), \
                     patch.object(c.signal, "SIGHUP", 1, create=True), \
                     patch.object(c.signal, "SIGALRM", 14, create=True):
                    if scenario == "success": queue.run("fixture", [], 7200, monitor=path)
                    else:
                        with self.assertRaises(RuntimeError): queue.run("fixture", [], 7200, monitor=path)
                    cleanup.assert_called_once_with(process, 123)

    def test_complete_analysis_rejects_missing_and_reports_negative(self):
        scenes = {"10": 77, "219": 11, "308": 27, "424": 19}
        table = []
        for scene, size in scenes.items():
            for i in range(size):
                table.append({"gamefile": scene + "/" + str(i), "scene": scene,
                    "outcomes": {str(seed): {"canonical": False, "uniform": True, "reference": False} for seed in c.SEEDS}})
        result = c.analyze_complete(table, scenes, draws=50)
        self.assertFalse(result["practical_replication_target_met"])
        self.assertEqual(result["contrasts"]["uniform"]["primary_mean_difference"], -1)
        self.assertEqual(result["contrasts"]["uniform"]["descriptive_crossed_bootstrap"]["percentile_interval_95"], [-1, -1])
        table[0]["outcomes"]["42"]["reference"] = None
        with self.assertRaisesRegex(RuntimeError, "Missing"):
            c.analyze_complete(table, scenes, draws=50)


if __name__ == "__main__":
    unittest.main()

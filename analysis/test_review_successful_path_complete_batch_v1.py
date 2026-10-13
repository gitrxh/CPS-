"""Synthetic local fixtures only; never opens actual experiment artifacts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from analysis import review_successful_path_complete_batch_v1 as m


def failure_episode(game="/toy/game"):
    trace = []
    for i in range(100):
        trace.append({"step": i, "observation": f"obs{i}", "admissible_commands": ["look"], "action": "invalid",
            "next_observation": f"obs{i + 1}", "won": False, "done": i == 99,
            "policy": {"selection": "generate", "raw_completion": " invalid ", "is_admissible": False, "generated_tokens": 1}})
    return {"gamefile": game, "seed": 20261012, "success": False, "environment_steps": 100,
        "stop_reason": "environment_done", "transitions": trace, "terminal": True, "final_observation": "obs100"}


def complete_fixture(root):
    """Entire artifact schema, synthetic outcomes, tiny fake weight receipts."""
    hashes = {}
    def put(name, value, jsonl=False):
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
        raw = ("".join(json.dumps(x) + "\n" for x in value) if jsonl else json.dumps(value)).encode()
        path.write_bytes(raw); hashes[name] = m.sha(raw); return hashes[name]
    paths = {"queue": "outputs/queue", "selection": "outputs/selection",
        "checkpoints": {a: "checkpoints/" + a for a in m.ARMS},
        "evaluations": {a: "outputs/eval_" + a for a in m.ARMS}}
    games = [f"/toy/valid_seen/{f}-object/trial{i}/game.tw-pddl" for f in sorted(m.FAMILIES) for i in range(4)]
    game_hashes = {g: "0" * 64 for g in games}
    manifest_hash = put(m.MANIFEST, {"split": "valid_seen", "games": games, "game_sha256": game_hashes})
    adoption = {"kind": "successful_path_learning_adoption", "adopted": True,
        "scope": "frozen60_single_seed_three_arm_exploratory_learning", "output_paths": paths,
        "file_sha256": {}, "original_file_sha256": game_hashes, "accepted_cpu_output": "outputs/cpu"}
    adoption_hash = put("adoption.json", adoption)
    put("outputs/queue/original_game_locks.json", game_hashes)
    stages = [{"name": n, "returncode": 0, "owned_cleanup_complete": True, "elapsed_seconds": 1}
        for n in ["select", "token_exposure_audit"] + ["train_" + a for a in m.ARMS] + ["eval_" + a for a in m.ARMS]]
    put("outputs/queue/status.json", {"complete": True, "trained_arms": list(m.ARMS), "evaluated_arms": list(m.ARMS),
                                    "adoption_sha256": adoption_hash, "stages": stages})
    audit = {"complete": True, "score_sft_token_mismatches": 0, "teacher_fields_used": False, "examples_per_arm": 800,
             "ordered_example_keys": list(range(800)), "sampled_indices": m.sample_indices(800),
             "examples": {}, "exposures": {}, "dataset_bytes_sha256": {}}
    successes = {"canonical": set(range(8)), "uniform": set(range(4, 12)), "reference": set(range(6)) | set(range(8, 12))}
    for arm in m.ARMS:
        cp = paths["checkpoints"][arm]; ev = paths["evaluations"][arm]
        data = paths["selection"] + "/" + m.DATA_FILES[arm]; data_hash = put(data, {"toy": arm})
        examples = [{"target_tokens": 8 + int(i < m.TARGET_TOKENS[arm] - 6400),
                     "input_tokens": 18 + int(i < m.TARGET_TOKENS[arm] - 6400)} for i in range(800)]
        audit["examples"][arm] = examples
        audit["exposures"][arm] = {"sampled_target_tokens": m.TARGET_TOKENS[arm],
                                   "sampled_input_tokens": m.TARGET_TOKENS[arm] + 8000}
        audit["dataset_bytes_sha256"][arm] = data_hash
        ts = {"matched_step_budget_complete": True, "optimizer_steps": 100, "requested_steps": 100,
              "stop_reason": "max_steps", "training_examples": 800, "target_tokens": m.TARGET_TOKENS[arm], "elapsed_seconds": 1}
        put(cp + "/summary.json", ts)
        put(cp + "/config.json", {**m.TRAIN, "protocol": m.PROTOCOL, "output": cp, "data": data, "data_sha256": data_hash})
        put(cp + "/final/training_state.json", {"optimizer_steps": 100, "target_tokens": m.TARGET_TOKENS[arm], "protocol": m.PROTOCOL})
        put(cp + "/final/adapter_config.json", {"base_model_name_or_path": m.MODEL, "r": 16, "lora_alpha": 32})
        put(cp + "/final/adapter_model.safetensors", {"synthetic": True})
        metrics = [{"optimizer_step": i, "target_tokens": sum(examples[j]["target_tokens"] for j in audit["sampled_indices"][:8 * i]),
                    "examples_per_step": 8, "action_token_nll": .1} for i in range(1, 101)] + [{"summary": ts}]
        put(cp + "/metrics.jsonl", metrics, jsonl=True)
        put(ev + "/config.json", {**m.EVAL, "protocol": m.PROTOCOL, "manifest_sha256": manifest_hash,
                                 "adapter": cp + "/final", "games_json": m.MANIFEST, "output": ev})
        episodes = []
        for i, game in enumerate(games):
            ep = failure_episode(game)
            if i in successes[arm]:
                ep["success"] = True; ep["transitions"][-1]["won"] = True
            episodes.append(ep)
        put(ev + "/episodes.jsonl", episodes, jsonl=True)
        put(ev + "/summary.json", {"complete": True, "technical_errors": 0, "planned_episodes": 24, "attempted_episodes": 24,
            "successes": len(successes[arm]), "environment_steps": 2400, "generated_tokens": 2400,
            "scored_tokens": 0, "forward_tokens": 3000, "elapsed_seconds": 1,
            "success_rate_attempted": len(successes[arm]) / 24, "success_rate_planned_lower_bound": len(successes[arm]) / 24})
    put("outputs/queue/token_exposure_audit.json", audit)
    put("outputs/queue/all_training_complete.json", {"arms": list(m.ARMS), "checkpoint_sha256": {k: v for k, v in hashes.items() if k.startswith("checkpoints/")}})
    put("outputs/queue/complete_batch_report.json", {"complete": True, "successes_out_of_24": {a: len(s) for a, s in successes.items()},
        "raw_initial_state_comparison_available": False, "reference_comparisons": {a: {"gains": 4, "losses": 2, "ties": 18, "rate_difference": 2 / 24} for a in ("canonical", "uniform")}})
    put("outputs/selection/selection_summary.json", {"complete": True, "training_gate_passed": True, "technical_errors": [],
        "scored_action_tokens": 100, "forward_tokens": 200, "cache_hits": 1, "model_calls": 2, "wall_seconds": 1})
    put("outputs/cpu/summary.json", {"complete": True, "coverage_gate_passed": True, "environment_step_calls_started": 100})
    put("outputs/successful_paths_cpu_v1/summary.json", {"complete": False, "environment_step_calls_started": 507})
    put("outputs/queue/locked_sha256.json", dict(hashes))


class ReviewTests(unittest.TestCase):
    def test_full_synthetic_batch_and_exact_exposure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); complete_fixture(root)
            result = m.review(root, "adoption.json")
            self.assertTrue(result["complete"])
            self.assertEqual({a: r["successes"] for a, r in result["arms"].items()}, {"canonical": 8, "uniform": 8, "reference": 10})
            self.assertEqual(result["reference_comparisons"]["canonical"]["gains"], 4)
            self.assertEqual(result["reference_comparisons"]["canonical"]["losses"], 2)
            self.assertEqual(result["arms"]["reference"]["training"]["target_tokens"], 6515)
            self.assertFalse(result["raw_initial_state_comparison_available"])
            m.write_outputs(result, root / "review")
            self.assertTrue((root / "review/paired_tasks.csv").exists())
    def test_all_summaries_read_before_refusing_and_no_episode_open(self):
        for bad in ("missing", "incomplete", "error"):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                paths = {"evaluations": {a: "eval/" + a for a in m.ARMS}}
                for arm in m.ARMS:
                    folder = root / paths["evaluations"][arm]; folder.mkdir(parents=True)
                    if bad == "missing" and arm == "uniform": continue
                    summary = {"complete": True, "technical_errors": 0, "planned_episodes": 24, "attempted_episodes": 24}
                    if arm == "uniform":
                        if bad == "incomplete": summary["complete"] = False
                        if bad == "error": summary["technical_errors"] = 1
                    (folder / "summary.json").write_text(json.dumps(summary))
                    (folder / "episodes.jsonl").write_text("MUST NEVER OPEN")
                reader = m.Reader(root)
                with patch.object(reader, "json", wraps=reader.json) as calls, patch.object(reader, "jsonl") as episodes:
                    with self.assertRaisesRegex(m.ReviewError, "episode reads refused"):
                        m.complete_summary_gate(reader, paths)
                    self.assertEqual(len(calls.call_args_list), 3)
                    self.assertTrue(all(str(c.args[0]).endswith("/summary.json") for c in calls.call_args_list))
                    episodes.assert_not_called()

    def test_scientific_failure_and_invalid_actions_remain_counted(self):
        result = m.validate_episode(failure_episode(), "/toy/game")
        self.assertFalse(result["success"])
        self.assertEqual(result["steps"], 100)
        self.assertEqual(result["invalid_actions"], 100)
        self.assertEqual(result["generated_tokens"], 100)

    def test_forged_success_admissibility_and_broken_history_rejected(self):
        for kind in ("success", "admissible", "chain"):
            ep = failure_episode()
            if kind == "success": ep["success"] = True
            elif kind == "admissible": ep["transitions"][0]["policy"]["is_admissible"] = True
            else: ep["transitions"][5]["observation"] = "copied wrong history"
            with self.subTest(kind=kind), self.assertRaises(m.ReviewError):
                m.validate_episode(ep, "/toy/game")

    def test_training_schedule_is_exactly_800_seeded_samples(self):
        self.assertEqual(len(m.sample_indices(250)), 800)
        self.assertEqual(m.sample_indices(250), m.sample_indices(250))
        self.assertEqual(set(m.sample_indices(250)[:250]), set(range(250)))

    def test_locked_byte_change_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "artifact.json").write_text("{}")
            reader = m.Reader(root); reader.locks = {"artifact.json": m.sha(b"{}")}
            reader.json("artifact.json")
            (root / "artifact.json").write_text('{"changed":true}')
            with self.assertRaises(m.ReviewError): reader.json("artifact.json")


if __name__ == "__main__": unittest.main()

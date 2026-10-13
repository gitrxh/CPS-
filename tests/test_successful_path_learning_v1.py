"""CPU fixtures for learning gates, exposure audit, and owned-child exit cleanup."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from gnrl import successful_path_token_audit_v1 as audit
from gnrl.policy import encode_action, encode_prompt
from scripts import run_successful_path_learning_v1 as queue


class Tokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return self.encode(json.dumps(messages, ensure_ascii=False)) + [1]


def dataset_fixture():
    datasets, selected, scores = {}, {}, {}
    tokenizer = Tokenizer()
    for arm in audit.ARMS:
        datasets[arm], selected[arm] = [], {}
        for index in range(2):
            game = "game" + str(index)
            actions = ["take a", "move a"] if arm != "reference" else ["take aaa", "move aaa"]
            cid = audit.digest(actions)
            selected[arm][game] = cid
            trace = [{"observation": "observation" + str(step), "admissible_commands": ["take a", "take aaa", "move a", "move aaa"],
                      "action": action, "hidden_future": "must not enter prompt"} for step, action in enumerate(actions)]
            row = {"gamefile": game, "task": "task", "initial_observation": "initial", "transitions": trace, "assignment": {"teacher": "secret"}}
            datasets[arm].append(row)
            history = []
            for step, t in enumerate(trace):
                prompt = encode_prompt(tokenizer, "task", history, t["observation"], t["admissible_commands"])
                target = encode_action(tokenizer, t["action"])
                scores[(game, cid, step)] = {"prompt_ids_sha256": audit.digest(prompt), "action_ids_sha256": audit.digest(target),
                                            "prompt_tokens": len(prompt), "action_tokens_including_eos": len(target)}
                history.append({"observation": t["observation"], "action": t["action"]})
    return tokenizer, datasets, selected, scores


class AuditTests(unittest.TestCase):
    def test_exact_schedule_and_measured_unequal_exposure(self):
        tokenizer, datasets, selected, scores = dataset_fixture()
        result = audit.audit(tokenizer, datasets, selected, scores)
        self.assertEqual(len(result["sampled_indices"]), 800)
        self.assertEqual(set(result["sampled_indices"]), {0, 1, 2, 3})
        self.assertEqual(result["exposures"]["canonical"]["sampled_target_tokens"], 5600)
        self.assertEqual(result["exposures"]["reference"]["sampled_target_tokens"], 7200)
        self.assertEqual(result["comparisons"]["reference_vs_canonical"]["changed_sampled_inputs"], 400)

    def test_teacher_metadata_cannot_change_student_tokens(self):
        tokenizer, datasets, selected, scores = dataset_fixture()
        before = audit.audit(tokenizer, datasets, selected, scores)
        datasets["reference"][0]["assignment"] = {"teacher": "different future"}
        datasets["reference"][0]["transitions"][0]["hidden_future"] = "different"
        after = audit.audit(tokenizer, datasets, selected, scores)
        self.assertEqual(before["examples"], after["examples"])

    def test_incomplete_or_changed_score_tokens_rejected(self):
        tokenizer, datasets, selected, scores = dataset_fixture()
        scores.pop(next(iter(scores)))
        with self.assertRaises(KeyError):
            audit.audit(tokenizer, datasets, selected, scores)
        tokenizer, datasets, selected, scores = dataset_fixture()
        next(iter(scores.values()))["prompt_ids_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "scoring/SFT"):
            audit.audit(tokenizer, datasets, selected, scores)

    def test_missing_task_or_length_mismatch_rejected(self):
        tokenizer, datasets, selected, scores = dataset_fixture()
        datasets["uniform"].pop()
        with self.assertRaisesRegex(ValueError, "ordered"):
            audit.audit(tokenizer, datasets, selected, scores)


class GateTests(unittest.TestCase):
    def test_candidate_and_cpu_receipts_are_not_learning_authorization(self):
        for value in ({"status": "candidate", "execution_authorized_by_this_file": False},
                      {"schema_version": 1, "kind": "successful_path_cpu_feasibility_adoption", "adopted": True}):
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "adoption.json"
                path.write_text(json.dumps(value))
                with self.assertRaisesRegex(RuntimeError, "Learning adoption gate"):
                    queue.validate_adoption(path)

    def test_incomplete_training_or_wrong_exposure_never_qualifies(self):
        config = {**queue.TRAIN, "data_sha256": "hash", "protocol": "alfworld-causal-v1"}
        summary = {"matched_step_budget_complete": True, "optimizer_steps": 100, "requested_steps": 100,
                   "stop_reason": "max_steps", "target_tokens": 1000}
        queue.validate_training(summary, config, "hash", 1000)
        for patch_value in ({"optimizer_steps": 99}, {"matched_step_budget_complete": False}, {"target_tokens": 999}):
            with self.assertRaises(RuntimeError):
                queue.validate_training({**summary, **patch_value}, config, "hash", 1000)

    def test_partial_eval_never_qualifies(self):
        with self.assertRaisesRegex(RuntimeError, "Incomplete evaluation"):
            queue.validate_evaluation({"complete": False, "attempted_episodes": 23}, [], {}, [], "x", "hash")

    def test_unknown_eval_stop_never_qualifies(self):
        games = [str(i) for i in range(24)]
        summary = {"complete": True, "planned_episodes": 24, "attempted_episodes": 24, "technical_errors": 0, "successes": 0}
        episodes = [{"gamefile": game, "seed": 20261012, "success": False, "stop_reason": "max_steps"} for game in games]
        config = {**queue.EVAL, "manifest_sha256": "hash", "adapter": "x", "protocol": "alfworld-causal-v1"}
        queue.validate_evaluation(summary, episodes, config, games, "x", "hash")
        episodes[0]["stop_reason"] = "unknown"
        with self.assertRaisesRegex(RuntimeError, "episode mismatch"):
            queue.validate_evaluation(summary, episodes, config, games, "x", "hash")

    def test_evaluation_requires_all_training(self):
        obj = queue.Queue.__new__(queue.Queue)
        obj.state = {"trained_arms": ["canonical", "uniform"]}
        with self.assertRaisesRegex(RuntimeError, "All three"):
            obj.evaluate("canonical")

    def test_model_inventory_addition_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            names = ["config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors.index.json", "weights.safetensors"]
            for name in names:
                (folder / name).write_text("{}")
            (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "weights.safetensors"}}))
            provenance = {"model_path": queue.MODEL, "algorithm": "SHA256", "files": [
                {"name": name, "bytes": (folder / name).stat().st_size, "sha256": queue.sha256(folder / name)} for name in names]}
            queue.verify_model(provenance, folder)
            (folder / "added_config.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "inventory"):
                queue.verify_model(provenance, folder)


def fake_stat(pid, session, ticks, state="S"):
    fields = [state, "1", str(session), str(session)] + ["0"] * 15 + [str(ticks)]
    return str(pid) + " (fixture process with spaces) " + " ".join(fields)


class OwnedCleanupTests(unittest.TestCase):
    def test_reaped_leader_owned_descendant_and_unrelated_session(self):
        with tempfile.TemporaryDirectory() as temporary:
            proc = Path(temporary)
            for pid, session, ticks in ((11, 10, 101), (12, 12, 110)):
                (proc / str(pid)).mkdir()
                (proc / str(pid) / "stat").write_text(fake_stat(pid, session, ticks))
            self.assertEqual([r["pid"] for r in queue.owned_members(10, 100, proc)], [11])
            (proc / "10").mkdir()
            (proc / "10" / "stat").write_text(fake_stat(10, 10, 999))
            with self.assertRaisesRegex(RuntimeError, "PID reused"):
                queue.owned_members(10, 100, proc)

    def test_owned_cleanup_on_success_nonzero_and_timeout(self):
        for outcome in (0, 3, subprocess.TimeoutExpired("fixture", 1)):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as temporary:
                obj = queue.Queue.__new__(queue.Queue)
                obj.folder = Path(temporary)
                obj.deadline = queue.time.monotonic() + 10
                obj.python, obj.env, obj.state = "python", {}, {"stages": []}
                obj.verify = lambda: None
                obj.update = lambda *args, **kwargs: None
                process = SimpleNamespace(pid=10, returncode=outcome if isinstance(outcome, int) else None)
                def wait(timeout):
                    if isinstance(outcome, Exception):
                        raise outcome
                    return outcome
                process.wait = wait
                with patch.object(queue.subprocess, "Popen", return_value=process), \
                     patch.object(queue, "process_record", return_value={"session": 10, "start_ticks": 100}), \
                     patch.object(queue, "cleanup_owned") as cleanup, \
                     patch.object(queue.signal, "pthread_sigmask", create=True), \
                     patch.object(queue.signal, "SIG_BLOCK", 0, create=True), \
                     patch.object(queue.signal, "SIG_SETMASK", 1, create=True), \
                     patch.object(queue.signal, "SIGHUP", 1, create=True):
                    if outcome == 0:
                        obj.run("fixture", [], gpu=False)
                    else:
                        with self.assertRaises((RuntimeError, subprocess.TimeoutExpired)):
                            obj.run("fixture", [], gpu=False)
                    cleanup.assert_called_once_with(process, 100)


if __name__ == "__main__":
    unittest.main()

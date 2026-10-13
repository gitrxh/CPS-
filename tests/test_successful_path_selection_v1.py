"""Pure toy scorer/exports; no real model, environment collection or adoption."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gnrl import successful_path_pools_v2 as pools_module
from gnrl import successful_path_selection_v1 as selection
from gnrl.policy import encode_prompt
from tests.test_successful_path_pools_v2 import make_pool, toy_domain


class Tokenizer:
    eos_token_id = 1
    def encode(self, text, add_special_tokens=False): return [ord(c) + 2 for c in text]
    def apply_chat_template(self, messages, **kwargs):
        return self.encode(json.dumps(messages, ensure_ascii=False))


class ToyPolicy:
    tokenizer = Tokenizer()
    max_action_tokens = 64
    def __init__(self, nonfinite=False):
        self.scored_tokens = self.forward_tokens = self.generated_tokens = 0
        self.nonfinite = nonfinite; self.seen = []
    def prompt_ids(self, task, history, observation, commands):
        allowed = json.dumps([task, history, observation, commands])
        if "secret_obj" in allowed or "PRIVATE_FUTURE" in allowed: raise AssertionError("Privileged prompt leak")
        self.seen.append(copy.deepcopy(history))
        return encode_prompt(self.tokenizer, task, history, observation, commands, 2048, 64)
    def score_encoded_actions(self, prompt, actions):
        action = actions[0]; count = len(self.tokenizer.encode(action)) + 1
        decoded = "".join(chr(i - 2) for i in prompt)
        value = -0.001 * count if "apple 2" in action or 'Action: take apple 2' in decoded else -0.1 * count
        self.scored_tokens += count; self.forward_tokens += len(prompt) + count
        return [{"action": action, "token_count": count, "logprob": float("nan") if self.nonfinite else value}]


def toy60():
    base, _, _ = make_pool()
    base["candidates"] = base["candidates"][:3]
    result = []
    for i in range(60):
        pool = copy.deepcopy(base); game = f"/toy/train/task{i}/game.tw-pddl"
        pool["gamefile"] = pool["canonical_source_row"]["gamefile"] = game
        raw = (json.dumps(pool["canonical_source_row"], ensure_ascii=False) + "\r\n").encode()
        pool["canonical_source_row_digest"] = pools_module.digest(pool["canonical_source_row"])
        pool["canonical_source_line_utf8"] = raw.decode()
        pool["canonical_source_line_sha256"] = hashlib.sha256(raw).hexdigest()
        result.append(pool)
    return result


class SelectorTests(unittest.TestCase):
    def setUp(self):
        domain = patch.object(pools_module, "LOCKED_DOMAIN_SHA256", hashlib.sha256(toy_domain().encode()).hexdigest())
        domain.start(); self.addCleanup(domain.stop)

    def test_tie_prefers_original_then_lexical_without_epsilon(self):
        pool, _, _ = make_pool(); ids = [c["candidate_id"] for c in pool["candidates"]]
        scores = {cid: {"logprob": -10.0} for cid in ids}
        self.assertEqual(selection.reference_choice(pool, scores)[0]["candidate_id"], pool["canonical_candidate_id"])
        scores[pool["canonical_candidate_id"]]["logprob"] = -10.000000000001
        self.assertEqual(selection.reference_choice(pool, scores)[0]["candidate_id"], min(set(ids) - {pool["canonical_candidate_id"]}))
        del scores[ids[-1]]
        with self.assertRaises(ValueError): selection.reference_choice(pool, scores)

    def test_uniform_exact_registered_derivation_and_includes_original(self):
        import random
        pools = toy60(); seen = set()
        for pool in pools:
            chosen, receipt = selection.uniform_choice(pool)
            seed = int(hashlib.sha256(("42:" + selection.digest(pool)).encode()).hexdigest(), 16)
            index = random.Random(seed).randrange(len(pool["candidates"]))
            self.assertEqual(receipt["selected_index"], index)
            self.assertEqual(chosen, pool["candidates"][index]); seen.add(index)
        self.assertIn(0, seen)

    def test_actual_causal_history_eos_and_no_private_prompt_fields(self):
        pool, _, _ = make_pool(); pool["PRIVATE_FUTURE"] = "PRIVATE_FUTURE"
        policy = ToyPolicy(); scorer = selection.Scorer(policy); ledger = io.StringIO()
        first = scorer.score(pool, pool["candidates"][0], ledger)
        second = scorer.score(pool, pool["candidates"][1], ledger)
        rows = [json.loads(line) for line in ledger.getvalue().splitlines()]
        length = len(pool["candidates"][0]["transitions"])
        self.assertEqual(policy.seen[0], []); self.assertEqual(policy.seen[length], [])
        self.assertEqual(policy.seen[-1], [{"observation": t["observation"], "action": t["action"]} for t in pool["candidates"][1]["transitions"][:-1]])
        self.assertEqual(first["logprob"], sum(first["per_action_logprobs"]))
        for row, t in zip(rows[:length], pool["candidates"][0]["transitions"]):
            ids = policy.tokenizer.encode(t["action"]) + [policy.tokenizer.eos_token_id]
            self.assertEqual(row["action_ids_sha256"], selection.token_digest(ids))
            self.assertEqual(row["action_tokens_including_eos"], len(ids))
        self.assertEqual(len(rows), len(first["per_action_logprobs"]) + len(second["per_action_logprobs"]))

    def test_complete60_bytes_pairing_and_independent_selection_recheck(self):
        pools = toy60()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "selection"
            summary = selection.export_selection(pools, out, ToyPolicy(), input_receipt={}, model_receipt={}, adoption_receipt={})
            self.assertTrue(summary["complete"])
            self.assertTrue(summary["training_gate_passed"])
            self.assertEqual((out / "canonical_baseline.jsonl").read_bytes(), b"".join(p["canonical_source_line_utf8"].encode() for p in pools))
            self.assertEqual(len(selection.validate_selection_artifacts(pools, out)["records"]), 60)
            # Rehash a false argmax event: the independent validator must still fail.
            events = selection.rows(out / "selection_events.jsonl")
            events[0]["reference_candidate_id"] = "invented"
            (out / "selection_events.jsonl").write_text("".join(json.dumps(r) + "\n" for r in events))
            summary["output_sha256"]["selection_events.jsonl"] = selection.sha_file(out / "selection_events.jsonl")
            selection.write_json(out / "selection_summary.json", summary)
            with self.assertRaises(ValueError): selection.validate_selection_artifacts(pools, out)

    def test_budget_and_nonfinite_failure_emit_no_training_prefix(self):
        pool, _, _ = make_pool()
        for policy, limits in ((ToyPolicy(), {**selection.LIMITS, "max_scored_action_tokens": 1}), (ToyPolicy(nonfinite=True), None)):
            with tempfile.TemporaryDirectory() as directory:
                out = Path(directory) / "selection"
                summary = selection.export_selection([pool], out, policy, input_receipt={}, model_receipt={}, adoption_receipt={}, limits=limits)
                self.assertFalse(summary["complete"]); self.assertFalse(summary["training_gate_passed"])
                self.assertTrue(summary["technical_errors"])
                self.assertFalse(any((out / name).exists() for name in selection.ARM_FILES.values()))

    def test_incomplete_cpu_gate_rejected_before_model(self):
        with tempfile.TemporaryDirectory() as directory:
            selection.write_json(Path(directory) / "summary.json", {"complete": False, "coverage_gate_passed": False})
            with self.assertRaisesRegex(ValueError, "CPU acquisition incomplete"):
                selection.validate_acquisition(directory)

    def test_model_manifest_rejects_mutated_or_shadowing_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); model = root / "model"; model.mkdir()
            contents = {"config.json": "{}", "tokenizer.json": "{}", "tokenizer_config.json": "{}",
                "model.safetensors.index.json": json.dumps({"weight_map": {"x": "part.safetensors"}}), "part.safetensors": "toy weights"}
            for name, value in contents.items(): (model / name).write_text(value)
            provenance = root / "provenance.json"
            selection.write_json(provenance, {"model_path": str(model), "resolved_path": str(model),
                "files": [{"name": name, "bytes": (model / name).stat().st_size, "sha256": selection.sha_file(model / name)} for name in contents]})
            inventory = root / "inventory.json"
            data = selection.read_json(provenance); data["algorithm"] = "SHA256"
            selection.write_json(inventory, data)
            receipt = selection.verify_model(provenance, model, inventory)
            self.assertIsNone(receipt["adapter"])
            (model / "part.safetensors").write_text("bad weights")
            with self.assertRaises(ValueError): selection.verify_model(provenance, model, inventory)
            (model / "part.safetensors").write_text(contents["part.safetensors"])
            (model / "model.safetensors").write_text("unbound replacement")
            with self.assertRaisesRegex(ValueError, "Unbound loader"):
                selection.verify_model(provenance, model, inventory)


if __name__ == "__main__": unittest.main()

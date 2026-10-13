"""Pure scheduler/schema fixtures; no tokenizer, torch, model or simulator calls."""
import copy
from collections import Counter
import json
from pathlib import Path
import random
import tempfile
import unittest

from gnrl import train_recovery_v1 as training


def label_fixture():
    result = {}
    for arm in training.ARMS:
        result[arm] = []
        for i in range(3):
            result[arm].append({"kind": "recovery_supervised_label", "schema_version": 1,
                "anchor_id": "anchor" + str(i), "gamefile": "game" + str(i), "seed": 20261012,
                "after_actions": 4, "task": "original task", "observation": "current observation",
                "history": [{"observation": "observation" + str(j), "action": "act" + str(j)} for j in range(4)],
                "admissible_commands": ["action a", "action b"],
                "action": "action b" if arm == training.ARMS[-1] else "action a", "arm": arm,
                "pool_sha256": "pool" + str(i), "selection_sha256": "selection" + str(i)})
    return result


class ScheduleTests(unittest.TestCase):
    def test_exact_4_plus_4_interleaving_100_updates(self):
        schedule = training.exact_schedule(349, 7)
        self.assertEqual(len(schedule), 800)
        self.assertEqual(Counter(s["source"] for s in schedule), {"bc": 400, "recovery": 400})
        for i in range(100):
            window = schedule[i * 8:(i + 1) * 8]
            self.assertEqual([s["source"] for s in window], ["bc", "recovery"] * 4)
            self.assertEqual([s["microstep"] for s in window], list(range(8)))
            self.assertEqual({s["optimizer_step"] for s in window}, {i + 1})

    def test_frozen_seed_golden_sequences_and_schedule_digest(self):
        self.assertEqual(training.shuffled_indices(349, 400, 613300)[:12],
                         [61, 34, 162, 115, 96, 168, 199, 28, 290, 146, 11, 59])
        self.assertEqual(training.shuffled_indices(7, 400, 613301)[:15],
                         [2, 3, 1, 6, 4, 5, 0, 3, 6, 1, 5, 0, 4, 2, 3])
        self.assertEqual(training.digest(training.exact_schedule(349, 7)),
                         "725b533849055cdf0959d9086752ddef2ee7f3e379051365c53b88e9b509b52d")

    def test_shuffle_on_wrap_keeps_each_full_epoch_a_permutation(self):
        bc = training.shuffled_indices(349, 400, 613300)
        self.assertEqual(set(bc[:349]), set(range(349)))
        self.assertEqual(len(set(bc[349:])), 51)
        anchors = training.shuffled_indices(7, 400, 613301)
        for start in range(0, 399, 7):
            self.assertEqual(set(anchors[start:start + 7]), set(range(7)))

    def test_private_rng_and_independent_source_schedules(self):
        before = random.getstate()
        first = training.exact_schedule(349, 7)
        self.assertEqual(random.getstate(), before)
        random.seed(999)
        self.assertEqual(first, training.exact_schedule(349, 7))
        second = training.exact_schedule(349, 31)
        self.assertEqual([s["index"] for s in first if s["source"] == "bc"],
                         [s["index"] for s in second if s["source"] == "bc"])
        random.setstate(before)

    def test_empty_population_rejected_and_single_anchor_repeat_explicit(self):
        for bc, anchors in [(0, 1), (349, 0), (-1, 7)]:
            with self.assertRaises(ValueError):
                training.exact_schedule(bc, anchors)
        self.assertEqual({s["index"] for s in training.exact_schedule(349, 1)
                          if s["source"] == "recovery"}, {0})


class LabelBoundaryTests(unittest.TestCase):
    def test_only_target_changes_accepted_same_order(self):
        data = label_fixture()
        self.assertEqual(training.validate_labels(data), ["anchor0", "anchor1", "anchor2"])
        for i in range(3):
            contexts = [{k: v for k, v in training.causal_row(data[a][i]).items() if k != "action"}
                        for a in training.ARMS]
            self.assertEqual(contexts[0], contexts[1])
            self.assertEqual(contexts[0], contexts[2])

    def test_context_or_identity_reorder_rejected(self):
        for field, value in [("task", "repaired task"), ("observation", "counterfactual"),
                             ("anchor_id", "another anchor"), ("pool_sha256", "changed"),
                             ("selection_sha256", "changed")]:
            data = label_fixture()
            data[training.ARMS[1]][0][field] = value
            with self.assertRaisesRegex(ValueError, "identity/context"):
                training.validate_labels(data)
        data = label_fixture()
        data[training.ARMS[1]].reverse()
        with self.assertRaisesRegex(ValueError, "identity/context"):
            training.validate_labels(data)

    def test_incomplete_history_duplicate_anchor_and_inadmissible_target_rejected(self):
        data = label_fixture()
        data[training.ARMS[0]][0]["history"].pop()
        with self.assertRaisesRegex(ValueError, "history"):
            training.validate_labels(data)
        data = label_fixture()
        data[training.ARMS[0]][1]["anchor_id"] = "anchor0"
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            training.validate_labels(data)
        data = label_fixture()
        data[training.ARMS[2]][0]["action"] = "not admissible"
        with self.assertRaisesRegex(ValueError, "inadmissible"):
            training.validate_labels(data)

    def test_teacher_fields_never_enter_causal_allowlist(self):
        row = label_fixture()[training.ARMS[0]][0]
        expected = training.causal_row(row)
        polluted = copy.deepcopy(row)
        polluted.update(expert_plan=["future"], future_observation="privileged", score=99)
        polluted["history"][0]["state_after"] = "privileged state"
        self.assertEqual(training.causal_row(polluted), expected)
        self.assertEqual(set(expected), {"task", "history", "observation", "admissible_commands", "action"})


class AdapterContractTests(unittest.TestCase):
    def test_existing_lora_contract_acceptance_and_rejection(self):
        config = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 16, "lora_alpha": 32,
                  "lora_dropout": 0.05, "bias": "none", "target_modules": sorted(training.TARGET_MODULES)}
        training.validate_adapter_config(config)
        for key, value in [("r", 8), ("lora_alpha", 16), ("lora_dropout", 0.0),
                           ("use_dora", True), ("modules_to_save", ["lm_head"]),
                           ("rank_pattern", {"q_proj": 8})]:
            with self.assertRaises(ValueError):
                training.validate_adapter_config({**config, key: value})


def score_sft_fixture():
    label = label_fixture()[training.ARMS[0]][0]
    label["anchor_id"] = training.digest([label["gamefile"], label["seed"], label["after_actions"]])
    encoded = {"prompt_ids_sha256": training.digest([11, 12, 13]),
               "target_ids_sha256": training.digest([21, 0]), "prompt_tokens": 3, "target_tokens": 2}
    score = {"logprob": -1.5, "prompt_ids_sha256": encoded["prompt_ids_sha256"],
             "action_ids_sha256": encoded["target_ids_sha256"], "prompt_tokens": 3,
             "action_tokens_including_eos": 2,
             "context_sha256": training.digest({k: label[k] for k in training.CONTEXT_FIELDS})}
    selected = {"kind": "recovery_selection", "schema_version": 1, "complete": True,
                "eligible": True, "status": "eligible", "anchor_id": label["anchor_id"],
                "pool_sha256": label["pool_sha256"], "labels": dict.fromkeys(training.ARMS, label["action"]),
                "actions": [label["action"]], "reference_scores": {label["action"]: score},
                "base_scores": {label["action"]: copy.deepcopy(score)}}
    selected["artifact_sha256"] = training.digest(selected)
    label["selection_sha256"] = selected["artifact_sha256"]
    return label, selected, encoded


class ScoreSFTBoundaryTests(unittest.TestCase):
    def test_exact_selected_score_and_sft_encoding_match(self):
        training.verify_score_sft(*score_sft_fixture())

    def test_corrupt_prompt_or_action_token_hash_rejected(self):
        for field in ("prompt_ids_sha256", "action_ids_sha256"):
            label, selected, encoded = score_sft_fixture()
            selected["reference_scores"][label["action"]][field] = "incorrect_hash"
            with self.assertRaisesRegex(ValueError, "Score/SFT token/context"):
                training.verify_score_sft(label, selected, encoded)

    def test_corrupt_target_count_or_context_rejected(self):
        for field, value in (("action_tokens_including_eos", 3), ("prompt_tokens", 4),
                             ("context_sha256", "counterfactual_context")):
            label, selected, encoded = score_sft_fixture()
            selected["reference_scores"][label["action"]][field] = value
            with self.assertRaisesRegex(ValueError, "Score/SFT token/context"):
                training.verify_score_sft(label, selected, encoded)

    def test_selected_label_identity_or_target_rejected(self):
        for field, value in (("anchor_id", "other_anchor"), ("pool_sha256", "other_pool"),
                             ("selection_sha256", "other_selection"), ("action", "action b")):
            label, selected, encoded = score_sft_fixture()
            label[field] = value
            with self.assertRaisesRegex(ValueError, "selection identity"):
                training.verify_score_sft(label, selected, encoded)

    def test_complete_artifact_binding_and_event_corruption_rejection(self):
        label, selected, encoded = score_sft_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = [selected]
            for i in range(179):
                row = {"kind": "recovery_selection", "schema_version": 1, "complete": True,
                       "eligible": False, "status": "terminated_before_anchor", "anchor_id": "missing" + str(i),
                       "pool_sha256": "pool" + str(i)}
                row["artifact_sha256"] = training.digest(row)
                records.append(row)
            def write_rows(name, rows):
                (root / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            write_rows("records.jsonl", records)
            events = [{"kind": "recovery_action_score", "anchor_id": label["anchor_id"],
                       "pool_sha256": label["pool_sha256"], "role": role, "action": label["action"],
                       **selected[role + "_scores"][label["action"]]} for role in ("reference", "base")]
            write_rows("events.jsonl", events)
            paths, datasets = {}, {}
            for arm in training.ARMS:
                paths[arm] = root / (arm + ".jsonl")
                datasets[arm] = [{**label, "arm": arm}]
                write_rows(arm + ".jsonl", datasets[arm])
            ledger = {"kind": "recovery_score_ledger", "active_call": None, "calls_started": 2,
                      "calls_completed": 2, "scored_target_tokens": 4, "forward_tokens": 10}
            summary = {"complete": True, "stage": "score", "labels_locked_before_Q": True,
                       "record_count": 180, "eligible_anchors": 1, "scoring": ledger,
                       "records_sha256": training.sha_file(root / "records.jsonl"),
                       "label_sha256": {a: training.sha_file(p) for a, p in paths.items()}}
            for name, value in (("score_ledger.json", ledger), ("summary.json", summary),
                                ("config.json", {}), ("cost_progress.json", {})):
                (root / name).write_text(json.dumps(value), encoding="utf-8")
            selected_by_id, inventory = training.selection_artifacts(root, paths, datasets)
            self.assertEqual(selected_by_id, {label["anchor_id"]: selected})
            self.assertEqual(len(inventory["files"]), 9)
            events[0]["prompt_ids_sha256"] = "wrong"
            write_rows("events.jsonl", events)
            with self.assertRaisesRegex(ValueError, "Scoring event differs"):
                training.selection_artifacts(root, paths, datasets)


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from gnrl.policy import encode_prompt, messages, supervised_example
from gnrl.train_sft import collate_examples, load_examples
from gnrl.evaluate import rollout


class Tokenizer:
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        return [ord(c) + 10 for c in text]

    def apply_chat_template(self, rows, tokenize, add_generation_prompt):
        return self.encode("".join(row["content"] for row in rows))


class PipelineTests(unittest.TestCase):
    def test_prompt_and_padding_are_unsupervised(self):
        row = {"task": "goal", "history": [], "observation": "room",
               "admissible_commands": ["look"], "action": "look"}
        example = supervised_example(Tokenizer(), row, 4096)
        self.assertEqual(example["labels"][-5:], Tokenizer().encode("look") + [2])
        self.assertTrue(all(x == -100 for x in example["labels"][:-5]))
        fake_torch = SimpleNamespace(long="long", tensor=lambda x, dtype: x)
        short = {"input_ids": [9, 2], "labels": [-100, 2]}
        batch = collate_examples([example, short], 2, fake_torch)
        self.assertEqual(batch["labels"][1][2:], [-100] * (len(example["input_ids"]) - 2))
        self.assertEqual(batch["attention_mask"][1][:3], [1, 1, 0])

    def test_privileged_keys_not_rendered(self):
        history = [{"observation": "before", "action": "look",
                    "expert_action": "SECRET", "state_after": "SECRET"}]
        rendered = json.dumps(messages("task", history, "now", ["go north"]))
        self.assertNotIn("SECRET", rendered)
        self.assertNotIn("expert_action", rendered)

    def test_history_truncation_keeps_complete_recent_turns(self):
        tokenizer = Tokenizer()
        current = encode_prompt(tokenizer, "task", [], "now", ["look"], 4096)
        ids = encode_prompt(tokenizer, "task", [{"observation": "x" * 1000, "action": "look"}],
                            "now", ["look"], len(current) + 64)
        self.assertEqual(ids, current)

    def test_loader_reconstructs_causal_history(self):
        trajectory = {"success": True, "replay_verified": True, "task": "goal", "initial_observation": "initial",
                      "transitions": [{"observation": "initial", "admissible_commands": ["a"], "action": "a"},
                                      {"observation": "fresh", "admissible_commands": ["b"], "action": "b"}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            path.write_text(json.dumps(trajectory), encoding="utf-8")
            examples = load_examples(str(path))
        self.assertEqual(examples[0]["history"], [])
        self.assertEqual(examples[1]["history"], [{"observation": "initial", "action": "a"}])
        self.assertEqual(examples[1]["observation"], "fresh")

    def test_loader_rejects_unverified_trajectory_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            path.write_text(json.dumps({"success": True, "transitions": []}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "replay_verified"):
                load_examples(str(path))
            path.write_text(json.dumps({"kind": "diagnostic", "nll": 0.1}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "diagnostic"):
                load_examples(str(path))

    def test_online_rollout_uses_actual_successors(self):
        def state(obs, commands, done=False, won=False):
            return SimpleNamespace(observation=obs, admissible_commands=commands,
                                   reward=float(won), done=done, won=won)

        class Env:
            def reset(self):
                self.actions = []
                return state("initial", ["a"])

            def step(self, action):
                self.actions.append(action)
                if len(self.actions) == 1:
                    return state("changed by a", ["b"])
                return state("goal reached", [], done=True, won=True)

        class Policy:
            scored_tokens = 0

            def choose_action(self, task, history, observation, commands, selection):
                if history:
                    assert observation == "changed by a"
                    assert history == [{"observation": "initial", "action": "a"}]
                return commands[0], {}

        env = Env()
        result = rollout(env, Policy())
        self.assertEqual(env.actions, ["a", "b"])
        self.assertTrue(result["success"])
        self.assertEqual(result["environment_steps"], 2)


if __name__ == "__main__":
    unittest.main()

"""Adapter contract tests use a tiny stub, not evidence about ALFWorld results."""
import random
import unittest
from types import SimpleNamespace

from gnrl.alfworld_env import AlfworldReplayEnv, ReplayMismatch, canonical_facts, replay_actions, information_only_action
from gnrl.alfworld_probe import audit_horizon_one, collect_expert, is_world_noop, reclassify_h1_probe


def fact(name, entity="object_1"):
    return SimpleNamespace(name=name, arguments=[SimpleNamespace(name=entity, type="object")])


class StubEngine:
    """Distinct aliases reach one state; feedback deliberately remembers the alias."""
    def reset(self):
        self.stage, self.moves, self.prefix = 0, 0, []
        return self.state("initial")

    def state(self, feedback):
        return {"_facts": [fact("stage_" + str(self.stage))], "moves": self.moves,
                "score": float(self.stage == 2), "won": self.stage == 2, "lost": False,
                "feedback": feedback, "admissible_commands": ["a", "alias", "look"] if self.stage == 0 else ["finish", "look"],
                "extra.expert_plan": ["a" if self.stage == 0 else "finish"]}

    def step(self, command):
        self.prefix.append(command)
        if command in {"a", "alias"}:
            self.stage = 1
        elif command == "finish":
            self.stage = 2
        self.moves += 1
        return self.state("fresh:" + "/".join(self.prefix)), float(self.stage == 2), self.stage == 2


def make_env(max_steps=10):
    env = object.__new__(AlfworldReplayEnv)
    env.env = StubEngine()
    env.max_steps, env.seed = max_steps, 7
    env._rng = random.Random(7)
    env.interactions, env.resets = 0, 0
    env.current, env.actions = None, []
    return env


class ReplayContractTests(unittest.TestCase):
    def test_information_actions_remain_excluded_if_checked_fact_changes(self):
        for action in ("look", "help", "inventory", "examine apple 1", "  EXAMINE   fridge 2 "):
            self.assertTrue(information_only_action(action))
        for action in ("use desklamp 1", "go to fridge 1", "open fridge 1", "lookalike"):
            self.assertFalse(information_only_action(action))
        # Models ALFWorld help/inventory adding the same checked(agent) predicate.
        probe = {"expert_action": "inventory", "expert_world_noop": False,
                 "candidates": [{"action": "help", "world_noop": False,
                                 "exact_successor_match": True, "eligible_nontrivial": True}],
                 "nontrivial_alternatives": 1}
        fixed = reclassify_h1_probe(probe)
        self.assertEqual(fixed["nontrivial_alternatives"], 0)
        self.assertTrue(fixed["candidates"][0]["exact_successor_match"])
        self.assertTrue(fixed["candidates"][0]["eligible_nontrivial_before_information_filter"])
        self.assertEqual(probe["nontrivial_alternatives"], 1)  # Input evidence unchanged.
        for expert, candidate in (("inventory", "take apple 1 from table 1"),
                                  ("take apple 1 from table 1", "examine apple 1")):
            probe["expert_action"] = expert
            probe["candidates"][0]["action"] = candidate
            self.assertFalse(reclassify_h1_probe(probe)["candidates"][0]["eligible_nontrivial"])

    def test_entity_ids_are_not_collapsed(self):
        self.assertNotEqual(canonical_facts([fact("at", "apple_1")]),
                            canonical_facts([fact("at", "apple_2")]))

    def test_restore_checks_time_and_terminal_counters(self):
        env = make_env()
        env.reset()
        state = env.step("a")
        self.assertEqual(env.restore(["a"], state), state)
        tampered = dict(state.canonical_state, elapsed_steps=2)
        with self.assertRaises(ReplayMismatch):
            env.restore(["a"], tampered)

    def test_noop_exclusion_ignores_counter_increments_only_for_filter(self):
        env = make_env()
        before = env.reset()
        after = env.step("look")
        self.assertTrue(is_world_noop(before.canonical_state, after.canonical_state))
        self.assertNotEqual(before.canonical_state, after.canonical_state)

    def test_truncation_is_in_strict_state_and_prohibits_extra_step(self):
        env = make_env(max_steps=1)
        env.reset()
        result = env.step("a")
        self.assertTrue(result.done)
        self.assertTrue(result.canonical_state["truncated"])
        self.assertFalse(result.canonical_state["terminated"])
        with self.assertRaises(RuntimeError):
            env.step("finish")

    def test_changed_branch_and_suffix_observations_are_executed_fresh(self):
        env = make_env()
        original = collect_expert(env)["transitions"]
        probes, branches, counts = audit_horizon_one(env, original, max_branch_states=2)
        self.assertEqual(counts["nontrivial_h1_matches"], 1)
        self.assertEqual(counts["identical_commands_excluded"], 2)
        self.assertEqual(counts["noop_commands_excluded"], 2)
        fresh = branches[0]["transitions"]
        self.assertEqual(fresh[0]["action"], "alias")
        self.assertEqual(fresh[1]["observation"], "fresh:alias")
        self.assertEqual(fresh[1]["next_observation"], "fresh:alias/finish")
        self.assertNotEqual(fresh[1]["next_observation"], original[1]["next_observation"])
        self.assertEqual(fresh[1]["state_after"], original[1]["state_after"])

    def test_replay_refuses_suffix_after_termination(self):
        with self.assertRaises(ReplayMismatch):
            replay_actions(make_env(), ["a", "finish", "look"])


if __name__ == "__main__":
    unittest.main()

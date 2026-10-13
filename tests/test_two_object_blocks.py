import copy
from collections import Counter
from types import SimpleNamespace
import unittest

from gnrl.alfworld_env import StepResult
from gnrl.macro_project import SearchBudget, validate_pool
from gnrl.two_object_blocks import (IneligibleBlock, build_pool, canonical_row,
                                    collect_planner, compile_delivery_order,
                                    inspect_deliveries, task_entities)


class FakeDeliveryEnv:
    """Contract simulator with one-slot inventory and a closed shared target."""
    def __init__(self, target_open=False):
        self.initial_target_open = target_open
        self.names = {"o1": "apple 1", "o2": "apple 2", "s1": "table 1",
                      "s2": "table 2", "dest": "cabinet 1", "other": "cabinet 2"}
        self.env = SimpleNamespace(state={"_entity_infos": {
            k: SimpleNamespace(name=v) for k, v in self.names.items()}})
        self.native = ["go to table 1", "take apple 1 from table 1", "go to cabinet 1"]
        if not target_open:
            self.native += ["open cabinet 1"]
        self.native += ["move apple 1 to cabinet 1", "go to table 2", "take apple 2 from table 2",
                        "go to cabinet 1", "move apple 2 to cabinet 1"]

    def reset(self):
        self.loc = "middle"
        self.objects = {"o1": "s1", "o2": "s2"}
        self.opened = {"dest"} if self.initial_target_open else set()
        self.checked = set()
        self.held = None
        self.n = 0
        return self.result("initial task: put two apples in cabinet")

    def result(self, observation):
        facts = []

        def fact(name, *args):
            facts.append([name, [[value, kind] for value, kind in args]])

        fact("atlocation", ("a", "agent"), (self.loc, "location"))
        for rec in ("s1", "s2", "dest", "other"):
            fact("receptacleatlocation", (rec, "receptacle"), (rec, "location"))
            fact("receptacletype", (rec, "receptacle"),
                 ("CabinetType" if rec in ("dest", "other") else "TableType", "rtype"))
        fact("openable", ("dest", "receptacle"))
        for obj in ("o1", "o2"):
            fact("objecttype", (obj, "object"), ("AppleType", "otype"))
            fact("pickupable", (obj, "object"))
        for obj, rec in self.objects.items():
            fact("inreceptacle", (obj, "object"), (rec, "receptacle"))
        for rec in self.opened:
            fact("opened", (rec, "receptacle"))
        for rec in self.checked:
            fact("checked", (rec, "receptacle"))
        if self.held:
            fact("holds", ("a", "agent"), (self.held, "object"))
            fact("holdsany", ("a", "agent"))
        won = self.objects.get("o1") == self.objects.get("o2") == "dest"
        commands = ["go to " + self.names[r] for r in ("s1", "s2", "dest", "other")]
        if self.loc == "dest" and "dest" not in self.opened:
            commands.append("open cabinet 1")
        if self.held is None:
            commands += ["take " + self.names[o] + " from " + self.names[r]
                         for o, r in self.objects.items() if r == self.loc and (r != "dest" or r in self.opened)]
        elif self.loc == "dest" and "dest" in self.opened:
            commands.append("move " + self.names[self.held] + " to cabinet 1")
        elif self.loc == "other":
            commands.append("move " + self.names[self.held] + " to cabinet 2")
        commands = sorted(commands)
        state = dict(facts=sorted(facts), moves=self.n, elapsed_steps=self.n, max_steps=100,
                     score=float(won), reward=float(won), won=won, lost=False,
                     terminated=won, truncated=False, done=won, admissible_commands=commands)
        expert = self.native[self.n] if self.n < len(self.native) else None
        return StepResult(observation, commands, expert, float(won), won, won, state)

    def step(self, action):
        current = self.result("")
        if current.done or action not in current.admissible_commands:
            raise ValueError("Invalid test action: " + action)
        reverse = {v: k for k, v in self.names.items()}
        if action.startswith("go to "):
            self.loc = reverse[action[6:]]
        elif action.startswith("open "):
            rec = reverse[action[5:]]
            self.opened.add(rec)
            self.checked.add(rec)
        elif action.startswith("take "):
            obj, _ = action[5:].split(" from ")
            self.held = reverse[obj]
            del self.objects[self.held]
        elif action.startswith("move "):
            obj, rec = action[5:].split(" to ")
            self.objects[reverse[obj]] = reverse[rec]
            self.held = None
        self.n += 1
        return self.result("Actual feedback: " + action)


METADATA = {"task_type": "pick_two_obj_and_place", "pddl_params": {
    "object_target": "Apple", "parent_target": "Cabinet", "object_sliced": False,
    "mrecep_target": ""}}


class TwoObjectBlockTests(unittest.TestCase):
    def source(self, env):
        initial, trace = collect_planner(env, SearchBudget(1000, 60), 24)
        return {"gamefile": "fake/train/pick_two_obj_and_place-Apple/trial/game.tw-pddl",
                "seed": 0, "split": "train", "success": False,
                "initial_observation": initial.observation, "transitions": trace}

    def test_closed_shared_target_reversed_blocks_have_exact_equal_endpoints(self):
        env = FakeDeliveryEnv()
        pool = build_pool(self.source(env), METADATA, env, SearchBudget(1000, 60))
        self.assertEqual(len(pool["candidates"]), 2)
        original, reverse = [c["transitions"] for c in pool["candidates"]]
        self.assertEqual(original[-1]["state_after"], reverse[-1]["state_after"])
        self.assertEqual(Counter(t["action"] for t in original), Counter(t["action"] for t in reverse))
        self.assertEqual(sum(t["action"] == "open cabinet 1" for t in reverse), 1)
        self.assertEqual(reverse[1]["action"], "take apple 2 from table 2")
        self.assertEqual(reverse[1]["observation"], "Actual feedback: go to table 2")
        self.assertTrue(validate_pool(pool))

    def test_open_target_adds_no_padding_or_redundant_open(self):
        env = FakeDeliveryEnv(target_open=True)
        pool = build_pool(self.source(env), METADATA, env, SearchBudget(1000, 60))
        self.assertEqual([len(c["transitions"]) for c in pool["candidates"]], [8, 8])
        self.assertFalse(any(t["action"].startswith("open") for c in pool["candidates"] for t in c["transitions"]))

    def test_different_receptacle_instances_are_rejected(self):
        env = FakeDeliveryEnv()
        initial, trace = collect_planner(env, SearchBudget(1000, 60), 24)
        trace = copy.deepcopy(trace)
        for fact in trace[-1]["state_after"]["facts"]:
            if fact[0] == "inreceptacle" and fact[1][0][0] == "o2":
                fact[1][1][0] = "other"
        objects, targets = task_entities(initial, METADATA)
        with self.assertRaisesRegex(IneligibleBlock, "same target instance"):
            inspect_deliveries(trace, objects, targets)

    def test_canonical_training_row_marks_verified_and_separates_baseline(self):
        env = FakeDeliveryEnv()
        pool = build_pool(self.source(env), METADATA, env, SearchBudget(1000, 60))
        row = canonical_row(pool, "test_hash")
        self.assertTrue(row["replay_verified"])
        self.assertEqual(row["selection"]["arm"], "canonical_native_planner")
        self.assertNotIn("teacher_search", row)
        self.assertTrue(row["replay_evidence"]["raw_handcoded_trace_is_not_this_baseline"])

    def test_plan_limit_is_finite(self):
        env = FakeDeliveryEnv()
        with self.assertRaisesRegex(IneligibleBlock, "plan-step budget"):
            collect_planner(env, SearchBudget(1000, 60), 2)


if __name__ == "__main__":
    unittest.main()

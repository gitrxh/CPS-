import copy
import unittest

from gnrl.macro_project import SearchBudget
from gnrl.two_object_blocks import build_pool, collect_planner
from gnrl.two_object_blocks_v2 import build_pool_from_seed
from test_two_object_blocks import FakeDeliveryEnv, METADATA


class PlannerFreeEnv(FakeDeliveryEnv):
    def result(self, observation):
        value = super().result(observation)
        value.expert_action = None
        return value


class CachedPlannerTests(unittest.TestCase):
    def source(self):
        env = FakeDeliveryEnv(target_open=True)
        initial, trace = collect_planner(env, SearchBudget(1000, 60), 24)
        return {"gamefile": "train/pick_two_obj_and_place-Apple/trial/game.tw-pddl",
                "seed": 0, "split": "train", "success": True, "replay_verified": True,
                "reset_replay_verified": True, "initial_observation": initial.observation,
                "transitions": trace}

    def test_three_passes_without_expert_api_match_v1_candidates(self):
        row = self.source()
        budget = SearchBudget(1000, 60)
        v2 = build_pool_from_seed(row, METADATA, PlannerFreeEnv(target_open=True), budget)
        v1 = build_pool(row, METADATA, FakeDeliveryEnv(target_open=True), SearchBudget(1000, 60))
        self.assertEqual(budget.resets, 3)
        self.assertEqual(budget.steps, 24)
        self.assertEqual(len(v2["candidates"]), 2)
        self.assertEqual([c["candidate_id"] for c in v1["candidates"]],
                         [c["candidate_id"] for c in v2["candidates"]])
        for left, right in zip(v1["candidates"], v2["candidates"]):
            self.assertEqual(left["transitions"][-1]["state_after"], right["transitions"][-1]["state_after"])
            self.assertEqual([t["observation"] for t in left["transitions"]],
                             [t["observation"] for t in right["transitions"]])

    def test_unverified_seed_refused(self):
        row = self.source()
        row["replay_verified"] = False
        with self.assertRaisesRegex(ValueError, "verified successful"):
            build_pool_from_seed(row, METADATA, PlannerFreeEnv(target_open=True), SearchBudget(1000, 60))

    def test_source_counter_drift_is_not_ignored(self):
        row = copy.deepcopy(self.source())
        row["transitions"][3]["state_after"]["moves"] += 1
        with self.assertRaisesRegex(RuntimeError, "Independent full replay differed"):
            build_pool_from_seed(row, METADATA, PlannerFreeEnv(target_open=True), SearchBudget(1000, 60))


if __name__ == "__main__":
    unittest.main()

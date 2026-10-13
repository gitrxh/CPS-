"""Local synthetic records only; no active artifacts, model or simulator."""
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from analysis import review_recovery_acquisition_complete_v1 as r
from gnrl import recovery_projection_v1 as p
from tests.test_recovery_projection_v1 import ToyEnv, ToyPolicy, GAME


def cost(stage, initial=None):
    return p.BudgetLedger(stage_limits=dict(zip(p.GLOBAL_LIMITS, r.LIMITS[stage])),
                         deadline=time.monotonic() + 60, initial_totals=initial)


def bundle(early=False):
    events = {s: [] for s in r.STAGES}
    collect_cost = cost("collect")
    episode = p.collect_learner_task(ToyEnv(early_win=early), ToyPolicy(), GAME, collect_cost, events["collect"].append)
    oracle_cost = cost("oracle", collect_cost.totals)
    pools = [p.certify_anchor(lambda g, e: ToyEnv(e), a, oracle_cost, events["oracle"].append)
             for a in episode["anchors"]]
    replay_cost = cost("replay", oracle_cost.totals)
    replays = [p.replay_certificates(lambda g, e: ToyEnv(e), pool, replay_cost, events["replay"].append) for pool in pools]
    costs = dict(zip(r.STAGES, (collect_cost.snapshot(), oracle_cost.snapshot(), replay_cost.snapshot())))
    return episode, pools, replays, events, costs


def review(data):
    episode, pools, replays, events, costs = data
    out = r.review_rows([GAME], [{"gamefile": GAME, "after_actions": 4, "family": r.family(GAME)}],
                        [episode], pools, replays, {s: iter(rows) for s, rows in events.items()})
    out["costs"] = r.review_costs(costs, out["recomputed_workload"])
    return out


class ReviewFixtures(unittest.TestCase):
    def test_complete_actual_branches_costs_and_descriptive_only(self):
        out = review(bundle())
        self.assertEqual(out["families"][r.family(GAME)]["slots"], 3)
        self.assertTrue(out["diagnostic36_availability"][0]["eligible"])
        self.assertEqual(out["costs"]["stage_counts"]["collect"]["environment_steps_started"], 100)
        self.assertTrue(out["costs"]["ledger_dependent"])
        self.assertNotIn("training_gate_passed", out)

    def test_early_termination_three_unavailable_slots_without_false_Q(self):
        out = review(bundle(early=True))
        self.assertEqual(out["families"][r.family(GAME)]["available_anchors"], 0)
        self.assertEqual(out["costs"]["stage_counts"]["oracle"]["environment_steps_started"], 0)
        self.assertNotIn("Q", out["diagnostic36_availability"][0])

    def test_missing_pool_or_replay_stops(self):
        for index in (1, 2):
            data = list(bundle())
            data[index] = data[index][:-1]
            with self.assertRaisesRegex(ValueError, "missing"):
                review(data)

    def test_extra_or_missing_actual_events_stops(self):
        for stage in r.STAGES:
            data = bundle()
            data[3][stage].pop()
            with self.assertRaisesRegex(ValueError, "event"):
                review(data)

    def test_false_structural_and_future_context_stops(self):
        for field in ("status", "student_context"):
            data = bundle()
            anchor = data[0]["anchors"][0]
            if field == "status": anchor[field] = "terminated_before_anchor"
            else: anchor[field]["future"] = "privileged"
            data[0].pop("artifact_sha256")
            data[0]["artifact_sha256"] = p.digest(data[0])
            with self.assertRaises((ValueError, RuntimeError)):
                review(data)

    def test_success_tampering_and_replay_successor_stops(self):
        data = bundle()
        data[1][0]["branches"][0]["success"] = False
        with self.assertRaises(RuntimeError): review(data)
        data = bundle()
        data[2][0]["candidates"][0]["actual_transition"]["after"]["observation"] = "invented"
        with self.assertRaises((ValueError, RuntimeError)): review(data)

    def test_cost_omitted_prefix_or_cumulative_reset_stops(self):
        for stage, field in (("oracle", "environment_steps_started"), ("replay", "resets_started")):
            data = bundle()
            data[4][stage]["stage_counts"][field] -= 1
            with self.assertRaisesRegex(ValueError, "cost lineage"):
                review(data)

    def test_incomplete_queue_never_opens_outcome_streams(self):
        adoption = {"kind": "recovery_acquisition_parent_adoption", "adopted": True,
                    "authorized_stages": list(r.STAGES), "physical_gpu": 2,
                    "output_paths": {"queue": "outputs/queue"}}
        def metadata(path):
            if str(path) == "adoption": return adoption
            if Path(path).name == "status.json": return {"complete": False}
            raise AssertionError("Read beyond incomplete barrier")
        with tempfile.TemporaryDirectory() as root, patch.object(r, "read", side_effect=metadata), \
             patch.object(r, "jsonl", side_effect=AssertionError("Outcome read forbidden")):
            with self.assertRaisesRegex(ValueError, "Queue incomplete"):
                r.run(root, "adoption", "receipt", "hash")

    def test_non_exited_controller_refused_before_outcomes(self):
        adoption = {"kind": "recovery_acquisition_parent_adoption", "adopted": True,
            "authorized_stages": list(r.STAGES), "physical_gpu": 2,
            "output_paths": {s: "outputs/" + s for s in (*r.STAGES, "queue")}}
        status = {"complete": True, "current_stage": "needs_astra_acquisition_review", "adoption_sha256": "hash",
            "stages": [{"name": s, "returncode": 0, "owned_cleanup_complete": True} for s in r.STAGES]}
        costs = {"kind": "recovery_cost_ledger", "active_call": None, "unknown_generation_after_error": False,
                 "totals": dict.fromkeys(p.COUNTERS, 0)}
        receipt = {"kind": "recovery_acquisition_postqueue_integrity", "complete": True, "errors": [],
            "adoption_sha256": "hash", "queue_status_sha256": "hash", "locked_sha256_file_sha256": "hash",
            "controller": {"pid": 7, "start_ticks": 9, "exited": False}, "owned_workers_gone": True}
        def metadata(path):
            path = Path(path)
            if path.name == "adoption": return adoption
            if path.name == "status.json": return status
            if path.name == "receipt": return receipt
            if path.name == "locked_sha256.json": return {}
            if path.name == "cost_progress.json": return costs
            if path.name == "summary.json":
                stage = path.parent.name
                return {"complete": True, "stage": stage, "record_count": r.EXPECTED[stage],
                        "planned_records": r.EXPECTED[stage], "costs": costs}
            raise AssertionError("Read beyond exited-process barrier")
        with tempfile.TemporaryDirectory() as root, patch.object(r, "read", side_effect=metadata), \
             patch.object(r, "sha", return_value="hash"), \
             patch.object(r, "jsonl", side_effect=AssertionError("Outcome read forbidden")):
            with self.assertRaisesRegex(ValueError, "have not exited"):
                r.run(root, "adoption", "receipt", "hash")


if __name__ == "__main__":
    unittest.main()

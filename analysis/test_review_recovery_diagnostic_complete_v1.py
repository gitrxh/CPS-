"""Pure synthetic complete/negative cases; never opens active experiment files."""
import copy
from collections import Counter
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analysis import review_recovery_diagnostic_complete_v1 as r
from gnrl import recovery_diagnostic_v1 as d
from gnrl import recovery_projection_v1 as p
from tests.test_recovery_diagnostic_v1 import GAME, make_pool, scores, full_fixture


def score_fixture():
    pool = make_pool()
    pools = [pool]
    for k in (8, 16):
        anchor = {"kind": "recovery_anchor", "schema_version": 1, "gamefile": GAME, "seed": p.SEED,
            "after_actions": k, "anchor_id": p.digest([GAME, p.SEED, k]), "status": "terminated_before_anchor"}
        pools.append(d.seal({"schema_version": 1, "kind": "recovery_oracle_pool", "complete": True,
            "anchor": anchor, "anchor_id": anchor["anchor_id"], "eligible": False, "status": anchor["status"],
            "expert_action": None, "proposed_actions": [], "branches": [], "retained_candidates": []}))
    selected = [d.selection(x, scores(x), scores(x, 0)) if x["eligible"] else d.selection(x, {}, {}) for x in pools]
    replays = [{"complete": True, "anchor_id": x["anchor_id"], "pool_sha256": x["artifact_sha256"],
        "candidates": [{"action": c["action"], "certificate_sha256": c["certificate_sha256"],
            "replay_verified": True, "actual_transition": c["first_transition"]} for c in x["retained_candidates"]]} for x in pools]
    events = [{"kind": "recovery_action_score", "anchor_id": pool["anchor_id"], "pool_sha256": pool["artifact_sha256"],
        "role": role, "action": action, **selected[0][role + "_scores"][action]}
        for role in ("reference", "base") for action in selected[0]["actions"]]
    labels = {arm: [d.label_rows(pool, selected[0])[arm]] for arm in d.ARMS}
    planned = [{"gamefile": GAME, "after_actions": 4, "family": "pick_and_place_simple"}]
    return planned, pools, selected, replays, events, labels


def score_review(data):
    planned, pools, selected, replays, events, labels = data
    return r.validate_scores([GAME], planned, pools, selected, replays, iter(events), labels)


def q_fixture():
    pools, selected, planned, rows = full_fixture()
    indexed = {(x["anchor"]["gamefile"], x["anchor"]["after_actions"]): x for x in pools}
    events = [{"kind": "diagnostic_transition", "anchor_id": row["anchor_id"], "candidate_action": row["action"],
               "transition": t} for row in rows for t in row["transitions"]]
    differences = {arm: sum(x["labels"][arm] != x["labels"][d.ARMS[-1]] for x in selected if x["eligible"])
                   for arm in d.ARMS[:2]}
    return pools, selected, planned, rows, indexed, events, differences


class CompleteDiagnosticReviewTests(unittest.TestCase):
    def test_scores_events_both_references_and_labels_match(self):
        _, _, _, work = score_review(score_fixture())
        self.assertEqual(work, {"calls": 2, "target_tokens": 8, "forward_tokens": 20})

    def test_changed_score_event_or_label_rejected(self):
        data = score_fixture()
        data[4][0]["logprob"] += 1
        with self.assertRaisesRegex(ValueError, "event"):
            score_review(data)
        data = score_fixture()
        data[5][d.ARMS[0]][0]["task"] = "future"
        with self.assertRaisesRegex(ValueError, "label"):
            score_review(data)

    def test_complete_true_Q_and_independent_gate_match_worker(self):
        pools, selected, planned, rows, indexed, events, differences = q_fixture()
        values, work = r.validate_true_q(planned, indexed, rows, iter(events))
        result = r.independent_gate(planned, indexed, {s["anchor_id"]: s for s in selected}, values, differences)
        self.assertEqual(result, d.diagnostic_summary(pools, selected, planned, rows))
        self.assertFalse(result["training_gate_passed"])
        self.assertIsNone(result["cells"][1]["Q"])
        self.assertEqual(work, {"resets": 2, "prefix_steps": 8, "forced_steps": 2, "learner_steps": 2, "generated_tokens": 4})

    def test_missing_actual_branch_or_event_is_not_zero(self):
        for cut_events in (False, True):
            _, _, planned, rows, indexed, events, _ = q_fixture()
            if cut_events: events.pop()
            else: rows.pop()
            with self.assertRaisesRegex(ValueError, "Missing|missing"):
                r.validate_true_q(planned, indexed, rows, iter(events))

    def test_false_Q_and_causal_history_rejected_even_after_resealing(self):
        for mutate_Q in (False, True):
            _, _, planned, rows, indexed, events, _ = q_fixture()
            if mutate_Q: rows[0]["Q"] = not rows[0]["Q"]
            else: rows[0]["transitions"][1]["context_sha256"] = "future"
            body = dict(rows[0]); body.pop("artifact_sha256")
            rows[0] = d.seal(body)
            with self.assertRaises((RuntimeError, ValueError)):
                r.validate_true_q(planned, indexed, rows, iter(events))

    def test_arithmetic_pass_and_futility_thresholds(self):
        planned, pools, selected, values = [], {}, {}, {}
        for i in range(36):
            anchor = {"gamefile": str(i), "after_actions": 4, "family": str(i // 6)}
            planned.append(anchor)
            identity = str(i)
            pools[(str(i), 4)] = {"anchor_id": identity, "eligible": True, "status": "eligible",
                "expert_action": "e", "retained_candidates": [{"action": "e"}, {"action": "q"}]}
            selected[identity] = {"actions": ["e", "q"], "q": {"e": .25, "q": .75}, "q_base": {"e": .5, "q": .5}}
            values[identity] = {"e": 0, "q": 1}
        labels = dict.fromkeys(d.ARMS[:2], 12)
        result = r.independent_gate(planned, pools, selected, values, labels)
        self.assertTrue(result["training_gate_passed"])
        self.assertEqual(result["mean_q_minus_uniform"], .25)
        for s in selected.values(): s["q"] = {"e": .5, "q": .5}
        self.assertFalse(r.independent_gate(planned, pools, selected, values, labels)["training_gate_passed"])
        values.pop("0")
        with self.assertRaises((KeyError, ValueError)):
            r.independent_gate(planned, pools, selected, values, labels)

    def test_costs_caps_and_no_prefix_omission(self):
        previous = dict.fromkeys(p.COUNTERS, 0)
        previous["environment_steps_started"] = previous["environment_steps_completed"] = 120
        score_work = Counter(calls=2, target_tokens=8, forward_tokens=20)
        q_work = Counter(resets=2, prefix_steps=8, forced_steps=2, learner_steps=2, generated_tokens=4)
        score_ledger = {"kind": "recovery_score_ledger", "active_call": None, "target_limit": 500000,
            "forward_limit": 7000000, "calls_started": 2, "calls_completed": 2, "scored_target_tokens": 8, "forward_tokens": 20}
        score = dict.fromkeys(p.COUNTERS, 0); score["forward_tokens"] = 20
        q = dict.fromkeys(p.COUNTERS, 0)
        for k, v in (("resets", 2), ("environment_steps", 12), ("policy_calls", 2)):
            q[k + "_started"] = q[k + "_completed"] = v
        q.update(generated_tokens=4, forward_tokens=24)
        costs = {}; current = previous
        for stage, count, limits, labels in (
            ("score", score, dict.fromkeys(p.GLOBAL_LIMITS, 0), {}),
            ("diagnostic", q, dict(zip(p.GLOBAL_LIMITS, (28800, 288, 1843200))),
             {"diagnostic_prefix:reset": 2, "diagnostic_prefix:step": 8, "diagnostic_forced_action:step": 2,
              "diagnostic_learner_action:step": 2, "diagnostic_continuation:generate": 2})):
            current = {k: current[k] + count[k] for k in p.COUNTERS}
            costs[stage] = {"stage_counts": count, "totals": current, "stage_limits": limits,
                            "global_limits": p.GLOBAL_LIMITS, "calls_by_label": labels}
        r.validate_costs(previous, costs, score_ledger, score_work, q_work)
        with self.assertRaisesRegex(ValueError, "caps"):
            r.validate_costs(previous, costs, dict(score_ledger, target_limit=1000000), score_work, q_work)
        costs["diagnostic"]["stage_counts"]["environment_steps_started"] -= 8
        with self.assertRaisesRegex(ValueError, "lineage"):
            r.validate_costs(previous, costs, score_ledger, score_work, q_work)

    def test_incomplete_queue_cannot_open_outcome_stream(self):
        adoption = {"kind": "recovery_diagnostic_parent_adoption", "adopted": True,
            "authorized_stages": ["score", "diagnostic"], "physical_gpu": 2, "output_paths": {"queue": "outputs/queue"}}
        def metadata(path):
            if str(path) == "adoption": return adoption
            if Path(path).name == "status.json": return {"complete": False}
            raise AssertionError("Read beyond failed complete barrier")
        with tempfile.TemporaryDirectory() as root, patch.object(r, "read", side_effect=metadata), \
             patch.object(r, "jsonl", side_effect=AssertionError("Outcomes forbidden")):
            with self.assertRaisesRegex(ValueError, "queue incomplete"):
                r.run(root, "adoption", "receipt", "sha")

    def test_complete_summaries_do_not_bypass_live_controller_barrier(self):
        adoption = {"kind": "recovery_diagnostic_parent_adoption", "adopted": True,
            "authorized_stages": list(r.STAGES), "physical_gpu": 2,
            "output_paths": {s: "outputs/" + s for s in (*r.STAGES, "queue")}}
        status = {"complete": True, "current_stage": "needs_astra_diagnostic_review", "adoption_sha256": "sha",
            "gate_evaluable": True, "training_gate_passed": False,
            "stages": [{"name": s, "returncode": 0, "owned_cleanup_complete": True} for s in r.STAGES]}
        costs = {"active_call": None, "unknown_generation_after_error": False,
                 "totals": dict.fromkeys(p.COUNTERS, 0), "stage_counts": dict.fromkeys(p.COUNTERS, 0)}
        receipt = {"kind": "recovery_diagnostic_postqueue_integrity", "complete": True, "errors": [],
            "adoption_sha256": "sha", "queue_status_sha256": "sha", "locked_sha256_file_sha256": "sha",
            "controller": {"pid": 7, "start_ticks": 9, "exited": False}, "owned_workers_gone": True}
        def metadata(path):
            path = Path(path)
            if path.name == "adoption": return adoption
            if path.name == "status.json": return status
            if path.name == "receipt": return receipt
            if path.name == "locked_sha256.json": return {}
            if path.name == "cost_progress.json": return costs
            if path.name == "summary.json":
                return {"complete": True, "stage": path.parent.name, "record_count": 180,
                    "labels_locked_before_Q": True, "costs": costs, "gate_evaluable": True,
                    "training_gate_passed": False, "outcome_unblinded": True}
            raise AssertionError("Read past process barrier")
        with tempfile.TemporaryDirectory() as root, patch.object(r, "read", side_effect=metadata), \
             patch.object(r, "sha", return_value="sha"), \
             patch.object(r, "jsonl", side_effect=AssertionError("Outcome read forbidden")):
            with self.assertRaisesRegex(ValueError, "have not exited"):
                r.metadata_barrier(Path(root), "adoption", "receipt", "sha")


if __name__ == "__main__":
    unittest.main()

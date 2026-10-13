"""Pure fake-policy/state-machine fixtures; no real model or environment."""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest

from gnrl.alfworld_env import StepResult
from gnrl import recovery_projection_v1 as r
from scripts import collect_recovery_projection_v1 as worker

GAME = "/data/toy/json_2.1.1/train/pick_and_place_simple-A-None-B-1/trial/game.tw-pddl"


class ToyEnv:
    seed, max_steps, gamefile = r.SEED, 100, GAME

    def __init__(self, expert_type=None, drift=False, early_win=False):
        self.expert_type = expert_type
        self.drift, self.early_win = drift, early_win
        self.closed = False
        self.calls = []

    def value(self, observation):
        done = self.won or self.n >= 100
        commands = [] if done else ["advance", "alt", "dead", "examine toy", "finish", "look"]
        state = {"facts": [["position", self.position + int(self.drift)]], "moves": self.moves,
            "elapsed_steps": self.n, "max_steps": 100, "score": int(self.won), "reward": int(self.won),
            "won": self.won, "lost": False, "terminated": self.won, "truncated": self.n >= 100 and not self.won,
            "done": done, "admissible_commands": commands}
        expert = "finish" if self.expert_type == "planner" and not self.dead and not done else None
        return StepResult(observation, commands, expert, float(self.won), done, self.won, state)

    def reset(self):
        self.calls.append("reset")
        self.n = self.moves = self.position = 0
        self.won = self.dead = False
        return self.value("Original task. Start here.")

    def step(self, action):
        self.calls.append(action)
        self.n += 1
        if action in ("finish", "alt") and not self.dead:
            self.won = True
        if action == "advance":
            self.position += 1
        if action == "dead":
            self.dead = True
        if action != "invalid":
            self.moves += 1
        if self.early_win and self.n == 3:
            self.won = True
        return self.value("actual:" + action + ":" + str(self.position))

    def close(self):
        self.closed = True


class ToyPolicy:
    def __init__(self, action="invalid"):
        self.generated_tokens = self.forward_tokens = 0
        self.action = action
        self.prompts = []

    def choose_action(self, task, history, observation, commands, selection):
        self.prompts.append(copy.deepcopy((task, history, observation, commands)))
        self.generated_tokens += 2
        self.forward_tokens += 12
        return self.action, {"selection": selection, "raw_completion": self.action,
            "generated_tokens": 2, "is_admissible": self.action in commands}


def ledger(**kwargs):
    limits = {"environment_steps_started": 10000, "resets_started": 100, "generated_tokens": 10000}
    limits.update(kwargs.pop("limits", {}))
    return r.BudgetLedger(stage_limits=limits, deadline=time.monotonic() + 60, **kwargs)


def episode():
    return r.collect_learner_task(ToyEnv(), ToyPolicy(), GAME, ledger())


class RecoveryFixtures(unittest.TestCase):
    def test_collection_preserves_invalid_actions_and_exact_causal_anchors(self):
        policy, env, costs = ToyPolicy(), ToyEnv(), ledger()
        row = r.collect_learner_task(env, policy, GAME, costs)
        r.validate_episode(row)
        self.assertEqual(row["environment_steps"], 100)
        self.assertFalse(row["success"])
        self.assertEqual(costs.totals["environment_steps_started"], 100)
        self.assertEqual(costs.totals["generated_tokens"], 200)
        for anchor, k in zip(row["anchors"], (4, 8, 16)):
            self.assertEqual(anchor["status"], "available")
            self.assertEqual(anchor["recorded_learner_next_action"], "invalid")
            self.assertEqual(len(anchor["student_context"]["history"]), k)
            self.assertEqual(set(anchor["student_context"]), {"task", "history", "observation", "admissible_commands"})
            self.assertEqual(anchor["student_context"]["history"][0]["observation"], row["initial"]["observation"])
            self.assertTrue(all(p["action"] == "invalid" for p in anchor["student_context"]["history"]))

    def test_early_termination_keeps_three_structural_slots_without_replacement(self):
        row = r.collect_learner_task(ToyEnv(early_win=True), ToyPolicy(), GAME, ledger())
        r.validate_episode(row)
        self.assertEqual(len(row["transitions"]), 3)
        self.assertEqual([a["status"] for a in row["anchors"]], ["terminated_before_anchor"] * 3)
        cost = ledger()
        def forbidden(*args):
            raise AssertionError("Unavailable slot launched an environment")
        pool = r.certify_anchor(forbidden, row["anchors"][0], cost)
        r.validate_pool(pool)
        self.assertFalse(pool["eligible"])
        self.assertEqual(cost.totals["resets_started"], 0)

    def test_proposal_keeps_teacher_and_admissible_learner_then_fixed_hash_order(self):
        anchor = episode()["anchors"][0]
        anchor["recorded_learner_next_action"] = "advance"
        actions = r.propose_actions(anchor, "look")
        self.assertEqual(actions[:2], ["look", "advance"])
        self.assertNotIn("examine toy", actions)
        self.assertEqual(actions, r.propose_actions(copy.deepcopy(anchor), "look"))
        self.assertEqual(len(actions), len(set(actions)))
        with self.assertRaisesRegex(RuntimeError, "inadmissible"):
            r.propose_actions(anchor, "invented")

    def test_oracle_runs_actual_success_and_failure_branches_with_prefix_costs(self):
        anchor = episode()["anchors"][0]
        costs, environments, events = ledger(), [], []
        def factory(game, expert):
            env = ToyEnv(expert)
            environments.append(env)
            return env
        pool = r.certify_anchor(factory, anchor, costs, events.append)
        r.validate_pool(pool)
        self.assertTrue(pool["eligible"])
        self.assertEqual(pool["proposed_actions"][0], "finish")
        self.assertNotIn("invalid", pool["proposed_actions"])
        outcomes = {b["action"]: b for b in pool["branches"]}
        self.assertFalse(outcomes["dead"]["success"])
        self.assertEqual(outcomes["dead"]["stop_reason"], "no_plan")
        self.assertEqual(outcomes["advance"]["transitions"][0]["after"]["observation"], "actual:advance:1")
        self.assertEqual(outcomes["advance"]["transitions"][1]["before"], outcomes["advance"]["transitions"][0]["after"])
        expected_steps = sum(4 + len(b["transitions"]) for b in pool["branches"])
        self.assertEqual(costs.totals["environment_steps_started"], expected_steps)
        self.assertEqual(costs.totals["resets_started"], len(pool["branches"]))
        self.assertTrue(environments[0].closed)
        self.assertTrue(events)

    def test_certificates_replay_without_expert_and_reject_hidden_state_drift(self):
        anchor = episode()["anchors"][1]
        pool = r.certify_anchor(lambda g, e: ToyEnv(e), anchor, ledger())
        costs = ledger()
        proof = r.replay_certificates(lambda g, e: ToyEnv(e), pool, costs)
        self.assertEqual(len(proof["candidates"]), len(pool["retained_candidates"]))
        self.assertEqual(costs.totals["environment_steps_started"], len(proof["candidates"]) * 9)
        with self.assertRaisesRegex(RuntimeError, "raw state differs"):
            r.replay_certificates(lambda g, e: ToyEnv(e, drift=True), pool, ledger())

    def test_budget_counts_failed_attempt_before_call_and_prevents_extra_call(self):
        snapshots = []
        costs = ledger(limits={"environment_steps_started": 1}, persist=lambda x: snapshots.append(copy.deepcopy(x)))
        class Broken:
            def __init__(self): self.calls = 0
            def step(self, action):
                self.calls += 1
                raise ValueError("synthetic environment failure")
        env = Broken()
        with self.assertRaises(ValueError): costs.step(env, "a", "fixture")
        self.assertEqual(costs.totals["environment_steps_started"], 1)
        self.assertEqual(costs.totals["environment_steps_completed"], 0)
        self.assertEqual(snapshots[-1]["active_call"]["operation"], "step")
        with self.assertRaisesRegex(RuntimeError, "budget exhausted"): costs.step(env, "b", "fixture")
        self.assertEqual(env.calls, 1)

    def test_cumulative_limit_and_deadline_apply_before_environment_call(self):
        initial = dict.fromkeys(r.COUNTERS, 0)
        initial["environment_steps_started"] = r.GLOBAL_LIMITS["environment_steps_started"]
        costs = ledger(initial_totals=initial)
        with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
            costs.step(None, "a", "never")
        costs.deadline = time.monotonic() - 1
        with self.assertRaisesRegex(RuntimeError, "wall budget"):
            costs.reset(None, "never")

    def test_technical_generation_error_is_not_structural_unavailability(self):
        class BrokenPolicy(ToyPolicy):
            def choose_action(self, *args): raise RuntimeError("OOM fixture")
        costs = ledger()
        with self.assertRaisesRegex(RuntimeError, "OOM"):
            r.collect_learner_task(ToyEnv(), BrokenPolicy(), GAME, costs)
        self.assertTrue(costs.unknown_generation_after_error)
        self.assertEqual(costs.totals["policy_calls_started"], 1)
        self.assertEqual(costs.totals["policy_calls_completed"], 0)

    def test_future_prompt_or_counterfactual_prefix_is_rejected(self):
        original = episode()["anchors"][0]
        for mutation in ("teacher", "history", "state"):
            anchor = copy.deepcopy(original)
            if mutation == "teacher": anchor["student_context"]["teacher_plan"] = ["finish"]
            if mutation == "history": anchor["student_context"]["history"][0]["observation"] = "future"
            if mutation == "state": anchor["prefix_transitions"][1]["before"]["canonical_state"]["moves"] = 9
            with self.assertRaises(RuntimeError): r.validate_anchor(anchor)

    def test_pool_success_or_candidate_tampering_is_rejected(self):
        pool = r.certify_anchor(lambda g, e: ToyEnv(e), episode()["anchors"][0], ledger())
        for mutation in ("success", "candidate"):
            changed = copy.deepcopy(pool)
            if mutation == "success": changed["branches"][0]["success"] = False
            else: changed["retained_candidates"][0]["action"] = "dead"
            with self.assertRaisesRegex(RuntimeError, "Pool bytes"):
                r.validate_pool(changed)

    def test_wrong_environment_seed_is_refused_before_reset(self):
        env = ToyEnv()
        env.seed = 42
        with self.assertRaisesRegex(RuntimeError, "identity"):
            r.collect_learner_task(env, ToyPolicy(), GAME, ledger())
        self.assertEqual(env.calls, [])

    def test_candidate_document_is_not_stage_adoption(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.json"
            path.write_text(json.dumps({"execution_authorized": False}))
            with self.assertRaisesRegex(RuntimeError, "adoption refused"):
                worker.validate_request(path)

    def test_unknown_status_or_false_structural_claim_cannot_hide_missing_work(self):
        for status in ("technical_incomplete", "terminated_at_anchor"):
            row = episode()
            row["anchors"][0]["status"] = status
            row.pop("artifact_sha256")
            row["artifact_sha256"] = r.digest(row)
            with self.assertRaisesRegex(RuntimeError, "status"):
                r.validate_episode(row)
        anchor = episode()["anchors"][0]
        anchor["status"] = "technical_incomplete"
        with self.assertRaisesRegex(RuntimeError, "technical"):
            r.certify_anchor(None, anchor, ledger())

    def test_explicit_lower_global_limits_and_generation_labels_are_enforced(self):
        limits = dict(r.GLOBAL_LIMITS, environment_steps_started=1)
        costs = ledger(global_limits=limits)
        env = ToyEnv()
        state = costs.reset(env, "fixture")
        ctx = r.context(state.observation, [], r.state_record(state))
        costs.generate(ToyPolicy(), ctx, label="diagnostic_continuation")
        self.assertEqual(costs.snapshot()["calls_by_label"]["diagnostic_continuation:generate"], 1)
        costs.step(env, "invalid", "fixture")
        with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
            costs.step(env, "invalid", "fixture")


if __name__ == "__main__":
    unittest.main()

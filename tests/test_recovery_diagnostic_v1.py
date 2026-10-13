"""Pure fake-state fixtures. No simulator, tokenizer/model load or remote access."""
import copy
import math
from pathlib import Path
import sys
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gnrl import recovery_diagnostic_v1 as d
from gnrl import recovery_projection_v1 as p
from gnrl.alfworld_env import StepResult

GAME = "/fixture/json_2.1.1/train/pick_and_place_simple-A-None-B-1/trial/game.tw-pddl"


def state(n, branch=None, terminal=False, won=False):
    commands = ["expert", "other"] if n == 4 else ["finish", "fail"] if n >= 5 else ["walk"]
    canonical = {"facts": [["position", [[str(n), "place"]]], ["branch", [[str(branch), "object"]]]],
        "moves": n, "elapsed_steps": n, "max_steps": 100, "score": float(won), "reward": float(won),
        "won": won, "lost": terminal and not won, "terminated": terminal, "truncated": False,
        "done": terminal, "admissible_commands": commands}
    return StepResult(f"observed {n} {branch}", commands, None, float(won), terminal, won, canonical)


def make_pool(game=GAME):
    initial = state(0)
    trace = [p.transition(state(i), "walk", state(i+1), i) for i in range(4)]
    anchor = {"schema_version": 1, "kind": "recovery_anchor", "gamefile": game, "seed": p.SEED,
        "after_actions": 4, "anchor_id": p.digest([game, p.SEED, 4]), "status": "available",
        "initial": p.state_record(initial), "prefix_transitions": trace, "anchor_state": p.state_record(state(4)),
        "student_context": p.context(initial.observation, trace, p.state_record(state(4))),
        "recorded_learner_next_action": "other"}
    branches, retained = [], []
    for index, action in enumerate(("expert", "other")):
        first = p.transition(state(4), action, state(5, action), 4)
        last = p.transition(state(5, action), "finish", state(6, action, True, True), 5)
        branch = {"candidate_index": index, "action": action, "success": True, "stop_reason": "won",
                  "transitions": [first, last], "final_state": last["after"]}
        branch["certificate_sha256"] = p.digest(branch)
        branches.append(branch)
        retained.append({"action": action, "candidate_index": index,
                         "certificate_sha256": branch["certificate_sha256"], "first_transition": first})
    pool = d.seal({"schema_version": 1, "kind": "recovery_oracle_pool", "anchor": anchor,
        "anchor_id": anchor["anchor_id"], "status": "eligible", "eligible": True, "expert_action": "expert",
        "proposed_actions": ["expert", "other"], "branches": branches, "retained_candidates": retained, "complete": True})
    p.validate_pool(pool)
    return pool


def scores(pool, other=math.log(3)):
    ctxhash = p.digest(pool["anchor"]["student_context"])
    return {a: {"logprob": other if a == "other" else 0.0, "action_tokens_including_eos": 2,
        "prompt_tokens": 3, "prompt_ids_sha256": "prompt", "action_ids_sha256": a,
        "context_sha256": ctxhash} for a in ("expert", "other")}


class FakeEnv:
    def __init__(self, game, expert_type):
        self.gamefile, self.expert_type = game, expert_type
        self.seed, self.max_steps = p.SEED, 100
        self.n = 0
        self.branch = None
        self.closed = False

    def reset(self):
        self.n, self.branch = 0, None
        return state(0)

    def step(self, action):
        self.n += 1
        if self.n == 5:
            self.branch = action
        return state(self.n, self.branch, self.n == 6, self.n == 6 and action == "finish")

    def close(self):
        self.closed = True


class FakePolicy:
    def __init__(self):
        self.generated_tokens = self.forward_tokens = 0
        self.contexts = []

    def choose_action(self, task, history, observation, commands, selection):
        self.contexts.append({"task": task, "history": copy.deepcopy(history), "observation": observation,
                              "admissible_commands": commands})
        action = "finish" if history[-1]["action"] == "other" else "fail"
        self.generated_tokens += 2
        self.forward_tokens += 12
        return action, {"selection": selection, "raw_completion": action, "generated_tokens": 2,
                        "is_admissible": action in commands}


def ledger():
    return p.BudgetLedger(stage_limits={"environment_steps_started": 28800, "resets_started": 288,
        "generated_tokens": 1843200}, deadline=time.monotonic()+30)


def actual_branches(pool):
    return [d.learner_branch(FakeEnv, FakePolicy(), pool, c, ledger()) for c in pool["retained_candidates"]]


def full_fixture():
    pools = [make_pool()]
    planned = [{"gamefile": GAME, "after_actions": 4, "family": "pick_and_place_simple"}]
    for i in range(35):
        game = GAME.replace("trial/", f"trial{i}/")
        anchor = {"schema_version": 1, "kind": "recovery_anchor", "gamefile": game,
            "seed": p.SEED, "after_actions": 4, "anchor_id": p.digest([game,p.SEED,4]),
            "status": "terminated_before_anchor"}
        pool = d.seal({"schema_version": 1, "kind": "recovery_oracle_pool", "anchor": anchor,
            "anchor_id": anchor["anchor_id"], "status": anchor["status"], "eligible": False,
            "expert_action": None, "proposed_actions": [], "branches": [], "retained_candidates": [], "complete": True})
        pools.append(pool)
        planned.append({"gamefile": game, "after_actions": 4, "family": "pick_and_place_simple"})
    selections = [d.selection(pool, scores(pool), scores(pool,0)) if pool["eligible"] else d.selection(pool,{}, {}) for pool in pools]
    return pools,selections,planned,actual_branches(pools[0])


class DiagnosticTests(unittest.TestCase):
    def test_stable_cdf_full_command_sum(self):
        pool = make_pool()
        a = d.selection(pool, scores(pool), scores(pool,0))
        b = d.selection(pool, dict(reversed(list(scores(pool).items()))), scores(pool,0))
        self.assertEqual(a,b)
        self.assertAlmostEqual(a["q"]["other"],.75)
        self.assertEqual(a["q_base"],{"expert":.5,"other":.5})
        self.assertEqual(a["conditional_draw"]["ordered_actions"],["expert","other"])
        self.assertGreaterEqual(a["conditional_draw"]["u"],0)
        self.assertLess(a["conditional_draw"]["u"],1)

    def test_nonfinite_or_incomplete_score_rejected(self):
        pool = make_pool()
        for score in ({"expert":0}, {"expert":0,"other":float("nan")}):
            with self.assertRaises(RuntimeError):
                d.probabilities(["expert","other"],score)
        a = scores(pool)
        a["other"]["action_ids_sha256"] = "different"
        with self.assertRaises(RuntimeError):
            d.selection(pool,a,scores(pool))

    def test_causal_label_allowlist_and_identical_context(self):
        pool = make_pool()
        pool["anchor"]["teacher_secret"] = "future success"
        pool.pop("artifact_sha256")
        pool = d.seal(pool)
        labels = d.label_rows(pool,d.selection(pool,scores(pool),scores(pool,0)))
        contexts = [{k:r[k] for k in ("task","history","observation","admissible_commands")} for r in labels.values()]
        self.assertTrue(all(ctx==contexts[0] for ctx in contexts))
        self.assertNotIn("future success",str(contexts))
        self.assertEqual(set(contexts[0]["history"][0]),{"observation","action"})

    def test_full_actual_history_forced_action_then_learner(self):
        pool, policy, budget = make_pool(), FakePolicy(), ledger()
        row = d.learner_branch(FakeEnv,policy,pool,pool["retained_candidates"][1],budget)
        self.assertTrue(row["Q"])
        self.assertEqual([t["action"] for t in policy.contexts[0]["history"]],["walk"]*4+["other"])
        self.assertEqual(policy.contexts[0]["observation"],state(5,"other").observation)
        self.assertEqual(budget.counts["environment_steps_started"],6)
        self.assertEqual(budget.counts["resets_started"],1)
        self.assertEqual(budget.counts["generated_tokens"],2)

    def test_teacher_success_is_not_learner_Q(self):
        pool = make_pool()
        rows = actual_branches(pool)
        self.assertTrue(all(b["success"] for b in pool["branches"]))
        self.assertEqual([r["Q"] for r in rows],[False,True])

    def test_fixed36_denominator_and_unavailable_null(self):
        args = full_fixture()
        result = d.diagnostic_summary(*args)
        self.assertEqual(result["denominator"],36)
        self.assertAlmostEqual(result["mean_q_minus_uniform"],.25/36)
        self.assertEqual(result["cells"][1]["Q"],None)
        self.assertEqual(result["cells"][1]["difference_q_uniform"],0)
        self.assertFalse(result["training_gate_passed"])

    def test_technical_missing_not_zero(self):
        pools, selected, planned, episodes = full_fixture()
        for invalid in (episodes[:1], episodes+episodes[:1]):
            with self.assertRaisesRegex(RuntimeError,"Technical missing"):
                d.diagnostic_summary(pools,selected,planned,invalid)

    def test_unknown_unavailable_and_tampered_Q_stop(self):
        pools, selected, planned, episodes = full_fixture()
        pools[1]["status"] = "tokenizer_error"
        pools[1].pop("artifact_sha256")
        pools[1] = d.seal(pools[1])
        with self.assertRaises(RuntimeError):
            d.pool_actions(pools[1])
        row = episodes[0]
        row["Q"] = True
        row.pop("artifact_sha256")
        with self.assertRaises(RuntimeError):
            d.validate_branch(pools[0],pools[0]["retained_candidates"][0],d.seal(row))

    def test_replay_mismatch_closes_own_env(self):
        made=[]
        class BadEnv(FakeEnv):
            def reset(self):
                return state(1)
        def factory(game,expert):
            env=BadEnv(game,expert);made.append(env);return env
        pool=make_pool()
        with self.assertRaises(RuntimeError):
            d.learner_branch(factory,FakePolicy(),pool,pool["retained_candidates"][0],ledger())
        self.assertTrue(made[0].closed)

    def test_history_forgery_and_budget_stop_rejected(self):
        pool=make_pool(); row=actual_branches(pool)[0]
        row["transitions"][1]["context_sha256"]="reset history"
        row.pop("artifact_sha256")
        with self.assertRaises(RuntimeError):
            d.validate_branch(pool,pool["retained_candidates"][0],d.seal(row))
        row=actual_branches(pool)[0];row["stop_reason"]="generated_token_budget";row.pop("artifact_sha256")
        with self.assertRaises(RuntimeError):
            d.validate_branch(pool,pool["retained_candidates"][0],d.seal(row))

    def test_score_budget_before_model_call_and_accounting(self):
        class Tokenizer:
            eos_token_id=7
            def encode(self,a,add_special_tokens=False):return [len(a)]
        class Scorer:
            tokenizer=Tokenizer()
            scored_tokens=forward_tokens=generated_tokens=0
            def prompt_ids(self,*args):return [1,2,3]
            def score_encoded_actions(self,prompt,actions):
                self.scored_tokens+=2*len(actions);self.forward_tokens+=5*len(actions)
                return [{"action":a,"logprob":-2.,"token_count":2}for a in actions]
        pool=make_pool();policy=Scorer()
        budget=d.ScoreLedger(target_limit=3,forward_limit=100,deadline=time.monotonic()+30)
        with self.assertRaises(RuntimeError):budget.score(policy,pool,"reference")
        self.assertEqual(policy.scored_tokens,0)
        budget=d.ScoreLedger(target_limit=4,forward_limit=10,deadline=time.monotonic()+30)
        result=budget.score(policy,pool,"reference")
        self.assertEqual(set(result),{"expert","other"})
        self.assertEqual(budget.snapshot()["forward_tokens"],10)
        self.assertIsNone(budget.active_call)

    def test_score_failure_retains_started_unknown_work(self):
        class Tokenizer:
            eos_token_id=7
            def encode(self,*args,**kwargs):return [2]
        class Broken:
            tokenizer=Tokenizer()
            scored_tokens=forward_tokens=generated_tokens=0
            def prompt_ids(self,*args):return [1]
            def score_encoded_actions(self,*args):raise RuntimeError("synthetic OOM")
        snapshots=[]
        budget=d.ScoreLedger(target_limit=100,forward_limit=100,deadline=time.monotonic()+30,
                             persist=lambda row:snapshots.append(copy.deepcopy(row)))
        with self.assertRaisesRegex(RuntimeError,"OOM"):
            budget.score(Broken(),make_pool(),"reference")
        self.assertEqual(snapshots[-1]["calls_started"],1)
        self.assertEqual(snapshots[-1]["calls_completed"],0)
        self.assertEqual(snapshots[-1]["active_call"]["reserved_target_tokens"],4)


if __name__ == "__main__":
    unittest.main()

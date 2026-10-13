"""Pure protocol fixtures: no simulator/model/remote execution."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gnrl import evaluate as old
from gnrl import recovery_evaluate_v1 as e
from gnrl import recovery_projection_v1 as p
from gnrl.alfworld_env import StepResult

GAME = "/fixture/json_2.1.1/valid_seen/pick_and_place_simple-A-None-B-1/trial/game.tw-pddl"


def state(n, done=False, won=False, commands=None):
    commands = ["finish", "walk"] if commands is None else commands
    canonical = {"facts": [["position", [[str(n), "place"]]]], "moves": n,
        "elapsed_steps": n, "max_steps": 100, "score": float(won), "reward": float(won),
        "won": won, "lost": done and not won and n < 100, "terminated": done and (n < 100 or won),
        "truncated": done and n == 100 and not won, "done": done, "admissible_commands": commands}
    return StepResult(f"observation {n}", commands, None, float(won), done, won, canonical)


class Env:
    gamefile, seed, max_steps, expert_type = GAME, 20261012, 100, None

    def __init__(self, terminal=3, win=True, empty=None):
        self.terminal, self.win, self.empty = terminal, win, empty
        self.actions = []

    def get(self):
        n = len(self.actions)
        return state(n, n == self.terminal, n == self.terminal and self.win,
                     [] if n == self.empty else None)

    def reset(self):
        self.actions = []
        return self.get()

    def step(self, action):
        self.actions.append(action)
        return self.get()

    def close(self):
        self.closed = True


class Policy:
    def __init__(self, action="INVALID"):
        self.action = action
        self.contexts = []
        self.generated_tokens = self.forward_tokens = self.scored_tokens = 0

    def choose_action(self, task, history, observation, commands, selection):
        self.contexts.append(copy.deepcopy({"task": task, "history": history,
            "observation": observation, "admissible_commands": commands, "selection": selection}))
        self.generated_tokens += 2
        self.forward_tokens += 12
        return self.action, {"selection": selection, "raw_completion": "\n " + self.action + " \n",
            "generated_tokens": 2, "is_admissible": self.action in commands}


def ledger(**changes):
    args = {"stage_limits": e.STAGE_LIMITS, "deadline": time.monotonic() + 60}
    args.update(changes)
    return p.BudgetLedger(**args)


def seal(row):
    row.pop("artifact_sha256", None)
    row["artifact_sha256"] = p.digest(row)
    return row


def model_fixture(root):
    from gnrl.train_recovery_v1 import SETTINGS
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
    canonical=root/e.CANONICAL
    initial_state={"optimizer_steps":100,"protocol":"alfworld-causal-v1","target_tokens":1000}
    write(canonical/"training_state.json",initial_state)
    write(canonical/"adapter_config.json",{})
    write(canonical/"adapter_model.bin",{"fake_weights":True})
    files=[{"name":f.name,"bytes":f.stat().st_size,"sha256":e.sha(f)} for f in sorted(canonical.iterdir())]
    inventory={"files":files,"files_sha256":p.digest(files)}
    audit={"kind":"recovery_joint_training_token_audit_v1","complete":True,"settings":SETTINGS,
        "score_sft_token_context_mismatches":0,"schedule_sha256":"schedule",
        "provenance":{"initial_adapter":inventory,"initial_training_state":initial_state},
        "arms":{arm:{"component_exposures":{"bc":400,"recovery":400},"totals":{"target_tokens":2000+i}}
                 for i,arm in enumerate(e.ARMS[1:])}}
    write(root/"audit.json",audit); audit_hash=e.sha(root/"audit.json")
    request={"joint_token_audit":"audit.json","file_sha256":{"audit.json":audit_hash},
             "arm":e.ARMS[0],"adapter":e.CANONICAL}
    barrier={"complete":True,"arms":list(e.ARMS),"joint_token_audit_sha256":audit_hash,"adapters":{}}
    for arm in e.ARMS:
        directory=canonical if arm==e.ARMS[0] else root/"checkpoints"/arm/"final"
        if arm!=e.ARMS[0]:
            expected=audit["arms"][arm]
            s={"optimizer_steps":100,"protocol":"alfworld-causal-v1","target_tokens":expected["totals"]["target_tokens"],
                "sampled_examples":800,"component_exposures":{"bc":400,"recovery":400},
                "initial_adapter_optimizer_steps":100,"cumulative_optimizer_steps":200,"optimizer_state_resumed":False,
                "token_audit_sha256":audit_hash,"schedule_sha256":"schedule",
                "initial_adapter_inventory_sha256":inventory["files_sha256"]}
            write(directory/"training_state.json",s);write(directory/"adapter_config.json",{})
            write(directory/"adapter_model.bin",{"fake_weights":arm})
            config={"arm":arm,"protocol":"alfworld-causal-v1","settings":SETTINGS,
                "token_audit_sha256":audit_hash,"schedule_sha256":"schedule"}
            summary={"complete":True,"matched_step_budget_complete":True,"optimizer_steps":100,"requested_steps":100,
                "stop_reason":"max_steps","technical_errors":[],"target_tokens":s["target_tokens"],
                "actual_backward_target_tokens":s["target_tokens"],"expected_target_tokens":s["target_tokens"],
                "component_exposures":{"bc":400,"recovery":400},"sampled_examples":800,
                "token_audit_sha256":audit_hash,"schedule_sha256":"schedule"}
            write(directory.parent/"config.json",config);write(directory.parent/"summary.json",summary)
            for name in ("config.json","summary.json"):
                f=directory.parent/name;request["file_sha256"][f.relative_to(root).as_posix()]=e.sha(f)
        hashes={f.relative_to(root).as_posix():e.sha(f)for f in directory.iterdir()}
        barrier["adapters"][arm]={"path":directory.relative_to(root).as_posix(),"file_sha256":hashes}
        request["file_sha256"].update(hashes)
    return request,barrier,audit


class EvaluationTests(unittest.TestCase):
    def test_matches_existing_actual_greedy_contexts(self):
        a, b, x, y = Env(), Env(), Policy(), Policy()
        before = old.rollout(a, x)
        costs, events = ledger(), []
        after = e.evaluate_task(b, y, GAME, costs, events.append)
        e.validate_episode(after, GAME)
        self.assertEqual(x.contexts, y.contexts)
        self.assertEqual(a.actions, b.actions)
        self.assertEqual(before["success"], after["success"])
        self.assertEqual(before["environment_steps"], after["environment_steps"])
        self.assertEqual(costs.counts["environment_steps_completed"], 3)
        self.assertEqual(costs.counts["generated_tokens"], 6)
        self.assertEqual(len(events), 4)
        self.assertTrue(all(t["action"] == "INVALID" for t in after["transitions"]))

    def test_hundred_action_loss_is_complete(self):
        row = e.evaluate_task(Env(100, False), Policy(), GAME, ledger())
        e.validate_episode(row, GAME)
        self.assertFalse(row["success"])
        self.assertEqual(row["environment_steps"], 100)
        self.assertTrue(row["final_state"]["canonical_state"]["truncated"])

    def test_empty_admissible_is_policy_failure(self):
        row = e.evaluate_task(Env(empty=1), Policy(), GAME, ledger())
        e.validate_episode(row, GAME)
        self.assertTrue(row["complete"])
        self.assertFalse(row["success"])
        self.assertEqual(row["stop_reason"], "no_admissible_commands")

    def test_initial_terminal_never_calls_policy(self):
        policy = Policy()
        row = e.evaluate_task(Env(0, True), policy, GAME, ledger())
        e.validate_episode(row, GAME)
        self.assertTrue(row["success"])
        self.assertEqual(policy.contexts, [])

    def test_budget_failure_preserves_started_call_and_inherited_cost(self):
        initial = dict.fromkeys(p.COUNTERS, 0)
        initial.update(environment_steps_started=9, environment_steps_completed=9)
        costs = ledger(stage_limits=dict(e.STAGE_LIMITS, environment_steps_started=1), initial_totals=initial)
        env = Env()
        with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
            e.evaluate_task(env, Policy(), GAME, costs)
        self.assertEqual(env.actions, ["INVALID"])
        self.assertEqual(costs.totals["environment_steps_started"], 10)
        # A real generation occurred before the next denied environment call.
        self.assertEqual(costs.counts["policy_calls_completed"], 2)
        self.assertEqual(costs.counts["generated_tokens"], 4)

    def test_model_error_does_not_become_policy_loss(self):
        class Failing(Policy):
            def choose_action(self, *args):
                self.generated_tokens += 1
                raise RuntimeError("fake model failure")
        costs = ledger()
        with self.assertRaisesRegex(RuntimeError, "fake model failure"):
            e.evaluate_task(Env(), Failing(), GAME, costs)
        self.assertTrue(costs.unknown_generation_after_error)
        self.assertEqual(costs.counts["generated_tokens"], 1)
        self.assertEqual(costs.counts["environment_steps_started"], 0)
        self.assertEqual(costs.active_call["operation"], "generate")

    def test_forged_outcome_context_or_repaired_command_rejected(self):
        row = e.evaluate_task(Env(), Policy(), GAME, ledger())
        mutations = [lambda r: r.update(success=False),
            lambda r: r["transitions"][1].update(context_sha256="reset-history"),
            lambda r: r["transitions"][0].update(action="walk"),
            lambda r: r["transitions"][0]["policy"].update(generated_tokens=65)]
        for mutation in mutations:
            wrong = copy.deepcopy(row)
            mutation(wrong)
            with self.assertRaises(RuntimeError):
                e.validate_episode(seal(wrong), GAME)

    def test_nonseen_or_expert_environment_rejected_before_reset(self):
        for game, expert in [(GAME.replace("valid_seen", "valid_unseen"), None), (GAME, "planner")]:
            env = Env()
            env.expert_type = expert
            costs = ledger()
            with self.assertRaises(RuntimeError):
                e.evaluate_task(env, Policy(), game, costs)
            self.assertEqual(costs.counts["resets_started"], 0)

    def test_broken_backend_max_steps_not_complete(self):
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            e.evaluate_task(Env(101, False), Policy(), GAME, ledger())

    def test_resealed_raw_flags_or_episode_token_total_rejected(self):
        row=e.evaluate_task(Env(),Policy(),GAME,ledger())
        def raw_final(r, **value):
            r["final_state"]["canonical_state"].update(value)
            r["transitions"][-1]["after"]["canonical_state"].update(value)
        def raw_initial(r):
            r["initial"]["canonical_state"]["admissible_commands"]=["invented"]
            r["transitions"][0]["before"]["canonical_state"]["admissible_commands"]=["invented"]
        for mutation in (lambda r:raw_final(r,won=False),
                         lambda r:raw_final(r,terminated=False), raw_initial,
                         lambda r:r.update(generated_tokens=999)):
            wrong=copy.deepcopy(row);mutation(wrong)
            with self.assertRaises(RuntimeError):e.validate_episode(seal(wrong),GAME)

    def test_all_four_model_barrier_accepts_complete_common_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);request,barrier,audit=model_fixture(root)
            with patch.object(e,"ROOT",root):e.validate_model_barrier(request,barrier,audit)

    def test_missing_or_aliased_model_and_wrong_training_arm_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);request,barrier,audit=model_fixture(root)
            with patch.object(e,"ROOT",root):
                missing=copy.deepcopy(barrier);del missing["adapters"][e.ARMS[-1]]
                with self.assertRaises(RuntimeError):e.validate_model_barrier(request,missing,audit)
                alias=copy.deepcopy(barrier);alias["adapters"][e.ARMS[1]]=alias["adapters"][e.ARMS[2]]
                with self.assertRaises(RuntimeError):e.validate_model_barrier(request,alias,audit)
                cfg=root/"checkpoints"/e.ARMS[1]/"config.json"
                obj=e.read(cfg);obj["arm"]=e.ARMS[2];cfg.write_text(json.dumps(obj))
                with self.assertRaisesRegex(RuntimeError,"arm/config"):
                    e.validate_model_barrier(request,barrier,audit)

    def test_model_target_token_mismatch_and_added_loader_file_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);request,barrier,audit=model_fixture(root)
            with patch.object(e,"ROOT",root):
                state_path=root/"checkpoints"/e.ARMS[1]/"final/training_state.json"
                obj=e.read(state_path);obj["target_tokens"]+=1;state_path.write_text(json.dumps(obj))
                with self.assertRaisesRegex(RuntimeError,"common audit"):
                    e.validate_model_barrier(request,barrier,audit)
                obj["target_tokens"]-=1;state_path.write_text(json.dumps(obj))
                (state_path.parent/"new_adapter.bin").write_bytes(b"fake")
                with self.assertRaisesRegex(RuntimeError,"inventory"):
                    e.validate_model_barrier(request,barrier,audit)

    def test_dangerous_artifact_paths_refused(self):
        for name in ("../outside",str(Path.cwd()/"absolute.json")):
            with self.assertRaises(RuntimeError):e.safe_path(name)

    def run_fixture(self, *, fail_in_episode):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);request_path=root/"request.json";request_path.write_text("{}")
            output=root/"new"
            request={"adapter":"fake_adapter","arm":e.ARMS[0],"global_deadline_unix":time.time()+1000}
            budget=ledger(stage_limits=dict(e.STAGE_LIMITS,environment_steps_started=0)) if fail_in_episode else ledger()
            def factory(game,**kwargs):
                env=Env();env.gamefile=game;return env
            def expire_after_hash(*args):budget.deadline=time.monotonic()-1
            games=[GAME.replace("trial/",f"trial{i}/")for i in range(48)]
            with patch.object(e,"validate_request",return_value=(request,games,output,dict.fromkeys(p.COUNTERS,0))), \
                 patch.object(e,"HFCausalPolicy",return_value=Policy()),patch.object(e.shared,"BudgetLedger",return_value=budget), \
                 patch("gnrl.alfworld_env.AlfworldReplayEnv",side_effect=factory), \
                 patch.object(e,"verify_bytes",side_effect=expire_after_hash),redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError):e.run(request_path)
            summary=e.read(output/"summary.json")
            self.assertFalse(summary["complete"])
            self.assertEqual(summary["technical_errors"],1)
            self.assertIsNone(summary["partial_performance_summary"])
            if fail_in_episode:
                episode=json.loads((output/"episodes.jsonl").read_text().splitlines()[-1])
                self.assertIsNone(episode["success"])
                self.assertFalse(episode["complete"])
            else:self.assertEqual(summary["completed_episodes"],48)

    def test_technical_episode_error_logs_null_success(self):
        self.run_fixture(fail_in_episode=True)

    def test_deadline_expiring_during_final_hashing_is_incomplete(self):
        self.run_fixture(fail_in_episode=False)


if __name__ == "__main__":
    unittest.main()

"""Pure fault fixtures only: no subprocess, GPU, tokenizer or real environment."""
import copy
import ast
import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch
from scripts import run_recovery_learning_queue_v1 as q
from tests.test_recovery_evaluate_v1 import Env, Policy, ledger, seal


class LearningFixtures(unittest.TestCase):
    def test_gate_requires_integrity_and_scientific_pass(self):
        status={"complete":True,"current_stage":"needs_astra_diagnostic_review","gate_evaluable":True,"training_gate_passed":True}
        result={"complete":True,"training_gate_passed":True}
        # Read the actual producer return shape, not a duplicated hand-written kind.
        tree=ast.parse((q.ROOT/"analysis/review_recovery_diagnostic_complete_v1.py").read_text(encoding="utf-8-sig"))
        run=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="run")
        produced=next(node.value for node in reversed(run.body) if isinstance(node,ast.Return))
        fields={key.value:value for key,value in zip(produced.keys,produced.values)}
        review={"kind":ast.literal_eval(fields["kind"]),"recovery_diagnostic_result_review_passed":True,
                "training_gate_passed":True,"complete":True,"gate_evaluable":True,"errors":[],"diagnostic_cost_sha256":"cost"}
        self.assertTrue(set(review)<=fields.keys())
        q.require_passed_diagnostic(status,result,review,{"diagnostic_cost_sha256":"cost"})
        for flag in (False,None):
            with self.assertRaises(RuntimeError):
                q.require_passed_diagnostic(status,result,dict(review,training_gate_passed=flag),{"diagnostic_cost_sha256":"cost"})
            with self.assertRaises(RuntimeError):
                q.require_passed_diagnostic(dict(status,training_gate_passed=flag),result,review,{"diagnostic_cost_sha256":"cost"})
        with self.assertRaisesRegex(RuntimeError,"binding"):
            q.require_passed_diagnostic(status,result,review,{"diagnostic_cost_sha256":"other"})

    def test_candidate_is_not_authorization(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"candidate.json";path.write_text('{"version":"v2"}')
            with self.assertRaisesRegex(RuntimeError,"Learning adoption refused"):q.validate_parent(path)

    def test_commands_share_initialization_labels_audit_and_fixed_arm_order(self):
        obj=q.LearningQueue.__new__(q.LearningQueue)
        obj.prior_paths={"score":"outputs/score"}
        obj.a={"bc_data":"bc.jsonl","output_paths":{"joint_token_audit":"audit.json",**{"train_"+a:"checkpoints/"+a for a in q.TRAIN}}}
        for arm in q.TRAIN:
            args=obj.trainer_args(arm)
            self.assertEqual(args[args.index("--initial-adapter")+1],q.evaluator.CANONICAL)
            self.assertEqual(args[args.index("--recovery-data")+1],"outputs/score/"+arm+".jsonl")
            self.assertNotIn("--peer-recovery-data",args)
        args=obj.trainer_args(q.TRAIN[0],audit=True)
        self.assertIn("--audit-only",args);self.assertNotIn("--output",args)
        self.assertEqual(args.count("--peer-recovery-data"),2)
        self.assertEqual(list(q.STAGES[1:4]),["train_"+a for a in q.TRAIN])
        self.assertEqual(list(q.STAGES[4:]),["eval_"+a for a in q.EVAL])

    def test_evaluation_refuses_missing_barrier_and_inherits_diagnostic_cost(self):
        with tempfile.TemporaryDirectory(dir=q.ROOT/"analysis") as folder:
            obj=q.LearningQueue.__new__(q.LearningQueue);obj.folder=Path(folder);obj.barrier=None
            with self.assertRaisesRegex(RuntimeError,"four models"):obj.evaluation_request(q.EVAL[0])
            obj.barrier={"adapters":{a:{"path":"adapter/"+a} for a in q.EVAL}}
            obj.adoption_hash="parent";obj.deadline_unix=1234;obj.locks={};obj.lock=lambda paths:None
            obj.prior_paths={"diagnostic":"outputs/Q"}
            obj.a={"candidate":"candidate.json","evaluator_review":"review.json","source_inventory":{},
                "original_file_sha256":{},"runtime_file_sha256":{},"output_paths":{"joint_token_audit":"audit.json",
                **{"eval_"+a:"outputs/"+a for a in q.EVAL}}}
            for i,arm in enumerate(q.EVAL):
                r=q.old.read_json(obj.evaluation_request(arm))
                expected="outputs/Q" if i==0 else "outputs/"+q.EVAL[i-1]
                self.assertEqual(r["previous_cost_json"],expected+"/cost_progress.json")
                self.assertEqual(r["global_deadline_unix"],1234)
                self.assertEqual(r["max_wall_seconds"],7200)

    def outcomes(self):
        games=[f"/fixture/json_2.1.1/valid_seen/family{i//8}-x/trial{i}/game.tw-pddl" for i in range(48)]
        meta={g:{"family":"family"+str(i//8),"scene":str(i%3)} for i,g in enumerate(games)}
        rows={}
        for arm in q.EVAL:
            rows[arm]=[]
            wins={q.EVAL[0]:2,"uniform_viable_recovery":4,"expert_recovery":5,"reference_conditional_recovery":8}[arm]
            for i,game in enumerate(games):
                env=Env(terminal=1,win=i<wins);env.gamefile=game
                rows[arm].append(q.evaluator.evaluate_task(env,Policy(),game,ledger()))
        return games,rows,meta

    def test_full192_actual_transition_table_fixed_denominators(self):
        games,rows,meta=self.outcomes()
        report=q.complete_table(games,rows,meta)
        self.assertTrue(report["exploratory_practical_screen_passed"])
        self.assertEqual(report["reference_minus_comparator"]["expert_recovery"]["difference"],3)
        self.assertEqual(len(report["table"]),48)
        self.assertEqual([g["denominator"] for g in report["groups"]["scene"].values()],[16,16,16])
        rows[q.EVAL[-1]][0]["success"]=False;seal(rows[q.EVAL[-1]][0])
        with self.assertRaises(RuntimeError):q.complete_table(games,rows,meta)

    def test_no_partial_table_or_invented_admissibility(self):
        games,rows,meta=self.outcomes()
        partial=copy.deepcopy(rows);partial[q.EVAL[0]].pop()
        with self.assertRaisesRegex(RuntimeError,"Partial"):q.complete_table(games,partial,meta)
        rows[q.EVAL[0]][0]["transitions"][0]["policy"]["is_admissible"]=True
        seal(rows[q.EVAL[0]][0])
        with self.assertRaises(RuntimeError):q.complete_table(games,rows,meta)

    def test_actual_event_stream_cannot_splice_stale_transition(self):
        games,rows,_=self.outcomes();episodes=rows[q.EVAL[0]]
        events=[]
        for row in episodes:
            events.append({"kind":"evaluation_initial","gamefile":row["gamefile"],"state":row["initial"]})
            events.extend({"kind":"evaluation_transition","gamefile":row["gamefile"],"transition":t} for t in row["transitions"])
        q.validate_events(episodes,events)
        changed=copy.deepcopy(events);changed[1]["transition"]["after"]["observation"]="stale counterfactual"
        with self.assertRaisesRegex(RuntimeError,"event stream"):q.validate_events(episodes,changed)

    def test_incomplete_all192_null_before_and_after_unblinding(self):
        with tempfile.TemporaryDirectory(dir=q.ROOT/"analysis") as folder:
            obj=q.LearningQueue.__new__(q.LearningQueue);obj.folder=Path(folder);obj.locks={}
            obj.previous_totals={};obj.games=[str(i) for i in range(48)]
            obj.update=lambda stage,**kw:obj.state.update(current_stage=stage,**kw)
            for unblinded in (False,True):
                obj.state={"stages":[],"outcome_unblinded":unblinded,"training_costs":{}}
                obj.incomplete(RuntimeError("fixture"))
                report=q.old.read_json(obj.folder/"result.json")
                self.assertEqual(len(report["cells"]),192)
                self.assertTrue(all(r["outcome"] is None for r in report["cells"]))
                self.assertEqual(report["outcome_unblinded"],unblinded)
                self.assertIsNone(report["contrasts"])

    def test_complete_barrier_required_before_any_episode_read(self):
        obj=q.LearningQueue.__new__(q.LearningQueue);obj.state={"stages":[],"outcome_unblinded":False}
        with patch.object(q.trainer,"read_rows") as read:
            with self.assertRaisesRegex(RuntimeError,"barrier"):obj.finalize()
            read.assert_not_called()
        self.assertFalse(obj.state["outcome_unblinded"])

    def fault(self,pending_signal):
        with tempfile.TemporaryDirectory() as folder:
            obj=q.LearningQueue.__new__(q.LearningQueue)
            obj.folder=Path(folder);obj.python="fake-python";obj.env={};obj.deadline=time.monotonic()+100
            obj.state={"stages":[]};obj.verify=lambda:None;obj.wait_gpu=lambda:None;obj.update=lambda *a,**k:None
            class Proc:
                pid=42;returncode=None
                def wait(self,timeout):raise subprocess.TimeoutExpired("fake",timeout)
            masks=[];cleanups=[]
            def mask(operation,previous):
                masks.append(operation)
                if pending_signal and len(masks)==2:raise InterruptedError("injected pending signal")
                return set()
            with patch.object(q.subprocess,"Popen",return_value=Proc()) as spawn, \
                 patch.object(q.old,"process_record",return_value={"session":42,"start_ticks":99}), \
                 patch.object(q.old,"cleanup_owned",side_effect=lambda process,ticks:cleanups.append((process.pid,ticks))), \
                 patch.object(q.signal,"pthread_sigmask",side_effect=mask,create=True), \
                 patch.object(q.signal,"SIG_BLOCK",0,create=True),patch.object(q.signal,"SIG_SETMASK",1,create=True), \
                 patch.object(q.signal,"SIGHUP",1,create=True),patch.object(q.signal,"SIGALRM",14,create=True):
                with self.assertRaises(InterruptedError if pending_signal else subprocess.TimeoutExpired):
                    obj.owned_run("joint_token_audit",["--fake"])
            self.assertEqual(spawn.call_count,1);self.assertEqual(cleanups,[(42,99)])
            self.assertTrue(obj.state["stages"][0]["owned_cleanup_complete"])
            self.assertEqual(spawn.call_args[0][0][2:4],["--gpus","2"])

    def test_wrong_final_target_or_protocol_refused_before_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);(folder/"final").mkdir()
            arm=q.TRAIN[0]
            audit={"arms":{arm:{"totals":{"target_tokens":123},"scheduled_examples":[]}},"schedule_sha256":"schedule"}
            summary={"complete":True,"matched_step_budget_complete":True,"stop_reason":"max_steps","technical_errors":[],
                "optimizer_steps":100,"requested_steps":100,"target_tokens":123,"sampled_examples":800,
                "component_exposures":{"bc":400,"recovery":400},"token_audit_sha256":"audit","schedule_sha256":"schedule"}
            state={"optimizer_steps":100,"protocol":"alfworld-causal-v1","target_tokens":124}
            config={"arm":arm,"settings":q.trainer.SETTINGS,"protocol":"alfworld-causal-v1"}
            for path,value in ((folder/"summary.json",summary),(folder/"config.json",config),(folder/"final/training_state.json",state)):
                path.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError,"state/audit"):q.validate_training(folder,arm,audit,"audit")
            state["protocol"]="different";(folder/"final/training_state.json").write_text(json.dumps(state))
            with self.assertRaisesRegex(RuntimeError,"protocol"):q.validate_training(folder,arm,audit,"audit")

    def test_training_metrics_must_match_all_actual_exposures(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);(folder/"final").mkdir();arm=q.TRAIN[0]
            slots=[{"index":i,"source":"bc" if i%2==0 else "recovery","target_tokens":2} for i in range(800)]
            audit={"arms":{arm:{"totals":{"target_tokens":1600,"input_tokens":9600,"prompt_tokens":8000},
                               "scheduled_examples":slots}},"schedule_sha256":"schedule"}
            common={"optimizer_steps":100,"target_tokens":1600,"sampled_examples":800,
                "component_exposures":{"bc":400,"recovery":400},"token_audit_sha256":"audit","schedule_sha256":"schedule"}
            (folder/"final/training_state.json").write_text(json.dumps(dict(common,protocol="alfworld-causal-v1")))
            (folder/"config.json").write_text(json.dumps({"arm":arm,"settings":q.trainer.SETTINGS,"protocol":"alfworld-causal-v1"}))
            inventory=q.trainer.inventory_tree(folder/"final")
            (folder/"final_adapter_inventory.json").write_text(json.dumps(inventory))
            summary=dict(common,complete=True,matched_step_budget_complete=True,stop_reason="max_steps",technical_errors=[],
                requested_steps=100,actual_backward_target_tokens=1600,expected_target_tokens=1600,
                final_adapter_files_sha256=inventory["files_sha256"],final_adapter_inventory_sha256=q.old.sha256(folder/"final_adapter_inventory.json"))
            (folder/"summary.json").write_text(json.dumps(summary))
            exposures=[]
            for slot in slots:
                exposures.extend([dict(slot,event="forward_started"),dict(slot,event="backward_completed",action_token_nll=1.0)])
            (folder/"example_exposures.jsonl").write_text("\n".join(json.dumps(r) for r in exposures))
            metrics=[{"optimizer_step":i,"window_target_tokens":16,"target_tokens":16*i,"sampled_examples":8*i,
                      "component_exposures":{"bc":4*i,"recovery":4*i},"action_token_nll":1.0} for i in range(1,101)]
            metrics.append({"summary":summary})
            def write_metrics(): (folder/"metrics.jsonl").write_text("\n".join(json.dumps(r) for r in metrics))
            write_metrics();self.assertEqual(q.validate_training(folder,arm,audit,"audit")["forward_calls"],800)
            metrics[30]["target_tokens"]+=1;write_metrics()
            with self.assertRaisesRegex(RuntimeError,"step/exposure"):q.validate_training(folder,arm,audit,"audit")
            metrics.pop();write_metrics()
            with self.assertRaisesRegex(RuntimeError,"metrics incomplete"):q.validate_training(folder,arm,audit,"audit")

    def test_all_trainings_and_model_lock_precede_any_evaluation(self):
        with tempfile.TemporaryDirectory(dir=q.ROOT/"analysis") as folder:
            folder=Path(folder);trace=[]
            obj=q.LearningQueue.__new__(q.LearningQueue)
            audit={"kind":"recovery_joint_training_token_audit_v1","complete":True,"settings":q.trainer.SETTINGS,
                "bc_examples":349,"anchor_count":1,"teacher_fields_used":False,"same_prompt_all_arms":True,
                "score_sft_selected_labels_checked":3,"score_sft_token_context_mismatches":0,"arms":dict.fromkeys(q.TRAIN,{})}
            path=folder/"audit.json";path.write_text(json.dumps(audit))
            obj.a={"output_paths":{"joint_token_audit":path.relative_to(q.ROOT).as_posix(),**{"train_"+a:"unused/"+a for a in q.TRAIN}}}
            obj.state={"training_costs":{}};obj.trained_inventories={}
            obj.owned_run=lambda stage,args,**kw:trace.append(stage)
            obj.trainer_args=lambda *a,**kw:[];obj.lock=lambda paths:None;obj.update=lambda *a,**kw:None
            obj.lock_models=lambda:trace.append("barrier")
            obj.evaluation_request=lambda arm:arm;obj.accept_evaluation=lambda *a:None
            obj.finalize=lambda:trace.append("unblind")
            with patch.object(q,"validate_training",return_value={}):obj.main()
            self.assertEqual(trace,[*q.STAGES[:4],"barrier",*q.STAGES[4:],"unblind"])

    def test_new_file_in_finished_adapter_refused_before_next_training(self):
        with tempfile.TemporaryDirectory(dir=q.ROOT/"analysis") as folder:
            folder=Path(folder);(folder/"adapter.bin").write_bytes(b"fixture")
            obj=q.LearningQueue.__new__(q.LearningQueue);obj.barrier=None;obj.deadline=time.monotonic()+100
            obj.trained_inventories={folder.relative_to(q.ROOT).as_posix():{(folder/"adapter.bin").relative_to(q.ROOT).as_posix()}}
            with patch.object(q.acquisition.Acquisition,"verify"):
                obj.verify()
                (folder/"new_loader_config.json").write_text("{}")
                with self.assertRaisesRegex(RuntimeError,"inventory changed"):obj.verify()

    def test_pending_signal_cleanup(self):self.fault(True)
    def test_timeout_cleanup_no_retry(self):self.fault(False)


if __name__ == "__main__":unittest.main()

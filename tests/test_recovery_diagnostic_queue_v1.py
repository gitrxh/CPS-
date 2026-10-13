"""Controller fault fixtures using mocks only: never spawns a real process."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_recovery_diagnostic_queue_v1 as q


def costs(previous, **increments):
    counts = dict.fromkeys(q.p.COUNTERS, 0)
    counts.update(increments)
    totals = {key: previous[key] + counts[key] for key in counts}
    result = {"kind": "recovery_cost_ledger", "active_call": None, "unknown_generation_after_error": False,
        "global_limits": q.p.GLOBAL_LIMITS, "stage_limits": q.p.GLOBAL_LIMITS,
        "totals": totals, "stage_counts": counts}
    return result, {"costs": copy.deepcopy(result)}


class QueueFixtures(unittest.TestCase):
    def test_deadline_inherits_elapsed_time_and_rejects_reset(self):
        candidate={"version":"v2","finite_budgets":{"global_wall_seconds_including_waits_verification_cleanup":171840}}
        status={"main_study_started_unix":1000,"global_deadline_unix":172840}
        with patch.object(q.time,"time",return_value=5000),patch.object(q.time,"monotonic",return_value=123):
            self.assertEqual(q.inherited_deadline(status,candidate),(172840,167963))
            with self.assertRaisesRegex(RuntimeError,"changed"):
                q.inherited_deadline(dict(status,global_deadline_unix=176840),candidate)
        with patch.object(q.time,"time",return_value=172820):
            with self.assertRaisesRegex(RuntimeError,"exhausted"):
                q.inherited_deadline(status,candidate)

    def test_prior_running_pid_reuse_and_orphan_refused_without_signalling(self):
        identity={"pid":50,"start_ticks":123,"owned_session":50}
        for current in ({"start_ticks":123,"state":"S"},{"start_ticks":124,"state":"Z"}):
            with patch.object(q.old,"process_record",return_value=current),patch.object(q.old,"signal_member") as signal:
                with self.assertRaises(RuntimeError):q.assert_exited(identity,session=True)
                signal.assert_not_called()
        with patch.object(q.old,"process_record",return_value=None),patch.object(q.old,"owned_members",return_value=[{"pid":51}]):
            with self.assertRaisesRegex(RuntimeError,"live members"):q.assert_exited(identity,session=True)
        with patch.object(q.old,"process_record",return_value=None),patch.object(q.old,"owned_members",return_value=[]):
            q.assert_exited(identity,session=True)

    def test_cost_inheritance_refuses_reset_and_incomplete_work(self):
        previous=dict.fromkeys(q.p.COUNTERS,0);previous["forward_tokens"]=100
        ledger,summary=costs(previous,forward_tokens=20)
        self.assertEqual(q.validate_cost_lineage(previous,ledger,summary)["forward_tokens"],120)
        changed=copy.deepcopy(ledger);changed["totals"]["forward_tokens"]=20
        with self.assertRaisesRegex(RuntimeError,"Cost reset"):
            q.validate_cost_lineage(previous,changed,summary)
        broken,summary=costs(previous,environment_steps_started=1)
        with self.assertRaisesRegex(RuntimeError,"Unfinished"):
            q.validate_cost_lineage(previous,broken,summary)

    def test_both_reference_score_cost_counted_exactly_once(self):
        row={"prompt_tokens":10,"action_tokens_including_eos":2}
        selected=[{"eligible":True,"actions":["a","b"],"reference_scores":{"a":row,"b":row},"base_scores":{"a":row,"b":row}}]
        scoring={"active_call":None,"calls_started":2,"calls_completed":2,"scored_target_tokens":8,"forward_tokens":48,
                 "target_limit":500000,"forward_limit":7000000}
        ledger,_=costs(dict.fromkeys(q.p.COUNTERS,0),forward_tokens=48)
        q.verify_score_cost(selected,scoring,ledger)
        for wrong in (24,96):
            changed=copy.deepcopy(ledger);changed["stage_counts"]["forward_tokens"]=wrong
            with self.assertRaisesRegex(RuntimeError,"shared costs"):
                q.verify_score_cost(selected,scoring,changed)
        with self.assertRaisesRegex(RuntimeError,"Scoring budget"):
            q.verify_score_cost(selected,dict(scoring,target_limit=1000000),ledger)

    def test_worker_unblinding_retained_after_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            state={"outcome_unblinded":False}
            q.preserve_worker_unblinding(Path(folder),state)
            self.assertFalse(state["outcome_unblinded"])
            (Path(folder)/"summary.json").write_text(json.dumps({"complete":False,
                "outcome_unblinded":True,"outcome_unblinded_unix":123,"gate_evaluable":False}))
            q.preserve_worker_unblinding(Path(folder),state)
            self.assertTrue(state["outcome_unblinded"])
            self.assertEqual(state["worker_outcome_unblinded_unix"],123)

    def test_cleanup_failure_fallback_and_unreadable_flag_never_false(self):
        obj=q.DiagnosticQueue.__new__(q.DiagnosticQueue)
        obj.a={"output_paths":{"diagnostic":"outputs/fake"}}
        obj.state={"outcome_unblinded":False}
        obj.update=lambda stage,**kw:obj.state.update(current_stage=stage,**kw)
        def crossed(folder,state):state.update(outcome_unblinded=True)
        with patch.object(q,"preserve_worker_unblinding",side_effect=crossed):
            q.mark_incomplete(obj,RuntimeError("cleanup failed"))
        self.assertTrue(obj.state["outcome_unblinded"])
        self.assertIsNone(obj.state["training_gate_passed"])
        obj.state["outcome_unblinded"]=False
        with patch.object(q,"preserve_worker_unblinding",side_effect=ValueError("bad receipt")):
            q.mark_incomplete(obj,RuntimeError("cleanup failed"))
        self.assertIsNone(obj.state["outcome_unblinded"])
        self.assertIn("cleanup failed",obj.state["error"])

    def test_request_contract_uses_replay_then_score_ledger_and_same_deadline(self):
        with tempfile.TemporaryDirectory(dir=q.ROOT/"analysis") as folder:
            obj=q.DiagnosticQueue.__new__(q.DiagnosticQueue)
            obj.folder=Path(folder);obj.adoption_hash="parent";obj.deadline_unix=123456
            obj.a={"candidate":"candidate.json","source_inventory":[],"original_file_sha256":{},"runtime_file_sha256":{},
                "implementation_review":"review.json","output_paths":{"score":"outputs/recovery_projection_score","diagnostic":"outputs/recovery_projection_Q"}}
            obj.locks={"candidate.json":"candidate-hash"};obj.prior_paths={"oracle":"outputs/oracle","replay":"outputs/replay"}
            obj.candidate={"finite_budgets":{"stage_wall_seconds":{"scoring":7200,"diagnostic_continuation":21600}}}
            obj.lock=lambda paths:None
            score=q.old.read_json(obj.request("score"));diagnostic=q.old.read_json(obj.request("diagnostic"))
            self.assertEqual(score["previous_summary"],score["replay_summary"])
            self.assertEqual(diagnostic["previous_summary"],diagnostic["selection_summary"])
            self.assertEqual(diagnostic["previous_cost_json"],"outputs/recovery_projection_score/cost_progress.json")
            self.assertEqual(set(diagnostic["label_files"]),set(q.d.ARMS))
            self.assertEqual(score["global_deadline_unix"],diagnostic["global_deadline_unix"])
            self.assertEqual(score["global_deadline_unix"],123456)
            self.assertEqual(score["physical_gpu"],2)

    def fault_fixture(self, *, pending_signal):
        with tempfile.TemporaryDirectory() as folder:
            obj=q.DiagnosticQueue.__new__(q.DiagnosticQueue);obj.folder=Path(folder);obj.python="fake";obj.env={}
            obj.deadline=time.monotonic()+100;obj.state={"stages":[]};obj.verify=lambda:None;obj.wait_gpu=lambda:None
            obj.update=lambda *args,**kw:None
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
                 patch.object(q.old,"read_json",return_value={"max_wall_seconds":100}), \
                 patch.object(q.signal,"pthread_sigmask",side_effect=mask,create=True), \
                 patch.object(q.signal,"SIG_BLOCK",0,create=True),patch.object(q.signal,"SIG_SETMASK",1,create=True), \
                 patch.object(q.signal,"SIGHUP",1,create=True),patch.object(q.signal,"SIGALRM",14,create=True):
                with self.assertRaises(InterruptedError if pending_signal else subprocess.TimeoutExpired):
                    obj.owned_run("score",Path("request.json"))
            self.assertEqual(spawn.call_count,1)
            self.assertEqual(cleanups,[(42,99)])
            self.assertTrue(obj.state["stages"][0]["owned_cleanup_complete"])
            command=spawn.call_args[0][0]
            self.assertEqual(command[command.index("--gpus")+1],"2")

    def test_pending_signal_after_identity_still_cleans_owned_session(self):
        self.fault_fixture(pending_signal=True)

    def test_stage_timeout_cleans_without_retry(self):
        self.fault_fixture(pending_signal=False)

    def test_scientific_failure_never_invokes_training(self):
        obj=q.DiagnosticQueue.__new__(q.DiagnosticQueue);called=[];updates=[]
        obj.request=lambda stage:stage
        obj.owned_run=lambda stage,request:called.append(stage)
        obj.verify=lambda:None
        obj.update=lambda stage,**kw:updates.append((stage,kw))
        obj.main()
        self.assertEqual(called,["score","diagnostic"])
        self.assertEqual(updates[-1][0],"needs_astra_diagnostic_review")

    def test_candidate_document_cannot_authorize_parent(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"candidate.json";path.write_text(json.dumps({"version":"v2","execution_authorized":False}))
            with self.assertRaisesRegex(RuntimeError,"Parent adoption refused"):
                q.validate_parent(path)


if __name__ == "__main__":
    unittest.main()

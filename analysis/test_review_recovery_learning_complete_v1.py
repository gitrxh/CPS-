"""Offline reviewer pure evidence/arithmetical fixtures; no real jobs."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from analysis import review_recovery_learning_complete_v1 as r
from tests import test_recovery_learning_queue_v1 as fixtures
from tests import test_recovery_evaluate_v1 as neutral
from tests.test_train_recovery_v1 import score_sft_fixture


class ReviewFixtures(unittest.TestCase):
    def test_bound_jsonl_rows_are_materialized_reusable_and_hash_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/"records.jsonl"
            path.write_text('{"index":0}\n{"index":1}\n',encoding="utf-8")
            locks={"records.jsonl":r.sha(path)}
            rows=r.bound_read(root,"records.jsonl",locks,True)
            self.assertEqual(len(rows),2);self.assertEqual(rows[1],{"index":1})
            self.assertEqual(list(rows),list(rows))
            path.write_text('{"index":2}\n',encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"hash differs"):r.bound_read(root,"records.jsonl",locks,True)

    def test_joint_score_sft_audit_and_actual_training_evidence(self):
        label,selected,encoded=score_sft_fixture();encoded["input_tokens"]=5
        anchors=[label["anchor_id"]];schedule=r.train.exact_schedule(349,1)
        labels={arm:[dict(label,arm=arm)] for arm in r.TRAIN}
        selections=[selected]+[{"anchor_id":"structural"+str(i)} for i in range(179)]
        examples={"bc":[dict(encoded) for _ in range(349)],"recovery":[dict(encoded)]}
        slots=[{**s,"anchor_id":anchors[s["index"]] if s["source"]=="recovery" else None,
                **examples[s["source"]][s["index"]]} for s in schedule]
        totals={key:sum(slot[key] for slot in slots) for key in ("target_tokens","prompt_tokens","input_tokens")}
        audit={"kind":"recovery_joint_training_token_audit_v1","complete":True,"settings":r.train.SETTINGS,
            "bc_examples":349,"anchor_count":1,"teacher_fields_used":False,"same_prompt_all_arms":True,
            "score_sft_token_context_mismatches":0,"score_sft_selected_labels_checked":3,"ordered_anchor_ids":anchors,
            "schedule":schedule,"schedule_sha256":r.train.digest(schedule),
            "provenance":{"initial_adapter":{"files_sha256":"initial"}},
            "arms":{arm:{"examples":copy.deepcopy(examples),"scheduled_examples":copy.deepcopy(slots),"totals":dict(totals)} for arm in r.TRAIN}}
        r.validate_joint_audit(audit,labels,selections)
        arm=r.TRAIN[0]
        config={"arm":arm,"settings":r.train.SETTINGS,"protocol":"alfworld-causal-v1",
            "token_audit_sha256":"audit","schedule_sha256":audit["schedule_sha256"]}
        state={"protocol":"alfworld-causal-v1","optimizer_steps":100,"target_tokens":1600,"sampled_examples":800,
            "component_exposures":{"bc":400,"recovery":400},"token_audit_sha256":"audit","schedule_sha256":audit["schedule_sha256"],
            "initial_adapter_optimizer_steps":100,"cumulative_optimizer_steps":200,"optimizer_state_resumed":False,
            "initial_adapter_inventory_sha256":"initial"}
        summary=dict(state,complete=True,matched_step_budget_complete=True,stop_reason="max_steps",requested_steps=100,
                     technical_errors=[],actual_backward_target_tokens=1600,expected_target_tokens=1600)
        exposures=[]
        for slot in slots:exposures.extend([dict(slot,event="forward_started"),dict(slot,event="backward_completed",action_token_nll=1.0)])
        metrics=[{"optimizer_step":i,"window_target_tokens":16,"target_tokens":16*i,"sampled_examples":8*i,
            "component_exposures":{"bc":4*i,"recovery":4*i},"action_token_nll":1.0} for i in range(1,101)]+[{"summary":summary}]
        costs=r.validate_training_evidence(arm,audit,"audit",config,state,summary,exposures,metrics)
        self.assertEqual(costs["target_tokens"],1600)
        metrics[17]["target_tokens"]+=1
        with self.assertRaisesRegex(ValueError,"optimizer-step"):r.validate_training_evidence(arm,audit,"audit",config,state,summary,exposures,metrics)
        audit["arms"][arm]["examples"]["recovery"][0]["target_ids_sha256"]="wrong"
        with self.assertRaisesRegex(ValueError,"Score/SFT token/context"):r.validate_joint_audit(audit,labels,selections)

    def test_actual_file_read_pipeline_complete192_and_corrupted_event(self):
        games,episodes,metadata=fixtures.LearningFixtures().outcomes()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);paths={"queue":"outputs/queue",**{"eval_"+a:"outputs/"+a for a in r.ARMS}}
            locks={};costs={}
            def write(name,value,rows=False):
                target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
                target.write_text("".join(json.dumps(x)+"\n" for x in value) if rows else json.dumps(value),encoding="utf-8")
                locks[name]=r.sha(target);return target
            for arm in r.ARMS:
                base=paths["eval_"+arm];events=[]
                for row in episodes[arm]:
                    events.append({"kind":"evaluation_initial","gamefile":row["gamefile"],"state":row["initial"]})
                    events.extend({"kind":"evaluation_transition","gamefile":row["gamefile"],"transition":t} for t in row["transitions"])
                write(base+"/episodes.jsonl",episodes[arm],True);write(base+"/events.jsonl",events,True)
                costs[arm]={"stage_counts":{"environment_steps_started":48,"policy_calls_started":48,
                    "generated_tokens":sum(row["generated_tokens"] for row in episodes[arm])}}
            stored=fixtures.q.complete_table(games,episodes,metadata)
            stored.update(outcome_unblinded=True,training_costs={},cumulative_costs={})
            write(paths["queue"]+"/result.json",stored)
            adoption=write("adoption.json",{});receipt=write("receipt.json",{})
            status={"adoption_sha256":r.sha(adoption),"exploratory_practical_screen_passed":True}
            status_path=write(paths["queue"]+"/status.json",status)
            lockpath=root/paths["queue"]/"locked_sha256.json"
            lockpath.write_text(json.dumps(locks),encoding="utf-8")
            barrier={"paths":paths,"locks":locks,"costs":costs,"games":games,"metadata":metadata,"status":status,
                "training_costs":{},"cumulative_costs":{},"status_sha256":r.sha(status_path),"locks_sha256":r.sha(lockpath)}
            progress={"controller_outcome_unblinded":True}
            with patch.object(r,"metadata_barrier",return_value=barrier):
                result=r.review(root,adoption,receipt,r.sha(receipt),progress)
            self.assertTrue(result["recovery_learning_result_review_passed"])
            self.assertEqual(result["independent_results"]["episodes"],192)
            event_name=paths["eval_"+r.ARMS[0]]+"/events.jsonl"
            changed=list(r.jsonl(root/event_name));changed[0]["state"]["observation"]="stale"
            write(event_name,changed,True)
            with patch.object(r,"metadata_barrier",return_value=barrier):
                with self.assertRaisesRegex(ValueError,"event stream"):r.review(root,adoption,receipt,r.sha(receipt),progress)

    def test_independent_arithmetic_exact_threshold_and_fixed_denominators(self):
        games,episodes,metadata=fixtures.LearningFixtures().outcomes()
        result=r.independent_results(games,episodes,metadata)
        self.assertTrue(result["exploratory_practical_screen_passed"])
        self.assertEqual(result["reference_minus_comparator"]["expert_recovery"],
                         {"denominator":48,"gains":3,"losses":0,"difference":3,"success_rate_difference":3/48})
        env=neutral.Env(terminal=1,win=False);env.gamefile=games[7]
        episodes[r.ARMS[-1]][7]=r.ev.evaluate_task(env,neutral.Policy(),games[7],neutral.ledger())
        result=r.independent_results(games,episodes,metadata)
        self.assertFalse(result["exploratory_practical_screen_passed"])
        self.assertEqual(len(result["table"]),48)
        self.assertEqual(sum(x["denominator"] for x in result["groups"]["scene"].values()),48)

    def test_partial_and_forged_success_rejected(self):
        games,episodes,metadata=fixtures.LearningFixtures().outcomes()
        missing=copy.deepcopy(episodes);missing[r.ARMS[0]].pop()
        with self.assertRaisesRegex(ValueError,"Partial"):r.independent_results(games,missing,metadata)
        episodes[r.ARMS[0]][0]["success"]=False;neutral.seal(episodes[r.ARMS[0]][0])
        with self.assertRaises(RuntimeError):r.independent_results(games,episodes,metadata)

    def test_incomplete_metadata_does_not_open_any_outcome_stream(self):
        adoption={"output_paths":{"queue":"outputs/queue"},"kind":"recovery_learning_parent_adoption",
                  "adopted":True,"authorized_stages":list(r.STAGES),"physical_gpu":2}
        progress={}
        with patch.object(r,"read",side_effect=[adoption,{"complete":False,"outcome_unblinded":True}]), \
             patch.object(r,"jsonl") as rows:
            with self.assertRaisesRegex(ValueError,"Queue incomplete"):
                r.metadata_barrier(r.ROOT,Path("adoption"),Path("receipt"),"hash",progress)
            rows.assert_not_called()
        self.assertTrue(progress["controller_outcome_unblinded"])

    def test_unblinding_is_persisted_before_first_stream_even_if_validation_fails(self):
        progress={"controller_outcome_unblinded":True,"reviewer_outcome_unblinded":False,"games":[str(i) for i in range(48)]}
        barrier={"paths":{"eval_"+a:"outputs/"+a for a in r.ARMS},"locks":{}}
        observations=[]
        def persist(value):observations.append(copy.deepcopy(value))
        with patch.object(r,"metadata_barrier",return_value=barrier),patch.object(r,"bound_read",side_effect=ValueError("corrupt trace")):
            try:r.review(r.ROOT,Path("a"),Path("b"),"sha",progress,persist)
            except ValueError as error:result=r.failure(progress,error)
        self.assertTrue(observations[0]["reviewer_outcome_unblinded"])
        self.assertTrue(result["reviewer_outcome_unblinded"])
        self.assertEqual(len(result["cells"]),192);self.assertTrue(all(x["outcome"] is None for x in result["cells"]))
        self.assertIsNone(result["contrasts"])

    def test_unknown_controller_unblinding_is_not_reported_false(self):
        result=r.failure({},ValueError("missing receipt"))
        self.assertIsNone(result["controller_outcome_unblinded"])
        self.assertFalse(result["reviewer_outcome_unblinded"])
        self.assertEqual(len(result["cells"]),192)

    def test_cost_inherits_scoring_once_and_rejects_partial_calls(self):
        previous=dict.fromkeys(r.p.COUNTERS,0);previous["forward_tokens"]=123456
        increments=dict.fromkeys(r.p.COUNTERS,0)
        increments.update(resets_started=48,resets_completed=48,environment_steps_started=96,environment_steps_completed=96,
                          policy_calls_started=96,policy_calls_completed=96,generated_tokens=192,forward_tokens=960)
        cost={"kind":"recovery_cost_ledger","active_call":None,"unknown_generation_after_error":False,
              "global_limits":r.p.GLOBAL_LIMITS,"stage_limits":r.ev.STAGE_LIMITS,"stage_counts":increments,
              "totals":{k:previous[k]+increments[k] for k in previous}}
        self.assertEqual(r.validate_costs(previous,cost,{"costs":cost})["forward_tokens"],124416)
        changed=copy.deepcopy(cost);changed["totals"]["forward_tokens"]=960
        with self.assertRaisesRegex(ValueError,"reset or duplicate"):r.validate_costs(previous,changed,{"costs":changed})
        changed=copy.deepcopy(cost);changed["active_call"]={"operation":"generate"}
        with self.assertRaisesRegex(ValueError,"metadata"):r.validate_costs(previous,changed,{"costs":changed})


if __name__=="__main__":unittest.main()

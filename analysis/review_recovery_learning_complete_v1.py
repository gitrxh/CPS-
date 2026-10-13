"""Offline, complete192-only independent arithmetic and stored-evidence review.

No environment/model construction. Large remote weights/runtime/original games
are verified by an independently hash-pinned postqueue receipt. Small source,
audit, training and evaluation artifacts are rehashed on this host. No evaluation
episode/event/result content is opened before the complete metadata barrier.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis import review_recovery_acquisition_complete_v1 as helper
from gnrl import recovery_evaluate_v1 as ev
from gnrl import recovery_projection_v1 as p
from gnrl import train_recovery_v1 as train

require,sha,read,local,jsonl=helper.require,helper.sha,helper.read,helper.local,helper.jsonl
TRAIN=tuple(train.ARMS)
ARMS=tuple(ev.ARMS)
STAGES=("joint_token_audit",*("train_"+a for a in TRAIN),*("eval_"+a for a in ARMS))
REPORT="outputs/recovery_seen48_freeze_v1/report.json"
CHECKED={}


def bound_read(root,name,locks,rows=False):
    path=local(root,name)
    require(sha(path)==locks[name],"Local evidence hash differs: "+name)
    CHECKED[str(path)]=locks[name]
    # Downstream evidence checks require ordered, reusable, indexed rows.
    return list(jsonl(path)) if rows else read(path)


def validate_training_evidence(arm,audit,audit_hash,config,state,summary,exposures,metrics):
    expected=audit["arms"][arm]
    slots=expected["scheduled_examples"]
    require(len(slots)==800 and len(exposures)==1600 and len(metrics)==101,"Incomplete training logs")
    require(config["arm"]==arm and config["settings"]==train.SETTINGS
            and config["protocol"]==state["protocol"]=="alfworld-causal-v1"
            and config["token_audit_sha256"]==audit_hash and config["schedule_sha256"]==audit["schedule_sha256"],"Training config mismatch")
    require(summary.get("complete") is True and summary.get("matched_step_budget_complete") is True
            and summary["stop_reason"]=="max_steps" and summary["requested_steps"]==100
            and summary["technical_errors"]==[] and metrics[-1]=={"summary":summary},"Incomplete final training")
    counts=Counter(); totals=Counter()
    for index,slot in enumerate(slots):
        for offset,event in enumerate(("forward_started","backward_completed")):
            row=exposures[index*2+offset]
            require(row["event"]==event and all(row.get(k)==v for k,v in slot.items()),"Training ordered exposure differs")
        require(math.isfinite(exposures[index*2+1]["action_token_nll"]),"Nonfinite actual training loss")
        counts[slot["source"]]+=1
        for k in ("target_tokens","prompt_tokens","input_tokens"):totals[k]+=slot[k]
    require(dict(counts)=={"bc":400,"recovery":400} and dict(totals)==expected["totals"]
            and 0<totals["target_tokens"]<=2000000,"Training token/exposure totals differ")
    cumulative=0
    for step,row in enumerate(metrics[:-1],1):
        indices=range((step-1)*8,step*8)
        tokens=sum(slots[i]["target_tokens"] for i in indices);cumulative+=tokens
        nll=sum(exposures[2*i+1]["action_token_nll"]*slots[i]["target_tokens"] for i in indices)/tokens
        require(row["optimizer_step"]==step and row["window_target_tokens"]==tokens and row["target_tokens"]==cumulative
                and row["sampled_examples"]==8*step and row["component_exposures"]=={"bc":4*step,"recovery":4*step}
                and math.isfinite(row["action_token_nll"]) and math.isclose(nll,row["action_token_nll"],rel_tol=1e-10,abs_tol=1e-10),
                "Training optimizer-step record differs")
    for r in (state,summary):
        require(r["optimizer_steps"]==100 and r["target_tokens"]==totals["target_tokens"] and r["sampled_examples"]==800
                and r["component_exposures"]==dict(counts) and r["token_audit_sha256"]==audit_hash
                and r["schedule_sha256"]==audit["schedule_sha256"],"Final target/protocol/audit mismatch")
    require(summary["actual_backward_target_tokens"]==summary["expected_target_tokens"]==totals["target_tokens"]
            and state["initial_adapter_optimizer_steps"]==100 and state["cumulative_optimizer_steps"]==200
            and state["optimizer_state_resumed"] is False
            and state["initial_adapter_inventory_sha256"]==audit["provenance"]["initial_adapter"]["files_sha256"],
            "Continuation initialization/final counts differ")
    return {"target_tokens":totals["target_tokens"],"prompt_tokens":totals["prompt_tokens"],"input_tokens":totals["input_tokens"],
            "forward_calls":800,"backward_calls":800,"optimizer_updates":100,"exposures":dict(counts)}


def validate_joint_audit(audit,labels,selections):
    require(audit.get("kind")=="recovery_joint_training_token_audit_v1" and audit.get("complete") is True
            and audit["settings"]==train.SETTINGS and audit["bc_examples"]==349 and audit["anchor_count"]>0
            and audit["teacher_fields_used"] is False and audit["same_prompt_all_arms"] is True
            and audit["score_sft_token_context_mismatches"]==0,"Invalid joint audit")
    anchors=train.validate_labels(labels)
    require(anchors==audit["ordered_anchor_ids"] and len(anchors)==audit["anchor_count"]
            and audit["score_sft_selected_labels_checked"]==3*len(anchors),"Joint labels differ")
    schedule=train.exact_schedule(349,len(anchors))
    require(audit["schedule"]==schedule and audit["schedule_sha256"]==train.digest(schedule),"Joint schedule differs")
    selected={s["anchor_id"]:s for s in selections}
    require(len(selected)==len(selections)==180,"Incomplete frozen selections")
    for arm in TRAIN:
        examples=audit["arms"][arm]["examples"]
        require(len(examples["bc"])==349 and len(examples["recovery"])==len(anchors),"Missing token examples")
        require(examples["bc"]==audit["arms"][TRAIN[0]]["examples"]["bc"],"Original BC token evidence differs across arms")
        for encoded in examples["bc"]+examples["recovery"]:
            require(type(encoded["target_tokens"]) is int and 0<encoded["target_tokens"]<=64
                    and type(encoded["prompt_tokens"]) is int and encoded["prompt_tokens"]>0
                    and encoded["input_tokens"]==encoded["prompt_tokens"]+encoded["target_tokens"]<=2048,
                    "Training encoded-token counts malformed")
        for i,label in enumerate(labels[arm]):
            train.verify_score_sft(label,selected[label["anchor_id"]],examples["recovery"][i])
        expected=[{**s,"anchor_id":anchors[s["index"]] if s["source"]=="recovery" else None,
                   **examples[s["source"]][s["index"]]} for s in schedule]
        require(audit["arms"][arm]["scheduled_examples"]==expected,"Recorded training schedule changed")
    for index in range(len(anchors)):
        require(len({audit["arms"][a]["examples"]["recovery"][index]["prompt_ids_sha256"] for a in TRAIN})==1,
                "Cross-arm prompt bytes differ")


def metadata_barrier(root,adoption_path,receipt_path,receipt_hash,progress):
    a=read(adoption_path);paths=a["output_paths"]
    require(a.get("kind")=="recovery_learning_parent_adoption" and a.get("adopted") is True
            and a["authorized_stages"]==list(STAGES) and a["physical_gpu"]==2,"Wrong learning adoption")
    status_path=local(root,paths["queue"]+"/status.json");locks_path=local(root,paths["queue"]+"/locked_sha256.json")
    status=read(status_path)
    progress["controller_outcome_unblinded"]=status.get("outcome_unblinded")
    require(status.get("complete") is True and status.get("current_stage")=="needs_astra_learning_result_review"
            and status.get("outcome_unblinded") is True and not status.get("error")
            and status["adoption_sha256"]==sha(adoption_path),"Queue incomplete; evaluation files remain closed")
    require([s["name"] for s in status["stages"]]==list(STAGES)
            and all(s.get("returncode")==0 and s.get("owned_cleanup_complete") is True for s in status["stages"]),"Incomplete eight-stage process barrier")
    locks=read(locks_path)
    require(sha(receipt_path)==receipt_hash,"Postqueue receipt changed")
    receipt=read(receipt_path)
    require(receipt.get("kind")=="recovery_learning_postqueue_integrity" and receipt.get("complete") is True
            and receipt.get("errors")==[] and receipt["adoption_sha256"]==sha(adoption_path)
            and receipt["queue_status_sha256"]==sha(status_path) and receipt["locked_sha256_file_sha256"]==sha(locks_path),"Wrong remote receipt")
    require(receipt["controller"].get("exited") is True and receipt.get("owned_workers_gone") is True
            and all(type(receipt["controller"].get(k)) is int and receipt["controller"][k]>0 for k in ("pid","start_ticks"))
            and len(receipt["stages"])==8,"Controller/workers not exited")
    for recorded,exited in zip(status["stages"],receipt["stages"]):
        require(all(recorded[k]==exited[k] for k in ("name","pid","start_ticks","owned_session"))
                and exited.get("exited") is True,"Exited identity differs")
    for dest,source in (("verified_original_file_sha256","original_file_sha256"),("verified_runtime_file_sha256","runtime_file_sha256"),
                        ("source_inventory","source_inventory")):
        require(receipt[dest]==a[source],"Remote frozen inventory differs")
    require(receipt["verified_file_sha256"]==locks and all(locks.get(n)==h for n,h in a["file_sha256"].items()),"Prior locks changed")
    # Verify the code actually imported locally; remote inventories alone cannot bind local validators.
    for name in a["source_inventory"]:
        path=local(root,name)
        require(sha(path)==locks[name],"Local validator/source differs: "+name)
        CHECKED[str(path)]=locks[name]
    require(sha(helper.__file__)=="42fe7a2f7c840d61514b58926edc935bfc2518dc10fa6eadbbe0ea55dc2390cd","Review utility source changed")
    candidate=bound_read(root,a["candidate"],locks)
    prior_status=bound_read(root,a["diagnostic_status"],locks)
    require(status["global_deadline_unix"]==prior_status["global_deadline_unix"]
            and status["main_study_started_unix"]==prior_status["main_study_started_unix"]
            and status["global_deadline_unix"]-status["main_study_started_unix"]==171840,"Deadline reset")
    require(status["main_study_started_unix"] <= datetime.fromisoformat(status["updated_utc"]).timestamp()
            < status["global_deadline_unix"], "Completed queue exceeded inherited deadline")
    diag_adoption=bound_read(root,a["diagnostic_parent_adoption"],locks);diagpaths=diag_adoption["output_paths"]
    prior_review=bound_read(root,a["diagnostic_result_review"],locks)
    require(prior_review.get("kind")=="recovery_complete_diagnostic_result_review"
            and prior_review.get("recovery_diagnostic_result_review_passed") is True
            and prior_review.get("training_gate_passed") is True and prior_status.get("training_gate_passed") is True,"Unpassed training gate")
    for field,name in (("adoption_sha256",a["diagnostic_parent_adoption"]),("diagnostic_status_sha256",a["diagnostic_status"]),
                       ("diagnostic_locks_sha256",a["diagnostic_locks"]),("diagnostic_result_sha256",diagpaths["diagnostic"]+"/diagnostic.json"),
                       ("diagnostic_cost_sha256",diagpaths["diagnostic"]+"/cost_progress.json"),("score_records_sha256",diagpaths["score"]+"/records.jsonl")):
        require(prior_review[field]==locks[name],"Prior diagnostic review binding differs")
    inventory=bound_read(root,a["input_inventory"],locks)
    require(receipt["base_inventory_sha256"]==inventory["base_full_inventory_sha256"]==locks[inventory["base_full_inventory"]]
            and receipt["base_files_verified"]==28 and receipt["initial_adapter_sha256"]==inventory["initial_adapter_sha256"],"Original base/adapter receipt differs")
    for field,flag,key,source in (("controller_review","recovery_learning_controller_review_passed","controller_sha256","scripts/run_recovery_learning_queue_v1.py"),
            ("training_review","recovery_training_review_passed","module_sha256","gnrl/train_recovery_v1.py"),
            ("evaluator_review","recovery_evaluator_review_passed","evaluator_sha256","gnrl/recovery_evaluate_v1.py")):
        review=bound_read(root,a[field],locks)
        require(review.get(flag) is True and review[key]==locks[source] and review["candidate_sha256"]==locks[a["candidate"]],"Component review differs")
    games=bound_read(root,ev.MANIFEST,locks)["games"]
    require(locks[ev.MANIFEST]==ev.MANIFEST_SHA256 and len(games)==len(set(games))==48,"Frozen48 differs")
    progress["games"]=games
    freeze=candidate["new_research_evaluation"]["frozen_metadata_artifacts"]
    metadata_report=bound_read(root,REPORT,locks)
    require(locks[REPORT]==freeze["report_sha256"] and [r["gamefile"] for r in metadata_report["selected"]]==games,"Frozen metadata order differs")
    metadata={}
    for r in metadata_report["selected"]:
        game=r["gamefile"]
        require(a["original_file_sha256"][game]==r["game_sha256"]
                and a["original_file_sha256"][str(Path(game).with_name("traj_data.json")).replace("\\","/")]==r["metadata_sha256"],"Frozen scene metadata differs")
        metadata[game]={"family":r["family"],"scene":r["scene"]}
    audit=bound_read(root,paths["joint_token_audit"],locks)
    labels={arm:bound_read(root,diagpaths["score"]+"/"+arm+".jsonl",locks,True) for arm in TRAIN}
    selections=bound_read(root,diagpaths["score"]+"/records.jsonl",locks,True)
    validate_joint_audit(audit,labels,selections)
    barrier=bound_read(root,paths["queue"]+"/all_models_locked.json",locks)
    require(barrier.get("complete") is True and barrier["arms"]==list(ARMS) and set(barrier["adapters"])==set(ARMS)
            and barrier["joint_token_audit_sha256"]==locks[paths["joint_token_audit"]]
            and receipt["all_models_locked_sha256"]==locks[paths["queue"]+"/all_models_locked.json"]
            and receipt["all_models_adapter_sha256"]=={arm:item["file_sha256"] for arm,item in barrier["adapters"].items()},"Four-model barrier differs")
    training_costs={}
    require(len({item["path"] for item in barrier["adapters"].values()})==4,"Repeated model path")
    for arm,item in barrier["adapters"].items():
        require(all(locks.get(n)==h for n,h in item["file_sha256"].items()),"Adapter bytes unbound")
        required={item["path"]+"/training_state.json",item["path"]+"/adapter_config.json"}
        require(required<=item["file_sha256"].keys()
                and any(n.endswith(("/adapter_model.bin","/adapter_model.safetensors")) for n in item["file_sha256"]),"Missing actual adapter files")
        state=bound_read(root,item["path"]+"/training_state.json",locks)
        if arm==ARMS[0]:
            initial=audit["provenance"]["initial_adapter"]
            require(item["path"]==ev.CANONICAL and item["file_sha256"]==inventory["initial_adapter_sha256"]
                    and item["file_sha256"]=={ev.CANONICAL+"/"+f["name"]:f["sha256"] for f in initial["files"]}
                    and p.digest(initial["files"])==initial["files_sha256"]
                    and state==audit["provenance"]["initial_training_state"],"Canonical initialization differs")
            continue
        folder=paths["train_"+arm]
        require(item["path"]==folder+"/final","Wrong trained checkpoint path")
        summary=bound_read(root,folder+"/summary.json",locks)
        training_costs[arm]=validate_training_evidence(arm,audit,locks[paths["joint_token_audit"]],
            bound_read(root,folder+"/config.json",locks),state,summary,
            bound_read(root,folder+"/example_exposures.jsonl",locks,True),bound_read(root,folder+"/metrics.jsonl",locks,True))
        saved=bound_read(root,folder+"/final_adapter_inventory.json",locks)
        require(p.digest(saved["files"])==saved["files_sha256"]==summary["final_adapter_files_sha256"]
                and summary["final_adapter_inventory_sha256"]==locks[folder+"/final_adapter_inventory.json"]
                and {item["path"]+"/"+f["name"]:f["sha256"] for f in saved["files"]}==item["file_sha256"],"Saved final inventory differs")
    require(status["training_costs"]==training_costs,"Training costs differ from full logs")
    costs={};summaries={};requests={}
    previous=bound_read(root,diagpaths["diagnostic"]+"/cost_progress.json",locks)["totals"]
    for index,arm in enumerate(ARMS):
        folder=paths["eval_"+arm]
        summary=summaries[arm]=bound_read(root,folder+"/summary.json",locks)
        cost=costs[arm]=bound_read(root,folder+"/cost_progress.json",locks)
        request_name=paths["queue"]+"/eval_"+arm+"_request.json"
        request=requests[arm]=bound_read(root,request_name,locks)
        require(summary.get("complete") is True and summary["arm"]==arm and summary["stage"]=="evaluation"
                and summary["planned_episodes"]==summary["attempted_episodes"]==summary["completed_episodes"]==48
                and summary["technical_errors"]==0 and summary["partial_performance_summary"] is None
                and summary["request_sha256"]==locks[request_name] and summary["episodes_sha256"]==locks[folder+"/episodes.jsonl"],"Incomplete48 model barrier")
        require(request["arm"]==arm and request["adapter"]==barrier["adapters"][arm]["path"]
                and request["parent_adoption_sha256"]==sha(adoption_path) and request["global_deadline_unix"]==status["global_deadline_unix"]
                and request["max_wall_seconds"]==7200 and request["output"]==folder and request["manifest"]==ev.MANIFEST
                and request["file_sha256"].get(paths["queue"]+"/all_models_locked.json")==locks[paths["queue"]+"/all_models_locked.json"]
                and all(locks.get(n)==h for n,h in request["file_sha256"].items()),"Evaluation request/barrier differs")
        predecessor=diagpaths["diagnostic"] if index==0 else paths["eval_"+ARMS[index-1]]
        require(request["previous_cost_json"]==predecessor+"/cost_progress.json" and request["previous_summary"]==predecessor+"/summary.json","Evaluation ledger order differs")
        config=bound_read(root,folder+"/config.json",locks)
        require(config["request"]==request and config["request_sha256"]==locks[request_name],"Evaluation config request differs")
        for key,value in {"protocol":"alfworld-causal-v1","model":ev.MODEL,"adapter":request["adapter"],"selection":"generate",
                "do_sample":False,"max_context_tokens":2048,"max_action_tokens":64,"max_steps":100,"seed":20261012,
                "expert_type":None,"max_generated_tokens":350000,"max_scored_tokens":10000000,"max_wall_seconds":7200}.items():
            require(type(config.get(key)) is type(value) and config[key]==value,"Evaluation setting differs: "+key)
        previous=validate_costs(previous,cost,summary)
        # Hash only; content remains unopened until every arm and all prerequisite evidence passes.
        for file in ("episodes.jsonl","events.jsonl"):require(sha(local(root,folder+"/"+file))==locks[folder+"/"+file],"Evaluation bytes differ")
    require(previous==status["cumulative_costs"],"Final cumulative costs differ")
    return {"adoption":a,"paths":paths,"status":status,"locks":locks,"games":games,"metadata":metadata,
            "costs":costs,"training_costs":training_costs,"cumulative_costs":previous,
            "status_sha256":sha(status_path),"locks_sha256":sha(locks_path)}


def validate_costs(previous,cost,summary):
    require(cost["kind"]=="recovery_cost_ledger" and cost["active_call"] is None
            and cost["unknown_generation_after_error"] is False and cost["global_limits"]==p.GLOBAL_LIMITS
            and cost["stage_limits"]==ev.STAGE_LIMITS and cost["totals"]==summary["costs"]["totals"]
            and cost["stage_counts"]==summary["costs"]["stage_counts"],"Cost metadata mismatch")
    require(set(cost["totals"])==set(cost["stage_counts"])==set(previous)==set(p.COUNTERS),"Missing counters")
    for key,amount in cost["stage_counts"].items():
        require(type(amount) is int and amount>=0 and cost["totals"][key]==previous[key]+amount,"Cost reset or duplicate")
    for key,limit in p.GLOBAL_LIMITS.items():
        require(cost["totals"][key]<=limit and cost["stage_counts"][key]<=ev.STAGE_LIMITS[key],"Budget exceeded")
    for stem in ("environment_steps","resets","policy_calls"):
        require(cost["totals"][stem+"_started"]==cost["totals"][stem+"_completed"],"Unfinished cost call")
    require(cost["stage_counts"]["resets_started"]==48,"Wrong reset count")
    return cost["totals"]


def independent_results(games,episodes,metadata):
    require(set(episodes)==set(ARMS) and len(games)==len(set(games))==48,"Missing192 outcomes")
    won={}
    for arm in ARMS:
        require(len(episodes[arm])==48,"Partial model result")
        won[arm]=[]
        for game,row in zip(games,episodes[arm]):
            ev.validate_episode(row,game)
            actual=row["initial"]["won"] or any(t["after"]["won"] for t in row["transitions"])
            require(actual is row["success"],"Success flag not actual won")
            won[arm].append(actual)
    table=[{"gamefile":g,**metadata[g],"success":{a:won[a][i] for a in ARMS}} for i,g in enumerate(games)]
    successes={a:sum(won[a]) for a in ARMS};ref=ARMS[-1]
    contrasts={}
    for arm in ARMS[:-1]:
        gains=sum(x and not y for x,y in zip(won[ref],won[arm]));losses=sum(y and not x for x,y in zip(won[ref],won[arm]))
        contrasts[arm]={"denominator":48,"gains":gains,"losses":losses,"difference":gains-losses,"success_rate_difference":(gains-losses)/48}
    groups={}
    for field in ("family","scene"):
        groups[field]={}
        for value in sorted({metadata[g][field] for g in games}):
            indices=[i for i,g in enumerate(games) if metadata[g][field]==value]
            groups[field][value]={"denominator":len(indices),"successes":{a:sum(won[a][i] for i in indices) for a in ARMS}}
    require(len(groups["family"])==6 and all(r["denominator"]==8 for r in groups["family"].values()),"Family denominators differ")
    passed=(successes[ref]-successes["expert_recovery"]>=3 and successes[ref]-successes["uniform_viable_recovery"]>=3
            and successes[ref]>successes["frozen_canonical_BC"])
    return {"complete":True,"episodes":192,"task_denominator":48,"table":table,"successes":successes,
            "reference_minus_comparator":contrasts,"groups":groups,"exploratory_practical_screen_passed":passed}


def review(root,adoption_path,receipt_path,receipt_hash,progress,persist=lambda value:None):
    CHECKED.clear()
    barrier=metadata_barrier(root,adoption_path,receipt_path,receipt_hash,progress)
    progress.update(reviewer_outcome_unblinded=True,reviewer_outcome_unblinded_unix=time.time())
    persist(progress)
    paths,locks=barrier["paths"],barrier["locks"];episodes={}
    for arm in ARMS:
        folder=paths["eval_"+arm]
        rows=episodes[arm]=bound_read(root,folder+"/episodes.jsonl",locks,True)
        events=bound_read(root,folder+"/events.jsonl",locks,True)
        expected=[]
        for row in rows:
            expected.append({"kind":"evaluation_initial","gamefile":row["gamefile"],"state":row["initial"]})
            expected.extend({"kind":"evaluation_transition","gamefile":row["gamefile"],"transition":t} for t in row["transitions"])
        require(events==expected,"Actual evaluation event stream differs")
        counts=barrier["costs"][arm]["stage_counts"]
        steps=sum(len(row["transitions"]) for row in rows)
        require(counts["environment_steps_started"]==counts["policy_calls_started"]==steps
                and counts["generated_tokens"]==sum(t["policy"]["generated_tokens"] for row in rows for t in row["transitions"]),"Trace cost sum differs")
    result=independent_results(barrier["games"],episodes,barrier["metadata"])
    stored=bound_read(root,paths["queue"]+"/result.json",locks)
    require(all(stored.get(k)==v for k,v in result.items()) and stored["outcome_unblinded"] is True
            and stored["training_costs"]==barrier["training_costs"] and stored["cumulative_costs"]==barrier["cumulative_costs"],"Independent result differs")
    require(barrier["status"]["exploratory_practical_screen_passed"] is result["exploratory_practical_screen_passed"],"Queue screen differs")
    require(sha(receipt_path)==receipt_hash and sha(adoption_path)==barrier["status"]["adoption_sha256"]
            and sha(local(root,paths["queue"]+"/status.json"))==barrier["status_sha256"]
            and sha(local(root,paths["queue"]+"/locked_sha256.json"))==barrier["locks_sha256"],"Review inputs changed")
    for name,h in CHECKED.items():require(sha(name)==h,"Local evidence changed during review: "+name)
    return {"kind":"recovery_complete_learning_result_review","complete":True,"errors":[],
        "recovery_learning_result_review_passed":True,"independent_results":result,
        "exploratory_practical_screen_passed":result["exploratory_practical_screen_passed"],
        "controller_outcome_unblinded":progress["controller_outcome_unblinded"],"reviewer_outcome_unblinded":True,
        "review_program_sha256":sha(__file__),"postqueue_receipt_sha256":receipt_hash,
        "adoption_sha256":sha(adoption_path),"learning_status_sha256":barrier["status_sha256"],
        "learning_locks_sha256":barrier["locks_sha256"],"learning_result_sha256":locks[paths["queue"]+"/result.json"],
        "training_costs":barrier["training_costs"],"cumulative_costs":barrier["cumulative_costs"],"execution_authorized":False,
        "interpretation":"Bounded one-seed exploratory seen-task evidence only; no independent-episode significance, unseen authorization or goal-completion claim.",
        "remote_evidence_scope":"Weight/model/game/runtime bytes and process exit rely on pinned independent remote receipt; local small artifacts and loaded validators are rehashed. Generated token and environment/call counts are recomputed from actual traces; forward token cost is inherited from the durable worker ledger because frozen generation details do not include prompt token IDs."}


def failure(progress,error):
    return {"kind":"recovery_complete_learning_result_review","complete":False,"recovery_learning_result_review_passed":False,
        "errors":[f"{type(error).__name__}: {error}"],"controller_outcome_unblinded":progress.get("controller_outcome_unblinded"),
        "reviewer_outcome_unblinded":progress.get("reviewer_outcome_unblinded",False),"contrasts":None,
        "exploratory_practical_screen_passed":None,"execution_authorized":False,
        "cells":[{"arm":arm,"gamefile":game,"outcome":None,"status":"technical_incomplete"}
                 for arm in ARMS for game in progress.get("games",[None]*48)]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adoption-json",type=Path,required=True)
    parser.add_argument("--postqueue-receipt",type=Path,required=True)
    parser.add_argument("--postqueue-sha256",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();require(not args.output.exists(),"Review output already exists")
    progress={"reviewer_outcome_unblinded":False,"controller_outcome_unblinded":None}
    journal=args.output.with_suffix(args.output.suffix+".progress.json")
    require(not journal.exists(),"Review progress journal already exists")
    def persist(value):
        tmp=journal.with_suffix(journal.suffix+".tmp")
        tmp.write_text(json.dumps(value,indent=2),encoding="utf-8");tmp.replace(journal)
    persist(progress)
    try:
        if sha(ROOT/ev.MANIFEST)==ev.MANIFEST_SHA256:
            progress["games"]=read(ROOT/ev.MANIFEST)["games"]
        result=review(ROOT,args.adoption_json,args.postqueue_receipt,args.postqueue_sha256,progress,persist)
    except Exception as error:result=failure(progress,error)
    persist(progress)
    with args.output.open("x",encoding="utf-8") as stream:json.dump(result,stream,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps({k:result[k] for k in ("complete","recovery_learning_result_review_passed","exploratory_practical_screen_passed","errors")}))
    return 0 if result["complete"] else 1


if __name__=="__main__":raise SystemExit(main())

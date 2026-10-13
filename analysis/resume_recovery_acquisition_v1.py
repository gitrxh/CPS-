"""Single explicitly adopted interruption recovery; CPU oracle/replay only.

Original acquisition files are immutable. A complete ordered prefix is copied
byte-for-byte, with its corresponding events checked against every certificate.
No scoring, training, evaluation, generic retry, or scientific summary occurs.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, closing
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_successful_path_learning_v1 as old
from scripts import collect_recovery_projection_v1 as worker
from gnrl import recovery_projection_v1 as m

SCRIPT = "analysis/resume_recovery_acquisition_v1.py"
TEST = "analysis/test_resume_recovery_acquisition_v1.py"
INVENTORY = "outputs/recovery_resume_inventory_v1.json"
INVENTORY_SHA = "38ca390d0c0375074de058751856e8ee952fd417d638e5bb32309b949ab26ae2"
CALLS = ("environment_steps", "resets", "policy_calls")
SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM) if sys.platform.startswith("linux") else ()


def lines(path, tail):
    """Yield complete original lines; only a final uncommitted fragment is dropped."""
    with Path(path).open("rb") as stream:
        for raw in stream:
            if not raw.endswith(b"\n"):
                tail.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
                return
            m.require(raw.strip(), "Blank complete JSONL record")
            yield raw, json.loads(raw)


def collection_anchors(path, games):
    tail, count = {}, 0
    for _, episode in lines(path, tail):
        m.require(count < len(games) and episode["gamefile"] == games[count], "Collection game order differs")
        m.validate_episode(episode)
        for anchor in episode["anchors"]:
            yield anchor
        count += 1
    m.require(not tail and count == 60 == len(games), "Incomplete collection")


def pool_events(pool):
    for branch in pool["branches"]:
        for transition in branch["transitions"]:
            yield {"kind": "oracle_transition", "anchor_id": pool["anchor_id"],
                   "candidate_index": branch["candidate_index"], "transition": transition}
        yield {"kind": "oracle_branch_complete", "anchor_id": pool["anchor_id"], "branch": branch}


def copy_prefix(records_path, events_path, anchors, records_out, events_out, expected):
    """Stream records/events in lockstep. Never select pools by outcome."""
    record_tail, event_tail = {}, {}
    with ExitStack() as stack:
        records = stack.enter_context(closing(lines(records_path, record_tail)))
        events = stack.enter_context(closing(lines(events_path, event_tail)))
        if hasattr(anchors, "close"):
            stack.enter_context(closing(anchors))
        return _copy_prefix(records, events, anchors, records_out, events_out, expected, record_tail, event_tail)


def _copy_prefix(records, events, anchors, records_out, events_out, expected, record_tail, event_tail):
    anchors = iter(anchors)
    count = copied_events = copied_bytes = 0
    for raw, pool in records:
        m.require(count < expected, "Unexpected additional complete pool")
        anchor = next(anchors, None)
        m.validate_pool(pool)
        m.require(pool["anchor"] == anchor, "Pool does not equal ordered complete collection anchor")
        for wanted in pool_events(pool):
            actual = next(events, None)
            m.require(actual is not None and actual[1] == wanted, "Oracle event missing, reordered, or changed")
            events_out.write(actual[0])
            copied_events += 1
            copied_bytes += len(actual[0])
        records_out.write(raw)
        count += 1
    m.require(count == expected, "Complete prefix count differs from adoption")
    next_anchor = next(anchors, None)
    discarded_events = discarded_bytes = 0
    discarded_hash = hashlib.sha256()
    for raw, event in events:
        m.require(next_anchor is not None and event.get("anchor_id") == next_anchor["anchor_id"]
                  and event.get("kind") in ("oracle_transition", "oracle_branch_complete"),
                  "Old event tail includes completed or out-of-order anchor")
        discarded_events += 1
        discarded_bytes += len(raw)
        discarded_hash.update(raw)
    # Exhaust collection validation, including its final newline and all60 identities.
    for _ in anchors:
        pass
    records_out.flush()
    events_out.flush()
    return {"copied_records": count, "copied_events": copied_events, "copied_event_bytes": copied_bytes,
            "record_tail_fragment": record_tail or None, "event_tail_fragment": event_tail or None,
            "interrupted_anchor_events_excluded": discarded_events,
            "interrupted_anchor_event_bytes": discarded_bytes,
            "interrupted_anchor_events_sha256": discarded_hash.hexdigest(),
            "old_artifacts_preserved": True}


def gaps(totals):
    return {kind: totals[kind + "_started"] - totals[kind + "_completed"] for kind in CALLS}


def remaining_limits(previous):
    m.require(previous["stage_limits"] == worker.STAGE_LIMITS["oracle"], "Old stage limits changed")
    result = {k: limit - previous["stage_counts"][k] for k, limit in worker.STAGE_LIMITS["oracle"].items()}
    m.require(all(type(v) is int and v >= 0 for v in result.values()), "Old stage budget exceeded")
    return result


def original_oracle_deadline(status, oracle_stage):
    return min(datetime.fromisoformat(oracle_stage["started_utc"]).timestamp() + worker.STAGE_WALL["oracle"],
               status["global_deadline_unix"] - 30)


def validate_historical_cost(cost, collection):
    m.require(cost["kind"] == "recovery_cost_ledger" and cost["global_limits"] == m.GLOBAL_LIMITS,
              "Wrong historical cost ledger")
    m.require(set(cost["totals"]) == set(cost["stage_counts"]) == set(m.COUNTERS), "Missing historical counters")
    m.require(all(type(v) is int and v >= 0 for d in (cost["totals"], cost["stage_counts"]) for v in d.values()),
              "Invalid historical counters")
    m.require(cost["totals"] == {k: collection["totals"][k] + cost["stage_counts"][k] for k in m.COUNTERS},
              "Historical cost lineage differs")
    m.require(cost["unknown_generation_after_error"] is False and
              cost["active_call"] == {"operation": "step", "label": "oracle_prefix"} and
              gaps(cost["totals"]) == {"environment_steps": 1, "resets": 0, "policy_calls": 0},
              "Historical interruption differs from reviewed recovery")
    remaining_limits(cost)


def validate_new_cost(cost, initial, historical_gap):
    m.require(cost["active_call"] is None and cost["unknown_generation_after_error"] is False,
              "New calls incomplete")
    m.require(all(gap == 0 for gap in gaps(cost["stage_counts"]).values()), "New stage calls unbalanced")
    m.require(cost["totals"] == {k: initial[k] + cost["stage_counts"][k] for k in m.COUNTERS},
              "Inherited counters changed")
    m.require(gaps(cost["totals"]) == historical_gap, "Historical unmatched call erased or enlarged")


class ResumeLedger(m.BudgetLedger):
    def __init__(self, *, provenance, **kwargs):
        self.provenance = provenance
        super().__init__(**kwargs)

    def snapshot(self):
        result = super().snapshot()
        result["resume_provenance"] = self.provenance
        result["historical_gap"] = self.provenance["historical_gap"]
        if self.provenance["stage"] == "oracle":
            result["combined_oracle_stage_counts"] = {k: self.provenance["old_oracle_stage_counts"][k] +
                result["stage_counts"][k] for k in m.COUNTERS}
            result["combined_oracle_execution_seconds"] = self.provenance["old_oracle_controller_elapsed_seconds"] + result["elapsed_seconds"]
        return result


def load_contract(path):
    a = old.read_json(path)
    for k, value in {"kind": "recovery_acquisition_resume_parent_adoption", "version": 1,
                     "adopted": True, "adopted_by": "root_after_independent_GPT_6_Astra_review",
                     "retry_authorized": True, "retry_reason": "user-requested interruption recovery",
                     "expected_completed_records": 131, "inventory": INVENTORY}.items():
        m.require(type(a.get(k)) is type(value) and a[k] == value, "Resume adoption gate: " + k)
    m.require(a["authorized_stages"] == ["oracle", "replay"], "Resume stages differ")
    hashes = a["file_sha256"]
    m.require({SCRIPT, TEST, a["old_parent"], a["inventory"], a["resume_review"]} <= hashes.keys(), "Missing resume bindings")
    m.require(hashes[INVENTORY] == INVENTORY_SHA, "Wrong interruption inventory")
    old.verify_hashes(hashes)
    review = old.read_json(ROOT / a["resume_review"])
    m.require(review.get("kind") == "recovery_acquisition_resume_astra_review" and
              review.get("resume_review_passed") is True and review.get("resume_script_sha256") == hashes[SCRIPT]
              and review.get("test_sha256") == hashes[TEST], "Missing exact independent Astra review")
    inv = old.read_json(ROOT / a["inventory"])
    parent = inv["documents"][a["old_parent"]]
    m.require(old.read_json(ROOT / a["old_parent"]) == parent and parent["kind"] == "recovery_acquisition_parent_adoption"
              and parent["adopted"] is True, "Old parent differs")
    paths = parent["output_paths"]
    docs = inv["documents"]
    status = docs[paths["queue"] + "/status.json"]
    m.require(status["adoption_sha256"] == hashes[a["old_parent"]], "Old controller adoption hash differs")
    oracle_stage = [s for s in status["stages"] if s["name"] == "oracle"]
    m.require(len(oracle_stage) == 1 and oracle_stage[0]["owned_cleanup_complete"] is True
              and oracle_stage[0]["returncode"] != 0 and status["complete"] is False
              and status["current_stage"] == "technical_incomplete", "Old controller state differs")
    old_stage = oracle_stage[0]
    deadline = status["global_deadline_unix"]
    m.require(deadline == status["main_study_started_unix"] + 171840, "Original global deadline differs")
    oracle_deadline = original_oracle_deadline(status, old_stage)
    cost = docs[paths["oracle"] + "/cost_progress.json"]
    collect_cost = docs[paths["collect"] + "/cost_progress.json"]
    validate_historical_cost(cost, collect_cost)
    summary = docs[paths["collect"] + "/summary.json"]
    m.require(summary["complete"] is True and summary["stage"] == "collect" and
              summary["record_count"] == summary["planned_records"] == 60 and
              summary["costs"]["totals"] == collect_cost["totals"] and
              summary["records_sha256"] == inv["files"][paths["collect"] + "/records.jsonl"]["sha256"],
              "Collection completion contract differs")
    outputs = a["output_paths"]
    m.require(set(outputs) == {"queue", "oracle", "replay"} and len(set(outputs.values())) == 3, "Bad output paths")
    resolved = [old.safe_relative(p) for p in outputs.values()]
    m.require(all(p.startswith("outputs/recovery_projection_") for p in outputs.values()) and
              not any(x != y and (x.is_relative_to(y) or y.is_relative_to(x)) for x in resolved for y in resolved),
              "Output namespace/overlap differs")
    m.require(not any(x == (ROOT / p).resolve() or x.is_relative_to((ROOT / p).resolve()) or
                       (ROOT / p).resolve().is_relative_to(x) for x in resolved for p in paths.values()),
              "Resume outputs overlap old outputs")
    identities = {(p["pid"], p["start_ticks"], p["session"]) for p in a["old_processes"]}
    required_processes = {(455646, 30336263, 455646)} | {
        (s["pid"], s["start_ticks"], s["owned_session"]) for s in status["stages"]}
    m.require(identities == required_processes and len(a["old_processes"]) == len(identities), "Old process identities incomplete")
    return a, inv, parent, status, old_stage, cost, oracle_deadline


def verify_inputs(contract, adoption_path, adoption_hash, extra=None, full_model=True):
    a, inv, parent, status, _, _, _ = contract
    m.require(time.time() < status["global_deadline_unix"] - 30, "Original global deadline exhausted")
    m.require(old.sha256(adoption_path) == adoption_hash, "Resume adoption changed")
    old.verify_hashes(a["file_sha256"])
    m.require(len(parent["source_inventory"]) == 50 and worker.source_inventory() == parent["source_inventory"],
              "Original source directory inventory changed")
    old.verify_hashes(parent["file_sha256"])
    lock = inv["documents"][parent["output_paths"]["queue"] + "/locked_sha256.json"]
    m.require(all(lock.get(k) == v for k, v in parent["file_sha256"].items()), "Old lock lost parent bindings")
    old.verify_hashes(lock)
    for name, entry in inv["files"].items():
        p = old.safe_relative(name)
        m.require(p.stat().st_size == entry["bytes"] and old.sha256(p) == entry["sha256"], "Interrupted artifact changed: " + name)
    for name, document in inv["documents"].items():
        m.require(name in inv["files"] and old.read_json(ROOT / name) == document, "Inventory document differs: " + name)
    candidate = old.read_json(ROOT / parent["candidate"])
    m.require(candidate["version"] == "v2" and candidate["finite_budgets"]["global_wall_seconds_including_waits_verification_cleanup"] == 171840,
              "Original candidate changed")
    for field, expected in (("runtime_file_sha256", 9), ("original_file_sha256", 336)):
        m.require(len(parent[field]) == expected, "Original inventory count differs")
        for name, value in parent[field].items():
            m.require(old.sha256(name) == value, "Original runtime/game changed: " + name)
    initial = old.read_json(ROOT / parent["input_inventory"])
    adapter_files = {p.relative_to(ROOT).as_posix() for p in (ROOT / worker.ADAPTER).rglob("*") if p.is_file()}
    m.require(initial["complete"] is True and initial["original_file_count"] == 336 and
              initial["original_file_sha256"] == parent["original_file_sha256"] and
              len(initial["initial_adapter_sha256"]) == 11 and adapter_files == set(initial["initial_adapter_sha256"]),
              "Original input/adapter inventory changed")
    old.verify_hashes(initial["initial_adapter_sha256"])
    model = old.read_json(ROOT / old.MODEL_INVENTORY)
    m.require(len(model["files"]) == 28, "Model inventory count differs")
    if full_model:
        old.verify_model(old.read_json(ROOT / old.PROVENANCE), full_inventory=model)
    for identity in a["old_processes"]:
        current = old.process_record(identity["pid"])
        m.require(current is None or current["start_ticks"] != identity["start_ticks"] or current["state"] == "Z",
                  "Old owned process still running")
        for entry in Path("/proc").iterdir():
            if entry.name.isdigit():
                member = old.process_record(int(entry.name))
                m.require(not member or member["session"] != identity["session"] or member["state"] == "Z",
                          "Old owned session still running")
    if extra:
        old.verify_hashes(extra)
    m.require(time.time() < status["global_deadline_unix"] - 30, "Original global deadline exhausted in verification")


def provenance(contract, stage):
    a, _, parent, status, previous_stage, cost, oracle_deadline = contract
    return {"stage": stage, "old_parent": a["old_parent"], "inventory": a["inventory"],
            "old_cost_json": parent["output_paths"]["oracle"] + "/cost_progress.json",
            "historical_active_call": cost["active_call"], "historical_gap": gaps(cost["totals"]),
            "old_oracle_stage_counts": cost["stage_counts"], "old_oracle_calls_by_label": cost["calls_by_label"],
            "old_oracle_controller_elapsed_seconds": previous_stage["elapsed_seconds"],
            "old_oracle_ledger_elapsed_seconds": cost["elapsed_seconds"],
            "original_global_deadline_unix": status["global_deadline_unix"],
            "original_oracle_deadline_unix": oracle_deadline,
            "wall_clock_policy": "Original absolute deadlines retained; user pause and recovery preparation count toward wall budget.",
            "retry_authorized": True, "retry_reason": a["retry_reason"], "automatic_retry_authorized": False}


def run_worker(adoption_path, request_path, stage):
    contract = load_contract(adoption_path)
    a, _, parent, status, _, historical, oracle_deadline = contract
    adoption_hash, request_hash = old.sha256(adoption_path), old.sha256(request_path)
    request = old.read_json(request_path)
    m.require(request_path.resolve() == (ROOT / a["output_paths"]["queue"] / (stage + "_request.json")).resolve()
              and request["adoption_sha256"] == adoption_hash and request["stage"] == stage,
              "Worker request differs")
    deadline = request["deadline_unix"]
    m.require(deadline == oracle_deadline if stage == "oracle" else
              deadline <= min(status["global_deadline_unix"] - 30, time.time() + worker.STAGE_WALL["replay"]),
              "Worker deadline enlarged")
    m.require(not os.environ.get("CUDA_VISIBLE_DEVICES") and all(os.environ.get(k) == "4" for k in
              ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")), "CPU cap/CUDA differs")
    verify_inputs(contract, adoption_path, adoption_hash, request["input_sha256"], full_model=False)
    m.require(time.time() < deadline - 20, "Stage deadline exhausted")
    output = ROOT / a["output_paths"][stage]
    output.mkdir(parents=True, exist_ok=False)
    initial = historical["totals"]
    limits = remaining_limits(historical) if stage == "oracle" else worker.STAGE_LIMITS["replay"]
    if stage == "replay":
        prior = ROOT / a["output_paths"]["oracle"]
        cost = old.read_json(prior / "cost_progress.json")
        summary = old.read_json(prior / "summary.json")
        m.require(summary["complete"] is True and summary["record_count"] == summary["planned_records"] == 180
                  and summary["records_sha256"] == old.sha256(prior / "records.jsonl") and summary["costs"]["totals"] == cost["totals"],
                  "Incomplete resumed oracle")
        validate_new_cost(cost, historical["totals"], gaps(historical["totals"]))
        initial = cost["totals"]
    ledger = ResumeLedger(provenance=provenance(contract, stage), stage_limits=limits, initial_totals=initial,
                          deadline=time.monotonic() + deadline - time.time() - 20,
                          persist=lambda value: old.atomic_json(output / "cost_progress.json", value))
    ledger.flush()
    old.atomic_json(output / "config.json", {"request": request, "request_sha256": request_hash,
                    "adoption_sha256": adoption_hash, "worker_executable": sys.executable,
                    "protocol": "alfworld-causal-v1", "environment_seed": m.SEED, "resume_provenance": ledger.provenance})
    games = old.read_json(ROOT / parent["candidate"])["training_cohort"]["frozen_gamefiles"]
    collect = ROOT / parent["output_paths"]["collect"] / "records.jsonl"
    count = 0
    summary = {"kind": "recovery_stage_summary", "schema_version": 1, "stage": stage, "complete": False,
               "planned_records": 180, "record_count": 0, "request_sha256": request_hash,
               "retry_authorized": True, "retry_reason": a["retry_reason"], "automatic_retry_authorized": False,
               "historical_gap": gaps(historical["totals"]), "resume_provenance": ledger.provenance}
    with (output / "records.jsonl").open("xb") as records, (output / "events.jsonl").open("xb") as events:
        def write(stream, row):
            stream.write((json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
            stream.flush()
        def factory(game, expert):
            from gnrl.alfworld_env import AlfworldReplayEnv
            return AlfworldReplayEnv(game, seed=m.SEED, max_steps=m.MAX_STEPS, expert_type=expert)
        try:
            if stage == "oracle":
                previous = ROOT / parent["output_paths"]["oracle"]
                summary["prefix_copy"] = copy_prefix(previous / "records.jsonl", previous / "events.jsonl",
                    collection_anchors(collect, games), records, events, a["expected_completed_records"])
                count = summary["prefix_copy"]["copied_records"]
                items = collection_anchors(collect, games)
                for _ in range(count):
                    next(items)
            else:
                def pools():
                    anchors = iter(collection_anchors(collect, games))
                    tail, n = {}, 0
                    for _, pool in lines(ROOT / a["output_paths"]["oracle"] / "records.jsonl", tail):
                        m.validate_pool(pool)
                        m.require(pool["anchor"] == next(anchors, None), "Replay pool/collection order differs")
                        n += 1
                        yield pool
                    m.require(not tail and n == 180 and next(anchors, None) is None, "Incomplete oracle input")
                items = pools()
            operation = m.certify_anchor if stage == "oracle" else m.replay_certificates
            for item in items:
                ledger.check()
                row = operation(factory, item, ledger, lambda value: write(events, value))
                write(records, row)
                count += 1
                print(json.dumps({"stage": stage, "completed_records": count}), flush=True)
            m.require(count == 180, "Incomplete stage count")
            validate_new_cost(ledger.snapshot(), initial, gaps(historical["totals"]))
            m.require(old.sha256(request_path) == request_hash, "Stage request changed")
            verify_inputs(contract, adoption_path, adoption_hash, request["input_sha256"], full_model=False)
            ledger.check()
            summary.update(complete=True, technical_complete=True, records_sha256=old.sha256(output / "records.jsonl"))
        except BaseException as error:
            summary.update(error=f"{type(error).__name__}: {error}", technical_incomplete=True)
            raise
        finally:
            summary.update(record_count=count, costs=ledger.snapshot())
            ledger.flush()
            old.atomic_json(output / "summary.json", summary)


class Controller:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.hash = old.sha256(path)
        self.contract = load_contract(path)
        self.a, _, _, status, _, historical, _ = self.contract
        self.deadline = status["global_deadline_unix"]
        for name in self.a["output_paths"].values():
            m.require(not (ROOT / name).exists(), "Resume output already exists; no automatic retry")
        self.folder = ROOT / self.a["output_paths"]["queue"]
        self.folder.mkdir(parents=True, exist_ok=False)
        self.locks = {}
        self.state = {"complete": False, "started_utc": old.utc(), "adoption_sha256": self.hash,
                      "global_deadline_unix": self.deadline, "original_oracle_deadline_unix": self.contract[-1],
                      "stages": [], "retry_authorized": True, "retry_reason": self.a["retry_reason"],
                      "automatic_retry_authorized": False, "historical_gap": gaps(historical["totals"])}
        self.update("preflight")

    def update(self, phase, **extra):
        self.state.update(current_stage=phase, updated_utc=old.utc(), **extra)
        old.atomic_json(self.folder / "status.json", self.state)
        print(json.dumps({"stage": phase, "complete": self.state["complete"]}), flush=True)

    def verify(self):
        verify_inputs(self.contract, self.path, self.hash, self.locks)

    def run(self, stage):
        self.verify()
        deadline = self.contract[-1] if stage == "oracle" else min(time.time() + worker.STAGE_WALL["replay"], self.deadline - 30)
        m.require(time.time() < deadline - 30, "Original stage deadline exhausted")
        request = self.folder / (stage + "_request.json")
        with request.open("x", encoding="utf-8") as stream:
            json.dump({"stage": stage, "adoption_sha256": self.hash, "deadline_unix": deadline,
                       "input_sha256": dict(self.locks)}, stream, indent=2)
        self.locks[request.relative_to(ROOT).as_posix()] = old.sha256(request)
        command = [str(ROOT / ".venv/bin/python"), SCRIPT, "--adoption-json", str(self.path),
                   "--worker", stage, "--request-json", str(request)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1",
                   HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4",
                   MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", NUMEXPR_NUM_THREADS="4")
        env.pop("GNRL_GPU_GUARD", None)
        record = {"name": stage, "command": command, "started_utc": old.utc(), "deadline_unix": deadline}
        self.state["stages"].append(record)
        self.update(stage)
        began, process, ticks, code = time.monotonic(), None, None, None
        with (self.folder / (stage + ".stdout.log")).open("xb") as out, (self.folder / (stage + ".stderr.log")).open("xb") as err:
            try:
                mask = signal.pthread_sigmask(signal.SIG_BLOCK, SIGNALS)
                try:
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                               stdout=out, stderr=err, start_new_session=True)
                    identity = old.process_record(process.pid)
                    m.require(identity and identity["session"] == process.pid, "Cannot establish owned session")
                    ticks = identity["start_ticks"]
                    record.update(pid=process.pid, start_ticks=ticks, owned_session=process.pid)
                finally:
                    signal.pthread_sigmask(signal.SIG_SETMASK, mask)
                self.update(stage)
                code = process.wait(timeout=max(.1, deadline - time.time() - 20))
            finally:
                if process is not None:
                    mask = signal.pthread_sigmask(signal.SIG_BLOCK, SIGNALS)
                    try:
                        old.cleanup_owned(process, ticks)
                    finally:
                        signal.pthread_sigmask(signal.SIG_SETMASK, mask)
                record.update(returncode=process.returncode if process else None,
                              elapsed_seconds=time.monotonic() - began, owned_cleanup_complete=True)
                self.update(stage)
        m.require(code == 0, "Incomplete resumed stage: " + stage)
        self.verify()
        m.require(time.time() <= deadline, "Original stage deadline exhausted during final verification")
        output = ROOT / self.a["output_paths"][stage]
        summary, cost = old.read_json(output / "summary.json"), old.read_json(output / "cost_progress.json")
        initial = self.contract[5]["totals"] if stage == "oracle" else old.read_json(ROOT / self.a["output_paths"]["oracle"] / "cost_progress.json")["totals"]
        m.require(summary["complete"] is True and summary["technical_complete"] is True and
                  summary["stage"] == stage and summary["record_count"] == summary["planned_records"] == 180 and
                  summary["request_sha256"] == old.sha256(request) and
                  summary["records_sha256"] == old.sha256(output / "records.jsonl") and summary["costs"]["totals"] == cost["totals"],
                  "Resumed stage completion contract failed")
        validate_new_cost(cost, initial, self.state["historical_gap"])
        for path in sorted(p for p in output.rglob("*") if p.is_file()):
            self.locks[path.relative_to(ROOT).as_posix()] = old.sha256(path)
        old.atomic_json(self.folder / "locked_sha256.json", self.locks)
        m.require(time.time() <= deadline, "Original stage deadline exhausted during output verification")
        self.update("completed_" + stage)

    def main(self):
        for stage in ("oracle", "replay"):
            self.run(stage)
        self.verify()
        self.update("needs_astra_resumed_acquisition_review", complete=True, technical_complete=True,
                    note="Complete180 records per stage and balanced new calls only; historical gap preserved. No scientific interpretation or further stages authorized.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adoption-json", type=Path, required=True)
    parser.add_argument("--worker", choices=("oracle", "replay"))
    parser.add_argument("--request-json", type=Path)
    args = parser.parse_args()
    m.require(sys.platform.startswith("linux") and ROOT.resolve() == Path("/home/jsy/GNRL_20261012").resolve(), "Wrong chemistry workspace")
    m.require(args.worker or (hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal")), "Controller requires system Python pidfd support")
    def stop(signum, frame):
        raise InterruptedError("Owned resumed acquisition interrupted")
    for sig in SIGNALS:
        signal.signal(sig, stop)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, SIGNALS)
    controller = None
    try:
        contract = load_contract(args.adoption_json)
        end = contract[3]["global_deadline_unix"]
        if args.worker:
            m.require(args.request_json is not None, "Worker request required")
            end = min(end, old.read_json(args.request_json)["deadline_unix"] - 20)
        m.require(end > time.time(), "Original deadline exhausted")
        signal.setitimer(signal.ITIMER_REAL, end - time.time())
        if args.worker:
            run_worker(args.adoption_json, args.request_json, args.worker)
        else:
            controller = Controller(args.adoption_json)
            controller.main()
    except BaseException as error:
        if controller is not None:
            controller.update("technical_incomplete", complete=False, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    main()

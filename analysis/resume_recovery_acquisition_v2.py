"""Single explicitly adopted server-reboot interruption recovery; CPU oracle/replay only.

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

from analysis import resume_recovery_acquisition_v1 as inherited

SCRIPT = "analysis/resume_recovery_acquisition_v2.py"
TEST = "analysis/test_resume_recovery_acquisition_v2.py"
INVENTORY = "outputs/recovery_reboot_resume_inventory_v2.json"
INVENTORY_SHA = "6f4ba069277bf6c77452cc9884960b1d4e5151d4d7eaa1fff9e9e24c8bec853f"
DESIGN = "analysis/recovery_reboot_resume_design_v2.json"
DESIGN_SHA = "29dc452803933b4c235e71d24182a7e8e1e672ad5cbb6b3c323a52d1bd1905e6"
CALLS = ("environment_steps", "resets", "policy_calls")
SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM) if sys.platform.startswith("linux") else ()


lines = inherited.lines
collection_anchors = inherited.collection_anchors
pool_events = inherited.pool_events
copy_prefix = inherited.copy_prefix


def gaps(totals):
    return {kind: totals[kind + "_started"] - totals[kind + "_completed"] for kind in CALLS}


def combined_counts(original, previous):
    return {k: original["stage_counts"][k] + previous["stage_counts"][k] for k in m.COUNTERS}


def remaining_limits(previous):
    counts = previous["combined_oracle_stage_counts"]
    result = {k: limit - counts[k] for k, limit in worker.STAGE_LIMITS["oracle"].items()}
    m.require(all(type(v) is int and v >= 0 for v in result.values()), "Historical oracle budget exceeded")
    return result


def validate_historical_cost(original, previous, collection):
    inherited.validate_historical_cost(original, collection)
    m.require(previous["kind"] == "recovery_cost_ledger" and previous["global_limits"] == m.GLOBAL_LIMITS,
              "Wrong resumed historical ledger")
    m.require(set(previous["totals"]) == set(previous["stage_counts"]) == set(m.COUNTERS), "Missing historical counters")
    m.require(all(type(v) is int and v >= 0 for d in (previous["totals"], previous["stage_counts"]) for v in d.values()),
              "Invalid historical counters")
    m.require(previous["totals"] == {k: original["totals"][k] + previous["stage_counts"][k] for k in m.COUNTERS},
              "Historical cost lineage differs")
    m.require(previous["stage_limits"] == inherited.remaining_limits(original), "First resume limits changed")
    m.require(previous["combined_oracle_stage_counts"] == combined_counts(original, previous), "Combined historical counts differ")
    m.require(previous["resume_provenance"]["old_oracle_stage_counts"] == original["stage_counts"] and
              previous["resume_provenance"]["old_oracle_calls_by_label"] == original["calls_by_label"], "Historical labels/counts lost")
    m.require(previous["unknown_generation_after_error"] is False and
              previous["active_call"] == {"operation": "step", "label": "oracle_prefix"} and
              gaps(previous["stage_counts"]) == {"environment_steps": 1, "resets": 0, "policy_calls": 0} and
              gaps(previous["totals"]) == {"environment_steps": 2, "resets": 0, "policy_calls": 0},
              "Reboot interruption differs from reviewed recovery")
    m.require(previous["stage_counts"]["environment_steps_started"] == 1003 and
              previous["stage_counts"]["environment_steps_completed"] == 1002 and
              previous["stage_counts"]["resets_started"] == previous["stage_counts"]["resets_completed"] == 55,
              "Frozen interrupted resume costs differ")
    remaining_limits(previous)


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
        return result


def new_oracle_deadline(adoption, global_deadline):
    start = adoption["new_window_started_unix"]
    m.require(type(start) in (int, float) and start > 0 and
              adoption["oracle_wall_seconds"] == 43200, "New oracle window differs")
    deadline = min(start + 43200, global_deadline - 30)
    m.require(adoption["new_oracle_deadline_unix"] == deadline, "New oracle deadline is not parent-bound")
    return deadline


def current_boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def current_boot_time():
    entries = [line.split()[1] for line in Path("/proc/stat").read_text().splitlines() if line.startswith("btime ")]
    m.require(len(entries) == 1, "Current boot time unavailable")
    return int(entries[0])


def validate_reboot(inv, previous_status, job, receipt, boot_id, boot_time):
    """Wall-time proof binds old launch to an earlier boot; never inspect reused IDs."""
    boot = inv["boot"]
    m.require(boot_id == boot["boot_id"] and boot_time == boot["boot_time_unix"],
              "Another reboot occurred; new adoption required")
    launched = datetime.fromisoformat(job["started_utc"]).timestamp()
    verified = datetime.fromisoformat(receipt["verified_utc"]).timestamp()
    m.require(launched < verified < boot["boot_time_unix"] < boot["observed_unix"],
              "Old launch is not proven to predate this boot")
    m.require(receipt["startup_verification_complete"] is True and
              receipt["controller"]["pid"] == job["pid"] and
              receipt["controller"]["start_ticks"] == int(job["start_ticks"]), "Old launch identity differs")
    stage = previous_status["stages"][0]
    m.require(receipt["worker"]["pid"] == stage["pid"] and
              receipt["worker"]["start_ticks"] == stage["start_ticks"] and
              receipt["worker"]["session"] == stage["owned_session"] and
              datetime.fromisoformat(stage["started_utc"]).timestamp() < verified,
              "Old worker identity differs")


def load_contract(path):
    a = old.read_json(path)
    required = {"kind": "recovery_acquisition_reboot_resume_parent_adoption", "version": 2,
                "adopted": True, "adopted_by": "root_after_independent_GPT_6_Astra_review",
                "retry_authorized": True, "retry_reason": "user-requested server-reboot interruption recovery",
                "expected_completed_records": 137, "inventory": INVENTORY,
                "original_deadlines_preserved": False, "original_global_deadline_preserved": True,
                "oracle_wall_budget_extended": True}
    for key, value in required.items():
        m.require(type(a.get(key)) is type(value) and a[key] == value, "Reboot adoption gate: " + key)
    m.require(a["authorized_stages"] == ["oracle", "replay"], "Reboot stages differ")
    hashes = a["file_sha256"]
    m.require({SCRIPT, TEST, DESIGN, a["old_parent"], a["resume_parent"], INVENTORY, a["resume_review"]} <= hashes.keys(),
              "Missing reboot adoption bindings")
    m.require(hashes[INVENTORY] == INVENTORY_SHA, "Wrong reboot inventory")
    m.require(a["design"] == DESIGN and hashes[DESIGN] == DESIGN_SHA, "Wrong reboot design")
    old.verify_hashes(hashes)
    review = old.read_json(ROOT / a["resume_review"])
    m.require(review.get("kind") == "recovery_acquisition_reboot_resume_astra_review" and
              review.get("resume_review_passed") is True and review.get("resume_script_sha256") == hashes[SCRIPT] and
              review.get("test_sha256") == hashes[TEST] and review.get("design_sha256") == DESIGN_SHA,
              "Missing exact independent reboot Astra review")
    # Explicitly reuse only the frozen v1 contract/integrity validation. Its
    # verify_inputs PID/session scanner is intentionally not called across boots.
    first = inherited.load_contract(ROOT / a["resume_parent"])
    first_adoption, first_inv, parent, status, old_stage, original_cost, original_deadline = first
    inv = old.read_json(ROOT / INVENTORY)
    docs = inv["documents"]
    m.require(inv["kind"] == "recovery_reboot_resume_inventory" and inv["version"] == 2 and
              inv["complete_record_lines"] == 137 and inv["record_tail_bytes"] == 0 and
              docs[first_adoption["inventory"]] == first_inv and docs[a["resume_parent"]] == first_adoption and
              docs[a["old_parent"]] == parent and first_adoption["old_parent"] == a["old_parent"],
              "Frozen original/resume ancestry differs")
    paths = first_adoption["output_paths"]
    previous = docs[paths["queue"] + "/status.json"]
    m.require(previous["complete"] is False and previous["current_stage"] == "oracle" and
              len(previous["stages"]) == 1 and previous["stages"][0]["name"] == "oracle" and
              "returncode" not in previous["stages"][0] and
              previous["adoption_sha256"] == hashes[a["resume_parent"]] and
              previous["global_deadline_unix"] == status["global_deadline_unix"] == a["global_deadline_unix"] == 1792005218.8364305 and
              previous["original_oracle_deadline_unix"] == original_deadline,
              "Frozen reboot controller state differs")
    m.require(a["expected_boot_id"] == inv["boot"]["boot_id"], "Adopted boot differs")
    job = docs["outputs/recovery_projection_resume_job_v1/job.json"]
    receipt = docs["outputs/recovery_projection_resume_job_v1/launch_verification.json"]
    validate_reboot(inv, previous, job, receipt, inv["boot"]["boot_id"], inv["boot"]["boot_time_unix"])
    cost = docs[paths["oracle"] + "/cost_progress.json"]
    collection_cost = docs[parent["output_paths"]["collect"] + "/cost_progress.json"]
    validate_historical_cost(original_cost, cost, collection_cost)
    expected_outputs = {"queue": "outputs/recovery_projection_reboot_resume_queue_v2",
                        "oracle": "outputs/recovery_projection_oracle_reboot_resume_v2",
                        "replay": "outputs/recovery_projection_replay_reboot_resume_v2"}
    m.require(a["output_paths"] == expected_outputs, "Reboot output namespace differs")
    protected = [old.safe_relative(p) for p in (*parent["output_paths"].values(), *paths.values())]
    for name in expected_outputs.values():
        output = old.safe_relative(name)
        m.require(not any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in protected),
                  "Reboot outputs overlap historical files")
    deadline = new_oracle_deadline(a, status["global_deadline_unix"])
    m.require(inv["boot"]["observed_unix"] <= a["new_window_started_unix"] and
              original_deadline < a["new_window_started_unix"], "Recovery anchor predates frozen inventory/expiry")
    return a, inv, parent, status, old_stage, cost, deadline


def verify_inputs(contract, adoption_path, adoption_hash, extra=None, full_model=True):
    """Integrity adapter retains v1 hash checks, replaces only its old-boot PID scan."""
    a, inv, parent, status, _, _, _ = contract
    m.require(time.time() < status["global_deadline_unix"] - 30, "Original global deadline exhausted")
    m.require(old.sha256(adoption_path) == adoption_hash, "Reboot adoption changed")
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
    first = inv["documents"][a["resume_parent"]]
    validate_reboot(inv, inv["documents"][first["output_paths"]["queue"] + "/status.json"],
                    inv["documents"]["outputs/recovery_projection_resume_job_v1/job.json"],
                    inv["documents"]["outputs/recovery_projection_resume_job_v1/launch_verification.json"],
                    current_boot_id(), current_boot_time())
    if extra:
        old.verify_hashes(extra)
    m.require(time.time() < status["global_deadline_unix"] - 30, "Original global deadline exhausted in verification")


def provenance(contract, stage):
    a, inv, parent, status, previous_stage, cost, oracle_deadline = contract
    original_cost = inv["documents"][parent["output_paths"]["oracle"] + "/cost_progress.json"]
    return {"stage": stage, "old_parent": a["old_parent"], "resume_parent": a["resume_parent"], "inventory": INVENTORY,
            "historical_gap": gaps(cost["totals"]), "historical_ledgers": [original_cost, cost],
            "historical_active_calls": [original_cost["active_call"], cost["active_call"]],
            "old_oracle_stage_counts": cost["combined_oracle_stage_counts"],
            "historical_calls_by_label": [original_cost["calls_by_label"], cost["calls_by_label"]],
            "historical_elapsed_observations": {
                "original_controller_elapsed_seconds": previous_stage["elapsed_seconds"],
                "original_ledger_elapsed_seconds": original_cost["elapsed_seconds"],
                "resume1_ledger_elapsed_lower_observation_seconds": cost["elapsed_seconds"],
                "resume1_wall_occupancy_upper_bound_seconds": inv["boot"]["boot_time_unix"] - datetime.fromisoformat(
                    inv["documents"][inv["documents"][a["resume_parent"]]["output_paths"]["queue"] + "/status.json"]["stages"][0]["started_utc"]).timestamp()},
            "historical_elapsed_note": "Separate observed times and start-to-boot upper bound; no exact resume1 final duration or combined compute time inferred.",
            "original_global_deadline_unix": status["global_deadline_unix"],
            "original_oracle_deadline_unix": inherited.original_oracle_deadline(status, previous_stage),
            "new_window_started_unix": a["new_window_started_unix"], "new_oracle_deadline_unix": oracle_deadline,
            "oracle_wall_seconds": 43200, "original_deadlines_preserved": False,
            "original_global_deadline_preserved": True, "oracle_wall_budget_extended": True,
            "boot_id": a["expected_boot_id"], "retry_authorized": True, "retry_reason": a["retry_reason"],
            "automatic_retry_authorized": False}


def preflight(adoption_path, full_model=True, validate_prefix=False):
    """Read-only deployment gate, producing structural metadata only."""
    adoption_path = Path(adoption_path)
    contract = load_contract(adoption_path)
    verify_inputs(contract, adoption_path, old.sha256(adoption_path), full_model=full_model)
    a, inv, parent, _, _, cost, deadline = contract
    m.require(time.time() < deadline - 30, "New oracle deadline exhausted")
    result = {"preflight_complete": True, "boot_id": a["expected_boot_id"],
              "expected_completed_records": 137, "historical_gap": gaps(cost["totals"]),
              "remaining_oracle_limits": remaining_limits(cost), "new_oracle_deadline_unix": deadline}
    if validate_prefix:
        result["inherited_prefix_validation"] = validate_inherited_prefix(contract)
        class Sink:
            def write(self, raw):
                return len(raw)
            def flush(self):
                pass
        first = inv["documents"][a["resume_parent"]]
        source = ROOT / first["output_paths"]["oracle"]
        collect = ROOT / parent["output_paths"]["collect"] / "records.jsonl"
        games = old.read_json(ROOT / parent["candidate"])["training_cohort"]["frozen_gamefiles"]
        result["prefix_validation"] = copy_prefix(source / "records.jsonl", source / "events.jsonl",
            collection_anchors(collect, games), Sink(), Sink(), 137)
    m.require(time.time() < deadline - 30, "New oracle deadline exhausted in preflight")
    return result


def compare_byte_prefix(path, expected_bytes, expected_sha256):
    digest = hashlib.sha256()
    remaining = expected_bytes
    with Path(path).open("rb") as stream:
        while remaining:
            chunk = stream.read(min(1024 * 1024, remaining))
            m.require(bool(chunk), "Inherited byte prefix truncated")
            digest.update(chunk)
            remaining -= len(chunk)
    m.require(digest.hexdigest() == expected_sha256, "Inherited byte prefix changed")


def validate_inherited_prefix(contract):
    """Prove original 131 committed records/events are raw prefixes of resume1."""
    a, inv, parent, _, _, _, _ = contract
    class HashSink:
        def __init__(self):
            self.bytes, self.digest = 0, hashlib.sha256()
        def write(self, raw):
            self.bytes += len(raw)
            self.digest.update(raw)
            return len(raw)
        def flush(self):
            pass
    old_source = ROOT / parent["output_paths"]["oracle"]
    new_source = ROOT / inv["documents"][a["resume_parent"]]["output_paths"]["oracle"]
    games = old.read_json(ROOT / parent["candidate"])["training_cohort"]["frozen_gamefiles"]
    collect = ROOT / parent["output_paths"]["collect"] / "records.jsonl"
    records, events = HashSink(), HashSink()
    metadata = copy_prefix(old_source / "records.jsonl", old_source / "events.jsonl",
        collection_anchors(collect, games), records, events, 131)
    for name, sink in (("records.jsonl", records), ("events.jsonl", events)):
        compare_byte_prefix(new_source / name, sink.bytes, sink.digest.hexdigest())
    return {"original_prefix_records": metadata["copied_records"], "original_prefix_bytes": records.bytes,
            "original_prefix_event_bytes": events.bytes, "byte_preservation_verified": True,
            "original_interruption_metadata": metadata}


def cleanup_current_boot(process, ticks, boot_id):
    m.require(current_boot_id() == boot_id, "Boot changed; refuse all old-boot cleanup")
    old.cleanup_owned(process, ticks)


def run_worker(adoption_path, request_path, stage):
    contract = load_contract(adoption_path)
    a, inv, parent, status, _, historical, oracle_deadline = contract
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
                previous = ROOT / inv["documents"][a["resume_parent"]]["output_paths"]["oracle"]
                summary["inherited_prefix_validation"] = validate_inherited_prefix(contract)
                print(json.dumps({"stage": stage, "event": "inherited_prefix_verified", "count": 131}), flush=True)
                summary["prefix_copy"] = copy_prefix(previous / "records.jsonl", previous / "events.jsonl",
                    collection_anchors(collect, games), records, events, a["expected_completed_records"])
                count = summary["prefix_copy"]["copied_records"]
                print(json.dumps({"stage": stage, "event": "prefix_copy_complete", "count": count}), flush=True)
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
            records_hash = old.sha256(output / "records.jsonl")
            ledger.check()
            summary.update(complete=True, technical_complete=True, records_sha256=records_hash)
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
                      "global_deadline_unix": self.deadline, "new_oracle_deadline_unix": self.contract[-1], "boot_id": self.a["expected_boot_id"],
                      "original_deadlines_preserved": False, "original_global_deadline_preserved": True,
                      "oracle_wall_budget_extended": True,
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
        m.require(time.time() < deadline - 30, "Adopted stage deadline exhausted")
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
        record = {"name": stage, "command": command, "started_utc": old.utc(), "deadline_unix": deadline, "boot_id": self.a["expected_boot_id"]}
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
                        cleanup_current_boot(process, ticks, self.a["expected_boot_id"])
                    finally:
                        signal.pthread_sigmask(signal.SIG_SETMASK, mask)
                record.update(returncode=process.returncode if process else None,
                              elapsed_seconds=time.monotonic() - began, owned_cleanup_complete=True)
                self.update(stage)
        m.require(code == 0, "Incomplete resumed stage: " + stage)
        self.verify()
        m.require(time.time() <= deadline, "Adopted stage deadline exhausted during final verification")
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
        m.require(time.time() <= deadline, "Adopted stage deadline exhausted during output verification")
        self.update("completed_" + stage)

    def main(self):
        for stage in ("oracle", "replay"):
            self.run(stage)
        self.verify()
        self.update("needs_astra_reboot_resumed_acquisition_review", complete=True, technical_complete=True,
                    note="Complete180 records per stage and balanced new calls only; historical gap preserved. No scientific interpretation or further stages authorized.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adoption-json", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--validate-prefix", action="store_true")
    parser.add_argument("--worker", choices=("oracle", "replay"))
    parser.add_argument("--request-json", type=Path)
    args = parser.parse_args()
    m.require(sys.platform.startswith("linux") and ROOT.resolve() == Path("/home/jsy/GNRL_20261012").resolve(), "Wrong chemistry workspace")
    if args.preflight:
        m.require(not args.worker and args.request_json is None, "Preflight cannot execute stages")
        print(json.dumps(preflight(args.adoption_json, validate_prefix=args.validate_prefix)))
        return
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

"""Parent-adopted collection/oracle/replay phase; never scores, trains or evaluates."""
from __future__ import annotations

import argparse
import csv
import io
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

STAGES = ("collect", "oracle", "replay")


def validate_parent(path):
    a = old.read_json(path)
    for k, expected in {"kind": "recovery_acquisition_parent_adoption", "version": 1,
                        "adopted": True, "physical_gpu": 2}.items():
        old.require(type(a.get(k)) is type(expected) and a[k] == expected, "Parent adoption: " + k)
    old.require(a["authorized_stages"] == list(STAGES), "Only three acquisition stages authorized")
    old.require(a["source_inventory"] == worker.source_inventory(), "Source inventory changed")
    old.require(set(a["source_inventory"]) <= a["file_sha256"].keys(), "Incomplete source locks")
    old.require({a["candidate"], a["implementation_review"], a["smoke_result_review"], a["input_inventory"],
                 old.PROVENANCE, old.MODEL_INVENTORY} <= a["file_sha256"].keys(), "Missing parent input/review bindings")
    old.verify_hashes(a["file_sha256"])
    review = old.read_json(ROOT / a["implementation_review"])
    old.require(review.get("recovery_collection_review_passed") is True
                and review.get("recovery_acquisition_controller_review_passed") is True,
                "Missing independent acquisition review")
    old.require(review["controller_sha256"] == old.sha256(__file__), "Review controller differs")
    smoke = old.read_json(ROOT / a["smoke_result_review"])
    old.require(smoke.get("kind") == "independent_stored_artifact_recovery_smoke_review_v1"
                and smoke.get("gate_passed") is True and all(smoke["gate"].values()), "CPU recovery smoke gate failed")
    inventory = old.read_json(ROOT / a["input_inventory"])
    old.require(inventory.get("complete") is True and inventory["original_file_count"] == 336,
                "Incomplete original input inventory")
    old.require(a["original_file_sha256"] == inventory["original_file_sha256"], "Original locks differ")
    for p, h in inventory["initial_adapter_sha256"].items():
        old.require(a["file_sha256"].get(p) == h, "Starting adapter not frozen")
    candidate = old.read_json(ROOT / a["candidate"])
    old.require(candidate["version"] == "v2" and candidate["finite_budgets"]["global_wall_seconds_including_waits_verification_cleanup"] == 171840,
                "Wrong fixed study candidate/deadline")
    paths = a["output_paths"]
    old.require(set(paths) == {"queue", *STAGES} and len(set(paths.values())) == 4, "Bad output paths")
    resolved = []
    for p in paths.values():
        q = old.safe_relative(p)
        old.require(p.startswith("outputs/recovery_projection_") and not q.exists(), "Output exists/namespace mismatch")
        resolved.append(q)
    old.require(not any(x != y and x.is_relative_to(y) for x in resolved for y in resolved), "Overlapping outputs")
    return a


class Acquisition:
    def __init__(self, path):
        self.started = time.monotonic()
        self.started_unix = time.time()
        self.deadline = self.started + 171840
        self.deadline_unix = self.started_unix + 171840
        self.path = Path(path).resolve()
        self.adoption_hash = old.sha256(path)
        self.a = validate_parent(path)
        self.locks = dict(self.a["file_sha256"])
        self.folder = ROOT / self.a["output_paths"]["queue"]
        self.folder.mkdir(parents=True, exist_ok=False)
        self.state = {"complete": False, "phase": "acquisition_only", "main_study_started_utc": old.utc(),
                      "main_study_started_unix": self.started_unix, "global_deadline_unix": self.deadline_unix,
                      "adoption_sha256": self.adoption_hash, "physical_gpu": 2, "stages": []}
        self.env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1",
                        HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                        OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
        self.env.pop("GNRL_GPU_GUARD", None)
        self.python = str(ROOT / ".venv/bin/python")
        self.update("preflight")

    def update(self, stage, **extra):
        self.state.update(current_stage=stage, updated_utc=old.utc(), **extra)
        old.atomic_json(self.folder / "status.json", self.state)
        print(json.dumps({"stage": stage, **extra}), flush=True)

    def verify(self):
        old.require(time.monotonic() < self.deadline - 30, "Global budget exhausted")
        old.require(old.sha256(self.path) == self.adoption_hash, "Parent adoption changed")
        old.require(worker.source_inventory() == self.a["source_inventory"], "Acquisition source inventory changed")
        old.verify_hashes(self.locks)
        old.verify_model(old.read_json(ROOT / old.PROVENANCE), full_inventory=old.read_json(ROOT / old.MODEL_INVENTORY))
        inventory = old.read_json(ROOT / self.a["input_inventory"])
        adapter = ROOT / worker.ADAPTER
        actual_adapter_files = {p.relative_to(ROOT).as_posix() for p in adapter.rglob("*") if p.is_file()}
        old.require(actual_adapter_files == set(inventory["initial_adapter_sha256"]), "Starting adapter directory inventory changed")
        for p, h in self.a["original_file_sha256"].items():
            old.require(old.sha256(p) == h, "Original game/metadata changed: " + p)
        for p, h in self.a["runtime_file_sha256"].items():
            old.require(old.sha256(p) == h, "Runtime file changed: " + p)
        old.require(time.monotonic() < self.deadline - 30, "Global budget exhausted during verification")

    def lock(self, paths):
        for p in paths:
            name = p.relative_to(ROOT).as_posix()
            h = old.sha256(p)
            old.require(name not in self.locks or self.locks[name] == h, "Artifact changed before lock")
            self.locks[name] = h
        old.atomic_json(self.folder / "locked_sha256.json", self.locks)

    def wait_gpu(self):
        from scripts.launch_gpu import resolve_devices
        self.update("waiting_for_gpu2")
        while time.monotonic() < self.deadline - 60:
            info = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used",
                                            "--format=csv,noheader,nounits"], text=True, timeout=30)
            active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid",
                                              "--format=csv,noheader,nounits"], text=True, timeout=30)
            try:
                resolve_devices([2], list(csv.reader(io.StringIO(info))), active_uuids=set(active.splitlines()))
                return
            except RuntimeError:
                time.sleep(min(20, max(0, self.deadline - 60 - time.monotonic())))
        raise TimeoutError("GPU2 unavailable within remaining study budget")

    def request(self, stage, previous):
        paths = self.a["output_paths"]
        prior = paths[previous] if previous else None
        request = {"schema_version": 1, "kind": "recovery_projection_stage_adoption", "adopted": True,
                   "adopted_by": "root_after_independent_GPT_6_Astra_review", "stage": stage,
                   "parent_adoption_sha256": self.adoption_hash,
                   "candidate": self.a["candidate"], "candidate_sha256": self.locks[self.a["candidate"]],
                   "physical_gpu": 2, "source_inventory": self.a["source_inventory"],
                   "file_sha256": dict(self.locks), "original_file_sha256": self.a["original_file_sha256"],
                   "implementation_review": self.a["implementation_review"], "output": paths[stage],
                   "previous_cost_json": prior + "/cost_progress.json" if prior else None,
                   "input_records": prior + "/records.jsonl" if prior else None,
                   "input_summary": prior + "/summary.json" if prior else None,
                   "global_deadline_unix": self.deadline_unix, "max_wall_seconds": worker.STAGE_WALL[stage]}
        path = self.folder / (stage + "_request.json")
        with path.open("x", encoding="utf-8") as f:
            json.dump(request, f, indent=2)
        self.lock([path])
        return path

    def run(self, stage, request):
        self.verify()
        gpu = stage == "collect"
        if gpu:
            self.wait_gpu()
        self.verify()
        command = [self.python, "scripts/collect_recovery_projection_v1.py", "--adoption-json", str(request)]
        if gpu:
            command = [self.python, "scripts/launch_gpu.py", "--gpus", "2", "--receipt",
                       str(self.folder / (stage + "_gpu.json")), "--", *command]
        began = time.monotonic()
        record = {"name": stage, "command": command, "started_utc": old.utc()}
        self.state["stages"].append(record)
        self.update(stage)
        process, ticks, code = None, None, None
        with (self.folder / (stage + ".stdout.log")).open("xb") as out, (self.folder / (stage + ".stderr.log")).open("xb") as err:
            try:
                spawn_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM})
                try:
                    process = subprocess.Popen(command, cwd=ROOT, env=self.env, stdin=subprocess.DEVNULL,
                                               stdout=out, stderr=err, start_new_session=True)
                    ident = old.process_record(process.pid)
                    old.require(ident and ident["session"] == process.pid, "Cannot establish owned session")
                    ticks = ident["start_ticks"]
                finally:
                    signal.pthread_sigmask(signal.SIG_SETMASK, spawn_mask)
                record.update(pid=process.pid, start_ticks=ticks, owned_session=process.pid)
                self.update(stage)
                code = process.wait(timeout=min(worker.STAGE_WALL[stage], max(.1, self.deadline - time.monotonic() - 30)))
            finally:
                if process is not None:
                    mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM})
                    try:
                        old.cleanup_owned(process, ticks)
                    finally:
                        signal.pthread_sigmask(signal.SIG_SETMASK, mask)
                record.update(returncode=process.returncode if process else None,
                              elapsed_seconds=time.monotonic() - began, owned_cleanup_complete=True)
                self.update(stage)
        old.require(code == 0, "Incomplete acquisition stage: " + stage)
        self.verify()
        output = ROOT / self.a["output_paths"][stage]
        summary = old.read_json(output / "summary.json")
        costs = old.read_json(output / "cost_progress.json")
        expected = 60 if stage == "collect" else 180
        old.require(summary.get("complete") is True and summary.get("stage") == stage and
                    summary.get("record_count") == expected and summary.get("planned_records") == expected,
                    "Stage completion contract failed")
        old.require(summary["request_sha256"] == old.sha256(request) and
                    summary["records_sha256"] == old.sha256(output / "records.jsonl"), "Stage provenance differs")
        old.require(costs["active_call"] is None and costs["unknown_generation_after_error"] is False,
                    "Incomplete cost ledger")
        total = costs["totals"]
        for kind in ("environment_steps", "resets", "policy_calls"):
            old.require(total[kind + "_started"] == total[kind + "_completed"], "Unfinished recorded call")
        self.lock(sorted(p for p in output.rglob("*") if p.is_file()))
        self.update("completed_" + stage, cumulative_costs=total)

    def main(self):
        previous = None
        for stage in STAGES:
            request = self.request(stage, previous)
            self.run(stage, request)
            previous = stage
        self.verify()
        self.update("needs_astra_acquisition_review", complete=True,
                    note="Acquisition complete only. No scoring,diagnostic,training,held-out or unseen jobs authorized.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adoption-json", required=True)
    a = p.parse_args()
    old.require(sys.platform.startswith("linux") and hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"),
                "Use system Python with pidfd support")
    old.require(ROOT.resolve() == Path("/home/jsy/GNRL_20261012").resolve(), "Wrong remote workspace")
    def stop(signum, frame):
        raise InterruptedError("Owned acquisition controller interrupted")
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM):
        signal.signal(s, stop)
    signal.setitimer(signal.ITIMER_REAL, 171840)
    q = None
    try:
        q = Acquisition(a.adoption_json)
        q.main()
    except BaseException as error:
        if q is not None:
            q.update("technical_incomplete", complete=False, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    main()

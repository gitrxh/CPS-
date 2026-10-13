"""System-Python CPU-only checks of the frozen queue's owned-session cleanup helpers.

Creates only benign short-lived Python processes. No model, simulator, CUDA,
queue main, or unrelated process is touched. Refuses an existing output receipt.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_successful_path_learning_v1 as q
from scripts.run_successful_path_learning_v2 import validate_controller

QUEUE_SHA256 = "99926d02303f56de560f1589109685ab3f863e058393d2eefb1c8f256c945f6c"
OUTPUT = ROOT / "outputs/successful_path_process_preflight_v2.json"


def child_code(mode):
    if mode == "normal_exit":
        return "import json,os; print(json.dumps({'ready':True,'pid':os.getpid(),'child':None}),flush=True)"
    sleeper = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('READY',flush=True); time.sleep(30)"
    return "\n".join([
        "import json,os,signal,subprocess,sys,time",
        "signal.signal(signal.SIGTERM,signal.SIG_IGN)",
        "child=subprocess.Popen([sys.executable,'-u','-c'," + repr(sleeper) + "],stdout=subprocess.PIPE,text=True)",
        "assert child.stdout.readline().strip()=='READY'",
        "print(json.dumps({'ready':True,'pid':os.getpid(),'child':child.pid}),flush=True)",
        "time.sleep(30)" if mode == "timeout_cleanup" else "sys.exit(0)"])


def run_case(mode, env):
    process = None
    ticks = None
    started = time.monotonic()
    record = {"case": mode, "started_utc": q.utc()}
    try:
        process = subprocess.Popen([sys.executable, "-u", "-c", child_code(mode)],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=env, start_new_session=True)
        identity = q.process_record(process.pid)
        q.require(identity is not None and identity["session"] == process.pid, "Cannot establish owned child identity")
        ticks = identity["start_ticks"]
        record.update(owned_leader=identity)
        q.require(select.select([process.stdout], [], [], 5)[0], "Fixture did not become ready")
        ready_line = process.stdout.readline()
        ready = json.loads(ready_line)
        q.require(ready["ready"] is True and ready["pid"] == process.pid, "Wrong child readiness identity")
        record["child_readiness"] = ready
        if ready["child"] is not None:
            child = q.process_record(ready["child"])
            q.require(child is not None and child["session"] == process.pid and child["start_ticks"] >= ticks,
                      "Fixture descendant escaped owned session")
            record["owned_descendant"] = child
        if mode == "timeout_cleanup":
            try:
                process.wait(timeout=0.2)
                raise RuntimeError("Timeout fixture unexpectedly exited")
            except subprocess.TimeoutExpired:
                record["timeout_observed"] = True
        else:
            record["leader_returncode_before_cleanup"] = process.wait(timeout=5)
            q.require(process.returncode == 0, "Normal/orphan fixture failed")
        before = q.owned_members(process.pid, ticks)
        record["live_members_before_cleanup"] = before
        if mode != "normal_exit":
            q.require(any(p["pid"] == ready["child"] for p in before), "Live descendant not established")
    finally:
        if process is not None:
            q.cleanup_owned(process, ticks, grace=0.3)
            after = q.owned_members(process.pid, ticks)
            record.update(live_members_after_cleanup=after, final_leader_returncode=process.returncode)
            q.require(not after, "Owned live members remained after cleanup")
            stdout, stderr = process.communicate(timeout=2)
            record.update(remaining_stdout=stdout, stderr=stderr)
    record.update(passed=True, elapsed_seconds=time.monotonic() - started)
    return record


def main():
    controller = validate_controller()
    q.require(sys.platform.startswith("linux") and ROOT.resolve() == Path("/home/jsy/GNRL_20261012").resolve(), "Preflight only in chemistry project")
    q.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and os.environ.get("GNRL_GPU_GUARD") != "1", "GPU must be disabled")
    q.require(q.sha256(ROOT / "scripts/run_successful_path_learning_v1.py") == QUEUE_SHA256, "Queue differs from reviewed frozen version")
    q.require(hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"), "Missing pidfd support")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8") as receipt:
        result = {"kind": "successful_path_process_control_cpu_preflight", "started_utc": q.utc(),
                  "queue_sha256": QUEUE_SHA256, "controller": controller,
                  "queue_v2_sha256": q.sha256(ROOT / "scripts/run_successful_path_learning_v2.py"), "preflight_sha256": q.sha256(__file__), "cpu_only": True,
                  "model_jobs": 0, "environment_jobs": 0, "gpu_jobs": 0, "cases": [], "complete": False}
        def interrupted(signum, frame):
            raise RuntimeError("Preflight interrupted by signal " + str(signum))
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGALRM, interrupted)
        signal.alarm(45)
        try:
            fd = os.pidfd_open(os.getpid())
            os.close(fd)
            result["pidfd_available"] = True
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", PYTHONUNBUFFERED="1")
            env.pop("GNRL_GPU_GUARD", None)
            for mode in ("normal_exit", "leader_exits_with_live_child", "timeout_cleanup"):
                result["cases"].append(run_case(mode, env))
            q.require(q.sha256(ROOT / "scripts/run_successful_path_learning_v1.py") == QUEUE_SHA256, "Queue changed during preflight")
            result["complete"] = True
        except BaseException as error:
            result["error"] = f"{type(error).__name__}: {error}"
        finally:
            signal.alarm(0)
            result["finished_utc"] = q.utc()
            json.dump(result, receipt, indent=2)
            receipt.flush()
        print(json.dumps(result), flush=True)
        return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

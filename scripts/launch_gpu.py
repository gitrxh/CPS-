"""Launch a job on explicitly authorized physical GPUs, holding local locks."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ALLOWED = frozenset({0, 1, 2, 3})


def resolve_devices(requested, rows, max_memory_mib=1024, active_uuids=()):
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("GPU list must be nonempty without duplicates")
    if not set(requested) <= ALLOWED:
        raise ValueError("Only physical GPU indices 0, 1, 2, 3 are authorized")
    by_index = {int(r[0].strip()): r for r in rows}
    devices = []
    for index in requested:
        row = by_index[index]
        if row[1].strip() in active_uuids:
            raise RuntimeError(f"Physical GPU {index} already has a compute process; not launching")
        if int(row[3].strip()) > max_memory_mib:
            raise RuntimeError(f"Physical GPU {index} is occupied; not launching")
        devices.append({"physical_index": index, "uuid": row[1].strip(), "name": row[2].strip()})
    return devices


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpus", required=True, help="Physical indices, restricted to 0–3")
    p.add_argument("--receipt", required=True)
    p.add_argument("--max-memory-mib", type=int, default=1024)
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args()
    requested = [int(x) for x in args.gpus.split(",")]
    if not set(requested) <= ALLOWED:
        p.error("Only physical GPU 0–3 are authorized")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        p.error("Missing child command")
    import fcntl
    lock_dir = Path.home() / "GNRL_20261012" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    locks = []
    for index in sorted(requested):
        lock = open(lock_dir / f"gpu_{index}.lock", "a+")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    smi = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.used", "--format=csv,noheader,nounits"
    ], text=True)
    running = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader,nounits"
    ], text=True)
    devices = resolve_devices(requested, list(csv.reader(io.StringIO(smi))),
                              args.max_memory_mib, set(running.splitlines()))
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=",".join(d["uuid"] for d in devices),
               CUDA_DEVICE_ORDER="PCI_BUS_ID", TOKENIZERS_PARALLELISM="false",
               GNRL_GPU_GUARD="1", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8")
    record = {"started_utc": datetime.now(timezone.utc).isoformat(), "devices": devices,
              "command": command, "launcher_pid": os.getpid()}
    path = Path(args.receipt)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)
    child = subprocess.Popen(command, env=env)
    record["child_pid"] = child.pid
    path.write_text(json.dumps(record, indent=2))
    try:
        rc = child.wait()
    except KeyboardInterrupt:
        child.terminate()
        rc = child.wait()
    record.update(returncode=rc, ended_utc=datetime.now(timezone.utc).isoformat())
    path.write_text(json.dumps(record, indent=2))
    sys.exit(rc)


if __name__ == "__main__":
    main()

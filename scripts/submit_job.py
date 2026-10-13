"""Start one named, detached job with PID and process start identity."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--name", required=True)
p.add_argument("command", nargs=argparse.REMAINDER)
a = p.parse_args()
assert a.name.replace("_", "").replace("-", "").isalnum()
root = Path(__file__).resolve().parents[1]
folder = root / "outputs" / a.name
folder.mkdir(parents=True, exist_ok=True)
receipt = folder / "job.json"
if receipt.exists():
    raise SystemExit("Job name already exists; inspect old job instead of duplicating it")
command = a.command[1:] if a.command[:1] == ["--"] else a.command
env = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", ALFWORLD_DATA="/home/jsy/zhangranxu-adrs-week-20260929/data/alfworld")
with open(folder / "stdout.log", "ab", buffering=0) as out, open(folder / "stderr.log", "ab", buffering=0) as err:
    proc = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
record = {"name": a.name, "pid": proc.pid, "command": command, "cwd": str(root), "started_utc": datetime.now(timezone.utc).isoformat(), "start_ticks": Path(f"/proc/{proc.pid}/stat").read_text().split(") ", 1)[1].split()[19]}
receipt.write_text(json.dumps(record, indent=2))
print(json.dumps(record))

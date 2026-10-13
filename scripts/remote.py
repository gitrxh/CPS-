"""SSH transport using an existing key; never accepts a password."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
C = json.loads((ROOT / "configs/server.json").read_text())
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "IdentitiesOnly=yes", "-i", C["ssh_key"], "-p", str(C["port"]), f'{C["user"]}@{C["host"]}']
SCP = ["scp", "-q", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-i", C["ssh_key"], "-P", str(C["port"])]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["exec", "deploy", "pull"])
    p.add_argument("paths", nargs="*")
    a = p.parse_args()
    if a.mode == "exec":
        script = sys.stdin.read() if not a.paths else Path(a.paths[0]).read_text(encoding="utf-8")
        raise SystemExit(subprocess.run(SSH + ["bash", "-s"], input=script.encode("utf-8")).returncode)
    if a.mode == "deploy":
        names = a.paths or ["gnrl", "scripts", "configs", "tests", "AGENTS.md"]
        for name in names:
            src = ROOT / name
            dst = f'{C["user"]}@{C["host"]}:{C["remote_root"]}/{Path(name).parent.as_posix()}/'
            subprocess.run(SCP + ["-r", str(src), dst], check=True)
    if a.mode == "pull":
        for name in a.paths:
            dst = ROOT / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(SCP + ["-r", f'{C["user"]}@{C["host"]}:{C["remote_root"]}/{name}', str(dst.parent)], check=True)

if __name__ == "__main__":
    main()

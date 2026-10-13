"""Filter success+replay-verified experts, preserving all exclusions in a receipt."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--input", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
counts = Counter()
rows = []
for line in Path(a.input).read_text().splitlines():
    row = json.loads(line)
    counts["rows"] += 1
    if row.get("kind") != "trajectory":
        counts["nontrajectory"] += 1
    elif not row.get("success"):
        counts["unsuccessful"] += 1
    elif not row.get("reset_replay_verified"):
        counts["unverified"] += 1
    else:
        counts["retained"] += 1
        counts["actions_retained"] += len(row["transitions"])
        rows.append(row)
out = Path(a.output)
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("x") as f:
    for row in rows:
        f.write(json.dumps(row) + "\n")
receipt = {"counts": dict(counts), "input": a.input, "input_sha256": hashlib.sha256(Path(a.input).read_bytes()).hexdigest(), "output_sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
out.with_suffix(".curation.json").write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt))

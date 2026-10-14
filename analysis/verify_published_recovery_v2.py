"""Verify the compact published evidence and recompute its descriptive means.

No SSH, model, simulator, or third-party dependencies are used. This verifies
the exported summaries; it does not independently audit the omitted raw traces.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "results/recovery_diagnostic_v2"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(left, right):
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def recompute():
    manifest = read_json(BUNDLE / "source_manifest.json")
    for entry in manifest["artifacts"]:
        payload = (ROOT / entry["published_path"]).read_bytes()
        require(len(payload) == entry["size_bytes"], entry["published_path"] + ": size mismatch")
        require(hashlib.sha256(payload).hexdigest() == entry["sha256"], entry["published_path"] + ": hash mismatch")
    diagnostic = read_json(BUNDLE / "diagnostic.json")
    summary = read_json(BUNDLE / "diagnostic_summary.json")
    scoring = read_json(BUNDLE / "score_summary.json")
    queue = read_json(BUNDLE / "queue_status.json")
    digest = hashlib.sha256((BUNDLE / "diagnostic.json").read_bytes()).hexdigest()
    require(digest == summary["diagnostic_sha256"], "Diagnostic receipt hash mismatch")
    require(diagnostic["complete"] and summary["complete"] and scoring["complete"] and queue["complete"], "Incomplete source evidence")
    require(diagnostic["gate_evaluable"] is True, "Gate is not evaluable")
    require(diagnostic["training_gate_passed"] == summary["training_gate_passed"] == queue["training_gate_passed"], "Gate receipt disagreement")
    cells = diagnostic["cells"]
    require(len(cells) == diagnostic["denominator"] == 36, "Fixed 36-slot denominator changed")
    available = [c for c in cells if c["Q"] is not None]
    unavailable = [c for c in cells if c["Q"] is None]
    for cell in available:
        require(cell["status"] == "eligible" and cell["Q"], "Invalid available cell")
        require(all(q in (0, 1) for q in cell["Q"].values()), "Non-binary actual Q")
        for key in ("Vq", "Vu", "Ve", "Vbase"):
            require(-1e-12 <= cell[key] <= 1 + 1e-12, "Utility outside [0,1] beyond floating-point tolerance")
        require(close(cell["Vu"], math.fsum(cell["Q"].values()) / len(cell["Q"])), "Uniform utility differs from actual Q mean")
        require(close(cell["difference_q_uniform"], cell["Vq"] - cell["Vu"]), "Uniform paired difference mismatch")
        require(close(cell["difference_q_expert"], cell["Vq"] - cell["Ve"]), "Expert paired difference mismatch")
    for cell in unavailable:
        require(all(cell[k] is None for k in ("Q", "Vq", "Vu", "Ve", "Vbase")), "Unavailable Q or utility was imputed")
        require(cell["difference_q_uniform"] == cell["difference_q_expert"] == 0, "Unavailable paired difference is not zero")
    branches = sum(len(c["Q"]) for c in available)
    require(branches == diagnostic["actual_learner_branches"] == summary["record_count"], "Branch count mismatch")
    means = {k: math.fsum(c[k] for c in available) / len(available) for k in ("Vq", "Vu", "Ve", "Vbase")}
    differences = {
        "q_minus_uniform": math.fsum(c["difference_q_uniform"] for c in cells) / len(cells),
        "q_minus_expert": math.fsum(c["difference_q_expert"] for c in cells) / len(cells),
    }
    for key, value in differences.items():
        require(close(value, diagnostic["mean_" + key]), "Published aggregate mismatch: " + key)
    return {
        "kind": "publication_descriptive_recomputation",
        "diagnostic_sha256": digest,
        "scope": "Fixed training-state diagnostic; not held-out task-success evaluation or a full-trajectory ECPS/MH experiment.",
        "fixed_slots": len(cells),
        "available_slots": len(available),
        "unavailable_slots": len(unavailable),
        "family_slot_counts": dict(sorted(Counter(c["family"] for c in cells).items())),
        "actual_learner_branches": branches,
        "scored_slots": scoring["record_count"],
        "scoring_calls": scoring["scoring"]["calls_completed"],
        "available_slot_means": means,
        "fixed_36_paired_differences": differences,
        "unavailable_handling": "Q and utilities remain null; only paired differences contribute zero to the predeclared 36-slot denominator.",
        "descriptive_breakdown": {
            "all_Q_zero": sum(set(c["Q"].values()) == {0} for c in available),
            "all_Q_one": sum(set(c["Q"].values()) == {1} for c in available),
            "mixed_Q": sum(len(set(c["Q"].values())) > 1 for c in available),
            "has_successful_action": sum(any(c["Q"].values()) for c in available),
        },
        "recorded_gate_components": diagnostic["gate_components"],
        "training_gate_passed": diagnostic["training_gate_passed"],
        "verification_limit": "Hashes and summary arithmetic are verified. Raw trajectories, action weights, model weights, and branch traces are not included or independently replayed by this verifier.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-summary", action="store_true", help="Write the derived summary instead of checking it")
    args = parser.parse_args()
    computed = recompute()
    target = BUNDLE / "computed_summary.json"
    if args.write_summary:
        target.write_text(json.dumps(computed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        require(read_json(target) == computed, "Derived summary differs from recomputation")
    print(json.dumps(computed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

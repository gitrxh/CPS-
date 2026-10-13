"""Read-only complete-result review; never launches model/environment work."""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads((ROOT / path).read_text())


def rows(path):
    return [json.loads(line) for line in (ROOT / path).read_text().splitlines() if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    status = read("outputs/phase1_queue_v2/status.json")
    assert status.get("current_stage") == "needs_astra_result_review" and status.get("queue_finished") is True
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "scope": "Complete single-seed development pilot review; no new rollout/model execution", "checks": {}}
    locked = read("outputs/phase1_queue_v2/locked_sha256.json")
    mismatches = [name for name, expected in locked.items() if not (ROOT / name).is_file() or sha(ROOT / name) != expected]
    assert not mismatches, mismatches
    report["checks"]["all_queue_locked_hashes_match"] = {"files": len(locked), "mismatches": mismatches}
    manifest = read("data/manifests/dev24.json"); games = manifest["games"]
    assert len(games) == len(set(games)) == 24
    assert all(sha(Path(g)) == manifest["game_sha256"][g] for g in games)
    report["checks"]["dev_manifest_and_24_game_bytes_match"] = True
    from scripts.run_phase1_queue import assert_pairing, ARM_DATA
    pools = rows("outputs/phase1_assembly_v1/candidate_pools.jsonl")
    receipt = assert_pairing(pools, *(rows(ARM_DATA[a]) for a in ("canonical", "uniform", "brp")), read("data/manifests/train120.json")["games"])
    assert receipt == read("outputs/phase1_queue_v2/pairing_receipt.json")
    report["checks"]["train120_full_pool_initial_state_terminal_multiset_pairing"] = True
    # Reuse the recorded analysis helper's function definitions without its CLI.
    source = ast.parse((ROOT / "scripts/summarize_pilot.py").read_text())
    module = ast.Module(body=[n for n in source.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))], type_ignores=[])
    helpers = {}; exec(compile(module, "scripts/summarize_pilot.py", "exec"), helpers)
    expected_tokens = {"canonical": 6532, "uniform": 6528, "brp": 6528}
    evidence = {}; report["training"] = {}; report["arms"] = {}
    for arm in expected_tokens:
        checkpoint = f"checkpoints/phase1_{arm}_seed42_v1"
        train = read(checkpoint + "/summary.json"); config = read(checkpoint + "/config.json")
        assert train["matched_step_budget_complete"] and train["optimizer_steps"] == train["requested_steps"] == 100
        assert train["target_tokens"] == expected_tokens[arm] and train["training_examples"] == 698
        assert config["data_sha256"] == sha(ROOT / ARM_DATA[arm])
        for k, value in {"seed":42,"max_steps":100,"batch_size":1,"gradient_accumulation":8,"learning_rate":0.0002,"lora_rank":16,"lora_alpha":32,"lora_dropout":0.05,"max_context_tokens":2048,"max_action_tokens":64}.items():
            assert config[k] == value, (arm, k)
        folder = ROOT / f"outputs/eval_phase1_{arm}_dev24_v1"
        ec = json.loads((folder / "config.json").read_text())
        assert ec["manifest_sha256"] == sha(ROOT / "data/manifests/dev24.json")
        assert ec["selection"] == "generate" and ec["seed"] == 20261012 and ec["max_steps"] == 100
        assert ec["max_context_tokens"] == 2048 and ec["max_action_tokens"] == 64
        assert Path(ec["adapter"]).resolve() == (ROOT / checkpoint / "final").resolve()
        assert ec["model"] == config["model"]
        records, summary = helpers["read_arm"](folder)
        assert list(records) == games and all(r["seed"] == 20261012 for r in records.values())
        assert all(r["transitions"] for r in records.values())
        evidence[arm] = records; report["arms"][arm] = summary
        report["training"][arm] = {"optimizer_steps":train["optimizer_steps"],"target_tokens":train["target_tokens"],"examples":train["training_examples"],"config_sha256":sha(ROOT / checkpoint / "config.json"),"final_checkpoint_all_files_locked":all(str(p.relative_to(ROOT)) in locked for p in (ROOT / checkpoint / "final").rglob("*") if p.is_file())}
        assert report["training"][arm]["final_checkpoint_all_files_locked"]
    assert all(evidence[a][g]["transitions"][0]["observation"] == evidence["canonical"][g]["transitions"][0]["observation"] and evidence[a][g]["transitions"][0]["admissible_commands"] == evidence["canonical"][g]["transitions"][0]["admissible_commands"] for a in evidence for g in games)
    report["checks"]["paired_initial_observations_and_admissible_commands"] = True
    report["checks"]["evaluation_raw_initial_state"] = "Not stored by the frozen evaluator. Direct cross-arm raw-state equality cannot be verified from these logs. Same game bytes/seed, initial observation and admissible commands are verified; do not upgrade that evidence to a recorded full-state comparison."
    provenance = read("outputs/reference_model_provenance_v1.json")
    model_checks = []
    for entry in provenance["files"]:
        path = Path(provenance["resolved_path"]) / entry["name"]; before = path.stat(); actual = sha(path); after = path.stat()
        assert before.st_size == after.st_size == entry["bytes"] and before.st_mtime_ns == after.st_mtime_ns and actual == entry["sha256"]
        model_checks.append({"name":entry["name"],"sha256":actual,"unchanged_during_read":True})
    report["checks"]["reference_model_current_bytes"] = {"manifest_sha256":sha(ROOT / "outputs/reference_model_provenance_v1.json"),"files":model_checks,"limitation":provenance["scope"]}
    report["comparisons"] = {right+"_vs_"+left:helpers["compare"](evidence[left], evidence[right]) for left,right in [("canonical","uniform"),("canonical","brp"),("uniform","brp")]}
    report["task_results"] = [{"gamefile":g,"family":Path(g).parents[1].name.split("-",1)[0],"success":{a:bool(evidence[a][g]["success"]) for a in evidence}} for g in games]
    positive = all(report["arms"]["brp"]["successes"] > report["arms"][a]["successes"] for a in ("canonical","uniform"))
    report["registered_directional_gate"] = {"rule":"learner-selected dev24 success count strictly exceeds both canonical and uniform","passed":positive,"scientific_decision":"Consider the prewritten phase2 fixed-seed confirmation candidate; not an efficacy claim or automatic execution approval." if positive else "Do not trigger phase2 confirmation or unseen evaluation. End the positive efficacy claim for this phase1 selector version; consider adopting only the separate broader successful-path finite CPU feasibility proposal."}
    report["interpretation_limits"] = ["Single training seed42 and24 development tasks; paired bootstrap/McNemar outputs are descriptive, unadjusted and subject to development-screen selection.","Only four development tasks are two-object tasks. This pilot cannot establish broad agent planning improvement or rule out small effects.","Same-pool training audit found8 learner/uniform trajectory differences,62/698 examples, and72/800 sampled inputs; target token exposures are near matched. Candidate coverage18 and selected changes9 are different quantities.","All arms share privileged native-planner generation. Positive ordinary-SFT or canonicalization effects are distinct from learner-aware finite-pool selection.","Exact-state candidates exchange useful physical delivery order; this does not establish hidden-object search, MH sampling or a unique causal reasoning mechanism."]
    output = ROOT / "outputs/phase1_independent_astra_review_v1.json"
    with output.open("x", encoding="utf-8") as f: json.dump(report, f, indent=2)
    print(json.dumps({"output":str(output),"checks_passed":True,"arms":report["arms"],"comparisons":report["comparisons"],"registered_directional_gate":report["registered_directional_gate"]},indent=2))


if __name__ == "__main__":
    main()

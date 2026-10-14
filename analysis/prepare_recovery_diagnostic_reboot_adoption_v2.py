"""Construct a review-bound parent adoption; never launches processes or CUDA.

Run on the deployment checkout after the exact new source/review files exist.
This creates only the explicit --output JSON. Queue validation still verifies
every inherited byte, model/runtime inventory, exit barrier and deadline.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_recovery_diagnostic_queue_reboot_v2 as queue

REBOOT = "configs/recovery_acquisition_reboot_resume_adoption_v2.json"
RESULT_REVIEW = "outputs/recovery_reboot_acquisition_completion_astra_review_v2.json"
ADAPTATION_REVIEW = "analysis/recovery_diagnostic_reboot_adaptation_astra_review_v2.json"
JOB = "outputs/recovery_projection_reboot_resume_job_v2/job.json"
LAUNCH = "outputs/recovery_projection_reboot_resume_job_v2/launch_verification.json"
SCHEMA = "analysis/recovery_diagnostic_reboot_adoption_schema_v2.json"
BUILDER = "analysis/prepare_recovery_diagnostic_reboot_adoption_v2.py"


def build(*, result_review=RESULT_REVIEW, adaptation_review=ADAPTATION_REVIEW):
    old = queue.old
    reboot = old.read_json(ROOT / REBOOT)
    original = old.read_json(ROOT / reboot["old_parent"])
    inventory = old.read_json(ROOT / reboot["inventory"])
    prior_queue = reboot["output_paths"]["queue"]
    status_path, locks_path = prior_queue + "/status.json", prior_queue + "/locked_sha256.json"
    hashes = {}

    def merge(mapping):
        for name, digest in mapping.items():
            old.require(name not in hashes or hashes[name] == digest, "Conflicting inherited hash: " + name)
            hashes[name] = digest

    merge(original["file_sha256"])
    merge(reboot["file_sha256"])
    merge(inventory["documents"][original["output_paths"]["queue"] + "/locked_sha256.json"])
    merge({name: info["sha256"] for name, info in inventory["files"].items()})
    merge(old.read_json(ROOT / locks_path))
    sources = queue.worker.source_inventory()
    additional = set(sources) | {REBOOT, status_path, locks_path, result_review, adaptation_review,
        JOB, LAUNCH, BUILDER, SCHEMA, original["candidate"], original["input_inventory"],
        "analysis/recovery_diagnostic_implementation_review_v1.json",
        "analysis/recovery_diagnostic_controller_astra_review_v1.json", old.PROVENANCE, old.MODEL_INVENTORY}
    merge({name: old.sha256(ROOT / name) for name in additional})
    result = old.read_json(ROOT / result_review)
    adaptation = old.read_json(ROOT / adaptation_review)
    old.require(result.get("recovery_reboot_acquisition_result_review_passed") is True
                and result.get("recovery_acquisition_result_review_passed") is True,
                "Complete acquisition review has not passed")
    old.require(adaptation.get("recovery_diagnostic_reboot_adaptation_review_passed") is True,
                "Exact adaptation review has not passed")
    for field,name in (("controller_sha256",queue.CONTROLLER),("worker_sha256",queue.worker.RUNNER),
                       ("lineage_sha256",queue.lineage.MODULE),("test_sha256",queue.TEST)):
        old.require(adaptation[field] == hashes[name], "Reviewed adaptation bytes differ: " + name)
    job = old.read_json(ROOT / JOB)
    final_cost = old.read_json(ROOT / reboot["output_paths"]["replay"] / "cost_progress.json")
    return {"kind":"recovery_diagnostic_parent_adoption", "version":2, "adopted":True,
        "adopted_by":"root_after_independent_GPT_6_Astra_review", "adopted_utc":old.utc(),
        "physical_gpu":2, "authorized_stages":["score","diagnostic"],
        "candidate":original["candidate"], "source_inventory":sources, "file_sha256":hashes,
        "original_file_sha256":original["original_file_sha256"],
        "runtime_file_sha256":original["runtime_file_sha256"], "input_inventory":original["input_inventory"],
        "implementation_review":"analysis/recovery_diagnostic_implementation_review_v1.json",
        "controller_review":"analysis/recovery_diagnostic_controller_astra_review_v1.json",
        "adaptation_review":adaptation_review, "acquisition_parent_adoption":REBOOT,
        "acquisition_status":status_path, "acquisition_locks":locks_path,
        "acquisition_result_review":result_review, "acquisition_job":JOB, "acquisition_launch_receipt":LAUNCH,
        "acquisition_controller_identity":{"pid":job["pid"],"start_ticks":int(job["start_ticks"])},
        "historical_gap":dict(queue.lineage.HISTORICAL_GAP),
        "acquisition_resume_provenance":queue.lineage.acquisition_provenance(final_cost),
        "output_paths":{"queue":"outputs/recovery_projection_diagnostic_reboot_queue_v2",
                        "score":"outputs/recovery_projection_score_reboot_v2",
                        "diagnostic":"outputs/recovery_projection_diagnostic_reboot_v2"},
        "execution_preferences":{"background":True,"external_monitoring":False,"scheduled_reports":False,
                                 "automatic_retries":False,"stop_after":"needs_astra_diagnostic_review"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--result-review", default=RESULT_REVIEW)
    parser.add_argument("--adaptation-review", default=ADAPTATION_REVIEW)
    args = parser.parse_args()
    output = queue.old.safe_relative(args.output)
    adoption = build(result_review=args.result_review, adaptation_review=args.adaptation_review)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(adoption, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"path":args.output,"sha256":queue.old.sha256(output),"execution_started":False}))


if __name__ == "__main__":
    main()

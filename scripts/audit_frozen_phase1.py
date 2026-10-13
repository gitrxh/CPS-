"""CPU-only audit of frozen phase1 choices and actual supervised token inputs."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gnrl.policy import supervised_example
from gnrl.train_sft import load_examples


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    files = {"canonical": "outputs/phase1_assembly_v1/canonical_baseline.jsonl",
             "uniform": "outputs/phase1_selection_v1/uniform_valid.jsonl",
             "brp": "outputs/phase1_selection_v1/learner_selected.jsonl"}
    trajectories = {k: rows(v) for k, v in files.items()}
    examples = {k: [supervised_example(tokenizer, r, 2048, 64) for r in load_examples(v)]
                for k, v in files.items()}
    identities = [(r["gamefile"], r["seed"]) for r in trajectories["canonical"]]
    assert len(identities) == len(set(identities)) == 120
    assert all([(r["gamefile"], r["seed"]) for r in data] == identities for data in trajectories.values())
    assert len({len(data) for data in examples.values()}) == 1
    n = len(examples["canonical"])
    rng = random.Random(42)
    order = list(range(n))
    rng.shuffle(order)
    exposure = []
    cursor = 0
    for _ in range(800):
        if cursor >= n:
            rng.shuffle(order)
            cursor = 0
        exposure.append(order[cursor])
        cursor += 1
    report = {"kind": "train_only_frozen_selection_audit", "evaluation_data_read": False,
              "source_sha256": {k: hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in files.items()},
              "trajectories_per_arm": 120, "action_examples_per_arm": n,
              "tokenizer": a.model, "context_budget": 2048, "seed": 42,
              "sampled_examples_for_100_steps_accum8": len(exposure), "arms": {}, "comparisons": {},
              "interpretation": "Input differences describe training exposure, not evidence of learning benefit. Each comparison uses the frozen identical task order and same reference candidate pools."}
    for arm, data in examples.items():
        report["arms"][arm] = {"unique_dataset_target_tokens": sum(r["target_tokens"] for r in data),
                               "predicted_100step_target_tokens": sum(data[i]["target_tokens"] for i in exposure),
                               "predicted_100step_input_tokens": sum(len(data[i]["input_ids"]) for i in exposure)}
    for left, right in [("canonical", "uniform"), ("canonical", "brp"), ("uniform", "brp")]:
        changed = []
        for l, r in zip(trajectories[left], trajectories[right]):
            assert Counter(t["action"] for t in l["transitions"]) == Counter(t["action"] for t in r["transitions"])
            assert l["selection"]["pool_sha256"] == r["selection"]["pool_sha256"]
            if l["selection"]["candidate_id"] != r["selection"]["candidate_id"]:
                changed.append(l["gamefile"])
        different = [l["input_ids"] != r["input_ids"] or l["labels"] != r["labels"]
                     for l,r in zip(examples[left], examples[right])]
        report["comparisons"][right + "_vs_" + left] = {
            "different_selected_trajectories": len(changed), "changed_tasks": changed,
            "different_tokenized_supervised_examples": sum(different),
            "different_sampled_examples_in_100steps": sum(different[i] for i in exposure)}
    with Path(a.output).open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    compact = dict(report)
    compact["comparisons"] = {k: {f:v for f,v in value.items() if f != "changed_tasks"}
                              for k,value in report["comparisons"].items()}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()

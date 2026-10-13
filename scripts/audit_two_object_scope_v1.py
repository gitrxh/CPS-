"""Read-only training-pool audit; no model or environment imports."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/two_blocks_v1/candidate_pools.jsonl"
OUTPUT = ROOT / "outputs/two_object_mechanism_scope_audit_v1.json"


def facts(state, predicate):
    return {tuple(str(a[0]) for a in args) for p, args in state["facts"]
            if str(p).lower() == predicate}


def pickups(trace):
    found = []
    for i, t in enumerate(trace):
        match = re.fullmatch(r"take (.+) from (.+)", t["action"])
        if not match:
            continue
        added = facts(t["state_after"], "holds") - facts(t["state_before"], "holds")
        assert len(added) == 1
        obj = next(iter(added))[-1]
        removed = facts(t["state_before"], "inreceptacle") - facts(t["state_after"], "inreceptacle")
        sources = [r for o, r in removed if o == obj]
        assert len(sources) == 1
        found.append({"step": i, "object_name": match[1], "source_name": match[2],
                      "object_id": obj, "source_id": sources[0]})
    assert len(found) == 2
    return found


def text_visible_before(trace, step, name, initial):
    # Diagnostic lexical evidence, not a claim about general semantic inference.
    observations = [initial] + [t["observation"] for t in trace[:step + 1]]
    return any(re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", x) for x in observations)


def main():
    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 20 and all(p["split"] == "train" for p in rows)
    tasks = []
    counters = Counter()
    for p in rows:
        original = next(c for c in p["candidates"] if c["candidate_id"] == p["original_candidate_id"])
        trace = original["transitions"]
        take = pickups(trace)
        init = p["initial_observation"]
        initial_commands = trace[0]["admissible_commands"]
        first_take = take[0]["step"]
        first_nav = next((i for i, t in enumerate(trace) if t["action"].startswith("go to ")), None)
        initial_named_objects_visible = [text_visible_before(trace, 0, x["object_name"], init) for x in take]
        row = {"gamefile": p["gamefile"], "seed": p["seed"], "pool_size": len(p["candidates"]),
               "canonical_pickups": take, "same_source_receptacle_id": take[0]["source_id"] == take[1]["source_id"],
               "initial_observation_contains_named_goal_objects": initial_named_objects_visible,
               "initial_admissible_has_goal_pickup": any(x.startswith("take " + t["object_name"] + " from ") for x in initial_commands for t in take),
               "first_navigation": trace[first_nav]["action"] if first_nav is not None else None,
               "both_objects_observed_before_first_pickup": all(text_visible_before(trace, first_take, x["object_name"], init) for x in take),
               "both_pickups_have_prior_observation_of_named_object": all(text_visible_before(trace, x["step"], x["object_name"], init) for x in take),
               "alternatives": []}
        counters["canonical_tasks"] += 1
        counters["canonical_same_source"] += row["same_source_receptacle_id"]
        counters["canonical_initial_named_objects_absent"] += not any(initial_named_objects_visible)
        counters["canonical_initial_goal_pickup_absent_from_admissibles"] += not row["initial_admissible_has_goal_pickup"]
        counters["canonical_both_objects_observed_before_first_pickup"] += row["both_objects_observed_before_first_pickup"]
        for c in p["candidates"]:
            if c is original:
                continue
            alt = c["transitions"]
            atake = pickups(alt)
            actions = [t["action"] for t in trace]
            alternative_actions = [t["action"] for t in alt]
            assert Counter(actions) == Counter(alternative_actions)
            assert trace[0]["state_before"] == alt[0]["state_before"]
            assert trace[-1]["state_after"] == alt[-1]["state_after"]
            assert all(t[-1]["won"] and t[-1]["done"] for t in (trace, alt))
            assert c["evidence"]["full_replay_verified"] and original["evidence"]["full_replay_verified"]
            names = [t["object_name"] for t in take]
            def swap_names(action):
                return re.sub(r"(?<!\w)(" + "|".join(map(re.escape, names)) + r")(?!\w)",
                              lambda m: names[1] if m[0] == names[0] else names[0], action)
            nav = [(i, x) for i, x in enumerate(actions) if x.startswith("go to ")]
            anav = [(i, x) for i, x in enumerate(alternative_actions) if x.startswith("go to ")]
            changes = [i for i, (a, b) in enumerate(zip(actions, alternative_actions)) if a != b]
            reversed_raw = [x["object_id"] for x in atake] == [x["object_id"] for x in take][::-1]
            item = {"candidate_id": c["candidate_id"], "canonical_actions": actions,
                    "alternative_actions": alternative_actions, "alternative_pickups": atake,
                    "navigation_sequence_identical": [x for _, x in nav] == [x for _, x in anav],
                    "navigation_step_positions_and_commands_identical": nav == anav,
                    "same_source_receptacle_id": row["same_source_receptacle_id"],
                    "raw_object_pickup_order_reversed": reversed_raw,
                    "entire_action_sequence_equals_swapping_two_object_names": list(map(swap_names, actions)) == alternative_actions,
                    "changed_action_indices": changes,
                    "changed_commands_are_only_take_or_move": all(actions[i].startswith(("take ", "move ")) and alternative_actions[i].startswith(("take ", "move ")) for i in changes),
                    "intermediate_raw_facts_differ_steps": [i for i, (a, b) in enumerate(zip(trace, alt)) if a["state_after"]["facts"] != b["state_after"]["facts"]],
                    "same_exact_action_multiset": True, "same_full_initial_and_terminal_state": True,
                    "both_successful_and_have_stored_replay_evidence": True,
                    "navigation_alias_only": not reversed_raw,
                    "alternative_both_objects_observed_before_first_pickup": all(text_visible_before(alt, atake[0]["step"], x["object_name"], init) for x in atake)}
            row["alternatives"].append(item)
            counters["alternatives"] += 1
            for k, v in item.items():
                if isinstance(v, bool):
                    counters["alternatives_" + k] += v
            counters["changed_action_positions"] += len(changes)
        tasks.append(row)
    report = {"version": 1, "scope": "train_only_stored_trace_audit_no_model_no_new_environment_execution",
              "source": str(SOURCE.relative_to(ROOT)), "source_sha256": sha256(SOURCE.read_bytes()).hexdigest(),
              "source_code_hashes": {name: sha256((ROOT / name).read_bytes()).hexdigest() for name in
                                     ["gnrl/two_object_blocks.py", "gnrl/policy.py", "gnrl/alfworld_env.py"]},
              "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
              "counts": dict(sorted(counters.items())), "tasks": tasks,
              "measurement_notes": [
                  "Sources are verified from grounded holds additions and inreceptacle removals, preserving distinct raw object/receptacle IDs.",
                  "Name-swap equality concerns action strings only. Raw object identities, emitted histories and intermediate physical states are not relabelled or asserted equal.",
                  "Observation visibility is exact named-object lexical presence in actual observations through the decision point, excluding hidden facts, expert fields and future observations. It does not measure all possible semantic inferences.",
                  "Replay success/state evidence is checked from stored records; this audit does not re-execute the environment or run a model.",
                  "Initial observations and admissible commands are both inspected; admissible lists are part of the common student prompt, so their information must not be ignored."
              ],
              "scientific_interpretation": {
                  "physical_difference": "Exchanging which distinct object is picked up and delivered first changes grounded holds/inreceptacle facts during the episode. Even when the two objects share a type and source and navigation is identical, this is real manipulation-order exchange, not a single-step navigation alias.",
                  "observed_subtypes": "Eight of18 alternatives have the same grounded source receptacle and identical navigation commands at identical steps; their full action strings equal a swap of the two object names. The other10 have different sources and changed navigation order, so the entire set is not merely object-label swapping. Those10 reorder complete deliveries from known distinct sources, without discovering new source assignments.",
                  "narrow_symmetry": "The eight same-source cases instantiate object-order symmetry within the same delivery skeleton. The ten different-source cases instantiate complete delivery-block order changes. Neither category alone establishes hidden-source discovery, treatment-order reasoning, or general planning diversity.",
                  "teacher_privilege": "The native planner and reversed-path compiler access world-state object locations and goal grounding before the student observes them. A student initially seeing only receptacle names and a task cannot directly infer the demonstrated correct first source from that evidence alone. Executable supervised traces can therefore contain privileged teacher decisions under partial observability.",
                  "where_student_choice_can_be_causal": "After a source is visited/opened and both named objects are observed, either first pickup can be a valid causal student decision. Reference selection over the entire path can reflect object-number/order preference and its subsequent observation history even when the earlier source navigation is identical.",
                  "shared_control": "All three phase1 arms use the same privileged teacher candidate generator and paired task/state/action-multiset constraints. This controls differential teacher access within the comparison, but does not prove that students acquire the planner's hidden-state search ability.",
                  "positive_claim_allowed": "A matched positive result would support the utility of choosing a reference-preferred, executable object-delivery order from this fixed set of privileged, state-equivalent teacher paths, under the tested ALFWorld prompt and training protocol.",
                  "positive_claim_not_established": ["General environment-consistent planning or broad action-block reordering beyond this two-object generator", "Improved exploration or hidden-object localization", "Discovery of new navigation strategies or global route optimization", "MH sampling correctness", "The unique causal explanation is learner-native reasoning rather than surface object-ID/order preference"],
                  "future_train_only_discriminator": "For a separately registered later study, audit score changes under consistent object-name permutations or separate manipulation and navigation score contributions. Such probes could test surface-label sensitivity but must not retroactively change this running protocol or reinterpret executable state equality."
              }}
    with OUTPUT.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"output": str(OUTPUT), "counts": report["counts"]}, indent=2))


if __name__ == "__main__":
    main()

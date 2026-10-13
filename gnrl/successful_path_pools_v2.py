"""CPU-only prototype for outcome-constrained successful paths, NOT exact-state BRP.

No acquisition/model CLI is exposed before this candidate is adopted. The API
executes recipes only when explicitly called with an environment. All student
inputs use the existing causal allowlist. Existing phase-0/1 modules are unchanged.
"""
from __future__ import annotations

import copy
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import re

from .alfworld_env import ReplayMismatch, information_only_action, transition_record
from .policy import messages
from .two_object_blocks import entity_names, predicates

LOCKED_DOMAIN_SHA256 = "0af55e7f92d12b09d28515aa730e4d03fddd17228ef1fd7b18a3ee8ba49b27b9"
FAMILIES = {"pick_and_place_simple", "look_at_obj_in_light", "pick_two_obj_and_place",
            "pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep"}
TREATMENTS = {"pick_clean_then_place_in_recep": ("clean", "cleanable", "isclean", {"sinktype", "sinkbasintype"}),
              "pick_heat_then_place_in_recep": ("heat", "heatable", "ishot", {"microwavetype"}),
              "pick_cool_then_place_in_recep": ("cool", "coolable", "iscool", {"fridgetype"})}
OPERATOR_MAP = {"gotolocation": ("go to", 1), "openobject": ("open", 1), "closeobject": ("close", 1),
                "pickupobject": ("take", 1), "putobject": ("move", 1), "cleanobject": ("clean", 5),
                "heatobject": ("heat", 5), "coolobject": ("cool", 5), "toggleobject": ("use", 5),
                "sliceobject": ("slice", 5), "look": ("look", 0), "inventory": ("inventory", 0),
                "help": ("help", 0), "examinereceptacle": ("examine", 0), "examineobject": ("examine", 0)}
TEMPLATES = {"openobject": "open {r}", "closeobject": "close {r}", "pickupobject": "take {o} from {r}",
             "putobject": "move {o} to {r}", "cleanobject": "clean {o} with {r}", "heatobject": "heat {o} with {r}",
             "coolobject": "cool {o} with {r}", "toggleobject": "use {o}", "sliceobject": "slice {co} with {ko}",
             "look": "look", "inventory": "inventory", "help": "help", "examinereceptacle": "examine {r}",
             "examineobject": "examine {o}", "gotolocation": "go to [{r.name | receptacleatlocation(r:receptacle, lend:location)}]"}


class UnsupportedTask(ValueError):
    pass


class RecipeRejected(ValueError):
    """Expected proposal rejection. Replay/environment failures are NOT this type."""


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def sexpr(text):
    """Parse original PDDL without rewriting the game; reject malformed input."""
    tokens = re.findall(r"\(|\)|[^\s()]+", re.sub(r";[^\n]*", "", text).lower())
    stack, roots = [], []
    for token in tokens:
        if token == "(":
            node = []
            (stack[-1] if stack else roots).append(node)
            stack.append(node)
        elif token == ")":
            if not stack:
                raise UnsupportedTask("Unbalanced PDDL close")
            stack.pop()
        else:
            if not stack:
                raise UnsupportedTask("PDDL token outside an expression")
            stack[-1].append(token)
    if stack or len(roots) != 1:
        raise UnsupportedTask("Malformed PDDL root")
    return roots[0]


def _nodes(node):
    if isinstance(node, list):
        yield node
        for child in node:
            yield from _nodes(child)


def parse_operator_costs(domain, grammar):
    """Check every active domain operator and its exact supported command map."""
    tree = sexpr(domain)
    actions = [n for n in tree if isinstance(n, list) and n and n[0] == ":action"]
    if Counter(a[1] for a in actions) != Counter(OPERATOR_MAP.keys()):
        raise UnsupportedTask("Unknown, missing or duplicate PDDL operator")
    templates = {m.group(1).lower(): m.group(2) for m in re.finditer(
        r'action\s+(\w+)\s*\{\s*template\s*::\s*"([^"]+)"', grammar, re.I)}
    for name, expected in TEMPLATES.items():
        if templates.get(name) != expected:
            raise UnsupportedTask("Unsupported operator/command grammar: " + name)
    costs = {}
    for action in actions:
        name = action[1]
        if action.count(":effect") != 1:
            raise UnsupportedTask("Missing or repeated operator effect")
        effect = action[action.index(":effect") + 1]
        direct = effect[1:] if effect and effect[0] == "and" else [effect]
        increments = [n for n in _nodes(effect) if n and n[0] in {"increase", "decrease", "assign", "scale-up", "scale-down"}]
        value = 0
        for inc in increments:
            if not any(inc is n for n in direct) or len(inc) != 3 or inc[0] != "increase" or inc[1] != ["total-cost"] or not re.fullmatch(r"\d+", str(inc[2])):
                raise UnsupportedTask("Nonconstant, conditional or unknown numeric cost effect")
            value += int(inc[2])
        command, expected = OPERATOR_MAP[name]
        if value != expected:
            raise UnsupportedTask("Changed cost for " + name)
        costs[command] = value
    return costs


@dataclass(frozen=True)
class GoalSpec:
    family: str
    object_type: str
    target_type: str | None
    toggle_type: str | None
    parsed_goal: list


def parse_goal(problem, metadata):
    """Validate the whole typed existential goal, including variable identity."""
    family = metadata.get("task_type")
    params = metadata.get("pddl_params", {})
    if family not in FAMILIES or params.get("object_sliced") or params.get("mrecep_target"):
        raise UnsupportedTask("Unsupported family/sliced/movable target")
    def typename(field, required):
        value = params.get(field, "")
        if not isinstance(value, str) or (required and not re.fullmatch(r"[A-Za-z]+", value)):
            raise UnsupportedTask("Missing or ambiguous metadata: " + field)
        return (value + "Type").lower() if value else None
    objtype = typename("object_target", True)
    targettype = typename("parent_target", family != "look_at_obj_in_light")
    toggletype = typename("toggle_target", family == "look_at_obj_in_light")
    root = sexpr(problem)
    goals = [n for n in root if isinstance(n, list) and n and n[0] == ":goal"]
    if len(goals) != 1 or len(goals[0]) != 2:
        raise UnsupportedTask("Expected exactly one original goal")
    goal = goals[0][1]
    types, atoms = {}, []
    def flatten(node, scope):
        if not isinstance(node, list) or not node:
            raise UnsupportedTask("Empty/invalid goal clause")
        op = node[0]
        if op == "and":
            for child in node[1:]: flatten(child, scope)
        elif op == "exists" and len(node) == 3:
            local = dict(scope); tokens = node[1]; pending = []; i = 0
            while i < len(tokens):
                if tokens[i] == "-":
                    if not pending or i + 1 >= len(tokens): raise UnsupportedTask("Invalid typed quantifier")
                    kind = tokens[i + 1]
                    for name in pending:
                        value = "v" + str(len(types)); types[value] = kind; local[name] = value
                    pending = []; i += 2
                else:
                    if not isinstance(tokens[i], str) or not tokens[i].startswith("?") or tokens[i] in pending:
                        raise UnsupportedTask("Invalid quantified variable")
                    pending.append(tokens[i]); i += 1
            if pending: raise UnsupportedTask("Untyped goal variables")
            flatten(node[2], local)
        else:
            if op == "not" and len(node) == 2 and isinstance(node[1], list) and node[1][:1] == ["="]:
                op, args = "neq", node[1][1:]
            else:
                args = node[1:]
            if op in {"or", "forall", "imply", "not", "="} or any(isinstance(x, list) for x in args):
                raise UnsupportedTask("Unsupported original goal clause")
            resolved = []
            for arg in args:
                if arg.startswith("?") and arg not in scope: raise UnsupportedTask("Unbound goal variable")
                resolved.append(scope.get(arg, arg))
            atoms.append((op, *resolved))
    flatten(goal, {})
    actual = set(atoms)
    def unique(values, what):
        if len(values) != 1: raise UnsupportedTask("Ambiguous goal " + what)
        return next(iter(values))
    if family == "look_at_obj_in_light":
        holds = unique({a for a in actual if a[0] == "holds" and len(a) == 3}, "holds")
        _, held_agent, obj = holds
        toggle = unique({a[1] for a in actual if a[0] == "istoggled" and len(a) == 2}, "device")
        rec = unique({a[2] for a in actual if a[:2] == ("inreceptacle", toggle) and len(a) == 3}, "device receptacle")
        loc = unique({a[2] for a in actual if a[:2] == ("receptacleatlocation", rec) and len(a) == 3}, "device location")
        at_agent = unique({a[1] for a in actual if a[0] == "atlocation" and len(a) == 3 and a[2] == loc}, "located agent")
        expected = {("holds", held_agent, obj), ("objecttype", obj, objtype), ("objecttype", toggle, toggletype),
                    ("toggleable", toggle), ("istoggled", toggle), ("inreceptacle", toggle, rec),
                    ("receptacleatlocation", rec, loc), ("atlocation", at_agent, loc)}
        required = {held_agent: "agent", obj: "object", toggle: "object", rec: "receptacle", loc: "location", at_agent: "agent"}
        if len({obj, toggle, rec, loc}) != 4: raise UnsupportedTask("Aliased goal roles")
    else:
        objects = {a[1] for a in actual if a[0] == "objecttype" and len(a) == 3 and a[2] == objtype}
        wanted = 2 if family == "pick_two_obj_and_place" else 1
        if len(objects) != wanted: raise UnsupportedTask("Goal object cardinality")
        rec = unique({a[1] for a in actual if a[0] == "receptacletype" and len(a) == 3 and a[2] == targettype}, "target")
        expected = {("receptacletype", rec, targettype)}
        required = {rec: "receptacle", **{o: "object" for o in objects}}
        for obj in objects:
            expected |= {("objecttype", obj, objtype), ("inreceptacle", obj, rec)}
            if family in TREATMENTS:
                _, cap, effect, _ = TREATMENTS[family]; expected |= {(cap, obj), (effect, obj)}
        if wanted == 2:
            pair = sorted(objects)
            if ("neq", *pair) in actual: expected.add(("neq", *pair))
            elif ("neq", *reversed(pair)) in actual: expected.add(("neq", *reversed(pair)))
            else: raise UnsupportedTask("Two objects must be explicitly distinct")
    if actual != expected or types != required:
        raise UnsupportedTask("Original goal differs from exact supported typed template")
    return GoalSpec(family, objtype, targettype, toggletype, goal)


def audit_game(game_data, metadata):
    if hashlib.sha256(game_data["pddl_domain"].encode()).hexdigest() != LOCKED_DOMAIN_SHA256:
        raise UnsupportedTask("Original domain hash differs from locked training domain")
    costs = parse_operator_costs(game_data["pddl_domain"], game_data["grammar"])
    goal = parse_goal(game_data["pddl_problem"], metadata)
    return goal, costs, {key + "_sha256": hashlib.sha256(game_data[key].encode()).hexdigest()
                         for key in ("pddl_domain", "pddl_problem", "grammar")}


def action_cost(action, costs):
    verb = "go to" if action.startswith("go to ") else action.split(" ", 1)[0]
    if verb not in costs: raise UnsupportedTask("Unaudited command cost: " + action)
    return costs[verb]


def physical_key(state):
    # Canonical state equality is untouched. This filter ignores bookkeeping and
    # explicit negative-cache facts under closed-world predicate semantics.
    return digest(sorted([f for f in state["facts"] if f[0].lower() not in {"checked", "examined"}
                          and not f[0].lower().startswith("not_")], key=lambda f: json.dumps(f)))


def assignment_key(a):
    return (tuple(a["objects"]), a.get("target"), a.get("device"))


def _unique(values, reason):
    values = set(values)
    if len(values) != 1: raise RecipeRejected(reason)
    return next(iter(values))


def eligible_assignments(initial, goal):
    f = predicates(initial)
    objects = sorted(o for o, t in f.get("objecttype", ()) if t.lower() == goal.object_type
                     and (o,) in f.get("pickupable", set())
                     and len({r for ob, r in f.get("inreceptacle", ()) if ob == o}) == 1)
    if goal.family in TREATMENTS:
        _, cap, _, kinds = TREATMENTS[goal.family]
        objects = [o for o in objects if (o,) in f.get(cap, set())]
        devices = sorted(r for r, t in f.get("receptacletype", ()) if t.lower() in kinds)
    elif goal.family == "look_at_obj_in_light":
        devices = sorted(o for o, t in f.get("objecttype", ()) if t.lower() == goal.toggle_type
                         and (o,) in f.get("toggleable", set())
                         and len({r for ob, r in f.get("inreceptacle", ()) if ob == o}) == 1)
    else:
        devices = [None]
    targets = [None] if goal.family == "look_at_obj_in_light" else sorted(
        r for r, t in f.get("receptacletype", ()) if t.lower() == goal.target_type
        and (t, goal.object_type) in f.get("cancontain", set()))
    object_groups = itertools.permutations(objects, 2) if goal.family == "pick_two_obj_and_place" else ((o,) for o in objects)
    for objs, target, device in itertools.product(object_groups, targets, devices):
        if target is not None and any((o, target) in f.get("inreceptacle", set()) for o in objs): continue
        if device in objs: continue
        yield {"objects": list(objs), "target": target, "device": device}


def proposal_axis(a, canonical):
    changed = []
    if tuple(a["objects"]) != tuple(canonical["objects"]):
        changed.append("pair_or_order" if len(a["objects"]) == 2 else "object")
    if a.get("target") != canonical.get("target"): changed.append("target")
    if a.get("device") != canonical.get("device"): changed.append("device")
    return changed[0] if len(changed) == 1 else "multi_axis" if changed else "canonical"


def enumerate_proposals(initial, goal, canonical, gamefile, limit=7, tuple_limit=4096):
    if not 0 <= limit <= 7 or not 1 <= tuple_limit <= 4096: raise ValueError("Registered assignment limits exceeded")
    buckets = defaultdict(list); seen = {assignment_key(canonical)}; considered = 0; capped = False
    for a in eligible_assignments(initial, goal):
        if considered >= tuple_limit: capped = True; break
        considered += 1
        key = assignment_key(a)
        if key in seen: continue
        seen.add(key); a["proposal_axis"] = proposal_axis(a, canonical); buckets[a["proposal_axis"]].append(a)
    for axis in buckets:
        buckets[axis].sort(key=lambda a: json.dumps(assignment_key(a), separators=(",", ":")))
    result = []; axes = ["object", "target", "device", "pair_or_order"]
    for i in range(max([len(buckets[a]) for a in axes] + [0])):
        for axis in axes:
            if i < len(buckets[axis]) and len(result) < limit: result.append(buckets[axis][i])
        if len(result) == limit: break
    multi = sorted(buckets["multi_axis"], key=lambda a: hashlib.sha256((gamefile + json.dumps(
        {k: a[k] for k in ("objects", "target", "device")}, sort_keys=True, separators=(",", ":"))).encode()).hexdigest())
    result.extend(multi[:limit - len(result)])
    return result, {"assignment_tuples_considered": considered, "assignment_enumeration_capped": capped,
                    "tuples_skipped_by_cap_lower_bound": int(capped), "eligible_by_axis": {k: len(v) for k, v in buckets.items()},
                    "fixed_attempt_count": len(result), "no_yield_dependent_replenishment": True}


def witness(state, goal, assignment):
    f = predicates(state); objects = assignment["objects"]
    if len(objects) != (2 if goal.family == "pick_two_obj_and_place" else 1) or len(set(objects)) != len(objects): return False
    if any((o, goal.object_type) not in f.get("objecttype", set()) for o in objects): return False
    if goal.family == "look_at_obj_in_light":
        device = assignment.get("device")
        agents = {a for a, _ in f.get("atlocation", set())}
        if len(agents) != 1: return False
        agent = next(iter(agents))
        if (agent, objects[0]) not in f.get("holds", set()) or (device, goal.toggle_type) not in f.get("objecttype", set()): return False
        if (device,) not in f.get("toggleable", set()) or (device,) not in f.get("istoggled", set()): return False
        return any((agent, loc) in f.get("atlocation", set()) for dev, rec in f.get("inreceptacle", set()) if dev == device
                   for r, loc in f.get("receptacleatlocation", set()) if r == rec)
    target = assignment.get("target")
    if (target, goal.target_type) not in f.get("receptacletype", set()): return False
    if any((o, target) not in f.get("inreceptacle", set()) for o in objects): return False
    if goal.family in TREATMENTS:
        _, cap, effect, _ = TREATMENTS[goal.family]
        return all((o,) in f.get(cap, set()) and (o,) in f.get(effect, set()) for o in objects)
    return True


def derive_canonical_assignment(trace, goal, names):
    final = predicates(trace[-1]["state_after"]); objset = {o for o, t in final.get("objecttype", ()) if t == goal.object_type}
    reverse_names = defaultdict(list)
    for identity, name in names.items(): reverse_names[name].append(identity)
    if goal.family == "look_at_obj_in_light":
        obj = _unique({o for _, o in final.get("holds", ()) if o in objset}, "Ambiguous canonical held object")
        used = [t["action"][4:] for t in trace if t["action"].startswith("use ")]
        devices = {o for o, typ in final.get("objecttype", ()) if typ == goal.toggle_type and (o,) in final.get("istoggled", set())}
        if used: devices &= set(reverse_names[_unique(used, "Canonical toggles multiple devices")])
        valid = [d for d in devices if witness(trace[-1]["state_after"], goal, {"objects": [obj], "target": None, "device": d})]
        return {"objects": [obj], "target": None, "device": _unique(valid, "Ambiguous canonical toggle witness")}
    deliveries = []
    for t in trace:
        before, after = predicates(t["state_before"]), predicates(t["state_after"])
        released = {o for _, o in before.get("holds", set()) - after.get("holds", set())}
        deliveries += [(o, r) for o, r in sorted(after.get("inreceptacle", set()) - before.get("inreceptacle", set()))
                       if o in released and o in objset and (r, goal.target_type) in final.get("receptacletype", set())]
    needed = 2 if goal.family == "pick_two_obj_and_place" else 1
    objects = [o for o, _ in deliveries]
    targets = {r for _, r in deliveries}
    if not deliveries and goal.family in TREATMENTS:
        # Some original games retain a second inReceptacle fact while the
        # object is held. Treatment alone can then satisfy the unchanged goal.
        # Recover that existing witness; never insert a fictitious delivery.
        verb, _, effect, _ = TREATMENTS[goal.family]
        treated = set()
        for t in trace:
            if not t["action"].startswith(verb + " "): continue
            surface = t["action"][len(verb) + 1:].rsplit(" with ", 1)[0]
            before, after = predicates(t["state_before"]), predicates(t["state_after"])
            new_effect = {o for (o,) in after.get(effect, set()) - before.get(effect, set())}
            held = {o for _, o in before.get("holds", set())}
            treated |= new_effect & held & objset & set(reverse_names[surface])
        obj = _unique(treated, "Canonical no-delivery treatment object ambiguous")
        targets = {r for o, r in final.get("inreceptacle", set()) if o == obj
                   and (r, goal.target_type) in final.get("receptacletype", set())
                   and all((obj, r) in predicates(t[side]).get("inreceptacle", set())
                           for t in trace for side in ("state_before", "state_after"))}
        objects = [obj]
    target = _unique(targets, "Ambiguous canonical target from grounded effects or retained witness")
    if len(objects) != needed or len(set(objects)) != needed: raise RecipeRejected("Canonical delivery witness/cardinality ambiguous")
    device = None
    if goal.family in TREATMENTS:
        verb = TREATMENTS[goal.family][0]
        surfaces = [t["action"].rsplit(" with ", 1)[-1] for t in trace if t["action"].startswith(verb + " ")]
        device = _unique(reverse_names[_unique(surfaces, "Canonical treatment device ambiguous")], "Ambiguous device surface")
    return {"objects": objects, "target": target, "device": device}


def terminal_semantics_audit(trace, goal, assignment):
    """Describe simulator witnesses without calling retained membership a delivery."""
    final = predicates(trace[-1]["state_after"])
    held = {o for _, o in final.get("holds", set())}
    retained = assignment.get("target") is not None and all(
        (o, assignment["target"]) in predicates(t[side]).get("inreceptacle", set())
        for o in assignment["objects"] for t in trace for side in ("state_before", "state_after"))
    return {"benchmark_goal_satisfied_while_held": bool(
                goal.family != "look_at_obj_in_light" and held.intersection(assignment["objects"])
                and witness(trace[-1]["state_after"], goal, assignment)),
            "retained_target_membership": bool(retained),
            "interpretation": "Original PDDL goal semantics; retained containment while held is not evidence of a physical placement."}


def _check_trace(trace, costs, max_actions=24):
    if not trace or len(trace) > max_actions: raise RecipeRejected("Empty/oversized recipe")
    seen = {physical_key(trace[0]["state_before"])}
    previous = trace[0]["state_before"]
    for i, t in enumerate(trace):
        if t["state_before"] != previous: raise ReplayMismatch("Broken trace state chain")
        if i and t["observation"] != trace[i - 1]["next_observation"]: raise ReplayMismatch("Broken observation chain")
        if t["state_before"]["done"] or t["action"] not in t["admissible_commands"]: raise RecipeRejected("Invalid/post-terminal action")
        if information_only_action(t["action"]) or t["action"].startswith(("close ", "slice ")): raise RecipeRejected("Information/padding/unsupported recipe action")
        action_cost(t["action"], costs)
        key = physical_key(t["state_after"])
        if key in seen: raise RecipeRejected("Physical noop or exact loop")
        seen.add(key); previous = t["state_after"]
    last = trace[-1]
    if not last["won"] or not last["done"] or not previous["won"] or not previous["terminated"] or previous["truncated"]:
        raise RecipeRejected("Original environment did not terminate successfully")
    return len(trace), sum(action_cost(t["action"], costs) for t in trace)


def replay_own_trace(env, trace, budget, initial_observation):
    """Verify against this candidate's own trace, never the canonical endpoint."""
    current = budget.reset(env)
    if current.observation != initial_observation or current.canonical_state != trace[0]["state_before"]:
        raise ReplayMismatch("Independent reset differs")
    for i, old in enumerate(trace):
        if current.done or old["action"] not in current.admissible_commands: raise ReplayMismatch("Replay action no longer executable")
        after = budget.step(env, old["action"]); fresh = transition_record(current, old["action"], after, i)
        for key in ("step", "observation", "admissible_commands", "action", "next_observation", "reward", "done", "won", "state_before", "state_after"):
            if old[key] != fresh[key]: raise ReplayMismatch("Independent candidate replay differs: " + key)
        current = after
    if not current.won or not current.done or current.canonical_state["truncated"]: raise ReplayMismatch("Replay did not win original game")
    return {"independent_full_replay_verified": True, "terminal_success": True,
            "own_trace_digest": digest(trace), "comparison": "candidate_to_own_discovery_trace"}


def execute_recipe(env, budget, initial, goal, assignment, costs, max_actions=24):
    current = budget.reset(env)
    if current.canonical_state != initial["state"] or current.observation != initial["observation"]:
        raise ReplayMismatch("Recipe reset differs from frozen canonical initial state")
    names = entity_names(env); trace = []; seen = {physical_key(current.canonical_state)}
    eligible = any(assignment_key(a) == assignment_key(assignment) for a in eligible_assignments(current.canonical_state, goal))
    if not eligible: raise RecipeRejected("Ineligible grounded assignment")
    def act(action):
        nonlocal current
        if current.done: raise RecipeRejected("Early success or terminal state before intended assignment")
        if len(trace) >= max_actions: raise RecipeRejected("Recipe action limit")
        if information_only_action(action) or action not in current.admissible_commands: raise RecipeRejected("Inadmissible/information action: " + action)
        after = budget.step(env, action)
        record = transition_record(current, action, after, len(trace)); trace.append(record); current = after
        key = physical_key(after.canonical_state)
        if key in seen: raise RecipeRejected("Physical noop or exact loop")
        seen.add(key)
        if current.done and not witness(current.canonical_state, goal, assignment): raise RecipeRejected("Early success from a different witness")
    def navigate(rec):
        f = predicates(current.canonical_state)
        location = _unique({l for r, l in f.get("receptacleatlocation", ()) if r == rec}, "Ambiguous receptacle location")
        present = _unique({l for _, l in f.get("atlocation", ())}, "Ambiguous agent location")
        if location != present: act("go to " + names[rec])
    def open_access(rec):
        f = predicates(current.canonical_state)
        if (rec,) in f.get("openable", set()) and (rec,) not in f.get("opened", set()): act("open " + names[rec])
    def take(obj):
        f = predicates(current.canonical_state)
        if f.get("holds") or f.get("holdsany"): raise RecipeRejected("Single-slot inventory occupied")
        source = _unique({r for o, r in f.get("inreceptacle", ()) if o == obj}, "Ambiguous live source")
        if source == assignment.get("target"): raise RecipeRejected("Object already in selected target")
        navigate(source); open_access(source); act("take " + names[obj] + " from " + names[source])
    def place(obj, target):
        if {o for _, o in predicates(current.canonical_state).get("holds", ())} != {obj}: raise RecipeRejected("Holding wrong object")
        navigate(target); open_access(target); act("move " + names[obj] + " to " + names[target])
    for obj in assignment["objects"]:
        if current.done: break
        take(obj)
        if current.done: break
        if goal.family in TREATMENTS:
            verb, cap, effect, _ = TREATMENTS[goal.family]; device = assignment["device"]
            if (obj,) in predicates(current.canonical_state).get(effect, set()): raise RecipeRejected("Unnecessary repeated treatment")
            navigate(device)
            if current.done: break
            act(verb + " " + names[obj] + " with " + names[device])
            if (obj,) not in predicates(current.canonical_state).get(effect, set()): raise RecipeRejected("Treatment failed on intended object")
        elif goal.family == "look_at_obj_in_light":
            device = assignment["device"]
            rec = _unique({r for o, r in predicates(current.canonical_state).get("inreceptacle", ()) if o == device}, "Ambiguous toggle source")
            navigate(rec)
            if not current.done:
                if (device,) in predicates(current.canonical_state).get("istoggled", set()): raise RecipeRejected("Useless repeated toggle")
                act("use " + names[device])
            break
        if not current.done: place(obj, assignment["target"])
    _check_trace(trace, costs, max_actions)
    if not witness(current.canonical_state, goal, assignment): raise RecipeRejected("Wrong intended terminal witness")
    return trace


def _task_history(trace):
    return [physical_key(t["state_after"]) for t in trace]


def candidate_relation(assignment, canonical_assignment, trace, canonical_trace):
    changed = assignment_key(assignment) != assignment_key(canonical_assignment)
    physical_changed = _task_history(trace) != _task_history(canonical_trace)
    order_only = (set(assignment["objects"]) == set(canonical_assignment["objects"])
                  and tuple(assignment["objects"]) != tuple(canonical_assignment["objects"])
                  and assignment.get("target") == canonical_assignment.get("target")
                  and assignment.get("device") == canonical_assignment.get("device"))
    return {"meaningful_assignment_change": changed and physical_changed and not order_only,
            "delivery_order_only": changed and physical_changed and order_only,
            "physical_history_changed": physical_changed, "assignment_changed": changed}


def validate_successful_pool(pool):
    """New outcome contract. Deliberately never invokes macro.validate_pool."""
    if pool.get("kind") != "successful_path_candidate_pool": raise ValueError("Wrong pool schema")
    candidates = pool["candidates"]
    if not 1 <= len(candidates) <= 8 or candidates[0]["candidate_id"] != pool["canonical_candidate_id"]: raise ValueError("Missing original / oversized pool")
    if candidates[0]["transitions"] != pool["canonical_source_row"]["transitions"]: raise ValueError("Canonical source content changed")
    if digest(pool["canonical_source_row"]) != pool["canonical_source_row_digest"]: raise ValueError("Canonical source row changed")
    source = pool["canonical_source_row"]
    for key in ("gamefile", "seed", "split", "initial_observation", "task"):
        if source[key] != pool[key]: raise ValueError("Pool identity differs from canonical source: " + key)
    line = pool["canonical_source_line_utf8"].encode("utf-8")
    if hashlib.sha256(line).hexdigest() != pool["canonical_source_line_sha256"] or json.loads(line) != source:
        raise ValueError("Original canonical byte content differs")
    if pool["operator_costs"] != {verb: cost for verb, cost in OPERATOR_MAP.values()}:
        raise ValueError("Cost table differs from locked audited domain")
    if pool["source_hashes"]["pddl_domain_sha256"] != LOCKED_DOMAIN_SHA256:
        raise ValueError("Pool domain differs from locked training domain")
    if hashlib.sha256(pool["original_problem"].encode()).hexdigest() != pool["source_hashes"]["pddl_problem_sha256"]:
        raise ValueError("Original problem hash differs")
    meta_line = pool["original_metadata_utf8"].encode("utf-8")
    if hashlib.sha256(meta_line).hexdigest() != pool["source_hashes"]["metadata_sha256"]:
        raise ValueError("Original metadata hash differs")
    goal = parse_goal(pool["original_problem"], json.loads(meta_line))
    if goal != GoalSpec(**pool["goal_spec"]) or goal.family != pool["family"] or goal.parsed_goal != pool["original_goal"]:
        raise ValueError("Family/goal schema differs from parsed original problem")
    if digest(pool["entity_names"]) != pool["entity_names_digest"]: raise ValueError("Grounding map differs")
    baseline_assignment = derive_canonical_assignment(candidates[0]["transitions"], goal, pool["entity_names"])
    if assignment_key(baseline_assignment) != assignment_key(candidates[0]["assignment"]): raise ValueError("Canonical assignment differs from grounded trace")
    seen = set()
    for c in candidates:
        actions = [t["action"] for t in c["transitions"]]
        if c["candidate_id"] != digest(actions) or c["candidate_id"] in seen: raise ValueError("Duplicate or incorrect action digest")
        seen.add(c["candidate_id"])
        if list(_check_trace(c["transitions"], pool["operator_costs"])) != pool["canonical_length_cost"]: raise ValueError("Outside canonical length/cost stratum")
        if c["transitions"][0]["state_before"] != pool["initial_state"] or c["transitions"][0]["observation"] != pool["initial_observation"]: raise ValueError("Different initial state/observation")
        if not c["evidence"].get("independent_full_replay_verified") or not c["evidence"].get("terminal_success"): raise ValueError("Missing independent original-goal replay")
        if c["evidence"].get("own_trace_digest") != digest(c["transitions"]): raise ValueError("Replay evidence no longer matches trace")
        if not witness(c["transitions"][-1]["state_after"], goal, c["assignment"]): raise ValueError("Candidate misses intended original-goal witness")
        actual_assignment = derive_canonical_assignment(c["transitions"], goal, pool["entity_names"])
        if assignment_key(actual_assignment) != assignment_key(c["assignment"]): raise ValueError("Claimed assignment differs from grounded actions/effects")
        if c.get("terminal_semantics_audit") != terminal_semantics_audit(c["transitions"], goal, actual_assignment):
            raise ValueError("Terminal simulator-semantics audit differs")
        relation = candidate_relation(c["assignment"], baseline_assignment, c["transitions"], candidates[0]["transitions"])
        for key in ("meaningful_assignment_change", "delivery_order_only"):
            if bool(c.get(key, False)) != relation[key]: raise ValueError("Candidate relation flag differs: " + key)
        if c is not candidates[0] and not (relation["assignment_changed"] and relation["physical_history_changed"]):
            raise ValueError("Alternative has no meaningful physical assignment/order change")
    return True


def build_successful_pool(row, game_data, metadata, env, budget, *, source_line_bytes=None,
                          game_bytes=None, metadata_bytes=None):
    """Prototype single-task API; caller must enforce frozen cohort/adoption gate."""
    if getattr(env, "expert_type", None) is not None: raise ValueError("All passes require expert_type=None")
    if budget.max_steps > 100000 or budget.max_seconds > 3600: raise ValueError("Registered CPU budget exceeded")
    if row.get("split") != "train" or not row.get("success") or not row.get("replay_verified"): raise ValueError("Unverified/non-training canonical seed")
    if row.get("task", row["initial_observation"]) != row["initial_observation"]: raise ValueError("Canonical task must be its actual initial observation")
    if source_line_bytes is None: raise ValueError("Original canonical JSONL bytes are required")
    if getattr(env, "seed", None) != row["seed"] or getattr(env, "max_steps", None) != 100: raise ValueError("Environment seed/episode limit differs")
    if Path(getattr(env, "gamefile", "")).resolve() != Path(row["gamefile"]).resolve(): raise ValueError("Environment game differs from canonical")
    start_all_steps = budget.steps
    game_bytes = Path(row["gamefile"]).read_bytes() if game_bytes is None else game_bytes
    metadata_bytes = Path(row["gamefile"]).with_name("traj_data.json").read_bytes() if metadata_bytes is None else metadata_bytes
    if json.loads(game_bytes) != game_data or json.loads(metadata_bytes) != metadata: raise ValueError("Source bytes differ from parsed input")
    if source_line_bytes is not None and json.loads(source_line_bytes) != row: raise ValueError("Canonical source line differs from parsed row")
    before = digest(row); goal, costs, hashes = audit_game(game_data, metadata)
    canonical = row["transitions"]
    stratum = _check_trace(canonical, costs)
    evidence = replay_own_trace(env, canonical, budget, row["initial_observation"])
    names = entity_names(env); assignment = derive_canonical_assignment(canonical, goal, names)
    if not witness(canonical[-1]["state_after"], goal, assignment): raise RecipeRejected("Canonical intended witness failed")
    baseline_id = digest([t["action"] for t in canonical])
    initial = {"state": canonical[0]["state_before"], "observation": row["initial_observation"]}
    proposals, enumeration = enumerate_proposals(initial["state"], goal, assignment, row["gamefile"])
    pool = {"kind": "successful_path_candidate_pool", "schema_version": 1, "method": "outcome_constrained_successful_path_selection",
            "gamefile": row["gamefile"], "seed": row["seed"], "split": "train", "task": row.get("task", row["initial_observation"]),
            "initial_observation": initial["observation"], "initial_state": copy.deepcopy(initial["state"]),
            "canonical_candidate_id": baseline_id, "canonical_source_row": copy.deepcopy(row), "canonical_source_row_digest": before,
            "canonical_source_line_sha256": hashlib.sha256(source_line_bytes).hexdigest() if source_line_bytes is not None else None,
            "canonical_source_line_utf8": source_line_bytes.decode("utf-8") if source_line_bytes is not None else None,
            "canonical_length_cost": list(stratum), "operator_costs": costs, "original_goal": goal.parsed_goal,
            "original_problem": game_data["pddl_problem"], "original_metadata_utf8": metadata_bytes.decode("utf-8"),
            "entity_names": names, "entity_names_digest": digest(names),
            "goal_spec": {"family": goal.family, "object_type": goal.object_type, "target_type": goal.target_type,
                          "toggle_type": goal.toggle_type, "parsed_goal": goal.parsed_goal},
            "family": goal.family, "source_hashes": {**hashes, "game_sha256": hashlib.sha256(game_bytes).hexdigest(),
                                                      "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest()},
            "candidates": [{"candidate_id": baseline_id, "origin": "unchanged_native_planner", "assignment": assignment,
                            "transitions": copy.deepcopy(canonical), "evidence": evidence, "meaningful_assignment_change": False,
                            "terminal_semantics_audit": terminal_semantics_audit(canonical, goal, assignment)}],
            "verifier_version": 2,
            "enumeration": enumeration, "attempts": [], "contract": "Same original success goal and canonical length/cost; terminal states may differ."}
    identities = {baseline_id}; start_steps = budget.steps
    for proposal in proposals:  # Fixed first seven, never replenish after rejection.
        audit = {"assignment": proposal, "accepted": False}; attempt_start = budget.steps
        try:
            trace = execute_recipe(env, budget, initial, goal, proposal, costs)
            audit["length_cost"] = list(_check_trace(trace, costs))
            audit["discovery_success"] = True
            if tuple(audit["length_cost"]) != stratum: raise RecipeRejected("Successful path outside canonical length/cost stratum")
            identity = digest([t["action"] for t in trace])
            if identity in identities: raise RecipeRejected("Duplicate canonical/candidate actions")
            if assignment_key(proposal) == assignment_key(assignment) or _task_history(trace) == _task_history(canonical): raise RecipeRejected("Assignment unchanged or alias-only physical history")
            evidence = replay_own_trace(env, trace, budget, initial["observation"])
            relation = candidate_relation(proposal, assignment, trace, canonical)
            pool["candidates"].append({"candidate_id": identity, "origin": "grounded_assignment_recipe", "assignment": proposal,
                                      "transitions": trace, "evidence": evidence, **relation,
                                      "terminal_semantics_audit": terminal_semantics_audit(trace, goal, proposal),
                                      "terminal_state_differs": trace[-1]["state_after"] != canonical[-1]["state_after"]})
            identities.add(identity); audit.update(accepted=True, candidate_id=identity)
        except RecipeRejected as error:
            audit["rejection_reason"] = str(error)
        audit["environment_steps_including_replay"] = budget.steps - attempt_start
        pool["attempts"].append(audit)
    pool["recipe_acquisition_steps_including_replay"] = budget.steps - start_steps
    pool["all_acquisition_steps_including_canonical_and_replay"] = budget.steps - start_all_steps
    if digest(row) != before: raise RuntimeError("Canonical input mutated")
    validate_successful_pool(pool)
    return pool


def student_messages(pool, candidate_index, step_index):
    """Only actual causal fields cross from teacher search to student prompts."""
    trace = pool["candidates"][candidate_index]["transitions"]; current = trace[step_index]
    history = [{"observation": t["observation"], "action": t["action"]} for t in trace[:step_index]]
    return messages(pool["task"], history, current["observation"], current["admissible_commands"])


def summarize_acquisition(pools, expected_gamefiles, technical_errors=(), stop_reason="complete"):
    ids = [p["gamefile"] for p in pools]; by_family = Counter(); errors = list(technical_errors)
    config = json.loads((Path(__file__).resolve().parents[1] / "configs/successful_path_selection_candidate_v1.json").read_text(encoding="utf-8"))
    registered = config["fixed_training_cohort"]["frozen_gamefiles"]
    seed = config["fixed_training_cohort"]["reset_seed"]
    if list(expected_gamefiles) != registered or len(registered) != 60 or len(set(registered)) != 60:
        errors.append("Requested cohort differs from the frozen 60 training tasks")
    if any(p.get("seed") != seed or p.get("split") != "train" or p.get("gamefile") not in registered for p in pools):
        errors.append("Pool task/seed/split differs from frozen training contract")
    for pool in pools:
        try:
            validate_successful_pool(pool)
        except Exception as error:
            errors.append(f"Pool validation failed for {pool.get('gamefile')}: {error}")
            continue
        if pool["family"] != "pick_two_obj_and_place" and any(c.get("meaningful_assignment_change") for c in pool["candidates"][1:]): by_family[pool["family"]] += 1
    complete = ids == registered and len(ids) == len(set(ids)) and not errors and stop_reason == "complete"
    return {"complete": complete, "planned_tasks": len(registered), "canonical_tasks": len(pools),
            "technical_errors": errors, "stop_reason": stop_reason,
            "eligible_non_two_object_tasks_by_family": dict(by_family),
            "coverage_gate_passed": complete and len(expected_gamefiles) == 60 and sum(by_family.values()) >= 10 and sum(n >= 2 for n in by_family.values()) >= 3,
            "interpretation": "Outcome-compatible finite pools are not exact-state BRP and are not evidence of downstream learning benefit."}


if __name__ == "__main__":
    raise SystemExit("Prototype only: no search/model CLI before separate adoption. Run the CPU toy tests.")

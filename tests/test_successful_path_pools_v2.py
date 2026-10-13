"""CPU fixtures only. Passing these is not an ALFWorld feasibility result."""
import copy
import hashlib
import itertools
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from gnrl.alfworld_env import ReplayMismatch, StepResult
from gnrl.macro_project import SearchBudget, BudgetExceeded
from gnrl import successful_path_pools_v2 as m


def metadata(family):
    return {"task_type": family, "pddl_params": {"object_target": "Apple", "parent_target": "" if family == "look_at_obj_in_light" else "Box",
                                                "toggle_target": "Lamp" if family == "look_at_obj_in_light" else "",
                                                "object_sliced": False, "mrecep_target": ""}}


def problem(family):
    if family == "look_at_obj_in_light":
        goal = """(and (exists (?d - object ?r - receptacle ?a - agent ?l - location)
                    (and (objectType ?d LampType) (toggleable ?d) (isToggled ?d)
                         (receptacleAtLocation ?r ?l) (atLocation ?a ?l) (inReceptacle ?d ?r)))
                       (exists (?o - object ?a - agent) (and (objectType ?o AppleType) (holds ?a ?o))))"""
    elif family == "pick_two_obj_and_place":
        goal = """(and (exists (?r - receptacle) (exists (?o1 - object)
            (and (objectType ?o1 AppleType) (receptacleType ?r BoxType) (inReceptacle ?o1 ?r)
                 (exists (?o2 - object) (and (not (= ?o1 ?o2)) (objectType ?o2 AppleType)
                    (receptacleType ?r BoxType) (inReceptacle ?o2 ?r)))))))"""
    else:
        terms = ""
        if family in m.TREATMENTS:
            _, cap, effect, _ = m.TREATMENTS[family]
            terms = f"({cap} ?o) ({effect} ?o)"
        goal = f"(and (exists (?r - receptacle) (exists (?o - object) (and (objectType ?o AppleType) (receptacleType ?r BoxType) (inReceptacle ?o ?r) {terms}))))"
    return f"(define (problem toy) (:domain alfred) (:goal {goal}))"


def toy_domain():
    ops = []
    for op, (_, cost) in m.OPERATOR_MAP.items():
        effect = f"(increase (total-cost) {cost})" if cost else ""
        ops.append(f"(:action {op} :parameters () :precondition () :effect (and (changed) {effect}))")
    return "(define (domain toy) " + " ".join(ops) + ")"


GRAMMAR = "\n".join(f'action {op} {{ template :: "{template}"; }}' for op, template in m.TEMPLATES.items())
COSTS = {verb: cost for verb, cost in m.OPERATOR_MAP.values()}


class ToyEnv:
    """Explicit state machine; original goal is fixed independently of recipes."""
    expert_type = None
    seed = 20261012
    max_steps = 100
    gamefile = "toy/game.tw-pddl"

    def __init__(self, family="pick_and_place_simple"):
        self.goal = m.parse_goal(problem(family), metadata(family))
        self.family = family
        self.names = {"secret_obj1": "apple 1", "secret_obj2": "apple 2", "secret_obj3": "apple 3",
                      "source1": "table 1", "source2": "table 2", "target1": "box 1", "target2": "box 2",
                      "treat1": "device 1", "treat2": "device 2", "lamp1": "lamp 1", "lamp2": "lamp 2"}
        self.env = SimpleNamespace(_entity_infos={k: SimpleNamespace(name=v) for k, v in self.names.items()})
        self.initial = {("atlocation", "agent", "l0"), ("cancontain", "boxtype", "appletype")}
        for obj, source in (("secret_obj1", "source1"), ("secret_obj2", "source2"), ("secret_obj3", "source2")):
            self.initial |= {("objecttype", obj, "appletype"), ("pickupable", obj), ("inreceptacle", obj, source)}
            for cap in ("cleanable", "heatable", "coolable"): self.initial.add((cap, obj))
        for i, rec in enumerate(("source1", "source2", "target1", "target2", "treat1", "treat2"), 1):
            self.initial.add(("receptacleatlocation", rec, "l" + str(i)))
            self.initial.add(("receptacletype", rec, "boxtype" if rec.startswith("target") else "tabletype"))
        kinds = m.TREATMENTS.get(family, (None, None, None, {"sinkbasintype"}))[3]
        for device in ("treat1", "treat2"):
            self.initial.add(("receptacletype", device, sorted(kinds)[0]))
        for lamp, rec in (("lamp1", "treat1"), ("lamp2", "treat2")):
            self.initial |= {("objecttype", lamp, "lamptype"), ("toggleable", lamp), ("inreceptacle", lamp, rec)}
        self.reset()

    def _raw(self):
        return sorted([[f[0], [[a, "symbol"] for a in f[1:]]] for f in self.facts], key=lambda f: json.dumps(f))

    def _won(self):
        # Independent toy reward logic: do not call the witness under test.
        f = self.facts; objects = {t[1] for t in f if t[0] == "objecttype" and t[2] == "appletype"}
        if self.family == "look_at_obj_in_light":
            if not any(("holds", "agent", o) in f for o in objects): return False
            lamps = {t[1] for t in f if t[0] == "objecttype" and t[2] == "lamptype"}
            for lamp in lamps:
                if ("toggleable", lamp) not in f or ("istoggled", lamp) not in f: continue
                for relation in f:
                    if relation[:2] != ("inreceptacle", lamp): continue
                    rec = relation[2]
                    if any(t[:2] == ("receptacleatlocation", rec) and ("atlocation", "agent", t[2]) in f for t in f): return True
            return False
        for target in ("target1", "target2"):
            placed = {o for o in objects if ("inreceptacle", o, target) in f}
            if self.family == "pick_two_obj_and_place":
                if len(placed) >= 2: return True
            elif self.family in m.TREATMENTS:
                _, cap, effect, _ = m.TREATMENTS[self.family]
                if any((cap, o) in f and (effect, o) in f for o in placed): return True
            elif placed: return True
        return False

    def _commands(self):
        commands = {"look", "help", "inventory"}; f = self.facts
        loc = next(t[2] for t in f if t[0] == "atlocation")
        holds = {t[2] for t in f if t[0] == "holds"}
        recs = {t[1]: t[2] for t in f if t[0] == "receptacleatlocation"}
        for rec, where in recs.items():
            commands.add("go to " + self.names[rec])
            if where != loc: continue
            if ("openable", rec) in f and ("opened", rec) not in f:
                commands.add("open " + self.names[rec]); accessible = False
            else: accessible = True
            for obj in self.names:
                if not holds and accessible and ("pickupable", obj) in f and ("inreceptacle", obj, rec) in f:
                    commands.add("take " + self.names[obj] + " from " + self.names[rec])
                if obj in holds and accessible and ("receptacletype", rec, "boxtype") in f and ("cancontain", "boxtype", "appletype") in f:
                    commands.add("move " + self.names[obj] + " to " + self.names[rec])
                if ("toggleable", obj) in f and ("inreceptacle", obj, rec) in f:
                    commands.add("use " + self.names[obj])
                for verb, cap, _, kinds in m.TREATMENTS.values():
                    if obj in holds and (cap, obj) in f and any(("receptacletype", rec, kind) in f for kind in kinds):
                        commands.add(verb + " " + self.names[obj] + " with " + self.names[rec])
        return sorted(commands)

    def _result(self, obs):
        won = self._won(); truncated = self.moves >= 100 and not won
        canonical = {"facts": self._raw(), "moves": self.moves, "elapsed_steps": self.moves, "max_steps": 100,
                     "score": float(won), "reward": float(won), "won": won, "lost": False,
                     "terminated": won, "truncated": truncated, "done": won or truncated,
                     "admissible_commands": self._commands()}
        return StepResult(obs, canonical["admissible_commands"], None, float(won), won or truncated, won, canonical)

    def reset(self):
        self.facts = set(self.initial); self.moves = 0; self.history = []
        self.current = self._result("Observed room and task, without hidden object coordinates.")
        return self.current

    def step(self, action):
        if action not in self.current.admissible_commands or self.current.done: raise RuntimeError("Toy invalid action")
        reverse = {v: k for k, v in self.names.items()}
        if action.startswith("go to "):
            rec = reverse[action[6:]]; loc = next(t[2] for t in self.facts if t[:2] == ("receptacleatlocation", rec))
            self.facts = {t for t in self.facts if t[0] != "atlocation"}; self.facts.add(("atlocation", "agent", loc))
        elif action.startswith("open "): self.facts.add(("opened", reverse[action[5:]]))
        elif action.startswith("take "):
            obj, rec = map(reverse.__getitem__, action[5:].split(" from "))
            self.facts.remove(("inreceptacle", obj, rec)); self.facts |= {("holds", "agent", obj), ("holdsany", "agent")}
        elif action.startswith("move "):
            obj, rec = map(reverse.__getitem__, action[5:].split(" to "))
            self.facts.remove(("holds", "agent", obj)); self.facts.remove(("holdsany", "agent")); self.facts.add(("inreceptacle", obj, rec))
        elif action.startswith("use "):
            obj = reverse[action[4:]]; self.facts.add(("istoggled", obj))
            if ("ison", obj) in self.facts: self.facts.remove(("ison", obj))
            else: self.facts.add(("ison", obj))
        elif any(action.startswith(verb + " ") for verb in ("heat", "cool", "clean")):
            verb, rest = action.split(" ", 1); obj, _ = map(reverse.__getitem__, rest.split(" with "))
            effect = {"heat": "ishot", "cool": "iscool", "clean": "isclean"}[verb]
            self.facts.add((effect, obj))
        else: self.facts.add(("checked", "agent"))
        self.moves += 1; self.history.append(action); self.current = self._result("Actually executed: " + " / ".join(self.history))
        return self.current


def canonical_fixture(family="pick_and_place_simple"):
    env = ToyEnv(family); first = env.reset()
    assignment = {"objects": ["secret_obj1", "secret_obj2"] if family == "pick_two_obj_and_place" else ["secret_obj1"],
                  "target": None if family == "look_at_obj_in_light" else "target1",
                  "device": "lamp1" if family == "look_at_obj_in_light" else "treat1" if family in m.TREATMENTS else None}
    trace = m.execute_recipe(env, SearchBudget(), {"state": first.canonical_state, "observation": first.observation}, env.goal, assignment, COSTS)
    for t in trace: t["expert_action"] = "untouched native planner annotation"
    row = {"kind": "trajectory", "gamefile": "toy/game.tw-pddl", "seed": 20261012, "split": "train", "success": True,
           "replay_verified": True, "reset_replay_verified": True, "initial_observation": first.observation,
           "task": first.observation, "transitions": trace}
    return env, row, assignment


def make_pool(family="pick_and_place_simple"):
    env, row, assignment = canonical_fixture(family)
    domain = toy_domain(); data = {"pddl_domain": domain, "pddl_problem": problem(family), "grammar": GRAMMAR}
    source = (json.dumps(row, ensure_ascii=False) + "\n").encode()
    with patch.object(m, "LOCKED_DOMAIN_SHA256", hashlib.sha256(domain.encode()).hexdigest()):
        pool = m.build_successful_pool(row, data, metadata(family), env, SearchBudget(), source_line_bytes=source,
                                       game_bytes=json.dumps(data).encode(), metadata_bytes=json.dumps(metadata(family)).encode())
    return pool, row, source


class SuccessfulPathPrototypeTests(unittest.TestCase):
    def setUp(self):
        # The toy domain is never claimed to be the locked production domain.
        self.domain_patch = patch.object(m, "LOCKED_DOMAIN_SHA256", hashlib.sha256(toy_domain().encode()).hexdigest())
        self.domain_patch.start()
        self.addCleanup(self.domain_patch.stop)

    def test_retained_membership_treatment_terminal_preserves_canonical(self):
        family = "pick_clean_then_place_in_recep"
        env = ToyEnv(family)
        env.initial.add(("inreceptacle", "secret_obj1", "target1"))
        before = first = env.reset(); trace = []
        for action in ("go to table 1", "take apple 1 from table 1", "go to device 1", "clean apple 1 with device 1"):
            after = env.step(action)
            trace.append(m.transition_record(before, action, after, len(trace)))
            before = after
        self.assertTrue(after.won)
        row = {"kind": "trajectory", "gamefile": env.gamefile, "seed": env.seed, "split": "train",
               "success": True, "replay_verified": True, "reset_replay_verified": True,
               "initial_observation": first.observation, "task": first.observation, "transitions": trace}
        original = (json.dumps(row) + "\n").encode()
        data = {"pddl_domain": toy_domain(), "pddl_problem": problem(family), "grammar": GRAMMAR}
        pool = m.build_successful_pool(row, data, metadata(family), env, SearchBudget(),
            source_line_bytes=original, game_bytes=json.dumps(data).encode(), metadata_bytes=json.dumps(metadata(family)).encode())
        canonical = pool["candidates"][0]
        self.assertEqual(pool["canonical_source_line_utf8"].encode(), original)
        self.assertEqual(canonical["transitions"], trace)
        self.assertEqual(pool["canonical_length_cost"], [4, 8])
        self.assertTrue(canonical["terminal_semantics_audit"]["retained_target_membership"])
        self.assertTrue(canonical["terminal_semantics_audit"]["benchmark_goal_satisfied_while_held"])
        self.assertFalse(canonical["meaningful_assignment_change"])
        self.assertTrue(all("secret_obj1" not in a["objects"] for a in m.eligible_assignments(first.canonical_state, env.goal)))
        self.assertEqual(len(pool["candidates"]), 1)
        corrupted = copy.deepcopy(pool)
        corrupted["candidates"][0]["terminal_semantics_audit"]["retained_target_membership"] = False
        with self.assertRaises(ValueError): m.validate_successful_pool(corrupted)
        # Removing the typed retained witness or the actual treatment effect
        # must not turn this fallback into an arbitrary terminal-object guess.
        for predicate in ("inreceptacle", "isclean"):
            broken = copy.deepcopy(trace)
            broken[-1]["state_after"]["facts"] = [f for f in broken[-1]["state_after"]["facts"]
                if not (f[0] == predicate and f[1][0][0] == "secret_obj1")]
            with self.assertRaises(m.RecipeRejected):
                m.derive_canonical_assignment(broken, env.goal, env.names)

    def test_all_six_original_goal_templates_and_live_recipes(self):
        for family in sorted(m.FAMILIES):
            with self.subTest(family=family):
                env, row, assignment = canonical_fixture(family)
                self.assertTrue(row["transitions"][-1]["won"])
                self.assertTrue(m.witness(row["transitions"][-1]["state_after"], env.goal, assignment))
                self.assertEqual(m.parse_goal(problem(family), metadata(family)).family, family)
                pool, _, _ = make_pool(family)
                self.assertTrue(m.validate_successful_pool(pool))

    def test_goal_rejects_unknown_clause_and_is_on_substitution(self):
        for malformed in (problem("look_at_obj_in_light").replace("isToggled", "isOn"),
                          problem("look_at_obj_in_light").replace("(toggleable ?d)", "(toggleable ?d) (checked ?r)")):
            with self.assertRaises(m.UnsupportedTask): m.parse_goal(malformed, metadata("look_at_obj_in_light"))
        bad = problem("pick_two_obj_and_place").replace("(inReceptacle ?o2 ?r)", "(inReceptacle ?o2 ?r2)")
        with self.assertRaises(m.UnsupportedTask): m.parse_goal(bad, metadata("pick_two_obj_and_place"))

    def test_cost_audit_rejects_changed_and_conditional_costs(self):
        self.assertEqual(m.parse_operator_costs(toy_domain(), GRAMMAR), COSTS)
        with self.assertRaises(m.UnsupportedTask): m.parse_operator_costs(toy_domain().replace("(increase (total-cost) 1)", "(increase (total-cost) 7)", 1), GRAMMAR)
        with self.assertRaises(m.UnsupportedTask): m.parse_operator_costs(toy_domain().replace("(increase (total-cost) 1)", "(when (condition) (increase (total-cost) 1))", 1), GRAMMAR)
        with self.assertRaises(m.UnsupportedTask): m.audit_game({"pddl_domain": toy_domain() + " ", "grammar": GRAMMAR, "pddl_problem": problem("pick_and_place_simple")}, metadata("pick_and_place_simple"))

    def test_treatment_holds_object_without_appliance_open_close_put(self):
        for family in m.TREATMENTS:
            env, row, _ = canonical_fixture(family)
            actions = [t["action"] for t in row["transitions"]]
            self.assertFalse(any(a.startswith(("open ", "close ")) for a in actions))
            self.assertEqual(sum(a.startswith("move ") for a in actions), 1)
            verb, _, _, _ = m.TREATMENTS[family]
            t = next(t for t in row["transitions"] if t["action"].startswith(verb + " "))
            self.assertIn(("agent", "secret_obj1"), m.predicates(t["state_before"])["holds"])

    def test_look_uses_is_toggled_even_if_device_ends_off(self):
        env = ToyEnv("look_at_obj_in_light"); env.initial.add(("ison", "lamp1")); first = env.reset()
        a = {"objects": ["secret_obj1"], "target": None, "device": "lamp1"}
        trace = m.execute_recipe(env, SearchBudget(), {"state": first.canonical_state, "observation": first.observation}, env.goal, a, COSTS)
        facts = m.predicates(trace[-1]["state_after"])
        self.assertIn(("lamp1",), facts["istoggled"]); self.assertNotIn(("lamp1",), facts.get("ison", set()))
        self.assertTrue(trace[-1]["won"])

    def test_two_objects_require_same_receptacle_instance_and_distinct_ids(self):
        env, row, a = canonical_fixture("pick_two_obj_and_place"); state = copy.deepcopy(row["transitions"][-1]["state_after"])
        self.assertTrue(m.witness(state, env.goal, a))
        for f in state["facts"]:
            if f[0] == "inreceptacle" and f[1][0][0] == "secret_obj2": f[1][1][0] = "target2"
        self.assertFalse(m.witness(state, env.goal, a))
        self.assertFalse(m.witness(row["transitions"][-1]["state_after"], env.goal, dict(a, objects=["secret_obj1", "secret_obj1"])))

    def test_inventory_containment_and_device_typing(self):
        env = ToyEnv("pick_heat_then_place_in_recep")
        env.initial.remove(("cancontain", "boxtype", "appletype"))
        self.assertEqual(list(m.eligible_assignments(env.reset().canonical_state, env.goal)), [])
        env = ToyEnv("pick_heat_then_place_in_recep"); first = env.reset()
        bad = {"objects": ["secret_obj1"], "target": "target1", "device": "source1"}
        with self.assertRaises(m.RecipeRejected): m.execute_recipe(env, SearchBudget(), {"state": first.canonical_state, "observation": first.observation}, env.goal, bad, COSTS)
        env.initial |= {("holds", "agent", "secret_obj2"), ("holdsany", "agent")}; first = env.reset()
        good = dict(bad, device="treat1")
        with self.assertRaises(m.RecipeRejected): m.execute_recipe(env, SearchBudget(), {"state": first.canonical_state, "observation": first.observation}, env.goal, good, COSTS)

    def test_pool_accepts_different_terminal_state_preserves_canonical_bytes(self):
        pool, row, source = make_pool()
        self.assertGreater(len(pool["candidates"]), 1)
        self.assertTrue(any(c.get("terminal_state_differs") for c in pool["candidates"][1:]))
        self.assertEqual(pool["canonical_source_line_utf8"].encode(), source)
        self.assertEqual(pool["candidates"][0]["transitions"], row["transitions"])
        self.assertTrue(all(c["transitions"][0]["expert_action"] == "untouched native planner annotation" for c in pool["candidates"][:1]))
        for c in pool["candidates"]:
            self.assertEqual(list(m._check_trace(c["transitions"], COSTS)), pool["canonical_length_cost"])
        pool["canonical_length_cost"][1] += 1
        with self.assertRaises(ValueError): m.validate_successful_pool(pool)

    def test_assignment_budget_is_fixed_before_live_success(self):
        env, row, a = canonical_fixture()
        first = row["transitions"][0]["state_before"]
        left, audit = m.enumerate_proposals(first, env.goal, a, "game", limit=7)
        right, _ = m.enumerate_proposals(first, env.goal, a, "game", limit=7)
        self.assertEqual(left, right); self.assertLessEqual(len(left), 7)
        self.assertTrue(audit["no_yield_dependent_replenishment"])
        with self.assertRaises(ValueError): m.enumerate_proposals(first, env.goal, a, "game", limit=8)

    def test_noops_information_padding_and_budget_are_rejected(self):
        env, row, _ = canonical_fixture()
        trace = copy.deepcopy(row["transitions"]); trace[0]["action"] = "look"
        with self.assertRaises(m.RecipeRejected): m._check_trace(trace, COSTS)
        trace = copy.deepcopy(row["transitions"]); trace[0]["state_after"] = trace[0]["state_before"]
        with self.assertRaises(m.RecipeRejected): m._check_trace(trace, COSTS)
        with self.assertRaises(BudgetExceeded): m.replay_own_trace(env, row["transitions"], SearchBudget(max_steps=1), row["initial_observation"])

    def test_fresh_replay_mismatch_is_technical_not_proposal_rejection(self):
        env, row, _ = canonical_fixture(); trace = copy.deepcopy(row["transitions"])
        trace[1]["next_observation"] = "copied counterfactual suffix"
        with self.assertRaises(ReplayMismatch): m.replay_own_trace(env, trace, SearchBudget(), row["initial_observation"])
        report = m.summarize_acquisition([], ["toy"], technical_errors=["reset mismatch"])
        self.assertFalse(report["complete"]); self.assertFalse(report["coverage_gate_passed"])

    def test_student_prompt_does_not_include_teacher_metadata(self):
        pool, _, _ = make_pool(); pool["private_future"] = "CANARY_FUTURE"
        text = json.dumps(m.student_messages(pool, 1, 1))
        self.assertNotIn("secret_obj", text); self.assertNotIn("CANARY_FUTURE", text)
        self.assertNotIn("assignment", text); self.assertNotIn("state_after", text)

    def test_validator_rejects_family_goal_device_cost_and_relation_injections(self):
        pool, _, _ = make_pool("pick_heat_then_place_in_recep")
        def bad_family(p): p["family"] = "invented_family"
        def bad_goal(p): p["original_goal"] = ["and", ["false"]]
        def bad_device(p): p["candidates"][1]["assignment"]["device"] = "nonexistent_device"
        def bad_cost(p): p["operator_costs"]["go to"] = 2; p["canonical_length_cost"][1] += 3
        def bad_relation(p): p["candidates"][1]["meaningful_assignment_change"] = False
        for mutate in (bad_family, bad_goal, bad_device, bad_cost, bad_relation):
            damaged = copy.deepcopy(pool); mutate(damaged)
            with self.subTest(mutation=mutate.__name__):
                with self.assertRaises(ValueError): m.validate_successful_pool(damaged)
        self.assertFalse(m.summarize_acquisition([pool], [pool["gamefile"]])["complete"])

    def test_live_won_is_required_and_other_goal_witness_is_rejected(self):
        env = ToyEnv(); env._won = lambda: False; first = env.reset()
        assignment = {"objects": ["secret_obj1"], "target": "target1", "device": None}
        with self.assertRaisesRegex(m.RecipeRejected, "did not terminate successfully"):
            m.execute_recipe(env, SearchBudget(), {"state": first.canonical_state, "observation": first.observation}, env.goal, assignment, COSTS)
        # Real toy goal is fulfilled by lamp1 at the selected lamp2's location;
        # success from that OTHER witness must not validate the lamp2 assignment.
        env = ToyEnv("look_at_obj_in_light")
        env.initial.remove(("inreceptacle", "lamp1", "treat1"))
        env.initial |= {("inreceptacle", "lamp1", "treat2"), ("istoggled", "lamp1")}
        first = env.reset(); assignment = {"objects": ["secret_obj1"], "target": None, "device": "lamp2"}
        with self.assertRaisesRegex(m.RecipeRejected, "different witness"):
            m.execute_recipe(env, SearchBudget(), {"state": first.canonical_state, "observation": first.observation}, env.goal, assignment, COSTS)


if __name__ == "__main__":
    unittest.main()

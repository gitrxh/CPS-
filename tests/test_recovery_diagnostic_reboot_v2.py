"""Pure post-reboot lineage and process faults; no model/simulator/remote work."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_recovery_diagnostic_queue_reboot_v2 as q
from scripts import score_recovery_diagnostic_reboot_v2 as w
from gnrl import recovery_diagnostic_lineage_reboot_v2 as lineage
from tests import test_recovery_diagnostic_queue_v1 as legacy


def initial():
    values = dict.fromkeys(q.p.COUNTERS, 0)
    values.update(environment_steps_started=29652, environment_steps_completed=29650,
                  resets_started=2148, resets_completed=2148, generated_tokens=28830,
                  policy_calls_started=3765, policy_calls_completed=3765, forward_tokens=5594038)
    return values


def cost(**increments):
    ledger, summary = legacy.costs(initial(), **increments)
    ledger = copy.deepcopy(ledger)
    ledger["stage_limits"] = dict(ledger["stage_limits"])
    ledger["historical_gap"] = dict(lineage.HISTORICAL_GAP)
    return ledger, {"costs": copy.deepcopy(ledger)}


class InheritedFaultFixtures(legacy.QueueFixtures):
    def setUp(self):
        self.binding = patch.object(legacy, "q", q)
        self.binding.start()
        self.addCleanup(self.binding.stop)

    def test_cost_inheritance_refuses_reset_and_incomplete_work(self):
        ledger, summary = cost(forward_tokens=20)
        self.assertEqual(q.validate_cost_lineage(initial(), ledger, summary)["forward_tokens"], 5594058)
        wrong = copy.deepcopy(ledger)
        wrong["totals"]["forward_tokens"] = 20
        with self.assertRaisesRegex(RuntimeError, "reset/double-count"):
            q.validate_cost_lineage(initial(), wrong, summary)
        broken, summary = cost(environment_steps_started=1)
        with self.assertRaisesRegex(RuntimeError, "unbalanced"):
            q.validate_cost_lineage(initial(), broken, summary)

    def test_request_contract_uses_replay_then_score_ledger_and_same_deadline(self):
        with tempfile.TemporaryDirectory(dir=q.ROOT/"analysis") as folder:
            obj = q.DiagnosticQueue.__new__(q.DiagnosticQueue)
            obj.folder = Path(folder)
            obj.adoption_hash = "parent"
            obj.deadline_unix = 1792005218.8364305
            obj.a = {"candidate":"candidate.json", "source_inventory":[], "original_file_sha256":{},
                "runtime_file_sha256":{}, "implementation_review":"legacy.json", "adaptation_review":"adapt.json",
                "acquisition_result_review":"result.json", "historical_gap":lineage.HISTORICAL_GAP,
                "acquisition_resume_provenance":{"proof":"preserved"},
                "output_paths":{"score":"outputs/recovery_projection_score_reboot_v2",
                                "diagnostic":"outputs/recovery_projection_diagnostic_reboot_v2"}}
            obj.locks = {"candidate.json":"hash"}
            obj.prior_paths = {"oracle":"outputs/recovery_projection_oracle_reboot_resume_v2",
                               "replay":"outputs/recovery_projection_replay_reboot_resume_v2"}
            obj.candidate = {"finite_budgets":{"stage_wall_seconds":{"scoring":7200,"diagnostic_continuation":21600}}}
            obj.lock = lambda paths: None
            score, diagnostic = [q.old.read_json(obj.request(stage)) for stage in q.STAGES]
            self.assertEqual(score["previous_summary"], score["replay_summary"])
            self.assertEqual(diagnostic["previous_summary"], diagnostic["selection_summary"])
            self.assertEqual(score["schema_version"], 2)
            self.assertEqual(score["historical_gap"], lineage.HISTORICAL_GAP)
            self.assertEqual(score["global_deadline_unix"], diagnostic["global_deadline_unix"])
            self.assertEqual(score["global_deadline_unix"], 1792005218.8364305)
            self.assertEqual([score["max_wall_seconds"], diagnostic["max_wall_seconds"]], [7200,21600])
            self.assertEqual(diagnostic["acquisition_resume_provenance"], {"proof":"preserved"})


class RebootLineageFixtures(unittest.TestCase):
    def test_two_historical_unknown_steps_stay_unknown(self):
        ledger, summary = cost(environment_steps_started=12, environment_steps_completed=12,
                               resets_started=1, resets_completed=1)
        totals = q.validate_cost_lineage(initial(), ledger, summary)
        self.assertEqual(lineage.gaps(totals), lineage.HISTORICAL_GAP)
        for delta in (-1,1,2):
            wrong = copy.deepcopy(ledger)
            wrong["totals"]["environment_steps_completed"] += delta
            with self.assertRaisesRegex(RuntimeError, "erased or enlarged"):
                lineage.validate_ledger(wrong, initial())

    def test_cannot_cancel_new_gap_by_changing_inherited_totals(self):
        ledger, _ = cost(environment_steps_started=1)
        ledger["totals"]["environment_steps_completed"] += 1
        with self.assertRaisesRegex(RuntimeError, "New stage calls unbalanced"):
            lineage.validate_ledger(ledger, initial())

    def test_double_counted_or_reset_history_rejected(self):
        for multiplier in (0,2):
            ledger, _ = cost(forward_tokens=10)
            ledger["totals"]["forward_tokens"] = initial()["forward_tokens"] * multiplier + 10
            with self.assertRaisesRegex(RuntimeError, "reset/double-count"):
                lineage.validate_ledger(ledger, initial())

    def test_active_unknown_generation_and_bad_counter_rejected(self):
        for key,value in (("active_call",{"operation":"step"}), ("unknown_generation_after_error",True)):
            ledger, _ = cost()
            ledger[key] = value
            with self.assertRaisesRegex(RuntimeError, "Incomplete"):
                lineage.validate_ledger(ledger)
        for bad in (-1, True, 0.5):
            ledger, _ = cost()
            ledger["stage_counts"]["forward_tokens"] = bad
            with self.assertRaisesRegex(RuntimeError, "Invalid cost"):
                lineage.validate_ledger(ledger)

    def test_budget_excess_and_bad_limit_rejected(self):
        for target in ("stage_counts", "totals"):
            ledger, _ = cost()
            ledger[target]["generated_tokens"] = 4000001
            with self.assertRaisesRegex(RuntimeError, "ceiling"):
                lineage.validate_ledger(ledger)
        ledger, _ = cost()
        ledger["stage_limits"]["generated_tokens"] = 4000001
        with self.assertRaisesRegex(RuntimeError, "ceiling"):
            lineage.validate_ledger(ledger)

    def test_summary_only_elapsed_time_may_differ(self):
        ledger, summary = cost()
        ledger["elapsed_seconds"] = 2
        summary["costs"]["elapsed_seconds"] = 1
        lineage.compare_summary(ledger, summary)
        summary["costs"]["historical_gap"]["environment_steps"] = 0
        with self.assertRaisesRegex(RuntimeError, "Summary"):
            lineage.compare_summary(ledger, summary)

    def test_real_snapshot_time_difference_is_not_cost_difference(self):
        provenance = {"historical_gap":lineage.HISTORICAL_GAP}
        with patch.object(q.p.time,"monotonic",side_effect=[100,101,102]):
            ledger = lineage.DiagnosticLedger(provenance=provenance, initial_totals=initial(),
                stage_limits=q.p.GLOBAL_LIMITS, deadline=200, persist=lambda row:None)
            first, second = ledger.snapshot(), ledger.snapshot()
        self.assertNotEqual(first["elapsed_seconds"], second["elapsed_seconds"])
        lineage.compare_summary(second, {"costs":first})

    def test_persisted_ledger_retains_provenance_and_balanced_new_calls(self):
        import time
        provenance = {"historical_gap":lineage.HISTORICAL_GAP,
            "original_global_deadline_unix":1792005218.8364305,"historical_ledgers":[{"a":1},{"b":2}]}
        ledger = lineage.DiagnosticLedger(provenance=provenance, initial_totals=initial(),
            stage_limits=q.p.GLOBAL_LIMITS, global_limits=q.p.GLOBAL_LIMITS,
            deadline=time.monotonic()+10, persist=lambda row:None)
        snapshot = ledger.snapshot()
        self.assertEqual(lineage.acquisition_provenance(snapshot), provenance)
        self.assertEqual(lineage.validate_ledger(snapshot, initial()), initial())

    def test_scientific_input_consumer_is_identical(self):
        def function(path, name):
            tree=ast.parse((q.ROOT/path).read_text(encoding="utf-8"))
            return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name))
        self.assertEqual(function(w.LEGACY_RUNNER,"inputs"),function(w.RUNNER,"inputs"))
        self.assertEqual(function(q.LEGACY_CONTROLLER,"verify_score_cost"),function(q.CONTROLLER,"verify_score_cost"))

    def test_unmask_wrapper_precedes_gpu_launcher(self):
        # The inherited fault fixture checks real command construction and cleanup.
        source=(q.ROOT/q.CONTROLLER).read_text(encoding="utf-8")
        self.assertIn('command = [self.python, "-c", unmask, *command]',source)
        self.assertIn('os.execv(sys.argv[1], sys.argv[1:])',source)
        self.assertIn('signal.SIG_UNBLOCK',source)


if __name__ == "__main__":
    unittest.main()

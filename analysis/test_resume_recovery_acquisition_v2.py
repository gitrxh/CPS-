"""Pure fixtures: no real simulator, model, remote execution, or GPU."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from analysis import resume_recovery_acquisition_v2 as resume
from gnrl import recovery_projection_v1 as m

spec = importlib.util.spec_from_file_location("recovery_fixtures", resume.ROOT / "tests/test_recovery_projection_v1.py")
toy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(toy)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False) + "\n").encode()


class ResumeFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.anchors = toy.episode()["anchors"]
        cls.events = []
        cls.pool = m.certify_anchor(lambda game, expert: toy.ToyEnv(expert), cls.anchors[0], toy.ledger(), cls.events.append)

    def copy(self, records=None, events=None, anchors=None):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "records").write_bytes(encoded(self.pool) if records is None else records)
            (folder / "events").write_bytes(b"".join(map(encoded, self.events)) if events is None else events)
            out, log = io.BytesIO(), io.BytesIO()
            meta = resume.copy_prefix(folder / "records", folder / "events", self.anchors if anchors is None else anchors, out, log, 1)
            return meta, out.getvalue(), log.getvalue()

    def test_preserves_exact_bytes_and_all_failed_branches(self):
        raw = (json.dumps(self.pool, separators=(",", ":")) + "\n").encode()
        meta, copied, events = self.copy(records=raw)
        self.assertEqual(copied, raw)
        self.assertEqual(events, b"".join(map(encoded, self.events)))
        self.assertTrue(any(not b["success"] for b in self.pool["branches"]))
        self.assertEqual(meta["copied_records"], 1)

    def test_only_uncommitted_final_fragments_may_be_discarded(self):
        record_tail, event_tail = b'{"broken":', b'{"kind":'
        suffix = {"kind": "oracle_transition", "anchor_id": self.anchors[1]["anchor_id"], "transition": {}}
        meta, _, copied_events = self.copy(records=encoded(self.pool) + record_tail,
            events=b"".join(map(encoded, self.events)) + encoded(suffix) + event_tail)
        self.assertEqual(meta["record_tail_fragment"], {"bytes": len(record_tail), "sha256": hashlib.sha256(record_tail).hexdigest()})
        self.assertEqual(meta["event_tail_fragment"]["bytes"], len(event_tail))
        self.assertEqual(meta["interrupted_anchor_events_excluded"], 1)
        self.assertEqual(copied_events, b"".join(map(encoded, self.events)))

    def test_complete_invalid_middle_record_is_fatal(self):
        with self.assertRaises((RuntimeError, json.JSONDecodeError)):
            self.copy(records=b'{"broken":\n' + encoded(self.pool))

    def test_full_anchor_payload_must_equal_collection(self):
        anchors = copy.deepcopy(self.anchors)
        anchors[0]["recorded_learner_next_action"] = "advance"
        with self.assertRaisesRegex(RuntimeError, "ordered complete collection anchor"):
            self.copy(anchors=anchors)

    def test_no_skipping_or_extra_complete_pools(self):
        with self.assertRaises(RuntimeError):
            self.copy(records=encoded(self.pool) * 2)
        with self.assertRaises(RuntimeError):
            self.copy(records=b"")
        with self.assertRaises(RuntimeError):
            self.copy(anchors=list(reversed(self.anchors)))

    def test_missing_changed_or_reordered_event_fails_closed(self):
        variants = [self.events[1:], [*self.events[:1], *self.events[2:]]]
        swapped = copy.deepcopy(self.events)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        variants.append(swapped)
        changed = copy.deepcopy(self.events)
        changed[0]["candidate_index"] = 99
        variants.append(changed)
        for events in variants:
            with self.subTest(events=len(events)), self.assertRaises(RuntimeError):
                self.copy(events=b"".join(map(encoded, events)))

    def test_fragment_cannot_replace_required_event(self):
        with self.assertRaises(RuntimeError):
            self.copy(events=b"".join(map(encoded, self.events[:-1])) + encoded(self.events[-1])[:-1])

    def test_completed_or_future_anchor_in_event_tail_fails(self):
        for anchor in (self.anchors[0], self.anchors[2]):
            extra = {"kind": "oracle_transition", "anchor_id": anchor["anchor_id"]}
            with self.assertRaises(RuntimeError):
                self.copy(events=b"".join(map(encoded, self.events)) + encoded(extra))

    def history(self):
        inv = json.loads((resume.ROOT / resume.INVENTORY).read_text())
        docs = inv["documents"]
        parent = docs["configs/recovery_acquisition_adoption_v1.json"]
        prior = docs["configs/recovery_acquisition_resume_adoption_v1.json"]
        original = docs[parent["output_paths"]["oracle"] + "/cost_progress.json"]
        historical = docs[prior["output_paths"]["oracle"] + "/cost_progress.json"]
        collection = docs[parent["output_paths"]["collect"] + "/cost_progress.json"]
        return inv, original, historical, collection

    def test_two_unknown_steps_and_all_started_calls_remain_charged(self):
        inv, original, historical, collection = self.history()
        resume.validate_historical_cost(original, historical, collection)
        self.assertEqual(resume.remaining_limits(historical), {
            "environment_steps_started": 132558, "resets_started": 663, "generated_tokens": 0})
        p = {"historical_gap": resume.gaps(historical["totals"]),
             "old_oracle_stage_counts": historical["combined_oracle_stage_counts"], "stage": "oracle"}
        ledger = resume.ResumeLedger(stage_limits=resume.remaining_limits(historical),
            deadline=time.monotonic() + 60, initial_totals=historical["totals"], provenance=p)
        env = toy.ToyEnv()
        ledger.reset(env, "oracle_prefix")
        ledger.step(env, "advance", "oracle_prefix")
        resume.validate_new_cost(ledger.snapshot(), historical["totals"], p["historical_gap"])
        self.assertEqual(resume.gaps(ledger.totals)["environment_steps"], 2)
        self.assertEqual(ledger.snapshot()["combined_oracle_stage_counts"]["environment_steps_started"], 11443)
        fake = ledger.snapshot()
        fake["totals"]["environment_steps_completed"] += 1
        with self.assertRaises(RuntimeError):
            resume.validate_new_cost(fake, historical["totals"], p["historical_gap"])
        ledger.counts["environment_steps_started"] = 132558
        with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
            ledger.step(env, "advance", "oracle_prefix")

    def test_historical_count_label_gap_and_lineage_mutation_rejected(self):
        _, original, historical, collection = self.history()
        for mutate in (lambda h: h["totals"].update(environment_steps_completed=15206),
                       lambda h: h["combined_oracle_stage_counts"].update(resets_started=776),
                       lambda h: h["resume_provenance"]["old_oracle_calls_by_label"].update(**{"oracle_prefix:step": 0}),
                       lambda h: h["stage_limits"].update(resets_started=719)):
            changed = copy.deepcopy(historical)
            mutate(changed)
            with self.assertRaises(RuntimeError):
                resume.validate_historical_cost(original, changed, collection)

    def test_parent_absolute_deadline_never_moves_with_process_start(self):
        a = {"new_window_started_unix": 1791885000., "oracle_wall_seconds": 43200,
             "new_oracle_deadline_unix": 1791928200.}
        for now in (1791885000, 1791928199, 1791990000):
            with mock.patch.object(resume.time, "time", return_value=now):
                self.assertEqual(resume.new_oracle_deadline(a, 1792005218.8364305), 1791928200.)
        a["new_oracle_deadline_unix"] += 1
        with self.assertRaisesRegex(RuntimeError, "parent-bound"):
            resume.new_oracle_deadline(a, 1792005218.8364305)
        a["new_oracle_deadline_unix"] = 1791888970
        self.assertEqual(resume.new_oracle_deadline(a, 1791889000), 1791888970)
        a["oracle_wall_seconds"] = 43201
        with self.assertRaises(RuntimeError):
            resume.new_oracle_deadline(a, 1791889000)

    def test_old_pid_and_session_collisions_are_never_inspected_after_proven_reboot(self):
        inv, _, _, _ = self.history()
        docs = inv["documents"]
        status = docs["outputs/recovery_projection_resume_queue_v1/status.json"]
        job = docs["outputs/recovery_projection_resume_job_v1/job.json"]
        receipt = docs["outputs/recovery_projection_resume_job_v1/launch_verification.json"]
        with mock.patch.object(resume.old, "process_record", side_effect=AssertionError("do not inspect old IDs")):
            resume.validate_reboot(inv, status, job, receipt, inv["boot"]["boot_id"], inv["boot"]["boot_time_unix"])
        for boot_id, btime in (("different", inv["boot"]["boot_time_unix"]), (inv["boot"]["boot_id"], 0)):
            with self.assertRaises(RuntimeError):
                resume.validate_reboot(inv, status, job, receipt, boot_id, btime)
        bad = copy.deepcopy(inv)
        bad["boot"]["boot_time_unix"] = 1
        with self.assertRaisesRegex(RuntimeError, "predate"):
            resume.validate_reboot(bad, status, job, receipt, bad["boot"]["boot_id"], 1)

    def test_cleanup_is_same_boot_and_only_current_owned_process(self):
        process = mock.Mock(pid=123)
        with mock.patch.object(resume, "current_boot_id", return_value="current"), \
             mock.patch.object(resume.old, "cleanup_owned") as cleanup:
            with self.assertRaisesRegex(RuntimeError, "refuse"):
                resume.cleanup_current_boot(process, 456, "earlier")
            cleanup.assert_not_called()
            resume.cleanup_current_boot(process, 456, "current")
            cleanup.assert_called_once_with(process, 456)

    def test_inherited_131_raw_prefix_integration_rejects_semantically_equal_reencoding(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "original").mkdir()
            (root / "resume1").mkdir()
            raw = encoded(self.pool)
            events = b"".join(map(encoded, self.events))
            for name, data in (("records.jsonl", raw), ("events.jsonl", events)):
                (root / "original" / name).write_bytes(data * 131)
                (root / "resume1" / name).write_bytes(data * 137)
            a = {"resume_parent": "first"}
            inv = {"documents": {"first": {"output_paths": {"oracle": "resume1"}}}}
            parent = {"candidate": "candidate", "output_paths": {"oracle": "original", "collect": "collect"}}
            contract = (a, inv, parent, None, None, None, None)
            with mock.patch.object(resume, "ROOT", root), \
                 mock.patch.object(resume.old, "read_json", return_value={"training_cohort": {"frozen_gamefiles": []}}), \
                 mock.patch.object(resume, "collection_anchors", side_effect=lambda *args: iter([self.anchors[0]] * 180)):
                result = resume.validate_inherited_prefix(contract)
                self.assertEqual(result["original_prefix_records"], 131)
                self.assertTrue(result["byte_preservation_verified"])
                changed = (json.dumps(self.pool, separators=(",", ":")) + "\n").encode()
                (root / "resume1/records.jsonl").write_bytes(changed + raw * 136)
                with self.assertRaisesRegex(RuntimeError, "byte prefix changed"):
                    resume.validate_inherited_prefix(contract)
                (root / "resume1/records.jsonl").write_bytes(raw * 137)
                (root / "resume1/events.jsonl").write_bytes(b" " + events * 137)
                with self.assertRaisesRegex(RuntimeError, "byte prefix changed"):
                    resume.validate_inherited_prefix(contract)

    def test_failed_new_call_cannot_be_hidden_by_historical_gap(self):
        totals = dict.fromkeys(m.COUNTERS, 0)
        totals["environment_steps_started"] = 1
        cost = m.BudgetLedger(stage_limits=resume.worker.STAGE_LIMITS["replay"], deadline=time.monotonic() + 60,
                              initial_totals=totals)
        class Broken:
            def step(self, action):
                raise RuntimeError("simulator failure")
        with self.assertRaisesRegex(RuntimeError, "simulator failure"):
            cost.step(Broken(), "advance", "test")
        with self.assertRaisesRegex(RuntimeError, "New calls incomplete"):
            resume.validate_new_cost(cost.snapshot(), totals, resume.gaps(totals))

    def test_controller_only_runs_oracle_then_replay_and_stops_on_failure(self):
        controller = resume.Controller.__new__(resume.Controller)
        controller.run, controller.verify, controller.update = mock.Mock(), mock.Mock(), mock.Mock()
        controller.main()
        self.assertEqual(controller.run.call_args_list, [mock.call("oracle"), mock.call("replay")])
        self.assertEqual(controller.update.call_args.args[0], "needs_astra_reboot_resumed_acquisition_review")
        controller.run = mock.Mock(side_effect=RuntimeError("stage failed"))
        controller.verify.reset_mock()
        with self.assertRaisesRegex(RuntimeError, "stage failed"):
            controller.main()
        self.assertEqual(controller.run.call_args_list, [mock.call("oracle")])
        controller.verify.assert_not_called()

    def test_worker_timeout_and_interrupt_always_clean_only_owned_process(self):
        for error in (resume.subprocess.TimeoutExpired("fixture", 1), InterruptedError("fixture")):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory(dir=resume.ROOT / "analysis") as folder:
                controller = resume.Controller.__new__(resume.Controller)
                controller.folder = Path(folder)
                controller.path, controller.hash = controller.folder / "adoption.json", "fixture"
                controller.a = {"output_paths": {"oracle": "outputs/unused_fixture_oracle"}, "expected_boot_id": "boot"}
                controller.contract = (None, None, None, None, None, None, time.time() + 100)
                controller.state, controller.locks = {"stages": []}, {}
                controller.verify, controller.update = mock.Mock(), mock.Mock()
                process = mock.Mock(pid=12345, returncode=-9)
                process.wait.side_effect = error
                with mock.patch.object(resume.subprocess, "Popen", return_value=process), \
                     mock.patch.object(resume.old, "process_record", return_value={"session": 12345, "start_ticks": 456}), \
                     mock.patch.object(resume, "cleanup_current_boot") as cleanup, \
                     mock.patch.object(resume.signal, "pthread_sigmask", create=True, return_value=set()), \
                     mock.patch.object(resume.signal, "SIG_BLOCK", create=True, new=0), \
                     mock.patch.object(resume.signal, "SIG_SETMASK", create=True, new=2):
                    with self.assertRaises(type(error)):
                        controller.run("oracle")
                    cleanup.assert_called_once_with(process, 456, "boot")
                self.assertTrue(controller.state["stages"][0]["owned_cleanup_complete"])


if __name__ == "__main__":
    unittest.main()

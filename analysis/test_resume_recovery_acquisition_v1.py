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

from analysis import resume_recovery_acquisition_v1 as resume
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

    def test_budget_and_historical_gap_are_never_refunded(self):
        inv = json.loads((resume.ROOT / resume.INVENTORY).read_text())
        parent = inv["documents"]["configs/recovery_acquisition_adoption_v1.json"]
        history = inv["documents"][parent["output_paths"]["oracle"] + "/cost_progress.json"]
        collect = inv["documents"][parent["output_paths"]["collect"] + "/cost_progress.json"]
        resume.validate_historical_cost(history, collect)
        limits = resume.remaining_limits(history)
        self.assertEqual(limits["environment_steps_started"], 144000 - 10439)
        self.assertEqual(limits["resets_started"], 1440 - 722)
        provenance = {"historical_gap": resume.gaps(history["totals"]), "old_oracle_stage_counts": history["stage_counts"],
                      "old_oracle_controller_elapsed_seconds": 22843.151138050947, "stage": "oracle"}
        ledger = resume.ResumeLedger(stage_limits=limits, deadline=time.monotonic() + 60,
            initial_totals=history["totals"], provenance=provenance)
        self.assertEqual(ledger.totals, history["totals"])
        env = toy.ToyEnv()
        ledger.reset(env, "oracle_prefix")
        ledger.step(env, "advance", "oracle_prefix")
        resume.validate_new_cost(ledger.snapshot(), history["totals"], provenance["historical_gap"])
        self.assertEqual(ledger.totals["environment_steps_started"] - ledger.totals["environment_steps_completed"], 1)
        fake = ledger.snapshot()
        fake["totals"]["environment_steps_completed"] += 1
        with self.assertRaises(RuntimeError):
            resume.validate_new_cost(fake, history["totals"], provenance["historical_gap"])
        ledger.counts["environment_steps_started"] = limits["environment_steps_started"]
        with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
            ledger.step(env, "advance", "oracle_prefix")

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

    def test_original_oracle_deadline_does_not_move_with_resume_time(self):
        inv = json.loads((resume.ROOT / resume.INVENTORY).read_text())
        status = inv["documents"]["outputs/recovery_projection_acquisition_queue_v1/status.json"]
        oracle = next(s for s in status["stages"] if s["name"] == "oracle")
        for now in (0, 1791864702, 1791878748):
            with mock.patch.object(resume.time, "time", return_value=now):
                self.assertEqual(resume.original_oracle_deadline(status, oracle), 1791878747.009107)
        earlier = dict(status, global_deadline_unix=1791864800)
        self.assertEqual(resume.original_oracle_deadline(earlier, oracle), 1791864770)

    def test_controller_only_runs_oracle_then_replay_and_stops_on_failure(self):
        controller = resume.Controller.__new__(resume.Controller)
        controller.run, controller.verify, controller.update = mock.Mock(), mock.Mock(), mock.Mock()
        controller.main()
        self.assertEqual(controller.run.call_args_list, [mock.call("oracle"), mock.call("replay")])
        self.assertEqual(controller.update.call_args.args[0], "needs_astra_resumed_acquisition_review")
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
                controller.a = {"output_paths": {"oracle": "outputs/unused_fixture_oracle"}}
                controller.contract = (None, None, None, None, None, None, time.time() + 100)
                controller.state, controller.locks = {"stages": []}, {}
                controller.verify, controller.update = mock.Mock(), mock.Mock()
                process = mock.Mock(pid=12345, returncode=-9)
                process.wait.side_effect = error
                with mock.patch.object(resume.subprocess, "Popen", return_value=process), \
                     mock.patch.object(resume.old, "process_record", return_value={"session": 12345, "start_ticks": 456}), \
                     mock.patch.object(resume.old, "cleanup_owned") as cleanup, \
                     mock.patch.object(resume.signal, "pthread_sigmask", create=True, return_value=set()), \
                     mock.patch.object(resume.signal, "SIG_BLOCK", create=True, new=0), \
                     mock.patch.object(resume.signal, "SIG_SETMASK", create=True, new=2):
                    with self.assertRaises(type(error)):
                        controller.run("oracle")
                    cleanup.assert_called_once_with(process, 456)
                self.assertTrue(controller.state["stages"][0]["owned_cleanup_complete"])


if __name__ == "__main__":
    unittest.main()

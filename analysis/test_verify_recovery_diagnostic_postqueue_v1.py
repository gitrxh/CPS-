"""Pure postqueue-verifier fixtures; no real processes or experiment outcomes."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analysis import verify_recovery_diagnostic_postqueue_v1 as v


class PostqueueTests(unittest.TestCase):
    def test_exited_and_zombie_with_empty_session(self):
        for process in (None, {"start_ticks": 55, "state": "Z"}):
            with patch.object(v.old, "process_record", return_value=process), patch.object(v.old, "owned_members", return_value=[]):
                v.exited(11, 55, 11)

    def test_live_reused_and_orphaned_session_rejected(self):
        for process, members in (({"start_ticks": 55, "state": "S"}, []),
                                 ({"start_ticks": 56, "state": "Z"}, []), (None, [17])):
            with patch.object(v.old, "process_record", return_value=process), patch.object(v.old, "owned_members", return_value=members):
                with self.assertRaises(RuntimeError):
                    v.exited(11, 55, 11)

    def test_identity_must_be_exact_positive_integers(self):
        for identity in ((0, 55, 0), (11, True, 11), (11, 55, 12), (True, 55, True)):
            with self.assertRaises(RuntimeError):
                v.exited(*identity)

    def test_measure_actual_bytes_and_reject_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "blob"
            path.write_bytes(b"real fixture bytes")
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            calls = []
            measured = v.measured({str(path): expected}, lambda: calls.append(1), absolute=True)
            self.assertEqual(measured, {str(path): expected})
            self.assertGreaterEqual(len(calls), 3)
            with self.assertRaisesRegex(RuntimeError, "Frozen bytes"):
                v.measured({str(path): "0" * 64}, lambda: None, absolute=True)

    def test_deadline_checked_before_file_access(self):
        def exhausted():
            raise TimeoutError("deadline")
        with self.assertRaisesRegex(TimeoutError, "deadline"):
            v.measure_one(Path("nonexistent fixture"), exhausted)

    def test_file_change_during_hash_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "blob"
            path.write_bytes(b"before")
            calls = []
            def mutate():
                calls.append(1)
                if len(calls) == 2:
                    path.write_bytes(b"different length after")
            with self.assertRaisesRegex(RuntimeError, "changed during"):
                v.measure_one(path, mutate)

    def test_deadline_inherited_and_never_reset(self):
        status = {"main_study_started_unix": 1000, "global_deadline_unix": 172840}
        candidate = {"version": "v2", "finite_budgets": {"global_wall_seconds_including_waits_verification_cleanup": 171840}}
        self.assertEqual(v.inherited_deadline(status, status, candidate, now=2000), 172840)
        for changed, now in ((dict(status, main_study_started_unix=1001), 2000),
                             (status, 172810), (status, 999),
                             (dict(status, global_deadline_unix=float("nan")), 2000)):
            with self.assertRaises(RuntimeError):
                v.inherited_deadline(changed, status, candidate, now=now)

    def test_workspace_escape_and_unnormalized_rejected(self):
        for value in ("../escape", "outputs/../escape", "/absolute", "outputs//repeat", None):
            with self.assertRaises(RuntimeError):
                v.relative(value)


if __name__ == "__main__":
    unittest.main()

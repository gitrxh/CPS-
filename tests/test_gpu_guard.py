import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("guard", Path(__file__).parents[1] / "scripts/launch_gpu.py")
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class GPUGuardTest(unittest.TestCase):
    def test_physical_allowlist(self):
        rows = [[str(i), f"GPU-{i}", "A100", "0"] for i in range(8)]
        for index in range(4, 8):
            with self.assertRaises(ValueError):
                guard.resolve_devices([index], rows)
        self.assertEqual(guard.resolve_devices([0, 3], rows)[1]["uuid"], "GPU-3")

    def test_busy_rejected(self):
        with self.assertRaises(RuntimeError):
            guard.resolve_devices([0], [["0", "GPU-0", "A100", "50000"]])
        with self.assertRaises(RuntimeError):
            guard.resolve_devices([0], [["0", "GPU-0", "A100", "100"]], active_uuids={"GPU-0"})

    def test_empty_duplicate_rejected(self):
        for indices in ([], [0, 0]):
            with self.assertRaises(ValueError):
                guard.resolve_devices(indices, [])


if __name__ == "__main__":
    unittest.main()

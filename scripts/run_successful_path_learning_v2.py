"""System-Python controller for the unchanged frozen learning queue.

Run with /usr/bin/python3, whose native pidfd bindings are available on the
chemistry server. The frozen v1 Queue still launches every model, tokenizer,
training and evaluation worker with the original project .venv/bin/python.
This entrypoint changes no methods, budgets, paths, receipt gates or settings.
It requires the same separately reviewed explicit parent learning adoption.
"""
from __future__ import annotations

import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_successful_path_learning_v1 as frozen

FROZEN_QUEUE_SHA256 = "99926d02303f56de560f1589109685ab3f863e058393d2eefb1c8f256c945f6c"


def validate_controller():
    frozen.require(sys.platform.startswith("linux"), "Controller requires Linux")
    frozen.require(Path(sys.executable).resolve() == Path("/usr/bin/python3").resolve(),
                   "Use /usr/bin/python3 for native pidfd controller; workers remain on project Python")
    frozen.require(hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"), "Controller lacks native pidfd APIs")
    frozen.require(frozen.sha256(ROOT / "scripts/run_successful_path_learning_v1.py") == FROZEN_QUEUE_SHA256,
                   "Frozen v1 queue bytes changed")
    fd = os.pidfd_open(os.getpid())
    os.close(fd)
    return {"controller_executable": sys.executable, "controller_resolved": str(Path(sys.executable).resolve()),
            "controller_python_version": sys.version, "worker_executable": str(ROOT / ".venv/bin/python"),
            "frozen_queue_sha256": FROZEN_QUEUE_SHA256, "native_pidfd_verified": True}


def main(argv=None):
    validate_controller()
    return frozen.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

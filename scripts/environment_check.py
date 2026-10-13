"""Record exact dependency/model availability without touching CUDA."""
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

report = {"python": sys.version, "executable": sys.executable, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "packages": {}}
for name in ["torch", "transformers", "peft", "alfworld", "textworld", "gym", "fast_downward"]:
    try:
        module = importlib.import_module(name)
        report["packages"][name] = {"version": getattr(module, "__version__", None), "file": module.__file__, "import_ok": True}
    except Exception as exc:
        report["packages"][name] = {"import_ok": False, "error": str(exc)}
print(json.dumps(report, indent=2))
Path("outputs/environment_check.json").write_text(json.dumps(report, indent=2))

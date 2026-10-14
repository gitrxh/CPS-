"""Send local verifier as stdin; capture only its receipt locally."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
audit = ROOT / 'analysis/review_acquisition_reboot_v2.py'
legacy = ROOT / 'analysis/review_recovery_acquisition_complete_v1.py'
source = audit.read_text(encoding='utf-8')
review = legacy.read_text(encoding='utf-8')
payload = "namespace={'__name__':'readonly_audit'}\nexec(" + repr(source) + ",namespace)\nresult=namespace['run'](" + repr(review) + ")\nprint(json.dumps(result,ensure_ascii=False,indent=2))"
script = "cd /home/jsy/GNRL_20261012\nPYTHONDONTWRITEBYTECODE=1 python3 - <<'GNRL_REBOOT_V2_READONLY'\nimport json\n" + payload + "\nGNRL_REBOOT_V2_READONLY\n"
proc = subprocess.run([sys.executable, str(ROOT / 'scripts/remote.py'), 'exec'], input=script, text=True, encoding='utf-8', capture_output=True)
if proc.returncode:
    print(proc.stderr)
    print(proc.stdout)
    raise SystemExit(proc.returncode)
result = json.loads(proc.stdout)
result['review_program_sha256'] = hashlib.sha256(audit.read_bytes()).hexdigest()
result['legacy_row_validator_sha256'] = hashlib.sha256(legacy.read_bytes()).hexdigest()
out = ROOT / 'outputs/recovery_reboot_acquisition_completion_astra_review_v2.json'
with out.open('x', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(json.dumps({k:result[k] for k in ['complete', 'errors', 'totals', 'cumulative_totals', 'historical_gap']}))
print('receipt_sha256=' + hashlib.sha256(out.read_bytes()).hexdigest())

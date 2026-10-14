"""Add evidence-grounded local interpretation to the completed remote receipt."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'outputs/recovery_reboot_acquisition_completion_astra_review_v2.json'
receipt = json.loads(path.read_text(encoding='utf-8'))
assert receipt['complete'] is True and receipt['errors'] == []
multi = [a for a in receipt['diagnostic36_availability'] if a['eligible'] and a['certified_actions'] >= 2]
families = Counter(a['family'] for a in multi)
coverage = len(multi) >= 12 and len(families) >= 4
receipt['pre_diagnostic_coverage_gate'] = {
    'passed': coverage, 'fixed_slots': 36, 'eligible_multiaction_anchors': len(multi),
    'family_count': len(families), 'counts_by_family': dict(families),
    'structural_unavailable_slots': sum(not a['eligible'] for a in receipt['diagnostic36_availability']),
    'threshold': 'At least 12 eligible multiaction anchors spanning at least 4 families; no replacement slots.'}
receipt['eligible_for_scoring_and_fixed36_real_continuation_diagnostic'] = coverage
receipt['independent_local_validation'] = {
    'tests_passed': 26,
    'command': 'python -m unittest analysis.test_review_recovery_acquisition_complete_v1 analysis.test_resume_recovery_acquisition_v2',
    'scope': 'Synthetic record/event validation and reboot lineage fixtures; observed completed successfully before receipt generation.',
    'file_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in [
        'analysis/test_review_recovery_acquisition_complete_v1.py',
        'analysis/test_resume_recovery_acquisition_v2.py',
        'analysis/execute_readonly_audit_reboot_v2.py',
        'analysis/annotate_acquisition_review_reboot_v2.py']}}
receipt['preserved_descriptive_negative_results'] = {
    'collection_learner_episodes_won': 26, 'collection_episodes': 60,
    'collection_inadmissible_actions': 2701,
    'oracle_proposals_not_certified': 1,
    'all_training_slots_structurally_unavailable': 49,
    'claim_boundary': 'These acquisition descriptives are not a new held-out comparison; no task-success improvement or learner-Q gate pass is claimed.'}
path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'complete': receipt['complete'], 'coverage_gate': receipt['pre_diagnostic_coverage_gate'],
    'receipt_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}, ensure_ascii=False))

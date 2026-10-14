"""Read-only remote audit payload; invoked through stdin, never launches env/model."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))


def run(review_source):
    import types
    r = types.ModuleType('independent_acquisition_review')
    r.__file__ = str(ROOT / 'analysis/review_recovery_acquisition_complete_v1.py')
    exec(compile(review_source, r.__file__, 'exec'), r.__dict__)
    from analysis import resume_recovery_acquisition_v2 as v
    req = r.require
    ap = ROOT / 'configs/recovery_acquisition_reboot_resume_adoption_v2.json'
    contract = v.load_contract(ap)
    a, inv, parent, old_status, _, historical, deadline = contract
    q = ROOT / a['output_paths']['queue']
    status = r.read(q / 'status.json')
    status_hash = r.sha(q / 'status.json')
    locks = r.read(q / 'locked_sha256.json')
    req(status['complete'] is True and status['technical_complete'] is True and
        status['current_stage'] == 'needs_astra_reboot_resumed_acquisition_review' and
        not status.get('error') and status['adoption_sha256'] == r.sha(ap), 'Queue barrier')
    req([s['name'] for s in status['stages']] == ['oracle', 'replay'], 'Stage order')
    req(status['boot_id'] == v.current_boot_id() == a['expected_boot_id'], 'Boot changed')
    job_path = ROOT / 'outputs/recovery_projection_reboot_resume_job_v2/job.json'
    launch_path = job_path.parent / 'launch_verification.json'
    job, launch = r.read(job_path), r.read(launch_path)
    req(launch['startup_verification_complete'] is True and launch['adoption_sha256'] == r.sha(ap)
        and launch['prefix_milestones'] == {'inherited_prefix_verified': 131, 'prefix_copy_complete': 137}, 'Startup provenance')
    owners = [{'name': 'controller', 'pid': job['pid'], 'start_ticks': int(job['start_ticks']), 'owned_session': job['pid']}] + status['stages']
    process_receipt = []
    for s in owners:
        now = v.old.process_record(s['pid'])
        req(not now or now['start_ticks'] != s['start_ticks'] or now['state'] == 'Z', 'Owned process still running')
        process_receipt.append({k: s[k] for k in ('name', 'pid', 'start_ticks', 'owned_session')})
    sessions = {s['owned_session'] for s in owners}
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            proc = v.old.process_record(int(path.name))
            req(not proc or proc['session'] not in sessions or proc['state'] == 'Z', 'Owned session remains active')
    costs, summaries = {}, {}
    for s in status['stages']:
        stage = s['name']
        req(s['returncode'] == 0 and s['owned_cleanup_complete'] is True, 'Stage cleanup')
        d = ROOT / a['output_paths'][stage]
        sm, c = r.read(d / 'summary.json'), r.read(d / 'cost_progress.json')
        req(sm['complete'] is True and sm['technical_complete'] is True and sm['record_count'] == sm['planned_records'] == 180
            and not sm.get('error') and not sm.get('technical_incomplete'), 'Stage metadata barrier')
        req(sm['costs']['totals'] == c['totals'] and sm['costs']['stage_counts'] == c['stage_counts'], 'Summary counters')
        v.validate_new_cost(c, historical['totals'] if stage == 'oracle' else costs['oracle']['totals'], {'environment_steps': 2, 'resets': 0, 'policy_calls': 0})
        req(c['stage_limits'] == (v.remaining_limits(historical) if stage == 'oracle' else v.worker.STAGE_LIMITS['replay']) and c['global_limits'] == r.p.GLOBAL_LIMITS, 'Budget limits')
        req(all(c['stage_counts'][k] <= n for k, n in c['stage_limits'].items()) and all(c['totals'][k] <= n for k, n in c['global_limits'].items()), 'Budget exceeded')
        request_path = q / (stage + '_request.json')
        request = r.read(request_path)
        config = r.read(d / 'config.json')
        req(request['stage'] == stage and request['adoption_sha256'] == r.sha(ap) and
            sm['request_sha256'] == r.sha(request_path) == config['request_sha256'] and config['request'] == request, 'Request binding')
        req(request['deadline_unix'] == s['deadline_unix'] and
            (s['deadline_unix'] == deadline if stage == 'oracle' else s['deadline_unix'] <= min(a['global_deadline_unix'] - 30, datetime.fromisoformat(s['started_utc']).timestamp() + 7200)), 'Deadline enlarged')
        req(datetime.fromisoformat(s['started_utc']).timestamp() + s['elapsed_seconds'] <= s['deadline_unix'], 'Observed runtime over deadline')
        costs[stage], summaries[stage] = c, sm
    # Rehash original source, games/runtime, adapters, historical artifacts, current outputs.
    # Full model bytes were verified by the frozen controller immediately before completion.
    v.verify_inputs(contract, ap, r.sha(ap), locks, full_model=False)
    frozen_hashes = {**parent['file_sha256'], **a['file_sha256'], **locks}
    old_locks = inv['documents'][parent['output_paths']['queue'] + '/locked_sha256.json']
    frozen_hashes.update(old_locks)
    candidate = r.read(ROOT / parent['candidate'])
    games = candidate['training_cohort']['frozen_gamefiles']
    train = r.read(ROOT / 'data/manifests/train120.json')['games']
    prior = r.read(ROOT / 'configs/successful_path_selection_candidate_v1.json')['fixed_training_cohort']['frozen_gamefiles']
    req(games == [g for g in train if g not in set(prior)] and len(set(games)) == 60 and Counter(map(r.family, games)) == Counter({f: 10 for f in r.TASK_TYPES}), 'Cohort freeze')
    planned = candidate['train_only_diagnostic_gate']['frozen_anchors']
    fixed = []
    for family in sorted(r.TASK_TYPES):
        ranked = sorted((g for g in games if r.family(g) == family), key=lambda g: hashlib.sha256(('recovery_projection_v1_diagnostic36|' + r.p.relative_game(g)).encode()).hexdigest())
        fixed.extend({'gamefile': g, 'family': family, 'after_actions': (4, 8, 16)[i // 2]} for i, g in enumerate(ranked[:6]))
    req(planned == fixed, 'Fixed 36 slots changed')
    # Metadata/hash/exit barrier passed. Parse all outcomes and actual events now.
    dirs = {'collect': ROOT / parent['output_paths']['collect'], **{s: ROOT / a['output_paths'][s] for s in ('oracle', 'replay')}}
    def raw_prefix(source, target, lines):
        digest = hashlib.sha256()
        size = 0
        with source.open('rb') as old, target.open('rb') as new:
            for _ in range(lines):
                line = old.readline()
                req(line.endswith(b'\n') and new.read(len(line)) == line, 'Raw prefix mismatch')
                digest.update(line); size += len(line)
        return {'lines': lines, 'bytes': size, 'sha256': digest.hexdigest()}
    first = inv['documents'][a['resume_parent']]
    generations = [ROOT / parent['output_paths']['oracle'], ROOT / first['output_paths']['oracle'], dirs['oracle']]
    prefix = {}
    for i, n in enumerate((131, 137)):
        prefix[str(n)] = raw_prefix(generations[i] / 'records.jsonl', generations[i+1] / 'records.jsonl', n)
    sm = summaries['oracle']
    prefix['original_events'] = raw_prefix(generations[0] / 'events.jsonl', generations[1] / 'events.jsonl', sm['inherited_prefix_validation']['original_interruption_metadata']['copied_events'])
    prefix['resumed_events'] = raw_prefix(generations[1] / 'events.jsonl', generations[2] / 'events.jsonl', sm['prefix_copy']['copied_events'])
    suffix = Counter()
    def pools():
        for i, pool in enumerate(r.jsonl(dirs['oracle'] / 'records.jsonl')):
            if i >= 137:
                anchor = pool['anchor']
                if anchor['status'] == 'available':
                    n = max(1, len(pool['branches']))
                    suffix.update(resets=n, prefix_steps=n * anchor['after_actions'])
                suffix['branch_steps'] += sum(len(b['transitions']) for b in pool['branches'])
            yield pool
    result = r.review_rows(games, planned, r.jsonl(dirs['collect'] / 'records.jsonl'), pools(), r.jsonl(dirs['replay'] / 'records.jsonl'), {s: r.jsonl(d / 'events.jsonl') for s, d in dirs.items()})
    collect_cost = r.read(dirs['collect'] / 'cost_progress.json')
    def counts(work, stage, cost):
        expected = dict.fromkeys(r.p.COUNTERS, 0)
        steps = work.get('steps', 0) if stage == 'collect' else work.get('prefix_steps', 0) + work.get('branch_steps', 0)
        for k, n in [('environment_steps', steps), ('resets', work['resets']), ('policy_calls', work.get('policy_calls', 0))]:
            expected[k + '_started'] = expected[k + '_completed'] = n
        expected['generated_tokens'] = work.get('tokens', 0)
        expected['forward_tokens'] = cost['stage_counts']['forward_tokens'] if stage == 'collect' else 0
        req(cost['stage_counts'] == expected, 'Recomputed stage costs differ: ' + stage)
        return expected
    counts(result['recomputed_workload']['collect'], 'collect', collect_cost)
    counts(suffix, 'oracle', costs['oracle'])
    counts(result['recomputed_workload']['replay'], 'replay', costs['replay'])
    canonical = result['recomputed_workload']['oracle']
    combined = costs['oracle']['combined_oracle_stage_counts']
    overhead = {'environment_steps_started': combined['environment_steps_started'] - canonical['prefix_steps'] - canonical['branch_steps'],
                'environment_steps_completed': combined['environment_steps_completed'] - canonical['prefix_steps'] - canonical['branch_steps'],
                'resets': combined['resets_started'] - canonical['resets']}
    req(all(n >= 0 for n in overhead.values()) and overhead['environment_steps_started'] - overhead['environment_steps_completed'] == 2, 'Historical overhead inconsistency')
    for name, expected in locks.items():
        req(r.sha(ROOT / name) == expected, 'Output changed during review')
    req(r.sha(q / 'status.json') == status_hash, 'Queue changed during review')
    totals = Counter()
    for stat in result['families'].values(): totals.update(stat)
    return {'kind': 'recovery_reboot_acquisition_completion_astra_review', 'version': 2,
        'reviewer': 'independent GPT-6 Astra completed_acquisition_v2_audit',
        'complete': True, 'errors': [], 'recovery_acquisition_result_review_passed': True,
        'recovery_reboot_acquisition_result_review_passed': True,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'boot_id': status['boot_id'], 'historical_gap': v.gaps(costs['replay']['totals']),
        'cumulative_totals': costs['replay']['totals'],
        'adoption_sha256': r.sha(ap), 'acquisition_status_sha256': status_hash,
        'acquisition_locks_sha256': r.sha(q / 'locked_sha256.json'),
        'replay_cost_sha256': r.sha(dirs['replay'] / 'cost_progress.json'),
        'candidate_sha256': r.sha(ROOT / parent['candidate']),
        'complete_record_counts': {'collect': 60, 'oracle': 180, 'replay': 180},
        'verified_source_and_output_sha256': frozen_hashes,
        'verified_runtime_file_sha256': parent['runtime_file_sha256'],
        'verified_original_file_sha256': parent['original_file_sha256'],
        'startup_sha256': r.sha(launch_path), 'job_sha256': r.sha(job_path),
        'exited_processes': process_receipt, 'owned_sessions_gone': True,
        'raw_prefix_verification': prefix, 'totals': dict(totals), **result,
        'cost_review': {'cumulative_totals': costs['replay']['totals'], 'new_oracle_stage_counts': costs['oracle']['stage_counts'],
            'replay_stage_counts': costs['replay']['stage_counts'], 'canonical_oracle_workload': canonical,
            'historical_interrupted_work_overhead': overhead, 'historical_gap': v.gaps(costs['replay']['totals']),
            'historical_active_calls': costs['replay']['resume_provenance']['historical_active_calls'],
            'historical_elapsed_observations': costs['oracle']['resume_provenance']['historical_elapsed_observations'],
            'current_stage_controller_elapsed_seconds': {s['name']: s['elapsed_seconds'] for s in status['stages']},
            'oracle_wall_budget_extended': True, 'original_global_deadline_preserved': True,
            'global_deadline_unix': a['global_deadline_unix']},
        'eligible_for_scoring_and_fixed36_real_continuation_diagnostic': True,
        'training_gate_passed': None, 'execution_authorized': False,
        'interpretation': 'Complete finite-pool acquisition and independent first-action replay. Acquisition availability and oracle certificates do not establish learner-Q utility or held-out task-success improvement. Training/evaluation requires the preregistered actual-continuation gate.',
        'verification_limits': ['No model or simulator calls. No remote writes. Frozen controller source and completion chain establish full 28-file model hash verification at stage/final verification; this read-only audit rehashes source/runtime/games/adapter/artifacts but does not repeat full model weight hashing.', 'Historical two unknown step returns remain charged. Overhead includes interrupted work excluded from canonical records. No exact combined historical compute time is claimed.', 'Forward-token counts use the durable ledger; raw text cannot independently reconstruct token IDs.']}


if __name__ == '__main__':
    source = sys.stdin.read()
    status_hash = hashlib.sha256((ROOT / 'outputs/recovery_projection_reboot_resume_queue_v2/status.json').read_bytes()).hexdigest()
    print(json.dumps(run(source), ensure_ascii=False, indent=2))

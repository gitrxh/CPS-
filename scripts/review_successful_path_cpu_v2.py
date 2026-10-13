"""Independent read-only audit of completed adopted CPU pools; no simulator/model calls."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gnrl.successful_path_pools_v2 import summarize_acquisition, validate_successful_pool


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out = ROOT / 'outputs/successful_paths_cpu_v2'
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['complete'] and summary['coverage_gate_passed'] and not summary['technical_errors']
    config = json.loads((ROOT / 'configs/successful_path_selection_candidate_v1.json').read_text())
    games = config['fixed_training_cohort']['frozen_gamefiles']
    pools = [json.loads(line) for line in (out / 'candidate_pools.jsonl').read_text().splitlines()]
    assert len(pools) == 60 and [p['gamefile'] for p in pools] == games
    recomputed = summarize_acquisition(pools, games)
    for key, value in recomputed.items():
        assert summary[key] == value, key
    adopted = json.loads((ROOT / 'configs/successful_path_cpu_adoption_v2.json').read_text())
    assert summary['adoption']['adoption_sha256'] == sha(ROOT / 'configs/successful_path_cpu_adoption_v2.json')
    for path, expected in adopted['file_sha256'].items():
        assert sha(ROOT / path) == expected, path
    native = {json.loads(line)['gamefile']: line for line in (ROOT / 'data/planner_train120_raw.jsonl').read_bytes().splitlines(keepends=True)}
    baseline = (out / 'canonical_baseline.jsonl').read_bytes().splitlines(keepends=True)
    index = json.loads((out / 'canonical_source_index.json').read_text())['records']
    assert len(baseline) == len(index) == 60
    families, meaningful, pool_sizes, rejections = Counter(), Counter(), Counter(), Counter()
    candidates = alternatives = differing_endpoints = order_only = attempts = steps = 0
    retained_semantics = []
    for i, (pool, raw, ix) in enumerate(zip(pools, baseline, index)):
        validate_successful_pool(pool)
        assert raw == native[pool['gamefile']] == pool['canonical_source_line_utf8'].encode()
        assert hashlib.sha256(raw).hexdigest() == ix['source_line_sha256'] == pool['canonical_source_line_sha256']
        assert ix['gamefile'] == pool['gamefile'] and ix['seed'] == pool['seed']
        game = Path(pool['gamefile'])
        assert sha(game) == pool['source_hashes']['game_sha256']
        assert sha(game.with_name('traj_data.json')) == pool['source_hashes']['metadata_sha256']
        assert pool['verifier_version'] == 2
        assert len(pool['attempts']) <= 7 and len(pool['candidates']) <= 8
        pool_steps = pool['all_acquisition_steps_including_canonical_and_replay']
        assert pool_steps == pool['acquisition_cost']['environment_steps_including_all_replay']
        assert pool_steps == len(pool['candidates'][0]['transitions']) + sum(a['environment_steps_including_replay'] for a in pool['attempts'])
        assert sum(a['accepted'] for a in pool['attempts']) == len(pool['candidates']) - 1
        steps += pool_steps; attempts += len(pool['attempts'])
        candidates += len(pool['candidates']); alternatives += len(pool['candidates']) - 1
        families[pool['family']] += 1; pool_sizes[len(pool['candidates'])] += 1
        if any(c['meaningful_assignment_change'] for c in pool['candidates'][1:]): meaningful[pool['family']] += 1
        for a in pool['attempts']:
            if not a['accepted']: rejections[a['rejection_reason']] += 1
        for c in pool['candidates'][1:]:
            differing_endpoints += c['transitions'][-1]['state_after'] != pool['candidates'][0]['transitions'][-1]['state_after']
            order_only += c.get('delivery_order_only', False)
        if pool['candidates'][0]['terminal_semantics_audit']['retained_target_membership']:
            retained_semantics.append({'task_index': i, 'gamefile': pool['gamefile'], 'pool_size': len(pool['candidates'])})
    assert steps == summary['environment_step_calls_started'] == summary['environment_step_calls_completed']
    assert summary['reset_calls_started'] == summary['reset_calls_completed']
    assert attempts == summary['recipe_attempts_in_completed_pools'] <= 420
    assert candidates == summary['retained_members'] <= 480
    assert steps <= 100000 and summary['wall_seconds'] <= 3600 and summary['gpu_jobs'] == 0
    prior = json.loads((ROOT / 'outputs/successful_paths_cpu_v1/summary.json').read_text())
    preflight = json.loads((ROOT / 'outputs/successful_path_canonical_preflight_v2.json').read_text())
    assert not prior['complete'] and prior['environment_step_calls_started'] == 507
    assert summary['cumulative_environment_step_calls_started'] == steps + 507
    assert summary['cumulative_reset_calls_started'] == summary['reset_calls_started'] + prior['reset_calls_started'] + preflight['actual_reset_calls_started']
    events = [json.loads(line) for line in (out / 'events.jsonl').read_text().splitlines()]
    done = [e for e in events if e['event'] == 'task_complete']
    assert len(done) == 60 and [e['gamefile'] for e in done] == games
    assert sum(e['environment_step_calls_started'] for e in done) == steps
    report = {'kind': 'independent_astra_completed_cpu_review', 'passed': True,
              'new_environment_calls': 0, 'new_model_calls': 0,
              'bound_files_rehashed': len(adopted['file_sha256']),
              'recomputed_gate': recomputed, 'tasks_by_family': dict(families),
              'meaningful_tasks_by_family_including_two': dict(meaningful), 'pool_size_counts': dict(pool_sizes),
              'retained_candidates': candidates, 'alternatives': alternatives,
              'alternatives_with_different_terminal_state': differing_endpoints,
              'delivery_order_only_alternatives': order_only, 'recipe_attempts': attempts,
              'rejection_reasons': dict(rejections), 'retained_membership_canonical': retained_semantics,
              'costs': {k: summary[k] for k in ('environment_step_calls_started', 'reset_calls_started', 'wall_seconds', 'historical_costs', 'cumulative_environment_step_calls_started', 'cumulative_reset_calls_started')},
              'artifact_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()},
              'recommendation': 'CPU feasibility gate passed. Review the separate selector and learning queue before any new learning adoption. Candidate support is outcome-constrained and length/cost matched; it is not exact-state BRP or evidence of learning benefit.',
              'limitations': ['The independent review validates saved real execution/replay evidence and source bytes, without rerunning environments.', 'Privileged teacher grounding and simulator retained-containment semantics remain disclosed.', 'Development evaluation would be exploratory; unseen evaluation is not authorized.']}
    target = ROOT / 'outputs/successful_path_cpu_v2_astra_result_review_v1.json'
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()

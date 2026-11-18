"""Read-only checks of published trail evidence; never starts a simulation/model."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'results/week12/controller_lookup_fix/seed42_452f3b2_20261111'
PILOT = ROOT / 'results/week12/asal_novelty_pilot/seed303_452f3b2_20261111'


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def check(condition, message):
    if not condition:
        raise ValueError(message)


def corrected(base, raw=False):
    configs = {n: read(base / n / 'input_config.json') for n in ('fixed', 'dynamic')}
    expected = json.loads(json.dumps(configs['fixed']))
    expected['env_options']['shift_interval'] = configs['dynamic']['env_options']['shift_interval']
    check(expected == configs['dynamic'], 'Treatments differ beyond reserve translation')
    check(configs['fixed']['seed'] == 42 and configs['fixed']['shape'] == 400 and
          configs['fixed']['steps'] == 1000, 'Wrong primary comparison')
    summary, budgets = read(base / 'event_summary.json'), read(base / 'resource_budget.json')
    measured, paired = rows(base / 'measurements.csv'), rows(base / 'paired_measurements.csv')
    results = {r['configuration']: r for r in rows(base / 'results.csv')}
    resources = {}
    for name, config in configs.items():
        resource = rows(base / name / 'run/resource_metrics.csv')
        events = rows(base / name / 'run/cell_events.csv')
        check([int(r['completed_update']) for r in resource] == list(range(1001)), 'Resource coverage')
        check([int(r['completed_update']) for r in events] == list(range(1, 1001)), 'Event coverage')
        totals = {k: 0 for k in events[0] if k.endswith(('_gains', '_losses', '_replacements'))}
        for update, event in enumerate(events, 1):
            e = {k: int(v) for k, v in event.items()}
            before, after = int(resource[update-1]['occupied_cells']), int(resource[update]['occupied_cells'])
            check((e['occupied_before'], e['occupied_after']) == (before, after), 'Event/count timing')
            phases = ('colonization', 'resource_death', 'chunk_death', 'radiation', 'cleanup', 'cull')
            check(sum(e[p+'_gains'] - e[p+'_losses'] for p in phases) == after-before, 'Phase accounting')
            check(e['net_gains'] - e['net_losses'] == after-before, 'Net accounting')
            if not (update > 1 and (update-1) % config['radiate_interval'] == 0):
                check(e['radiation_gains'] == e['radiation_losses'] == 0, 'Radiation timing')
            for key in totals:
                totals[key] += e[key]
        check(all(summary[name][k] == v for k, v in totals.items()), 'Event totals')
        for key in ('colonization_gains', 'radiation_gains', 'net_gains', 'net_losses'):
            check(int(results[name][key]) == totals[key], 'Results/event summary')
        check(int(results[name]['initial_occupied']) == int(resource[0]['occupied_cells']) == 32186, 'Initial occupancy')
        check(int(results[name]['final_occupied']) == int(resource[-1]['occupied_cells']), 'Final occupancy')
        for key, csv_key in [('cumulative_delivered', 'actual_energy_delivered'), ('cumulative_requested', 'requested_energy')]:
            check(float(resource[-1][key]) == budgets['runs'][name][key] == float(results[name][csv_key]), 'Budget summary')
        captures = [r for r in measured if r['configuration'] == name]
        check([int(r['completed_update']) for r in captures] == list(range(0, 1001, 25)), 'Capture labels')
        for capture in captures:
            source = resource[int(capture['completed_update'])]
            check(all(capture[k] == source[k] for k in source), 'Capture/resource measurement timing')
        resources[name] = resource
    maximum = 0.
    for left, right in zip(resources['fixed'], resources['dynamic']):
        check(all(left[k] == right[k] for k in ('completed_update', 'cumulative_requested', 'remaining_capacity')), 'Nominal budgets differ')
        check(float(left['cumulative_clipped']) == float(right['cumulative_clipped']) == 0., 'Clipping')
        maximum = max(maximum, abs(float(left['cumulative_delivered'])-float(right['cumulative_delivered'])))
    check(maximum == budgets['maximum_actual_delivery_difference'] and maximum < 0.0001, 'Actual supply mismatch')
    check([int(r['completed_update']) for r in paired] == list(range(0, 1001, 25)), 'Paired coverage')
    check(all(float(r['maximum_abs_state_difference']) == int(r['differing_occupied_locations']) == 0
              for r in paired if int(r['completed_update']) < 125), 'Pre-motion differences')
    check(next(int(r['completed_update']) for r in paired if int(r['differing_occupied_locations'])) == 150, 'First differing capture')
    validation = read(base / 'validation.json')
    check(validation['decoded_gif_occupied_panel_matches'] == validation['state_rgb_source_measurement_matches'] == 82, 'Original raw verification record')
    panels = 0
    with Image.open(base / 'comparison.gif') as animation:
        check(animation.n_frames == 41 and animation.size == (696, 1237) and animation.info['loop'] == 0, 'Animation geometry/coverage')
        for index in range(animation.n_frames):
            animation.seek(index)
            check(animation.info['duration'] == 180, 'Playback timing')
            decoded = np.asarray(animation.convert('RGB'))
            if raw:
                # The compact archive omits these states. --raw requires the local run or regeneration.
                for column, name in enumerate(('fixed', 'dynamic')):
                    state = np.load(base / name / 'run/frame_states' / f'update_{index*25:07d}.npz')['state']
                    occupied = Image.fromarray(((state[0, 13] >= 0).T * 255).astype(np.uint8))
                    expected = np.asarray(occupied.resize((320, 320), Image.Resampling.NEAREST)) > 127
                    observed = decoded[453:773, 12+column*348:332+column*348].mean(axis=-1) > 127
                    check(np.array_equal(expected, observed), 'Raw state/GIF panel synchronization')
                    panels += 1
    return {'resource_rows_checked': 2002, 'event_updates_checked': 2000, 'matched_capture_rows': 82,
            'maximum_actual_delivery_difference': maximum, 'animation_frames_decoded': 41,
            'playback_seconds': 7.38, 'raw_occupancy_panels_rechecked': panels,
            'raw_checks': 'Original validation.json records state/resource/controller checks; compact files cannot repeat them without local raw data or a fresh run.'}


def pilot(base):
    proposals, selection = read(base / 'candidates.json'), read(base / 'selection.json')
    check(len(proposals) == 8 and selection['selected_id'] == 'hand' and selection['loss_improvement_vs_hand'] == 0, 'Pilot selection')
    configurations, trajectories = set(), set()
    for identity in ['fixed', 'hand', *['candidates/'+p['id'] for p in proposals]]:
        directory = base / identity
        config, score = read(directory / 'effective_config.json'), read(directory / 'scores.json')
        z = np.load(directory / 'embeddings.npy', allow_pickle=False)
        check(z.shape == (10, 512) and np.isfinite(z).all() and np.allclose(np.linalg.norm(z, axis=1), 1, atol=1e-4), 'Embeddings')
        loss = float(np.tril(z @ z.T, k=-1).max(axis=1).mean())
        check(abs(loss-score['loss_oe']) < 1e-6 and abs(0.9-loss-score['novelty']) < 1e-6, 'Saved ASAL objective')
        trajectory = rows(directory / 'measurements.csv')
        check([int(r['completed_update']) for r in trajectory] == list(range(0, 201, 20)), 'Pilot sample coverage')
        check(trajectory == [{k: str(v) if v is not None else '' for k,v in r.items()} for r in score['measurements']], 'Pilot table/score mismatch')
        if identity.startswith('candidates/'):
            proposal = next(p for p in proposals if identity.endswith(p['id']))
            effective = tuple(config['env_options'][k] for k in ('shift_interval', 'shift_cells'))
            check(effective == (proposal['shift_interval'], proposal['shift_cells']), 'Rounded proposal/config mismatch')
            check(proposal['loss_oe'] == score['loss_oe'], 'Candidate objective')
            configurations.add(effective)
            trajectories.add(tuple(r['occupied_cells'] for r in trajectory))
    check('PRE-FIX' in (base / 'usage.txt').read_text(), 'Historical label')
    return {'scope': 'PRE-FIX controller; no corrected novelty-search result', 'candidates': 8,
            'unique_effective_proposal_configurations': len(configurations),
            'unique_sampled_occupancy_count_trajectories': len(trajectories),
            'saved_embedding_sequences_checked': 10, 'selection': 'hand; no improvement'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=DEFAULT)
    parser.add_argument('--pilot', type=Path, default=PILOT)
    parser.add_argument('--raw', action='store_true', help='Also read omitted local raw states to recheck GIF occupancy panels')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = {'corrected': corrected(args.results, args.raw), 'historical_pilot': pilot(args.pilot)}
    if args.output:
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

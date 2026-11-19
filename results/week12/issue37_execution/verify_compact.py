"""Check the shared issue-37 evidence without raw captures or model weights."""
import ast
import csv
import json
import math
from pathlib import Path
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'results/week12/issue37_novelty_multiseed'


def load(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label, tolerance=1e-12):
    require(math.isfinite(float(actual)) and abs(float(actual) - float(expected)) <= tolerance,
            f'{label}: {actual} != {expected}')


def main():
    settings = load(OUTPUT / 'search_settings.json')
    base = load(OUTPUT / 'base_config.json')
    seeds = settings['search_seeds']
    updates = [0, *settings['sampled_updates']]
    require(seeds == [303, 7, 13] and settings['comparison_seeds'] == [101, 202, 404], 'Seed plan')
    require(settings['spread_penalty'] == 1.0 and settings['scoring_view'] == 'occupancy', 'Objective settings')
    require(updates == list(range(0, 201, 20)), 'Sampling plan')
    checked = []

    def run(directory, seed, parameters):
        score = load(directory / 'scores.json')
        config = load(directory / 'effective_config.json')
        expected = dict(base['env_options'], **parameters)
        require(config['seed'] == seed and config['env_options'] == expected, f'Configuration: {directory}')
        for key in ('steps', 'shape', 'backend', 'device', 'env', 'env_param', 'infra_decay',
                    'defense_coeff', 'radiate_interval', 'cleanup_interval', 'emergency_pop_cap'):
            require(config[key] == base[key], f'Scientific setting {key}: {directory}')
        require([m['completed_update'] for m in score['measurements']] == updates, 'Completed updates')
        embeddings = np.load(directory / 'embeddings.npy', allow_pickle=False)
        require(embeddings.shape == (10, 512) and np.isfinite(embeddings).all(), 'Embedding shape/values')
        require(np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-4), 'Embedding normalization')
        loss = np.tril(embeddings @ embeddings.T, k=-1).max(axis=1).mean()
        close(score['loss_oe'], loss, 'Embedding rescore', 3e-7)
        close(score['novelty'], 0.9 - score['loss_oe'], 'Novelty display')
        checked.append(directory)
        return score, config

    def world(directory, parameters):
        values = [run(directory if index == 0 else directory / f'seed_{seed}', seed, parameters)[0]['loss_oe']
                  for index, seed in enumerate(seeds)]
        mean = statistics.mean(values)
        spread = statistics.stdev(values)
        return dict(per_seed_loss=values, mean_loss=mean, loss_std=spread, selection_loss=mean + spread)

    def summary_matches(saved, expected):
        for key in ('mean_loss', 'loss_std', 'selection_loss'):
            close(saved[key], expected[key], key)
        require(saved['per_seed_loss'] == expected['per_seed_loss'], 'Per-seed losses')

    fixed, _ = run(OUTPUT / 'fixed', 303, {'shift_interval': 0})
    hand = world(OUTPUT / 'hand', base['env_options'])
    selection = load(OUTPUT / 'selection.json')
    summary_matches(selection['hand'], hand)
    candidates = load(OUTPUT / 'candidates.json')
    with (OUTPUT / 'candidates.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    require(len(candidates) == len(rows) == 64 and len({c['id'] for c in candidates}) == 64, 'Proposal coverage')
    unique = set()
    best_id, best = 'hand', hand
    parameters_by_id = {}
    for candidate, row in zip(candidates, rows):
        parameters = {}
        for value, (key, bounds) in zip(candidate['vector'], settings['parameters'].items()):
            value = bounds['min'] + min(1.0, max(0.0, value)) * (bounds['max'] - bounds['min'])
            parameters[key] = int(round(value)) if bounds['integer'] else value
            require(candidate[key] == parameters[key] == int(row[key]), 'Effective rounding')
        unique.add(tuple(parameters.items()))
        parameters_by_id[candidate['id']] = parameters
        expected = world(OUTPUT / 'candidates' / candidate['id'], parameters)
        summary_matches(candidate, expected)
        require(row['id'] == candidate['id'], 'Candidate CSV identity')
        for key in ('mean_loss', 'loss_std', 'selection_loss'):
            close(row[key], candidate[key], f'Candidate CSV {key}')
        require(ast.literal_eval(row['per_seed_loss']) == expected['per_seed_loss'], 'Candidate CSV seeds')
        if expected['selection_loss'] < best['selection_loss'] - 1e-7:
            best_id, best = candidate['id'], expected
    require(len(unique) == 45 and best_id == selection['selected_id'] == 'g02_c02', 'Unique configurations/selection')
    summary_matches(selection['selected'], best)
    close(selection['selection_loss_improvement_vs_hand'], hand['selection_loss'] - best['selection_loss'], 'Selection gain')
    selected = parameters_by_id[best_id]
    require(selected == {'shift_interval': 26, 'shift_cells': 6}, 'Selected parameters')
    require(load(OUTPUT / 'selected_config.json')['env_options'] == dict(base['env_options'], **selected), 'Selected config')

    with (OUTPUT / 'comparison.csv').open(newline='') as stream:
        primary_rows = list(csv.DictReader(stream))
    require(len(primary_rows) == 3, 'Primary comparison coverage')
    primary = [('fixed', OUTPUT / 'fixed'), ('hand', OUTPUT / 'hand'), ('selected', OUTPUT / 'candidates' / best_id)]
    for row, (label, directory) in zip(primary_rows, primary):
        score = load(directory / 'scores.json')
        require(row['configuration'] == label, 'Primary comparison label')
        close(row['loss_oe_lower_is_better'], score['loss_oe'], 'Primary comparison score')
        require(int(row['final_occupied']) == score['measurements'][-1]['occupied_cells'], 'Primary endpoint count')

    native = load(OUTPUT / 'heldout/per_seed.json')
    with (OUTPUT / 'heldout/comparison.csv').open(newline='') as stream:
        heldout_rows = list(csv.DictReader(stream))
    require(len(native) == len(heldout_rows) == 3, 'Held-out coverage')
    hand_values, selected_values = [], []
    for seed, saved, row in zip(settings['comparison_seeds'], native, heldout_rows):
        require(seed == saved['seed'] == int(row['seed']), 'Held-out seed identity')
        pair = []
        for label, parameters, values in (('hand', base['env_options'], hand_values), ('selected', selected, selected_values)):
            score, config = run(OUTPUT / 'heldout' / f'seed_{seed}' / label, seed, parameters)
            values.append(score['loss_oe'])
            close(saved[label + '_loss'], score['loss_oe'], 'Held-out saved loss')
            close(row[label + '_loss'], score['loss_oe'], 'Held-out CSV loss')
            pair.append({k: v for k, v in config.items() if k not in ('env_options', 'output_dir')})
        require(pair[0] == pair[1], 'Matched held-out conditions')
        close(row['paired_difference_hand_minus_selected'], hand_values[-1] - selected_values[-1], 'Paired CSV difference')
    differences = [a - b for a, b in zip(hand_values, selected_values)]
    heldout = load(OUTPUT / 'heldout/summary.json')
    for key, value in (('hand_mean_loss', statistics.mean(hand_values)), ('selected_mean_loss', statistics.mean(selected_values)),
                       ('paired_difference_mean', statistics.mean(differences)), ('paired_difference_sample_std', statistics.stdev(differences))):
        close(heldout[key], value, key)
    require(heldout['paired_differences'] == differences, 'Paired summary differences')
    require(heldout['improved_seeds'] == sum(d > 1e-7 for d in differences) == 1 and heldout['held_up'] is False, 'Held-out result')
    require(len(checked) == 202, 'Rollout coverage')
    completed = load(OUTPUT / 'completed.json')
    require(completed['candidate_worlds'] == 64 and completed['rollouts_per_world'] == 3 and completed['heldout_ran'], 'Completion record')
    print('PASS: 202 saved rollouts; 64 proposals / 45 unique configurations; selected 26/6; held-out improvement 1/3.')
    print('Compact scores, configurations, embeddings and summaries agree. Raw-state/render checks remain in the prior validation reports.')


if __name__ == '__main__':
    main()

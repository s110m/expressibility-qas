"""Table of the compiler-baseline comparison over the nine classification cells
(Access revision, Reviewer 1 #7 / Reviewer 2 #5).

Joins <run>/compiler_baselines/<backend>.json (each tool's layout, its success
rate rank, its 300-sample sweep rank) with the full-split accuracies that
run_compiler_baseline_evals.py measured locally (eval_results/*.local_*.json),
matching tools to measurements by layout, and writes
success_rate_ablation/compiler_baselines_summary.json.

Usage (repo root): python examples/summarize_compiler_baselines.py
"""
import glob
import json
import os

import numpy as np
from scipy import stats

TOOLS = ('joint_search', 'enumeration', 'oracle', 'qiskit_l2', 'qiskit_l3',
         'mapomatic', 'tket')


def main():
    cells = {}
    for path in sorted(glob.glob('runs/*nsga2_best_s10?_seed42/compiler_baselines/*V2.json')):
        run = path.split(os.sep + 'compiler_baselines')[0].split('/compiler_baselines')[0]
        rec = json.load(open(path))
        by_layout = {}
        for p in glob.glob(os.path.join(run, 'eval_results', '*.local_*.json')):
            r = json.load(open(p))
            if r.get('layout') is not None and 'qiskit_layout' not in p:
                by_layout[tuple(r['layout'])] = r['accuracy_percent']
        free = {lvl: json.load(open(glob.glob(os.path.join(
            run, 'eval_results', f'*.qiskit_layout.local_l{lvl}.json'))[0]))['accuracy_percent']
            for lvl in (2, 3)}
        oracle = json.load(open(glob.glob(os.path.join(
            run, 'eval_results', '*.local_oracle_layout.json'))[0]))
        tools = dict(rec['tools'])
        tools['oracle'] = {'layout': oracle['layout']}
        cell = {}
        for t in TOOLS:
            lay = tuple(tools[t]['layout'])
            cell[t] = {'layout': list(lay), 'accuracy_percent': by_layout.get(lay),
                       'success_rate_rank': tools[t].get('success_rate_rank'),
                       'sweep300_rank': tools[t].get('sweep300_rank')}
        cell['qiskit_default'] = {'accuracy_percent': free[2]}
        cell['qiskit_default_l3'] = {'accuracy_percent': free[3]}
        name = f"{rec['backend']}_s{run.split('nsga2_best_s')[1][:3]}"
        cells[name] = cell

    conds = list(next(iter(cells.values())))
    enum = np.array([c['enumeration']['accuracy_percent'] for c in cells.values()])
    summary = {}
    for k in conds:
        acc = np.array([c[k]['accuracy_percent'] for c in cells.values()])
        ranks = [c[k].get('success_rate_rank') for c in cells.values()]
        d = enum - acc
        t = stats.ttest_rel(enum, acc) if k != 'enumeration' else None
        summary[k] = {
            'mean_accuracy': float(acc.mean()),
            'mean_success_rate_rank': float(np.mean(ranks)) if None not in ranks else None,
            'n_same_layout_as_enumeration': int(sum(
                c[k].get('layout') == c['enumeration']['layout'] for c in cells.values())),
            'enumeration_minus': float(d.mean()),
            'enumeration_wins_losses_ties': [int((d > 1e-9).sum()), int((d < -1e-9).sum()),
                                             int((abs(d) <= 1e-9).sum())],
            't': None if t is None else float(t.statistic),
            'p': None if t is None else float(t.pvalue)}
        print(f"{k:18s} acc {acc.mean():6.2f}  rank {summary[k]['mean_success_rate_rank']}  "
              f"same {summary[k]['n_same_layout_as_enumeration']}/9  enum-minus {d.mean():+.2f}  "
              f"w/l/t {summary[k]['enumeration_wins_losses_ties']}  "
              f"t={summary[k]['t']}  p={summary[k]['p']}")
    for name, c in cells.items():
        print(name, '  '.join(f"{k}={c[k]['accuracy_percent']:.2f}" for k in conds))
    os.makedirs('success_rate_ablation', exist_ok=True)
    json.dump({'cells': cells, 'summary': summary},
              open('success_rate_ablation/compiler_baselines_summary.json', 'w'), indent=1)


if __name__ == '__main__':
    main()

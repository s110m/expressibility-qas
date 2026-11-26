"""Genome-ablation table (tab:genome) and summary numbers from the result files.

Analytic part: every setting x device x search seed.  Accuracy part: the three
retrained settings, from eval_cell.sh results (gj_ = deployed layout, ge_ =
enumerated layout on the same checkpoint and device)."""
import glob, json, os, collections
import numpy as np
from scipy import stats
GA = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(GA)
ORDER = [('p0.05', r'$0.05$', r'$0.05$'), ('p0.1', r'$0.1$', r'$0.1$'),
         ('p0.2', r'$0.2$', r'$0.2$'), ('p0.4', r'$0.4$', r'$0.4$'),
         ('tournament', r'$0.4$', r'$0.4$'), ('split', r'$0.1$', r'$0.5$'),
         ('staged', r'$0.4$', r'$0.5$')]
cov = json.load(open(os.path.join(GA, 'deployed_coverage.json')))
DEV = {'FakeQuitoV2': 'quito', 'FakeAthensV2': 'athens', 'FakeBelemV2': 'belem'}

canon = {}
for f in glob.glob(os.path.join(GA, 'eval3_jobs_*.txt')):
    for line in open(f):
        if line.strip():
            s, dev, seed, cs, cdev = line.split()
            canon[(s, dev, int(seed))] = (cs, cdev)

def acc(s, dev, seed):
    if (s, dev, seed) not in canon:
        return None
    cs, cdev = canon[(s, dev, seed)]
    run = glob.glob(os.path.join(ROOT, 'runs', f'*.scratch.{cdev}.*ga_{cs.replace(".", "")}_s{seed}_seed42'))
    if not run:
        return None
    tag = f'{s.replace(".", "")}_{dev}_s{seed}'
    r = {}
    for k in ('gj', 'ge'):
        g = glob.glob(os.path.join(run[0], 'eval_results', f'*.{k}_{tag}.json'))
        if not g:
            return None
        r[k] = json.load(open(g[0]))['accuracy_percent']
    return r

summary, rows = {}, []
for s, pa, pl in ORDER:
    runs = []
    for f in sorted(glob.glob(os.path.join(GA, s, '*_s1*.json'))):
        if f.endswith(('partial.json', 'fromlog.json')):
            continue
        d = json.load(open(f))
        a = d['deployed']
        runs.append({'backend': d['backend'], 'seed': d['seed'],
                     'rank': a['success_rate_rank'], 'blocks': a['n_blocks'],
                     'params': a['n_params'], 'events': d['same_arch_events'],
                     'acc': acc(s, DEV[d['backend']], d['seed'])})
    rk = np.array([r['rank'] for r in runs], float)
    accs = [r['acc'] for r in runs if r['acc']]
    gap = np.array([x['ge'] - x['gj'] for x in accs]) if accs else None
    summary[s] = {'n_runs': len(runs), 'rank_mean': rk.mean(), 'rank_median': float(np.median(rk)),
                  'top5pct': int((rk <= 6).sum()),
                  'events_mean': float(np.mean([r['events'] for r in runs])),
                  'layouts_tried_median': cov[s]['layouts_tried_median'],
                  'blocks_mean': float(np.mean([r['blocks'] for r in runs])),
                  'n_acc': len(accs), 'runs': runs}
    if accs:
        summary[s].update(acc_deployed=float(np.mean([x['gj'] for x in accs])),
                          acc_enum=float(np.mean([x['ge'] for x in accs])),
                          gap_mean=float(gap.mean()),
                          gap_t=float(stats.ttest_1samp(gap, 0).statistic) if len(gap) > 1 else None)
    g = summary[s]
    accs_txt = (f"${g['acc_deployed']:.2f}$ & ${g['gap_mean']:.2f}$" if accs else r'--- & ---')
    rows.append(f"        {s if s not in ('split','staged','tournament') else s.capitalize()} & {pa} & {pl} & "
                f"${g['layouts_tried_median']:.0f}$ & ${g['rank_mean']:.1f}$ & ${g['top5pct']}/{g['n_runs']}$ & {accs_txt} \\\\")
json.dump(summary, open(os.path.join(GA, 'table_summary.json'), 'w'), indent=1, default=float)
print('\n'.join(rows))
for s, g in summary.items():
    print(s, {k: v for k, v in g.items() if k != 'runs'})

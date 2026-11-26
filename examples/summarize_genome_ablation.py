"""Summary table of the NSGA-II genome ablation (nsga2_genome_ablation.py).

Per setting, over the 3 backends x 3 search seeds: same-architecture layout
comparisons counted in the search logs, against the Eq. (mutation) prediction
(n_offspring x (1-p_arch)^17 x (1-(1-p_layout)^4), which ignores crossover);
improving mapping moves; the deployed solution's size, expressibility and the
analytic rank of its layout among all 120; and, where retrained, full-split
accuracy at the searched and at the enumerated layout.

Writes genome_ablation/summary.json.

Usage (repo root): python examples/summarize_genome_ablation.py
"""
import glob
import json
import os

import numpy as np
from scipy import stats

ORDER = ('p0.4', 'p0.2', 'p0.1', 'p0.05', 'split', 'tournament', 'staged')
DEV = {'FakeAthensV2': 'athens', 'FakeQuitoV2': 'quito', 'FakeBelemV2': 'belem'}


def accuracy(setting, backend, seed, suffix):
    leaf = f"ga_{setting.replace('.', '')}_s{seed}"
    run = (f"runs/mnist.four0123.train.searched.scratch.{DEV[backend]}.noise.opt2."
           f"setting0.u3cu3_s0.ldiff_blkexpand.{leaf}_seed42")
    hits = glob.glob(os.path.join(run, 'eval_results', f'*.{suffix}.json'))
    return json.load(open(hits[0]))['accuracy_percent'] if hits else None


def main():
    out = {}
    base = {}
    for setting in ORDER:
        recs = [json.load(open(p)) for p in sorted(glob.glob(f'genome_ablation/{setting}/*.json'))]
        if not recs:
            continue
        rows = []
        for r in recs:
            pa, pl = r['rates']['arch'], r['rates']['layout']
            pred = None if setting == 'staged' else \
                r['n_offspring_total'] * (1 - pa) ** 17 * (1 - (1 - pl) ** 4)
            d = r['deployed']
            rows.append({
                'cell': f"{r['backend']}_s{r['seed']}",
                'events': r['same_arch_events'], 'predicted': pred,
                'improving': r['improving_moves'],
                'blocks': d['n_blocks'], 'params': d['n_params'],
                'expr': d['expressibility'], 'sr_rank': d['success_rate_rank'],
                'front': r['front_size'], 'minutes': r['search_seconds'] / 60,
                'acc_joint': accuracy(setting, r['backend'], r['seed'], 'ga_joint'),
                'acc_enum': accuracy(setting, r['backend'], r['seed'], 'ga_enum'),
            })
        m = lambda k: float(np.mean([x[k] for x in rows if x[k] is not None])) \
            if any(x[k] is not None for x in rows) else None
        summ = {k: m(k) for k in ('events', 'predicted', 'improving', 'blocks', 'params',
                                  'expr', 'sr_rank', 'front', 'minutes',
                                  'acc_joint', 'acc_enum')}
        summ['n'] = len(rows)
        summ['sr_rank_top10'] = int(sum(x['sr_rank'] <= 10 for x in rows))
        summ['sr_rank_is_1'] = int(sum(x['sr_rank'] == 1 for x in rows))
        if setting == 'p0.4':
            base = {x['cell']: x for x in rows}
        elif base:
            paired = [(x, base[x['cell']]) for x in rows if x['cell'] in base]
            dr = [x['sr_rank'] - b['sr_rank'] for x, b in paired]
            summ['paired_sr_rank_diff'] = float(np.mean(dr))
            summ['paired_sr_rank_t'] = float(stats.ttest_rel(
                [x['sr_rank'] for x, _ in paired], [b['sr_rank'] for _, b in paired]).statistic) \
                if len(paired) > 1 else None
        gaps = [x['acc_enum'] - x['acc_joint'] for x in rows
                if x['acc_enum'] is not None and x['acc_joint'] is not None]
        summ['enum_minus_joint'] = float(np.mean(gaps)) if gaps else None
        summ['n_retrained'] = len(gaps)
        out[setting] = {'summary': summ, 'rows': rows}
        print(f"{setting:10s} n={summ['n']} events {summ['events']:.1f} (pred "
              f"{summ['predicted'] if summ['predicted'] is None else round(summ['predicted'], 1)}) "
              f"improving {summ['improving']:.1f} blocks {summ['blocks']:.1f} params "
              f"{summ['params']:.0f} Expr {summ['expr']:.4f} S-rank {summ['sr_rank']:.1f} "
              f"(top10 {summ['sr_rank_top10']}, =1 {summ['sr_rank_is_1']}) "
              f"acc joint {summ['acc_joint']} enum {summ['acc_enum']} "
              f"gap {summ['enum_minus_joint']} [{summ['minutes']:.0f} min]")
    json.dump(out, open('genome_ablation/summary.json', 'w'), indent=1)


if __name__ == '__main__':
    main()

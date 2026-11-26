"""Does any M4 arm still beat M3 once circuit size is controlled for?

The raw M4 table is dominated by a size effect the paper already characterised:
noisy accuracy rises with ln(params), so an arm that selects smaller circuits
loses accuracy for a reason that has nothing to do with whether its search
ranked candidates well. `esd/cap` is the extreme case -- its p50 depth cap
selected 24-39 parameter circuits against M3's 54-156, and reads -12.05 pp raw.
This regresses size (and device) out and asks what is left of each arm.

The answer, on the 2026-09-01 runs: almost nothing survives on either side.
`esd/cap`'s -12.05 pp collapses to -2.12 pp (t=-1.70) -- it was the size law,
not a search failure. `esd/best`'s +0.40 pp rises only to +1.08 pp (t=+1.24),
still short of significance, so the best arm remains a tie with M3. The one
arm that is genuinely worse than its size predicts is `ess/cap` at -6.09 pp
(t=-3.96): capping swap_count drags expressibility from ~0.030 to 0.10-0.18,
well off the optimum, and the circuits underperform accordingly.

Sizes come from `n_sampled_params`, which eval.py records in every
eval_results/*.json, so nothing here needs a checkpoint or a model rebuild.
The fit is calibrated on all 45 rows (36 M4 + 9 M3) rather than per arm: arms
barely overlap in size, so a per-arm fit would extrapolate.

Run after `collect_m4_results.py`, from the repo root.
"""
import glob
import json
import math
import os
import statistics

import numpy as np

DEVICES = ('athens', 'quito', 'belem')
SEEDS = (100, 101, 102)
BASE = 'mnist.four0123'
ARMS = ('M3', 'esd/best', 'esd/cap', 'ess/best', 'ess/cap')


def run_dir(fam, dev, seed, arm):
    return (f'runs/{BASE}.train.searched.scratch.{dev}.noise.opt2.setting0.'
            f'u3cu3_s0.ldiff_blkexpand.nsga2{fam}_{arm}_s{seed}_seed42')


def m3_run_dir(dev, seed):
    return (f'runs/{BASE}.train.searched.scratch.{dev}.noise.opt2.setting0.'
            f'u3cu3_s0.ldiff_blkexpand.nsga2_best_s{seed}_seed42')


def full_split_eval(rdir, suffix):
    """The n=4157 eval for ONE layout variant -- never a max over the dir.

    Same discipline as collect_m4_results.full_split_acc: taking a max would
    silently report the enumerated or oracle layout as if the search had found
    it. `suffix` is eval.py's --result-suffix.
    """
    matches = sorted(glob.glob(
        os.path.join(rdir, 'eval_results', f'*.all.{suffix}.json')))
    if not matches:
        return None
    try:
        d = json.load(open(matches[0]))
    except Exception:
        return None
    return d if d.get('n_samples') == 4157 else None


def load_rows():
    rows = []
    for fam in ('esd', 'ess'):
        for dev in DEVICES:
            for seed in SEEDS:
                for arm in ('best', 'cap'):
                    d = full_split_eval(run_dir(fam, dev, seed, arm), 'm4_layout')
                    if d:
                        rows.append(dict(arm=f'{fam}/{arm}', dev=dev, seed=seed,
                                         acc=d['accuracy_percent'],
                                         params=d['n_sampled_params'],
                                         blocks=d['solution']['arch'][-1],
                                         model_size=d['model_size']))
    for dev in DEVICES:
        for seed in SEEDS:
            d = full_split_eval(m3_run_dir(dev, seed), 'nsga2_layout')
            if d:
                rows.append(dict(arm='M3', dev=dev, seed=seed,
                                 acc=d['accuracy_percent'],
                                 params=d['n_sampled_params'],
                                 blocks=d['solution']['arch'][-1],
                                 model_size=d['model_size']))
    return rows


def design(rows):
    """ln(params) plus device fixed effects; athens is the reference level."""
    cols = [np.ones(len(rows)),
            np.array([math.log(r['params']) for r in rows])]
    for dev in ('belem', 'quito'):
        cols.append(np.array([1.0 if r['dev'] == dev else 0.0 for r in rows]))
    return np.column_stack(cols), np.array([r['acc'] for r in rows])


def main():
    rows = load_rows()
    if not rows:
        print('no finished n=4157 evals found -- run the pipeline first')
        return

    sizes = sorted({r['model_size'] for r in rows})
    print(f'model_size values present: {sizes}   (192 = pure quantum)')
    if sizes != [192]:
        print('  !! a hybrid run is mixed in -- the size law does not transfer')

    print(f"\n=== what each arm selected ===")
    print(f"{'arm':<10} {'n':>2} {'params':>12} {'mean':>7} {'blocks':>7} "
          f"{'raw acc':>8}")
    for a in ARMS:
        g = [r for r in rows if r['arm'] == a]
        if not g:
            continue
        p = [r['params'] for r in g]
        print(f"{a:<10} {len(g):>2} {min(p):>5}-{max(p):<6} "
              f"{statistics.mean(p):>7.1f} "
              f"{statistics.mean([r['blocks'] for r in g]):>7.1f} "
              f"{statistics.mean([r['acc'] for r in g]):>8.2f}")

    X, y = design(rows)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    r2 = 1 - resid.var() / y.var()
    print(f"\n=== size + device fit, calibrated on all {len(rows)} rows ===")
    print(f"  acc = {beta[0]:.2f} + {beta[1]:.2f}*ln(params) "
          f"{beta[2]:+.2f}*belem {beta[3]:+.2f}*quito     R2={r2:.3f}")
    print("  (the pure track's published fit is acc = 35.5 + 9.4*ln N)")

    print(f"\n=== residual after removing size and device ===")
    for a in ARMS:
        idx = [i for i, r in enumerate(rows) if r['arm'] == a]
        if not idx:
            continue
        v = resid[idx]
        sd = v.std(ddof=1) if len(v) > 1 else 0.0
        print(f"  {a:<10} n={len(idx):2}  {v.mean():+6.2f} pp  (sd {sd:.2f})")

    print(f"\n=== size-controlled paired difference vs M3, same (backend, seed) ===")
    print('    a difference still needs ~1.5 pp to be worth reporting')
    m3idx = {(r['dev'], r['seed']): i for i, r in enumerate(rows)
             if r['arm'] == 'M3'}
    for a in ARMS[1:]:
        d = [resid[i] - resid[m3idx[(r['dev'], r['seed'])]]
             for i, r in enumerate(rows)
             if r['arm'] == a and (r['dev'], r['seed']) in m3idx]
        if len(d) < 2:
            print(f"  {a:<10} -- not enough paired cells")
            continue
        se = statistics.stdev(d) / math.sqrt(len(d))
        t = statistics.mean(d) / se if se else float('nan')
        print(f"  {a:<10} n={len(d):2}  {statistics.mean(d):+6.2f} pp  "
              f"SE {se:4.2f}  t={t:+5.2f}")


if __name__ == '__main__':
    main()

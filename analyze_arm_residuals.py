"""Does expressibility still beat loss_succ once circuit size is controlled for?

Raw full-split noisy accuracy favours the expressibility arms (B/C) by ~9 pp,
but they also select circuits 2.7x larger, and PAPER_STORY_PURE's own fit says
noisy_acc rises with ln(params). This regresses size out and asks what is left.

Size calibration optionally borrows the `_nopool` retrains: their SEARCH is
invalid (defective supernet) but their from-scratch accuracies are real
observations of the size-accuracy relation, and they are the only data covering
the 40-84 param band where arms A/D and B/C would otherwise barely overlap.
"""
import json
import math
import sys

import numpy as np

DEV = {'athens': 'FakeAthensV2', 'belem': 'FakeBelemV2', 'quito': 'FakeQuitoV2'}
ARMNAME = {'A': 'A loss_succ+evo', 'B': 'B expr+evo',
           'C': 'C expr+random', 'D': 'D loss_succ+random'}
NOPOOL = sys.argv[1] if len(sys.argv) > 1 else None


def load_arms():
    """One row per (circuit, eval device). Arms A/D search per device, so their
    circuit differs by device; arms B/C search once on athens and transfer."""
    rows = []
    for r in json.load(open('all_arms.json')):
        arm = r['arm']
        devs = [r['search_dev']] if arm in 'AD' else list(DEV)
        for dev in devs:
            ev = r['evals'].get(f'{DEV[dev]}|4157')
            if not ev:
                continue
            rows.append(dict(arm=arm, seed=r['seed'], dev=dev, acc=ev['acc'],
                             params=ev['params'], score=r['score'],
                             circuit=f"{arm}{r['seed']}{r['search_dev']}"))
    return rows


def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    r2 = 1 - resid.var() / y.var()
    return beta, resid, r2


def design(rows, device_terms):
    lp = np.array([math.log(r['params']) for r in rows])
    cols = [np.ones(len(rows)), lp]
    if device_terms:                      # athens is the reference level
        for d in ('belem', 'quito'):
            cols.append(np.array([1.0 if r['dev'] == d else 0.0 for r in rows]))
    return np.column_stack(cols), np.array([r['acc'] for r in rows])


def report(rows, calib, device_terms, label):
    """Fit on `calib`, score residuals for `rows`."""
    Xc, yc = design(calib, device_terms)
    beta, _, r2 = ols(Xc, yc)
    Xr, yr = design(rows, device_terms)
    resid = yr - Xr @ beta
    terms = f"{beta[0]:.2f} + {beta[1]:.2f}*ln(params)"
    if device_terms:
        terms += f" + {beta[2]:+.2f}*belem {beta[3]:+.2f}*quito"
    print(f"\n{label}\n  fit (n={len(calib)}): noisy_acc = {terms}   R2={r2:.3f}")
    for arm in 'ABCD':
        idx = [i for i, r in enumerate(rows) if r['arm'] == arm]
        if not idx:
            continue
        # cluster to the circuit: arms B/C reuse one circuit across 3 devices
        by_circ = {}
        for i in idx:
            by_circ.setdefault(rows[i]['circuit'], []).append(resid[i])
        cl = np.array([np.mean(v) for v in by_circ.values()])
        print(f"  {ARMNAME[arm]:20} residual {cl.mean():+6.2f} pp  "
              f"(sd {cl.std(ddof=1) if len(cl) > 1 else 0:4.2f}, "
              f"{len(cl)} circuits, {len(idx)} evals)")
    grp = {}
    for g, arms in (('expressibility (B+C)', 'BC'), ('loss_succ (A+D)', 'AD')):
        by_circ = {}
        for i, r in enumerate(rows):
            if r['arm'] in arms:
                by_circ.setdefault(r['circuit'], []).append(resid[i])
        grp[g] = np.array([np.mean(v) for v in by_circ.values()])
    a, b = grp['expressibility (B+C)'], grp['loss_succ (A+D)']
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    print(f"  --> expressibility - loss_succ = {a.mean() - b.mean():+.2f} pp "
          f"(SE {se:.2f}, {a.mean() - b.mean():+.2f}/{se:.2f} = "
          f"{(a.mean() - b.mean()) / se:+.1f} sigma)")


def main():
    rows = load_arms()
    calib = list(rows)
    if NOPOOL:
        extra = [r for r in json.load(open(NOPOOL)) if r['variant'] == 'nopool'
                 and r.get('eval_n') == 4157]
        for r in extra:
            calib.append(dict(arm='N', seed=r['seed'], dev=r['backend'],
                              acc=r['noisy_acc'], params=r['n_params'],
                              score=r['es_score'], circuit=f"N{r['seed']}{r['backend']}"))
        print(f"borrowed {len(extra)} _nopool rows for size calibration only")

    print("=== common support: params selected by each arm ===")
    for arm in 'ABCD':
        p = sorted({r['params'] for r in rows if r['arm'] == arm})
        print(f"  {ARMNAME[arm]:20} n={len([r for r in rows if r['arm']==arm]):2}  "
              f"params {min(p):3}-{max(p):3}  {p}")
    ad = [r['params'] for r in rows if r['arm'] in 'AD']
    bc = [r['params'] for r in rows if r['arm'] in 'BC']
    print(f"  loss_succ range {min(ad)}-{max(ad)} vs expressibility {min(bc)}-{max(bc)}"
          f"  ->  overlap {max(min(ad), min(bc))}-{min(max(ad), max(bc))}")
    if NOPOOL:
        np_ = [r['params'] for r in calib if r['arm'] == 'N']
        print(f"  _nopool calibration adds params {min(np_)}-{max(np_)}")

    report(rows, rows, False, "=== [1] pooled fit on the 36 arm rows (prior method) ===")
    report(rows, rows, True, "=== [2] + device fixed effects ===")
    if NOPOOL:
        report(rows, calib, True, "=== [3] + device FE, size curve calibrated on arms + _nopool ===")

    print("\n=== matched-size spot check (circuits within +/-15% params) ===")
    ad_r = [r for r in rows if r['arm'] in 'AD']
    bc_r = [r for r in rows if r['arm'] in 'BC']
    pairs = [(x, y) for x in ad_r for y in bc_r
             if x['dev'] == y['dev'] and abs(math.log(x['params'] / y['params'])) < 0.14]
    if not pairs:
        print("  none -- the two families select disjoint circuit sizes")
    for x, y in pairs:
        print(f"  {x['dev']:7} {x['arm']}{x['seed']} {x['params']:3}p {x['acc']:6.2f}  vs  "
              f"{y['arm']}{y['seed']} {y['params']:3}p {y['acc']:6.2f}   "
              f"delta {y['acc'] - x['acc']:+.2f} pp")


if __name__ == '__main__':
    main()

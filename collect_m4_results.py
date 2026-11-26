"""Collect the M4 three-objective results into one table.

Two families, two deployment rules each, against the two-objective M3 baseline
at the same search seeds:

    esd  [expressibility, success_rate, depth]        architecture counterweight
    ess  [expressibility, success_rate, swap_count]   mapping signal
    arm "best"  argmin expressibility          -- the paper's existing rule
    arm "cap"   argmin expressibility, 3rd objective <= p50

Reports, per cell: the selected architecture's block count, all three objective
values, the layout, the full-split (n=4157) noisy accuracy, and -- when the
120-layout sweep has run -- the rank of the selected layout among all 120.

THE RANK COLUMN IS THE ONE THAT ANSWERS THE MAPPING QUESTION -- accuracy alone
cannot separate a better mapping from a better architecture. Judge it against
the PER-RUN null this script computes, not the flat 60.5 uniform-draw figure:
competition ranking counts ties as not-better, so a tie-heavy run's null sits
near 53-55, and 60.5 would credit an arm for its tie structure. M3 is random
either way (59.8 against a 54.9 null). M4's `ess/cap` is the arm this changes:
z=-2.06 "below null" against 60.5, but t=-1.91 against its own 53.6, which does
not survive four-arm correction. See `layout_rank_stats` for the full argument.

RESULT, 2026-09-01 (all 36 runs + sweep complete): no arm beats M3. Best is
esd/best at +0.40 pp (t=+0.38), under the 1.5 pp bar -- a tie. esd/cap's -12 pp
is almost all the size law (only -2.12 pp survives `analyze_m4_size.py`); its
p50 depth cap selected 1.3-1.7 blocks. No arm moves the layout rank. That
confirms the paper's Sec. VII-A diagnosis: the 17-to-4 gene ratio, not the quality
of the mapping signal.

Missing pieces print as '-' rather than crashing, so this is safe to run while
the sweep is still going.
"""
import glob
import json
import math
import os
import statistics
import sys

sys.path.insert(0, 'examples')

try:
    from torchpack.utils import io as tio
except Exception:                                    # torch not importable
    tio = None

DEVICES = ('athens', 'quito', 'belem')
SEEDS = (100, 101, 102)
FAMILIES = (('esd', 'depth'), ('ess', 'swap_count'))
ARMS = (('best', ''), ('cap', '_cap'))
BASE = 'mnist.four0123'

# Uniform-draw null for a rank among 120, from the paper: mean 60.5, sd 34.6.
NULL_RANK_MEAN = 60.5
NULL_RANK_SD = 34.6


def es_dir(fam, dev, seed, suffix=''):
    return (f'es_runs/{BASE}.es.{dev}.noise.opt2.setting0.u3cu3_s0.'
            f'ldiff_blkexpand.nsga2_{fam}.blk8s1.1.1_diff7_chu3_sta40_'
            f'seed{seed}{suffix}')


def run_dir(fam, dev, seed, arm):
    return (f'runs/{BASE}.train.searched.scratch.{dev}.noise.opt2.setting0.'
            f'u3cu3_s0.ldiff_blkexpand.nsga2{fam}_{arm}_s{seed}_seed42')


def m3_run_dir(dev, seed):
    return (f'runs/{BASE}.train.searched.scratch.{dev}.noise.opt2.setting0.'
            f'u3cu3_s0.ldiff_blkexpand.nsga2_best_s{seed}_seed42')


def load_pt(path):
    if tio is None or not os.path.exists(path):
        return None
    try:
        return tio.load(path, weights_only=False)
    except Exception:
        return None


def full_split_acc(rdir, suffix):
    """Full-split (n=4157) noisy accuracy for ONE layout variant of this run.

    `suffix` is eval.py's --result-suffix, which distinguishes the searched
    layout from the enumerated / oracle / qiskit-default layouts evaluated on
    the same checkpoint. Taking a max over the directory would silently report
    the oracle layout as if the search had found it -- Athens seed 101 reads
    79.07 by its own layout and 80.27 by enumeration.
    """
    matches = sorted(glob.glob(
        os.path.join(rdir, 'eval_results', f'*.all.{suffix}.json')))
    if not matches:
        return None
    try:
        d = json.load(open(matches[0]))
    except Exception:
        return None
    return d.get('accuracy_percent') if d.get('n_samples') == 4157 else None


def layout_rank_stats(rdir):
    """Rank of the selected layout among all 120, plus the null it must beat.

    `rank` is ONE PLUS THE NUMBER OF STRICTLY BETTER LAYOUTS (competition
    ranking), which is what tab:joint reports. This is NOT the sweep summary's
    own `selected_layout_rank`: that is `order.index(selected) + 1`, an ordinal
    position that breaks ties by whichever equal-accuracy layout happens to sort
    first. At n=300 every accuracy is a multiple of 1/300, so ties are common
    and the two conventions diverge -- Belem seed 100 reads 32 by the paper's
    definition and 40 by the sweep's. Mixing them would make a mapping look
    worse than it is and would not be comparable to the 59.8 baseline.

    THE NULL IS PER-RUN, NOT 60.5, AND THIS MATTERS. A uniform draw from 120
    distinct values averages rank 60.5, but competition ranking counts a tie as
    NOT strictly better, so ties pull every rank down -- the selected layout's
    included. The honest null is therefore the mean strict rank over a uniform
    draw from *this run's own* 120 accuracies, which lands at 53-55 here rather
    than 60.5, and lower still for a run with more ties. Judging a tie-heavy arm
    against 60.5 credits it for the tie structure: M4's `ess/cap` averages 14.4
    ties against M3's 8.6 and reads z=-2.06 "below null" against 60.5, but only
    t=-1.91 (p~0.09, uncorrected for four arms) against its own 53.6. M3 itself
    is unaffected -- 59.8 against a 54.9 null is still random -- so this does not
    disturb the published two-objective result.

    `midrank` (strict rank plus half the ties beyond the first) is reported as a
    tie-fair cross-check; it moves the same arm to 43.5.

    Recomputed from the per-layout JSONL so every convention is explicit here
    rather than inherited.
    """
    matches = sorted(glob.glob(
        os.path.join(rdir, 'layout_sweep', '*.300.jsonl')))
    if not matches:
        return None
    accs, selected = [], None
    try:
        for line in open(matches[0]):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            accs.append(d['accuracy'])
            if d.get('is_selected_layout'):
                selected = d['accuracy']
    except Exception:
        return None
    if selected is None or not accs:
        return None

    def strict(v):
        return 1 + sum(1 for a in accs if a > v)

    ties = sum(1 for a in accs if a == selected)
    return dict(rank=strict(selected),
                ties=ties,
                midrank=strict(selected) + (ties - 1) / 2.0,
                null=statistics.mean(strict(a) for a in accs),
                spread=(max(accs) - min(accs)) * 100.0)


def fmt(v, spec='{:.2f}'):
    return '-' if v is None else spec.format(v)


def collect():
    rows = []
    for fam, third in FAMILIES:
        for dev in DEVICES:
            for seed in SEEDS:
                for arm, suffix in ARMS:
                    sol = load_pt(os.path.join(
                        es_dir(fam, dev, seed, suffix),
                        'checkpoints/best_solution.pt'))
                    obj = (sol or {}).get('objectives', {})
                    sel = (sol or {}).get('selection', {})
                    arch = (sol or {}).get('solution', {}).get('arch')
                    rd = run_dir(fam, dev, seed, arm)
                    rows.append(dict(
                        fam=fam, third=third, dev=dev, seed=seed, arm=arm,
                        blocks=arch[-1] if arch else None,
                        expr=obj.get('expressibility'),
                        srate=obj.get('success_rate'),
                        third_val=obj.get(third),
                        feasible=sel.get('n_feasible'),
                        layout=(sol or {}).get('solution', {}).get('layout'),
                        acc=full_split_acc(rd, 'm4_layout'),
                        sweep=layout_rank_stats(rd)))

    m3 = []
    for dev in DEVICES:
        for seed in SEEDS:
            rd = m3_run_dir(dev, seed)
            m3.append(dict(fam='M3', third=None, dev=dev, seed=seed,
                           arm='2-obj', blocks=None, expr=None, srate=None,
                           third_val=None, feasible=None, layout=None,
                           acc=full_split_acc(rd, 'nsga2_layout'),
                           sweep=layout_rank_stats(rd)))
    return rows, m3


def main():
    rows, m3 = collect()

    hdr = (f"{'fam':4} {'backend':8} {'seed':>4} {'arm':<5} {'feas':>4} "
           f"{'blk':>3} {'Expr':>8} {'Srate':>7} {'3rd':>7} "
           f"{'layout':<14} {'acc@4157':>9} {'rank/120':>8}")
    print(hdr)
    print('-' * len(hdr))
    for r in rows + m3:
        print(f"{r['fam']:4} {r['dev']:8} {r['seed']:>4} {r['arm']:<5} "
              f"{fmt(r['feasible'], '{:d}'):>4} {fmt(r['blocks'], '{:d}'):>3} "
              f"{fmt(r['expr'], '{:.5f}'):>8} {fmt(r['srate'], '{:.4f}'):>7} "
              f"{fmt(r['third_val'], '{:.0f}'):>7} "
              f"{str(r['layout'] or '-'):<14} {fmt(r['acc']):>9} "
              f"{fmt((r['sweep'] or {}).get('rank'), '{:d}'):>8}")

    groups = [(f'{fam}/{arm}', [r for r in rows if r['fam'] == fam and r['arm'] == arm])
              for fam, _ in FAMILIES for arm, _ in ARMS]
    groups.append(('M3 (2-obj)', m3))

    print('\n=== accuracy, full split (n=4157) ===')
    for name, g in groups:
        v = [r['acc'] for r in g if r['acc'] is not None]
        if v:
            sd = statistics.stdev(v) if len(v) > 1 else 0.0
            print(f"  {name:<11} n={len(v):2}  {statistics.mean(v):6.2f} %  (sd {sd:.2f})")
        else:
            print(f"  {name:<11} n= 0  -- nothing finished yet")

    print('\n=== paired vs M3 at the same (backend, seed) ===')
    print('    a difference needs ~1.5 pp to clear two standard errors at n=9')
    for name, g in groups[:-1]:
        d = []
        for r in g:
            b = next((x['acc'] for x in m3 if x['dev'] == r['dev']
                      and x['seed'] == r['seed']), None)
            if r['acc'] is not None and b is not None:
                d.append(r['acc'] - b)
        if len(d) >= 2:
            sd = statistics.stdev(d)
            se = sd / math.sqrt(len(d))
            t = statistics.mean(d) / se if se else float('nan')
            print(f"  {name:<11} n={len(d):2}  mean {statistics.mean(d):+6.2f} pp"
                  f"  SE {se:4.2f}  t={t:+5.2f}"
                  f"  better in {sum(1 for x in d if x > 0)} of {len(d)}")
        else:
            print(f"  {name:<11} -- not enough paired cells yet")

    print('\n=== selected-layout rank among 120 (1 = best) ===')
    print('    Judged against a PER-RUN null, not the flat 60.5: competition')
    print('    ranking counts a tie as not-better, so a tie-heavy run has a')
    print('    lower null and 60.5 would credit it for its tie structure.')
    print('    "excess" = selected rank minus that run\'s own null; negative is')
    print('    a better-than-random mapping. Paired t, df=n-1, |t|>2.31 for')
    print('    p<.05 at n=9 -- and four arms are being tested, so a bare 0.05')
    print('    does not survive correction.')
    hdr2 = (f"  {'':<11} {'n':>2}  {'rank':>5} {'null':>5} {'excess':>7} "
            f"{'t':>6}  {'midrank':>7} {'ties':>5}  {'z vs 60.5':>9}")
    print(hdr2)
    for name, g in groups:
        v = [r['sweep'] for r in g if r['sweep'] is not None]
        if not v:
            print(f"  {name:<11} n= 0  -- sweep not run yet")
            continue
        rank = statistics.mean(s['rank'] for s in v)
        null = statistics.mean(s['null'] for s in v)
        mid = statistics.mean(s['midrank'] for s in v)
        ties = statistics.mean(s['ties'] for s in v)
        exc = [s['rank'] - s['null'] for s in v]
        if len(exc) > 1:
            se = statistics.stdev(exc) / math.sqrt(len(exc))
            t = statistics.mean(exc) / se if se else float('nan')
        else:
            t = float('nan')
        z = (rank - NULL_RANK_MEAN) / (NULL_RANK_SD / math.sqrt(len(v)))
        print(f"  {name:<11} {len(v):>2}  {rank:>5.1f} {null:>5.1f} "
              f"{statistics.mean(exc):>+7.1f} {t:>+6.2f}  {mid:>7.1f} "
              f"{ties:>5.1f}  {z:>+9.2f}")
    print('    The last column is the superseded flat-60.5 z, kept only so the')
    print('    two can be compared. Do not report it: it reads ess/cap as')
    print('    z=-2.06 "below null" where the correct null gives t=-1.91.')

    print('\n=== mean blocks selected (M3 was 5.3 athens / 4.0 quito / 7.7 belem) ===')
    for name, g in groups[:-1]:
        for dev in DEVICES:
            v = [r['blocks'] for r in g if r['dev'] == dev and r['blocks']]
            if v:
                print(f"  {name:<11} {dev:8} {statistics.mean(v):.1f}")


if __name__ == '__main__':
    main()

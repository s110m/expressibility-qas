"""
Picks one solution off a saved NSGA-II Pareto front and exports it as a
`best_solution.pt` checkpoint -- the exact same `{'solution', 'score'}` format
run_es.py writes -- so the NSGA-II pipeline hands off to train.py/eval.py in the
same currency the single-objective ES pipeline already uses.

Why this exists
---------------
A Pareto front has no single "best" entry, so run_nsga2.py deliberately saves
all of them (checkpoints/pareto_front.pt) and score_pareto_front.py adds an
accuracy/loss estimate per entry (checkpoints/pareto_front_scored.pt). Before
this script the only way to train the chosen candidate was to hand-copy its
'arch' list into a training config with `ckpt.load_ckpt: False`. That works for
the architecture but silently drops the searched qubit **layout**: nothing ever
writes a 'solution' key into the training run's checkpoints, so eval.py's
`state_dict['solution']['layout']` lookup finds nothing and the noisy evaluation
falls back to the transpiler's default mapping -- throwing away exactly the half
of the search that NSGA-II added over a pure-expressibility score.

Exporting instead means train.py takes its normal `ckpt.load_ckpt: True` path
(train.py:210-215 -> trainer.solution -> trainers.py:167-169), so arch AND
layout end up inside every checkpoint the training run saves, and eval.py picks
both up with no config edits at all.

Usage
-----
    # inspect the front (replaces eyeballing it with a python -c one-liner)
    python examples/export_pareto_solution.py <es_dir>/checkpoints/pareto_front_scored.pt --list

    # export the highest-accuracy solution (default criterion when scored)
    python examples/export_pareto_solution.py <es_dir>/checkpoints/pareto_front_scored.pt

    # or pick by any objective / by position
    python examples/export_pareto_solution.py <front>.pt --select success_rate
    python examples/export_pareto_solution.py <front>.pt --index 7

    # or take the knee -- the balanced compromise across the objectives, which
    # is the right rule when the front's accuracy/loss columns are worthless
    # (pure substrate: the SuperCircuit's shared weights sit at chance, so
    # --select loss picks whichever architecture got luckiest on 500 samples)
    python examples/export_pareto_solution.py <front>.pt --select knee

The checkpoint is written to <es_dir>/checkpoints/best_solution.pt, i.e. right
next to the front it came from and at exactly the path run_es.py would have used,
so Step 6 is simply:

    python examples/train.py <searched/scratch/.../nsga2_best.yml> --ckpt-dir=<es_dir>
"""
import argparse
import math
import os

from torchpack.utils import io
from torchpack.utils.logging import logger

# Direction of "better" for every metric that can appear on a scored front.
# Objectives come from run_nsga2.py (es.nsga2.objectives); accuracy/loss are
# added by score_pareto_front.py.
LOWER_IS_BETTER = ('expressibility', 'depth', 'swap_count', 'loss', 'loss_noisy')
HIGHER_IS_BETTER = ('success_rate', 'accuracy', 'accuracy_noisy')

# Order matters: it fixes the column order of --list and the tie-break order of
# --select. accuracy/loss are score_pareto_front.py's ideal-simulator scores and
# are LAYOUT-BLIND (identical for two solutions sharing an arch); the _noisy
# pair comes from its --noisy pass and does depend on the layout.
SCORING_METRICS = ('accuracy', 'loss', 'accuracy_noisy', 'loss_noisy')


def entry_metrics(entry):
    """All comparable numbers attached to one front entry: its NSGA-II
    objectives plus score_pareto_front.py's accuracy/loss when present."""
    metrics = dict(entry.get('objectives', {}))
    for extra in SCORING_METRICS:
        if extra in entry:
            metrics[extra] = entry[extra]
    return metrics


def signed(metric, value):
    """Value rewritten so that smaller is always better, for sort keys."""
    if metric in HIGHER_IS_BETTER:
        return -value
    if metric in LOWER_IS_BETTER:
        return value
    raise ValueError(
        f"unknown metric '{metric}' -- add it to LOWER_IS_BETTER or "
        f"HIGHER_IS_BETTER so its 'better' direction is unambiguous")


def knee_distances(pareto_front, objectives):
    """Per-entry normalised distance to the ideal point, over the NSGA-II
    OBJECTIVES ONLY -- smaller is better, and the minimum is the knee.

    Each objective is oriented so that lower is better, then min-max scaled
    across the front, so objectives on incomparable scales (expressibility is a
    KL divergence, success_rate a probability, depth a gate count) contribute
    equally instead of whichever has the larger raw magnitude dominating. The
    chosen entry is the one closest to the unattainable corner where every
    objective simultaneously takes its best value on this front -- i.e. the
    balanced compromise, rather than either extreme.

    Deliberately ignores accuracy/loss even on a scored front: on the pure
    substrate those come from SuperCircuit shared weights that sit at chance
    (measured: 40 random architectures averaged 0.2463 against a 0.25 chance
    level), so they carry no ranking signal, and consulting them would also
    contradict the training-free claim this selection is meant to support.

    An objective that is constant across the front contributes nothing (its
    scaled range is zero), which is the sensible degenerate behaviour: it
    cannot discriminate between these solutions.
    """
    columns = {}
    for objective in objectives:
        values = [signed(objective, entry_metrics(e)[objective])
                  for e in pareto_front]
        lo, hi = min(values), max(values)
        span = hi - lo
        columns[objective] = [0.0 if span == 0 else (v - lo) / span
                              for v in values]

    return [sum(columns[o][i] ** 2 for o in objectives) ** 0.5
            for i in range(len(pareto_front))]


def parse_cap(spec, parser):
    """Parse one --cap METRIC=VALUE spec into (metric, raw_value_string)."""
    if '=' not in spec:
        parser.error(f"--cap must look like METRIC=VALUE, got '{spec}'")
    metric, _, value = spec.partition('=')
    metric, value = metric.strip(), value.strip()
    if metric not in LOWER_IS_BETTER + HIGHER_IS_BETTER:
        parser.error(f"--cap metric '{metric}' is unknown; choose from "
                     f"{LOWER_IS_BETTER + HIGHER_IS_BETTER}")
    return metric, value


def resolve_cap(metric, raw, values, parser):
    """Turn a cap value into an absolute threshold.

    Accepts an absolute number ('depth=140') or a percentile of THIS front
    ('depth=p50'). The percentile form is what keeps a cap comparable across
    backends: transpiled depth on Quito is not on the same scale as on Athens,
    so one fixed number would bind hard on one device and not at all on
    another, whereas "the shallower half of this front" means the same thing
    everywhere.
    """
    if raw.lower().startswith('p'):
        try:
            pct = float(raw[1:])
        except ValueError:
            parser.error(f"--cap {metric}={raw}: '{raw[1:]}' is not a number")
        if not 0 <= pct <= 100:
            parser.error(f"--cap {metric}={raw}: percentile must be 0-100")
        ordered = sorted(values)
        # nearest-rank percentile, so the threshold is a value that actually
        # occurs on the front and p50 keeps at least half of it
        rank = max(1, math.ceil(pct / 100.0 * len(ordered)))
        return ordered[rank - 1], f"{raw} of this front"
    try:
        return float(raw), 'absolute'
    except ValueError:
        parser.error(f"--cap {metric}={raw}: not a number or pNN percentile")


def apply_caps(pareto_front, caps, parser):
    """Indices of the front entries satisfying every cap.

    A cap is a hard feasibility constraint applied BEFORE --select, so the
    exported solution is the best by the selection criterion *among solutions
    that satisfy it*. Direction follows LOWER_IS_BETTER/HIGHER_IS_BETTER: a cap
    on depth is an upper bound, a cap on success_rate is a lower bound.
    """
    indices = list(range(len(pareto_front)))
    for metric, raw in caps:
        values = []
        for entry in pareto_front:
            metrics = entry_metrics(entry)
            if metric not in metrics:
                parser.error(
                    f"--cap {metric}=... but '{metric}' is not on this front; "
                    f"available: {sorted(metrics)}. Add it to "
                    f"es.nsga2.objectives and re-search, or cap a metric you "
                    f"already have.")
            values.append(metrics[metric])

        threshold, how = resolve_cap(metric, raw, values, parser)
        upper = metric in LOWER_IS_BETTER
        kept = [i for i in indices
                if (values[i] <= threshold if upper else values[i] >= threshold)]
        logger.info(
            f"Cap {metric} {'<=' if upper else '>='} {threshold:g} ({how}): "
            f"{len(kept)} of {len(indices)} solutions survive "
            f"(front spans {min(values):g}..{max(values):g})")
        if not kept:
            parser.error(
                f"--cap {metric}={raw} excludes every solution on this front")
        indices = kept
    return indices


def sort_key(metrics, criterion, tie_break_order):
    """Primary criterion first, then every other available metric in a fixed
    order, so a tie on the primary criterion still resolves deterministically
    (and identically across runs) instead of depending on front ordering."""
    key = [signed(criterion, metrics[criterion])]
    for metric in tie_break_order:
        key.append(signed(metric, metrics[metric]))
    return tuple(key)


def format_front(pareto_front, metric_names, best_index=None):
    header = f"{'idx':>4}  " + '  '.join(f"{m:>14}" for m in metric_names) + \
             '   layout            arch'
    lines = [header, '-' * len(header)]
    for i, entry in enumerate(pareto_front):
        metrics = entry_metrics(entry)
        cells = []
        for metric in metric_names:
            value = metrics.get(metric)
            cells.append(f"{value:>14.5f}" if value is not None
                         else f"{'-':>14}")
        marker = ' *' if i == best_index else '  '
        lines.append(
            f"{i:>4}{marker}" + '  '.join(cells) +
            f"   {str(entry['solution']['layout']):<16}"
            f"  {entry['solution']['arch']}")
    return '\n'.join(lines)


def validate_solution(solution):
    for field in ('arch', 'layout'):
        if field not in solution:
            raise ValueError(f"solution is missing '{field}': {solution}")
        if not all(isinstance(v, int) for v in solution[field]):
            raise ValueError(f"solution['{field}'] must be a list of plain "
                             f"ints, got {solution[field]}")
    layout = solution['layout']
    if len(set(layout)) != len(layout):
        raise ValueError(f"layout must assign distinct physical qubits, "
                         f"got {layout}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('pareto_front', metavar='FILE',
                        help='pareto_front.pt (run_nsga2.py) or '
                             'pareto_front_scored.pt (score_pareto_front.py)')
    parser.add_argument('--select', type=str, default=None,
                        help="metric to pick the best solution by; any NSGA-II "
                             "objective on the front, plus accuracy/loss on a "
                             "scored front, or 'knee' for the balanced "
                             "compromise across all objectives (uses the "
                             "objectives only, never the supernet-weight "
                             "accuracy). Default: accuracy if the front is "
                             "scored, otherwise you must choose explicitly.")
    parser.add_argument('--cap', type=str, action='append', default=None,
                        metavar='METRIC=VALUE',
                        help="hard constraint applied BEFORE --select, so the "
                             "result is the best by --select among solutions "
                             "that satisfy it. VALUE is an absolute number or "
                             "a percentile of this front (e.g. "
                             "'--cap depth=p50' keeps the shallower half). "
                             "Repeatable. Upper bound for depth/swap_count/"
                             "expressibility, lower bound for success_rate/"
                             "accuracy.")
    parser.add_argument('--index', type=int, default=None,
                        help='pick this position in the front instead of '
                             'optimizing a metric (mutually exclusive with '
                             '--select)')
    parser.add_argument('--list', action='store_true',
                        help='print the whole front (with indices) and exit '
                             'without writing anything')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='directory to write <out-dir>/checkpoints/'
                             '<out-name> into; default is the search run '
                             'directory the front came from, so the export '
                             'lands exactly where run_es.py puts its own '
                             'best_solution.pt')
    parser.add_argument('--out-name', type=str, default='best_solution.pt',
                        help="checkpoint filename; keep the default unless you "
                             "also override ckpt.name in the training config "
                             "(train/searched/default.yml expects "
                             "'checkpoints/best_solution.pt')")
    args = parser.parse_args()

    if args.index is not None and args.select is not None:
        parser.error('--index and --select are mutually exclusive')

    checkpoints_dir = os.path.dirname(os.path.abspath(args.pareto_front))
    es_dir = os.path.dirname(checkpoints_dir)

    front_data = io.load(args.pareto_front, weights_only=False)
    pareto_front = front_data['pareto_front']
    objectives = front_data['objectives']

    if not pareto_front:
        raise ValueError(f"{args.pareto_front} contains an empty Pareto front")

    available = entry_metrics(pareto_front[0])
    is_scored = 'accuracy' in available
    # objectives first (search order), then the scoring extras
    metric_names = [m for m in objectives if m in available] + \
                   [m for m in SCORING_METRICS if m in available]

    logger.info(f"Loaded {len(pareto_front)} solutions from "
                f"{args.pareto_front}\n"
                f"  objectives: {objectives}\n"
                f"  available metrics: {metric_names}"
                f"{'' if is_scored else '  (unscored front -- run score_pareto_front.py first for accuracy/loss)'}")

    caps = [parse_cap(c, parser) for c in (args.cap or [])]
    candidates = apply_caps(pareto_front, caps, parser) if caps \
        else list(range(len(pareto_front)))

    if args.list:
        print(format_front(pareto_front, metric_names))
        return

    if args.index is not None:
        if not 0 <= args.index < len(pareto_front):
            parser.error(f"--index {args.index} out of range for a front of "
                         f"{len(pareto_front)} solutions")
        if args.index not in candidates:
            parser.error(f"--index {args.index} is excluded by --cap")
        best_index = args.index
        criterion = f'index={args.index}'
        # a positional pick has no optimized criterion; report accuracy when the
        # front is scored so 'score' still means something downstream
        score = entry_metrics(pareto_front[best_index]).get(
            'accuracy', math.nan)
    elif args.select == 'knee':
        criterion = 'knee'
        objective_names = [m for m in objectives if m in available]
        if len(objective_names) < 2:
            parser.error(
                f"--select knee needs at least 2 objectives on the front, "
                f"found {objective_names}")
        distances = knee_distances(pareto_front, objective_names)
        best_index = min(candidates, key=distances.__getitem__)
        score = distances[best_index]
        logger.info(
            f"Knee over objectives {objective_names}: solution {best_index} "
            f"at normalised distance {score:.5f} to the ideal point "
            f"(front spans {min(distances):.5f}..{max(distances):.5f})")
    else:
        criterion = args.select
        if criterion is None:
            if not is_scored:
                parser.error(
                    f"this front is not scored, so there is no default "
                    f"criterion -- pass --select from {metric_names}, or "
                    f"--select knee, or --index N, or run "
                    f"score_pareto_front.py first to make accuracy available")
            criterion = 'accuracy'
            logger.info("No --select given; defaulting to accuracy.")
        if criterion not in metric_names:
            parser.error(f"--select {criterion} is not available on this "
                         f"front; choose from {metric_names} or 'knee'")

        tie_break_order = [m for m in metric_names if m != criterion]
        keys = [sort_key(entry_metrics(e), criterion, tie_break_order)
                for e in pareto_front]
        best_index = min(candidates, key=keys.__getitem__)

        n_tied = sum(1 for i in candidates
                     if entry_metrics(pareto_front[i])[criterion] ==
                     entry_metrics(pareto_front[best_index])[criterion])
        if n_tied > 1:
            logger.info(f"{n_tied} solutions tie on {criterion}; broke the tie "
                        f"deterministically using {tie_break_order}.")
        score = entry_metrics(pareto_front[best_index])[criterion]

    best_entry = pareto_front[best_index]
    solution = {'arch': list(best_entry['solution']['arch']),
                'layout': list(best_entry['solution']['layout'])}
    validate_solution(solution)

    print(format_front(pareto_front, metric_names, best_index=best_index))

    out_dir = args.out_dir or es_dir
    out_path = os.path.join(out_dir, 'checkpoints', args.out_name)
    io.save(out_path, {
        'solution': solution,
        'score': float(score),
        # provenance -- ignored by train.py/eval.py, which only read
        # 'solution'/'score', but it makes an exported checkpoint
        # self-explanatory months later
        'objectives': entry_metrics(best_entry),
        'selection': {
            'criterion': criterion,
            'index': best_index,
            'n_candidates': len(pareto_front),
            'caps': [f"{m}={v}" for m, v in caps],
            'n_feasible': len(candidates),
            'source': os.path.abspath(args.pareto_front),
        },
    })

    logger.info(
        f"Selected solution {best_index}/{len(pareto_front)} by {criterion}\n"
        f"  arch:       {solution['arch']}\n"
        f"  layout:     {solution['layout']}\n"
        f"  metrics:    {entry_metrics(best_entry)}\n"
        f"  saved to:   {out_path}\n\n"
        f"Train it with:\n"
        f"  python examples/train.py <train/searched/scratch/.../nsga2_best.yml> "
        f"--ckpt-dir={out_dir}")


if __name__ == '__main__':
    main()

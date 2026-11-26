"""How robust is the enumerated layout to calibration drift?
(Access revision, Reviewer 1 #1.)

Real devices drift between calibrations, so the calibration table used to rank
layouts is always somewhat stale. This script treats the fake backend's snapshot
as the truth -- it is the noise model every task outcome was measured under --
and ranks the 120 layouts of each cell with a *perturbed* copy of it: every gate
error and readout probability multiplied, and every T1/T2 divided, by an
independent log-normal factor exp(sigma * z), z ~ N(0, 1). For each sigma it
reports, over N_DRAWS draws per cell,

  * the Spearman correlation between the stale and the true S_hat ranking,
  * how often the stale arg-max is the true arg-max,
  * the task rank of the layout the stale table selects (the cost of deciding
    with drifted data), next to the undrifted pick and a uniform draw (60.5).

Transpilation with an initial layout does not consult error rates (with a layout
given, Qiskit skips VF2Layout/VF2PostLayout), so each layout's transpiled
program is fixed and is computed once; only the calibration table changes.

Writes success_rate_ablation/drift/summary.json. Task outcomes come from the
ablation's per-cell files, keyed by layout.

Usage (repo root), same arguments as success_rate_ablation.py:
    python examples/success_rate_drift.py --classification runs/<run> ... \
        --vqe vqe_runs/tfim4/blk1_seed0 ...
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore")

import argparse
import copy
import itertools
import json

import numpy as np
from scipy.stats import spearmanr

import compiler_layout_baselines as cb
import success_rate_ablation as ab
import vqe_layout_experiment as vqe

SIGMAS = (0.1, 0.25, 0.5, 1.0)
N_DRAWS = 200


def programs(circ, backend_name):
    """layout -> list of (gate name, physical qubits), measurements dropped."""
    processor = vqe.make_processor(backend_name, ab.EVAL_CFG.format(dev='quito'))
    backend = cb.get_backend(backend_name)
    out = {}
    for layout in itertools.permutations(range(backend.num_qubits), 4):
        processor.set_layout(list(layout))
        processor.optimization_level = 2
        t = processor.transpile(circ)
        t = ab.to_device_width(t[0] if isinstance(t, list) else t,
                               backend.num_qubits)
        bare = t.remove_final_measurements(inplace=False)
        out[layout] = [(i.operation.name,
                        tuple(bare.find_bit(q).index for q in i.qubits))
                       for i in bare.data]
    return out, ab.Calibration(backend.properties())


def s_hat(cal, prog):
    s, used = 1.0, set()
    for name, qs in prog:
        used.update(qs)
        g = 1 - cal.err[name][qs]
        s *= g if g != 0 else 1e-5
        d = cal.dur.get(name, {}).get(qs)
        if d is not None:
            for q in qs:
                t1, t2, _ = cal.t12[q]
                s *= np.exp(-d / t1) * np.exp(-d / t2)
    for q in used:
        ro = cal.readout[q]
        if ro is not None:
            r = 1 - (ro[0] + ro[1]) / 2
            s *= r if r != 0 else 1e-5
    return s


def perturb(cal, sigma, rng):
    c = copy.deepcopy(cal)
    f = lambda: float(np.exp(sigma * rng.standard_normal()))
    for name in c.err:
        for qs, e in c.err[name].items():
            if e is not None:  # uncalibrated ops (e.g. reset) carry no error
                c.err[name][qs] = min(e * f(), 0.999)
    for q, (t1, t2, *rest) in list(c.t12.items()) if isinstance(c.t12, dict) \
            else enumerate(c.t12):
        c.t12[q] = (t1 / f(), t2 / f(), *rest)
    for q, ro in (c.readout.items() if isinstance(c.readout, dict)
                  else enumerate(c.readout)):
        if ro is not None:
            c.readout[q] = tuple(min(p * f(), 0.5) for p in ro)
    return c


def analyse(label, circ, backend_name, src_dir, higher_is_better, rng):
    src = json.load(open(os.path.join(src_dir, f'{label}.json')))
    task = {tuple(r['layout']): r['task'] for r in src['rows']}
    progs, cal = programs(circ, backend_name)
    layouts = list(progs)
    sign = 1 if higher_is_better else -1
    tasks = np.array([sign * task[l] for l in layouts])
    rank_of = lambda i: 1 + int((tasks > tasks[i]).sum())
    true = np.array([s_hat(cal, progs[l]) for l in layouts])
    i0 = int(np.argmax(true))
    rec = {'label': label, 'undrifted_pick_rank': rank_of(i0), 'sigma': {}}
    for sigma in SIGMAS:
        rhos, same, ranks = [], 0, []
        for _ in range(N_DRAWS):
            c = perturb(cal, sigma, rng)
            stale = np.array([s_hat(c, progs[l]) for l in layouts])
            rhos.append(spearmanr(stale, true).correlation)
            i = int(np.argmax(stale))
            same += i == i0
            ranks.append(rank_of(i))
        rec['sigma'][str(sigma)] = {'rho_stale_true': float(np.mean(rhos)),
                                    'same_pick': same / N_DRAWS,
                                    'pick_rank': float(np.mean(ranks))}
    print(f"  {label}: undrifted {rec['undrifted_pick_rank']}  " + '  '.join(
        f"s={s}: rho {v['rho_stale_true']:.2f} same {v['same_pick']:.2f} "
        f"rank {v['pick_rank']:.1f}" for s, v in rec['sigma'].items()), flush=True)
    return rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classification', nargs='*', default=[])
    parser.add_argument('--vqe', nargs='*', default=[])
    parser.add_argument('--src-dir', default='success_rate_ablation')
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    out_dir = os.path.join(args.src_dir, 'drift')
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for run_dir in args.classification:
        dev = run_dir.split('scratch.')[1].split('.')[0]
        seed = run_dir.split('_best_s')[1][:3]
        circs, backend_name, _ = ab.classification_inputs(run_dir, dev)
        results.append(analyse(f'cls_{dev}_s{seed}', circs[0], backend_name,
                               args.src_dir, True, rng))
    for vqe_dir in args.vqe:
        tag = os.path.basename(os.path.normpath(vqe_dir))
        for backend_name in vqe.BACKENDS:
            circs, _ = ab.vqe_inputs(vqe_dir, backend_name)
            results.append(analyse(f'vqe_{tag}_{backend_name}', circs[0],
                                   backend_name, args.src_dir, False, rng))
    summary = {'sigmas': list(SIGMAS), 'n_draws': N_DRAWS, 'cells': results}
    for group, sel in (('classification', 'cls_'), ('vqe', 'vqe_')):
        rs = [r for r in results if r['label'].startswith(sel)]
        if rs:
            summary[group] = {
                'n_cells': len(rs),
                'undrifted_pick_rank': float(np.mean([r['undrifted_pick_rank'] for r in rs])),
                **{str(s): {k: float(np.mean([r['sigma'][str(s)][k] for r in rs]))
                            for k in ('rho_stale_true', 'same_pick', 'pick_rank')}
                   for s in SIGMAS}}
    json.dump(summary, open(os.path.join(out_dir, 'summary.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != 'cells'}, indent=1))


if __name__ == '__main__':
    main()

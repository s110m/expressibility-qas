"""The success-rate estimate with relaxation charged over scheduled time
(Access revision, Reviewer 1 #1/#4 follow-up to success_rate_ablation.py).

get_success_rate charges exp(-t/T1) exp(-t/T2) only for the gates a qubit takes
part in, so a qubit idling while a SWAP runs elsewhere accrues nothing. Here the
same programs are ALAP-scheduled with idle periods padded by delays, exactly as
success_rate_ablation.py does for its "_sched" fidelities, and the relaxation
factor is charged for every gate *and* delay on each touched qubit up to its
measurement:

    S_sched = S_gate * prod_q exp(-D_q/T1_q) exp(-D_q/T2_q) * S_read

Gate and readout terms are unchanged. The exact fidelities and task outcomes
are read back from success_rate_ablation/<label>.json, keyed by layout, so this
script only transpiles and schedules; it writes

    success_rate_ablation/sched/<label>.json  and  .../sched/summary.json

Usage (repo root), with the same arguments as success_rate_ablation.py:
    python examples/success_rate_sched.py --classification runs/<run> ... \
        --vqe vqe_runs/tfim4/blk1_seed0 ...
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore")

import argparse
import itertools
import json
import time

import numpy as np
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDelay
from scipy.stats import spearmanr

import compiler_layout_baselines as cb
import success_rate_ablation as ab
import vqe_layout_experiment as vqe


_UNIT_TO_NS = {'s': 1e9, 'ms': 1e6, 'us': 1e3, 'ns': 1.0, 'ps': 1e-3}


def scheduled_relax(cal, sched, dt):
    """Relaxation survival over each touched qubit's scheduled time, measurements
    excluded. Returns three variants, all in the calibration's units (ns):

      'all'    -- every gate and every delay on the qubit (incl. leading delays)
      'active' -- as 'all', but delays before the qubit's first gate are skipped
                  (a qubit waiting in |0> cannot relax)
      'gates'  -- gates only; must reproduce get_success_rate's relaxation term

    "Touched" means acted on by at least one gate, the same qubit set
    get_success_rate charges (routing ancillas included).

    Fixed 2026-11-17: the first version converted delays with backend.dt, i.e.
    to seconds, while gate lengths and T1/T2 are in ns, so idle time was charged
    1e9 times too little; it also charged only measured qubits, dropping the
    routing ancillas that get_success_rate charges.
    """
    busy = {'all': {}, 'active': {}, 'gates': {}}
    started, touched = set(), set()
    for instr in sched.data:
        name = instr.operation.name
        if name in ('measure', 'barrier'):
            continue
        qs = tuple(sched.find_bit(q).index for q in instr.qubits)
        if name == 'delay':
            op = instr.operation
            d = (op.duration * dt * 1e9 if op.unit == 'dt'
                 else float(op.duration) * _UNIT_TO_NS[op.unit])
            for q in qs:
                busy['all'][q] = busy['all'].get(q, 0.0) + d
                if q in started:
                    busy['active'][q] = busy['active'].get(q, 0.0) + d
        else:
            d = cal.dur.get(name, {}).get(qs)
            touched.update(qs)
            started.update(qs)
            if d is None:
                continue
            for q in qs:
                for k in busy:
                    busy[k][q] = busy[k].get(q, 0.0) + d
    out = {}
    for k, b in busy.items():
        s = 1.0
        for q in touched:
            t1, t2, _ = cal.t12[q]
            s *= np.exp(-b.get(q, 0.0) / t1) * np.exp(-b.get(q, 0.0) / t2)
        out[k] = s
    return out


KEYS = ('S_sched', 'S_sched_active', 'S_check', 'S_full')


def evaluate(label, circs, backend_name, out_dir, src_dir, higher_is_better):
    out_path = os.path.join(out_dir, f'{label}.json')
    if os.path.exists(out_path):
        return json.load(open(out_path))
    t0 = time.time()
    src = json.load(open(os.path.join(src_dir, f'{label}.json')))
    by_layout = {tuple(r['layout']): r for r in src['rows']}
    processor = vqe.make_processor(backend_name, ab.EVAL_CFG.format(dev='quito'))
    backend = cb.get_backend(backend_name)
    cal = ab.Calibration(backend.properties())
    scheduler = PassManager([ALAPScheduleAnalysis(target=backend.target),
                             PadDelay(target=backend.target)])
    rows = []
    for layout in itertools.permutations(range(backend.num_qubits), 4):
        processor.set_layout(list(layout))
        processor.optimization_level = 2
        vals = {'S_sched': [], 'S_sched_active': [], 'S_check': []}
        for c in circs:
            t = processor.transpile(c)
            t = ab.to_device_width(t[0] if isinstance(t, list) else t,
                                   backend.num_qubits)
            fac = ab.survival_factors(cal, t.remove_final_measurements(inplace=False))
            r = scheduled_relax(cal, scheduler.run(t), backend.dt)
            base = fac['gate'] * fac['read']
            vals['S_sched'].append(base * r['all'])
            vals['S_sched_active'].append(base * r['active'])
            vals['S_check'].append(base * r['gates'])
        ref = by_layout[layout]
        rows.append({'layout': list(layout),
                     **{k: float(np.mean(v)) for k, v in vals.items()},
                     **{k: ref[k] for k in ('S_full', 'F_out', 'F_out_sched', 'task')}})

    sign = 1 if higher_is_better else -1
    tasks = [sign * r['task'] for r in rows]
    corr, pick = {}, {}
    for k in KEYS:
        x = [r[k] for r in rows]
        corr[k] = {m: float(spearmanr(x, [r[m] for r in rows]).correlation)
                   for m in ('F_out', 'F_out_sched')}
        corr[k]['task'] = float(spearmanr(x, tasks).correlation)
        best = max(rows, key=lambda r: r[k])
        pick[k] = 1 + sum(t > sign * best['task'] for t in tasks)
    rec = {'label': label, 'backend': backend_name, 'spearman': corr,
           'argmax_task_rank': pick, 'rows': rows,
           'elapsed_seconds': time.time() - t0}
    json.dump(rec, open(out_path, 'w'), indent=1)
    print(f"  {label}: rho(S_sched, F_sched) {corr['S_sched']['F_out_sched']:+.2f} "
          f"vs S_full {corr['S_full']['F_out_sched']:+.2f}  "
          f"[{rec['elapsed_seconds']:.0f}s]", flush=True)
    return rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classification', nargs='*', default=[])
    parser.add_argument('--vqe', nargs='*', default=[])
    parser.add_argument('--src-dir', default='success_rate_ablation')
    args = parser.parse_args()
    parser_out = 'sched_fixed'  # the pre-fix outputs in sched/ are kept as evidence
    out_dir = os.path.join(args.src_dir, parser_out)
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for run_dir in args.classification:
        dev = run_dir.split('scratch.')[1].split('.')[0]
        seed = run_dir.split('_best_s')[1][:3]
        circs, backend_name, _ = ab.classification_inputs(run_dir, dev)
        results.append(evaluate(f'cls_{dev}_s{seed}', circs, backend_name,
                                out_dir, args.src_dir, True))
    for vqe_dir in args.vqe:
        tag = os.path.basename(os.path.normpath(vqe_dir))
        for backend_name in vqe.BACKENDS:
            circs, _ = ab.vqe_inputs(vqe_dir, backend_name)
            results.append(evaluate(f'vqe_{tag}_{backend_name}', circs,
                                    backend_name, out_dir, args.src_dir, False))
    summary = {}
    for group, sel in (('classification', 'cls_'), ('vqe', 'vqe_')):
        rs = [r for r in results if r['label'].startswith(sel)]
        if rs:
            summary[group] = {
                'n_cells': len(rs),
                'mean_spearman': {k: {m: float(np.mean([r['spearman'][k][m] for r in rs]))
                                      for m in ('F_out', 'F_out_sched', 'task')}
                                  for k in KEYS},
                'mean_argmax_task_rank': {k: float(np.mean(
                    [r['argmax_task_rank'][k] for r in rs])) for k in KEYS},
            }
    json.dump(summary, open(os.path.join(out_dir, 'summary.json'), 'w'), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()

"""How much does the success-rate estimate lose by treating error channels as
independent and multiplicative? (Access revision, Reviewer 1 #4 / Reviewer 2 #2)

For every (circuit, backend, initial layout) this compares

  * the analytic estimate, decomposed into its three survival factors
        S_gate   prod over gates of (1 - gate error)
        S_relax  prod over gates and their qubits of exp(-t/T1) exp(-t/T2)
        S_read   prod over touched qubits of (1 - mean readout error)
    so that every single-channel and pairwise product can be ranked on its
    own; their full product is asserted equal to get_success_rate().
  * the exact answer from a density-matrix simulation of the same transpiled
    circuit under NoiseModel.from_backend() -- the noise model the evaluations
    use -- in which the channels act jointly, not independently:
        F_state  <psi_ideal| rho |psi_ideal>    (no readout)
        F_out    Hellinger fidelity of the measured distribution, readout
                 confusion applied, against the ideal one
    each computed twice: on the circuit as evaluated (unscheduled, so
    relaxation only acts during gates, exactly as in the evaluations), and
    ALAP-scheduled with delays padded, so idle qubits decohere as on hardware
    -- which tests the estimate's per-gate relaxation simplification too.
  * the task outcome already measured for that layout: 300-sample accuracy
    from the classification layout sweep, or the VQE energy.

Circuits come from exactly the evaluation pipeline: QiskitProcessor transpiles
with initial_layout (optimization level 2, seed 42), as the sweeps did.

Writes success_rate_ablation/<label>.json per (circuit, backend) and a
summary of rank correlations to success_rate_ablation/summary.json.

Usage (repo root):
    python examples/success_rate_ablation.py --classification runs/<run> ... \
        --vqe vqe_runs/tfim4/blk2_seed0 ...
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore")

import argparse
import glob
import itertools
import json
import time

import numpy as np
import torch
from qiskit.quantum_info import Statevector, hellinger_fidelity
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDelay
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from scipy.stats import spearmanr
from torchpack.utils.config import configs

from torchquantum.plugin import tq2qiskit
from torchquantum.util import get_success_rate
from torchquantum.util.utils import (
    gate_error_values,
    gate_length_values,
    readout_error_values,
    thermal_relaxation_values,
)

import compiler_layout_baselines as cb
import vqe_layout_experiment as vqe

EVAL_CFG = 'examples/configs/mnist/four0123/eval/{dev}/noise/opt2/all.yml'
N_SAMPLES = 4  # classification inputs averaged per layout (F depends on input)
VARIANTS = {
    'full': ('gate', 'relax', 'read'),
    'gate': ('gate',), 'relax': ('relax',), 'read': ('read',),
    'gate+relax': ('gate', 'relax'), 'gate+read': ('gate', 'read'),
    'relax+read': ('relax', 'read'),
}


# ---- the estimate, factor by factor -------------------------------------------
class Calibration:
    def __init__(self, properties):
        self.err, self.dur = {}, {}
        for name, qargs, v in gate_error_values(properties):
            self.err.setdefault(name, {})[tuple(qargs)] = v
        for name, qargs, v in gate_length_values(properties):
            self.dur.setdefault(name, {})[tuple(qargs)] = v
        self.t12 = thermal_relaxation_values(properties)
        self.readout = readout_error_values(properties)


def survival_factors(cal, circ):
    """Mirror of get_success_rate's loop, keeping the three factors apart."""
    s = {'gate': 1.0, 'relax': 1.0, 'read': 1.0}
    used = set()
    for instr in circ.data:
        name = instr.operation.name
        qs = tuple(circ.find_bit(q).index for q in instr.qubits)
        used.update(qs)
        g = 1 - cal.err[name][qs]
        s['gate'] *= g if g != 0 else 1e-5
        d = cal.dur.get(name, {}).get(qs)
        if d is not None:
            for q in qs:
                t1, t2, _ = cal.t12[q]
                s['relax'] *= np.exp(-d / t1) * np.exp(-d / t2)
    for q in used:
        ro = cal.readout[q]
        if ro is not None:
            r = 1 - (ro[0] + ro[1]) / 2
            s['read'] *= r if r != 0 else 1e-5
    return s


# ---- the exact answer -------------------------------------------------------------
def to_device_width(tc, n_qubits):
    """Drop the idle ancilla slots QiskitProcessor's AerSimulator target pads
    the register with (30 qubits), keeping physical qubits 0..n_qubits-1."""
    from qiskit import QuantumCircuit
    if tc.num_qubits == n_qubits:
        return tc
    out = QuantumCircuit(n_qubits, tc.num_clbits)
    for instr in tc.data:
        qs = [tc.find_bit(q).index for q in instr.qubits]
        assert max(qs) < n_qubits, f'{instr.operation.name} acts on {qs}'
        out.append(instr.operation, qs, [tc.find_bit(c).index for c in instr.clbits])
    return out


def exact_fidelities(sim, cal, t_meas, t_sched_meas):
    """F_state and F_out for an unscheduled and a scheduled transpiled circuit."""
    out = {}
    for tag, tc in (('', t_meas), ('_sched', t_sched_meas)):
        # physical qubit -> classical bit, read off the final measurements
        q2c = {tc.find_bit(i.qubits[0]).index: tc.find_bit(i.clbits[0]).index
               for i in tc.data if i.operation.name == 'measure'}
        bare = tc.remove_final_measurements(inplace=False)
        psi = Statevector(bare)
        dm_circ = bare.copy()
        dm_circ.save_density_matrix()
        rho = np.asarray(sim.run(dm_circ).result().data()['density_matrix'])
        vec = np.asarray(psi.data)
        out['F_state' + tag] = float(np.real(vec.conj() @ rho @ vec))

        # measured qubits ordered by classical bit: bit k <-> qubits[k], which
        # is also Qiskit's bitstring convention (bit 0 rightmost)
        qubits = sorted(q2c, key=q2c.get)
        width = len(qubits)
        p_ideal = psi.probabilities_dict(qubits)
        marg = np.zeros(2 ** width)
        for idx, p in enumerate(np.real(np.diag(rho))):
            marg[sum(((idx >> q) & 1) << k for k, q in enumerate(qubits))] += p
        # per-qubit readout confusion, P(out | in), from the same calibration
        # numbers NoiseModel.from_backend uses for its readout errors. In a
        # C-order reshape to [2]*width, bit k lives on axis width-1-k.
        tensor = marg.reshape([2] * width)
        for k, q in enumerate(qubits):
            p10, p01 = cal.readout[q] or (0.0, 0.0)
            conf = np.array([[1 - p10, p01], [p10, 1 - p01]])
            axis = width - 1 - k
            tensor = np.moveaxis(
                np.tensordot(conf, np.moveaxis(tensor, axis, 0), axes=1), 0, axis)
        p_noisy_d = {format(i, f'0{width}b'): float(v)
                     for i, v in enumerate(tensor.reshape(-1)) if v > 1e-15}
        out['F_out' + tag] = float(hellinger_fidelity(p_ideal, p_noisy_d))
    return out


# ---- per circuit ---------------------------------------------------------------------
def evaluate(label, circs, backend_name, task_metric, out_dir, higher_is_better,
             limit=None):
    out_path = os.path.join(out_dir, f'{label}.json')
    if os.path.exists(out_path):
        print(f'  {label}: exists, skipping')
        return json.load(open(out_path))
    t0 = time.time()
    processor = vqe.make_processor(backend_name, EVAL_CFG.format(dev='quito'))
    backend = cb.get_backend(backend_name)
    properties = backend.properties()
    cal = Calibration(properties)
    noise_model = NoiseModel.from_backend(backend)
    sim = AerSimulator(method='density_matrix', noise_model=noise_model)
    # both passes need the timing source; PadDelay looks durations up itself
    scheduler = PassManager([ALAPScheduleAnalysis(target=backend.target),
                             PadDelay(target=backend.target)])

    rows = []
    layouts = list(itertools.permutations(range(backend.num_qubits), 4))
    for layout in layouts[:limit]:
        processor.set_layout(list(layout))
        processor.optimization_level = 2
        recs = []
        for c in circs:
            t = processor.transpile(c)
            t = to_device_width(t[0] if isinstance(t, list) else t,
                                backend.num_qubits)
            bare = t.remove_final_measurements(inplace=False)
            fac = survival_factors(cal, bare)
            s_full = fac['gate'] * fac['relax'] * fac['read']
            ref = get_success_rate(properties, bare)
            assert abs(s_full - ref) <= 1e-9 * max(1.0, abs(ref)), (s_full, ref)
            recs.append({**{f'S_{k}': float(np.prod([fac[c_] for c_ in v]))
                            for k, v in VARIANTS.items()},
                         **exact_fidelities(sim, cal, t, scheduler.run(t))})
        row = {k: float(np.mean([r[k] for r in recs])) for k in recs[0]}
        row['layout'] = list(layout)
        row['task'] = task_metric[tuple(layout)]
        rows.append(row)

    sign = 1 if higher_is_better else -1
    keys = [k for k in rows[0] if k not in ('layout', 'task')]
    corr = {}
    for k in keys:
        x = [r[k] for r in rows]
        corr[k] = {
            'vs_task': float(spearmanr(x, [sign * r['task'] for r in rows]).correlation),
            'vs_F_out': float(spearmanr(x, [r['F_out'] for r in rows]).correlation),
            'vs_F_out_sched': float(spearmanr(x, [r['F_out_sched'] for r in rows]).correlation),
        }
    bias = {
        'S_full_over_F_out_median': float(np.median([r['S_full'] / r['F_out'] for r in rows])),
        'S_gate_relax_over_F_state_median': float(np.median(
            [r['S_gate+relax'] / r['F_state'] for r in rows])),
    }
    # does choosing the arg-max of each quantity pick a good layout?
    pick = {}
    tasks = [sign * r['task'] for r in rows]
    for k in keys:
        best = max(rows, key=lambda r: r[k])
        pick[k] = {'layout': best['layout'], 'task': best['task'],
                   'task_rank': 1 + sum(t > sign * best['task'] for t in tasks)}
    rec = {'label': label, 'backend': backend_name, 'n_layouts': len(rows),
           'n_input_samples': len(circs), 'spearman': corr, 'bias': bias,
           'argmax_pick': pick, 'rows': rows,
           'elapsed_seconds': time.time() - t0}
    json.dump(rec, open(out_path, 'w'), indent=1)
    print(f"  {label}: rho(S_full, task) {corr['S_full']['vs_task']:+.2f}  "
          f"rho(S_full, F_out) {corr['S_full']['vs_F_out']:+.2f}  "
          f"rho(F_out, task) {corr['F_out']['vs_task']:+.2f}  "
          f"S/F_out {bias['S_full_over_F_out_median']:.2f}  "
          f"[{rec['elapsed_seconds']:.0f}s]", flush=True)
    return rec


def classification_inputs(run_dir, dev):
    configs.clear()
    configs.load(os.path.join(run_dir, 'metainfo', 'configs.yaml'))
    configs.load(EVAL_CFG.format(dev=dev), recursive=True)
    configs.ckpt.weight_from_scratch = False
    sd = torch.load(os.path.join(run_dir, configs.ckpt.name),
                    map_location='cpu', weights_only=False)
    model = cb.builder.make_model()
    model.load_state_dict(sd['model'], strict=False)
    model.set_sample_arch(sd['solution']['arch'])
    model.eval()
    split = cb.builder.make_dataset()[configs.dataset.split]
    x = torch.stack([split[i]['image'] for i in range(N_SAMPLES)])
    cap = cb._CaptureProcessor()
    model.set_qiskit_processor(cap)
    with torch.no_grad():
        model(x, use_qiskit=True)
    sweep = {}
    (p,) = glob.glob(os.path.join(run_dir, 'layout_sweep', '*.300.jsonl'))
    for line in open(p):
        r = json.loads(line)
        sweep[tuple(r['layout'])] = r['accuracy_percent']
    return cap.circs, configs.qiskit.backend_name, sweep


def vqe_inputs(vqe_dir, backend_name):
    ck = torch.load(os.path.join(vqe_dir, 'checkpoint.pt'), weights_only=False)
    layer, _ = vqe.build_ansatz(ck['arch_cfg'], ck['sample_arch'][-1])
    # the checkpoint also carries the q_device state buffers a forward pass
    # attaches to each layer; every trained parameter must still be present
    missing, _ = layer.load_state_dict(ck['state_dict'], strict=False)
    assert not missing, missing
    import torchquantum as tq
    qdev = tq.QuantumDevice(n_wires=4, bsz=1)
    with torch.no_grad():
        layer(qdev)
    cz, _ = vqe.measurement_circuits(tq2qiskit(qdev, layer))
    res = json.load(open(os.path.join(vqe_dir, f'{backend_name}.json')))
    energy = {tuple(r['layout']): r['energy'] for r in res['layouts']}
    return [cz], energy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classification', nargs='*', default=[])
    parser.add_argument('--vqe', nargs='*', default=[])
    parser.add_argument('--backends', nargs='+', default=list(vqe.BACKENDS),
                        help='backends for the VQE circuits')
    parser.add_argument('--out-dir', default='success_rate_ablation')
    parser.add_argument('--limit', type=int, default=None,
                        help='pilot only: first N layouts per cell')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    results = []
    for run_dir in args.classification:
        dev = run_dir.split('scratch.')[1].split('.')[0]
        seed = run_dir.split('_best_s')[1][:3]
        circs, backend_name, sweep = classification_inputs(run_dir, dev)
        results.append(evaluate(f'cls_{dev}_s{seed}', circs, backend_name,
                                sweep, args.out_dir, higher_is_better=True,
                                limit=args.limit))
    for vqe_dir in args.vqe:
        for backend_name in args.backends:
            circs, energy = vqe_inputs(vqe_dir, backend_name)
            tag = os.path.basename(os.path.normpath(vqe_dir))
            results.append(evaluate(f'vqe_{tag}_{backend_name}', circs,
                                    backend_name, energy, args.out_dir,
                                    higher_is_better=False, limit=args.limit))

    summary = {}
    for group, sel in (('classification', 'cls_'), ('vqe', 'vqe_')):
        rs = [r for r in results if r['label'].startswith(sel)]
        if not rs:
            continue
        keys = rs[0]['spearman'].keys()
        summary[group] = {
            'n_cells': len(rs),
            'mean_spearman': {k: {m: float(np.mean([r['spearman'][k][m] for r in rs]))
                                  for m in ('vs_task', 'vs_F_out', 'vs_F_out_sched')}
                              for k in keys},
            'mean_argmax_task_rank': {k: float(np.mean(
                [r['argmax_pick'][k]['task_rank'] for r in rs])) for k in keys},
            'median_bias': {b: float(np.median([r['bias'][b] for r in rs]))
                            for b in rs[0]['bias']},
        }
    json.dump(summary, open(os.path.join(args.out_dir, 'summary.json'), 'w'), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()

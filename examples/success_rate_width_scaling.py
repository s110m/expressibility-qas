"""Does the success-rate ranking of layouts hold at other circuit widths?
(Access revision, Reviewer 2 #4 (2).)

The paper ranks layouts for four-qubit circuits on five-qubit devices. Here
circuits of the same U3+CU3 design space are built at n = 2..7 qubits, at two
depths (2 and 6 blocks, full layer width), with parameters drawn uniformly at
random (the first step of the ranking argument, from S_hat to the program's
output fidelity, does not involve training). For every layout of each circuit
on a five-qubit (FakeQuitoV2) and a seven-qubit (FakeLagosV2) device, the
analytic estimate is set against the exact state fidelity
F_state = <psi_ideal|rho|psi_ideal> from a density-matrix simulation under
NoiseModel.from_backend, computed with success_rate_ablation.py's code.

F_state rather than the output-distribution fidelity F_out: for random-parameter
circuits the ideal output is close to Porter-Thomas, and a fully depolarized
(uniform) output still has Hellinger fidelity ~0.79 to it, so F_out is not
monotone in the noise and ranks heavily routed, nearly depolarized layouts near
the top. F_state = f + (1 - f)/d under global depolarizing noise of strength
1 - f, monotone in f. (The trained circuits of success_rate_ablation.py have
peaked output distributions, where F_out is informative.) Both are stored.

Writes success_rate_ablation/width_scaling.json.

Usage (repo root): python examples/success_rate_width_scaling.py
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
import torch
import torchquantum as tq
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDelay
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from scipy.stats import spearmanr

from torchquantum.algorithm.quantumnas.super_layers import SuperU3CU3ShareFrontLayer0
from torchquantum.plugin import tq2qiskit

import compiler_layout_baselines as cb
import success_rate_ablation as ab
import vqe_layout_experiment as vqe

CELLS = {  # backend -> widths
    'FakeQuitoV2': (2, 3, 4, 5),
    'FakeLagosV2': (3, 5, 7),
}
DEPTHS = (2, 6)
MAX_LAYOUTS = 840  # random subset when P(N, n) is larger (FakeLagosV2, n >= 5)


def build_circuit(n, blocks, seed):
    layer = SuperU3CU3ShareFrontLayer0(arch=dict(
        n_wires=n, n_blocks=blocks, n_layers_per_block=2,
        n_front_share_blocks=1, n_front_share_wires=1, n_front_share_ops=1))
    layer.set_sample_arch([n] * (2 * blocks) + [blocks])
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in layer.parameters():
            p.copy_(2 * np.pi * torch.rand(p.shape, generator=g))
    qdev = tq.QuantumDevice(n_wires=n, bsz=1)
    circ = tq2qiskit(qdev, layer)
    circ.measure_all()
    return circ


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='success_rate_ablation/width_scaling.json')
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    results = []
    for backend_name, widths in CELLS.items():
        backend = cb.get_backend(backend_name)
        props = backend.properties()
        cal = ab.Calibration(props)
        sim = AerSimulator(method='density_matrix',
                           noise_model=NoiseModel.from_backend(backend))
        sched = PassManager([ALAPScheduleAnalysis(target=backend.target),
                             PadDelay(target=backend.target)])
        processor = vqe.make_processor(backend_name, ab.EVAL_CFG.format(dev='quito'))
        processor.optimization_level = 2
        for n in widths:
            layouts = list(itertools.permutations(range(backend.num_qubits), n))
            if len(layouts) > MAX_LAYOUTS:
                idx = rng.choice(len(layouts), MAX_LAYOUTS, replace=False)
                layouts = [layouts[i] for i in sorted(idx)]
            for blocks in DEPTHS:
                t0 = time.time()
                circ = build_circuit(n, blocks, seed=100 * n + blocks)
                rows = []
                for lay in layouts:
                    processor.set_layout(list(lay))
                    t = processor.transpile(circ)
                    t = ab.to_device_width(t[0] if isinstance(t, list) else t,
                                           backend.num_qubits)
                    fac = ab.survival_factors(cal, t.remove_final_measurements(inplace=False))
                    f = ab.exact_fidelities(sim, cal, t, sched.run(t))
                    rows.append({'layout': list(lay),
                                 'S': fac['gate'] * fac['relax'] * fac['read'],
                                 'S_gate_relax': fac['gate'] * fac['relax'], **f})
                S = np.array([r['S'] for r in rows])
                F = np.array([r['F_state'] for r in rows])
                Fs = np.array([r['F_state_sched'] for r in rows])
                Fo = np.array([r['F_out'] for r in rows])
                pick = int(np.argmax(S))
                rec = {
                    'backend': backend_name, 'n_qubits': n, 'blocks': blocks,
                    'n_params': 2 * blocks * n * 3, 'n_layouts': len(rows),
                    'rho_S_F': float(spearmanr(S, F).correlation),
                    'rho_S_F_sched': float(spearmanr(S, Fs).correlation),
                    'rho_SGR_F': float(spearmanr([r['S_gate_relax'] for r in rows], F).correlation),
                    'rho_S_F_out': float(spearmanr(S, Fo).correlation),
                    'pick_rank_F': 1 + int((F > F[pick] + 1e-12).sum()),
                    'pick_gap_F': float(F.max() - F[pick]),
                    'F_spread': float(F.max() - F.min()),
                    'F_median': float(np.median(F)),
                    'S_over_F_median': float(np.median(S / F)),
                    'uniform_rank': (len(rows) + 1) / 2,
                    'seconds': time.time() - t0, 'rows': rows,
                }
                results.append(rec)
                print(f"{backend_name} n={n} b={blocks}: {len(rows)} layouts  "
                      f"rho(S,F) {rec['rho_S_F']:+.2f}  sched {rec['rho_S_F_sched']:+.2f}  "
                      f"pick rank {rec['pick_rank_F']}/{len(rows)}  "
                      f"gap {rec['pick_gap_F']:.3f} of spread {rec['F_spread']:.3f}  "
                      f"[{rec['seconds']:.0f}s]", flush=True)
                json.dump(results, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()

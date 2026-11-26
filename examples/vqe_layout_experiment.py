"""Layout selection on a non-classification task: VQE for a 4-qubit spin chain
(Access revision, Reviewer 1 #6).

The mapping-side claim of the paper -- rank layouts by the analytic success
rate and enumerate them -- never looks at the loss being optimized, only at
the transpiled circuit and the calibration tables. This script tests it where
the objective is an energy instead of a classification accuracy.

Problem: transverse-field Ising chain, 4 qubits, open boundary, at the critical
point,
    H = -J * sum_{i=0}^{2} Z_i Z_{i+1}  -  h * sum_{i=0}^{3} X_i,   J = h = 1,
whose exact ground energy comes from diagonalizing the 16x16 matrix.

Ansatz: the paper's own u3cu3_s0 design space, built from the SuperCircuit
arch block of an existing run's metainfo/configs.yaml, at full layer width and
a chosen number of blocks (1 / 2 / 3 blocks = 24 / 48 / 72 parameters). Deeper
is pointless here: noise-free, 2 blocks already reach E0 to 1e-4, while under
noise 4 blocks (60 CNOTs) sit near the mixed-state energy of 0 on FakeQuitoV2
for every layout, leaving no layout signal to rank.

Per (blocks, training seed):
  1. train noise-free (exact statevector, Adam) to minimize <H>;
  2. convert to Qiskit and, for each backend, measure <H> under that device's
     noise model through QiskitProcessor -- the same transpile and simulator
     path the classification evals use -- for every one of the 120 initial
     layouts, plus Qiskit's free (calibration-blind) choice at levels 2 and 3;
  3. score every layout by get_success_rate, the same estimator the paper's
     enumeration uses, and ask Qiskit's calibration-aware layout (levels 2/3,
     transpiled against the fake backend's own Target), mapomatic and tket
     for their placements.
<H> needs two measurement settings: Z basis for the ZZ terms, X basis (a
Hadamard on every qubit before measuring) for the X terms.

Writes vqe_runs/tfim4/blk<B>_seed<S>/{checkpoint.pt, <backend>.json}.

Usage (repo root):
    python examples/vqe_layout_experiment.py --arch-from runs/<any u3cu3_s0 run> \
        --blocks 1 2 3 --seeds 0 1 2
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
from qiskit import QuantumCircuit
from torchpack.utils.config import configs

import torchquantum as tq
from torchquantum.algorithm.quantumnas.super_layers import super_layer_name_dict
from torchquantum.plugin import tq2qiskit
from torchquantum.util import get_success_rate

from core import builder
import compiler_layout_baselines as cb

N_WIRES = 4
J, H_FIELD = 1.0, 1.0
BACKENDS = ('FakeQuitoV2', 'FakeBelemV2', 'FakeAthensV2')
N_SHOTS = 8192


# ---- Hamiltonian ------------------------------------------------------------
ZZ_TERMS = [(-J, (i, i + 1)) for i in range(N_WIRES - 1)]
X_TERMS = [(-H_FIELD, i) for i in range(N_WIRES)]


def hamiltonian_matrix():
    """Dense H in torchquantum's basis order (wire 0 = most significant bit)."""
    I2 = np.eye(2)
    Z = np.diag([1.0, -1.0])
    X = np.array([[0.0, 1.0], [1.0, 0.0]])

    def kron_at(ops):
        m = np.array([[1.0]])
        for w in range(N_WIRES):
            m = np.kron(m, ops.get(w, I2))
        return m

    H = sum(c * kron_at({a: Z, b: Z}) for c, (a, b) in ZZ_TERMS)
    H = H + sum(c * kron_at({w: X}) for c, w in X_TERMS)
    return H


def energy_from_counts(counts_z, counts_x):
    """<H> from Z- and X-basis counts. Qiskit bitstrings put qubit 0 rightmost."""
    def expval(counts, wires):
        tot, acc = 0, 0.0
        for bits, n in counts.items():
            b = bits.replace(' ', '')[::-1]  # b[q] is qubit q's outcome
            parity = sum(int(b[w]) for w in wires) % 2
            acc += n * (1 - 2 * parity)
            tot += n
        return acc / tot

    e = sum(c * expval(counts_z, ab) for c, ab in ZZ_TERMS)
    e += sum(c * expval(counts_x, (w,)) for c, w in X_TERMS)
    return float(e)


# ---- ansatz -------------------------------------------------------------------
def build_ansatz(arch_cfg, n_blocks):
    arch = dict(arch_cfg)
    layer = super_layer_name_dict[arch['q_layer_name']](arch)
    # per-layer choices, as SuperQFCModel0.arch_space assembles them
    full_width = [max(sub.arch_space) for sub in layer.super_layers_all]
    sample = full_width + [n_blocks]
    layer.set_sample_arch(sample)
    return layer, sample


def train(layer, H, steps, lr, seed):
    torch.manual_seed(seed)
    for p in layer.parameters():
        torch.nn.init.uniform_(p, -np.pi, np.pi)
    Ht = torch.tensor(H, dtype=torch.complex64)
    opt = torch.optim.Adam(layer.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        qdev = tq.QuantumDevice(n_wires=N_WIRES, bsz=1)
        layer(qdev)
        psi = qdev.get_states_1d()[0]
        e = torch.real(torch.conj(psi) @ Ht @ psi)
        opt.zero_grad()
        e.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        qdev = tq.QuantumDevice(n_wires=N_WIRES, bsz=1)
        layer(qdev)
        psi = qdev.get_states_1d()[0]
        return float(torch.real(torch.conj(psi) @ Ht @ psi)), qdev


def measurement_circuits(ansatz):
    cz = ansatz.copy()
    cz.measure_all()
    cx = ansatz.copy()
    cx.h(range(N_WIRES))
    cx.measure_all()
    return cz, cx


# ---- noisy evaluation through the classification pipeline's own processor -------
def make_processor(backend_name, seed_eval_cfg):
    configs.clear()
    configs.load(seed_eval_cfg, recursive=True)
    configs.qiskit.backend_name = backend_name
    configs.qiskit.noise_model_name = backend_name
    configs.qiskit.initial_layout = None
    configs.qiskit.n_shots = N_SHOTS
    return builder.make_qiskit_processor()


def measured_energy(processor, cz, cx, layout, level=2):
    processor.set_layout(None if layout is None else list(layout))
    processor.optimization_level = level
    counts = processor.process_ready_circs_get_counts([cz, cx], parallel=False)
    return energy_from_counts(counts[0], counts[1])


def rank_lower_better(value, values):
    return 1 + sum(v < value for v in values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch-from', required=True,
                        help='run dir whose metainfo/configs.yaml supplies the '
                             'u3cu3_s0 SuperCircuit arch block')
    parser.add_argument('--eval-config',
                        default='examples/configs/mnist/four0123/eval/quito/noise/opt2/all.yml',
                        help='config whose qiskit block (shots, seeds, opt level) '
                             'the processor inherits; backend is overridden')
    parser.add_argument('--blocks', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--backends', nargs='+', default=list(BACKENDS))
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.02)
    parser.add_argument('--limit-layouts', type=int, default=None,
                        help='pilot only: evaluate the first N layouts')
    parser.add_argument('--out-root', default='vqe_runs/tfim4')
    args = parser.parse_args()

    configs.load(os.path.join(args.arch_from, 'metainfo', 'configs.yaml'))
    arch_cfg = dict(configs.model.arch)
    assert arch_cfg['n_wires'] == N_WIRES, arch_cfg['n_wires']

    H = hamiltonian_matrix()
    e0 = float(np.linalg.eigvalsh(H)[0])
    print(f'TFIM-4 (J={J}, h={H_FIELD}, open): exact ground energy E0 = {e0:.6f}')

    for n_blocks in args.blocks:
        for seed in args.seeds:
            out_dir = os.path.join(args.out_root, f'blk{n_blocks}_seed{seed}')
            os.makedirs(out_dir, exist_ok=True)
            layer, sample = build_ansatz(arch_cfg, n_blocks)
            n_params = layer.count_sample_params()
            t0 = time.time()
            e_train, qdev = train(layer, H, args.steps, args.lr, seed)
            ansatz = tq2qiskit(qdev, layer)
            from qiskit.quantum_info import Statevector
            e_qiskit = float(np.real(Statevector(ansatz).expectation_value(
                _qiskit_hamiltonian())))
            assert abs(e_qiskit - e_train) < 1e-3, (
                f'tq2qiskit changed the state: {e_train} vs {e_qiskit}')
            torch.save({'state_dict': layer.state_dict(), 'sample_arch': sample,
                        'arch_cfg': arch_cfg, 'e_train': e_train, 'e0': e0},
                       os.path.join(out_dir, 'checkpoint.pt'))
            print(f'### blk{n_blocks} seed{seed}: {n_params} params, '
                  f'noise-free E = {e_train:.5f} (E0 {e0:.5f}, gap '
                  f'{e_train - e0:.2e}), trained in {time.time() - t0:.0f}s',
                  flush=True)
            cz, cx = measurement_circuits(ansatz)

            for backend_name in args.backends:
                out_path = os.path.join(out_dir, f'{backend_name}.json')
                if os.path.exists(out_path):
                    print(f'  {backend_name}: exists, skipping')
                    continue
                t1 = time.time()
                processor = make_processor(backend_name, args.eval_config)
                backend = cb.get_backend(backend_name)
                properties = backend.properties()
                layouts = list(itertools.permutations(
                    range(backend.num_qubits), N_WIRES))
                if args.limit_layouts:
                    layouts = layouts[:args.limit_layouts]

                rows = {}
                for layout in layouts:
                    processor.set_layout(list(layout))
                    processor.optimization_level = 2
                    t = processor.transpile(ansatz)
                    t = t[0] if isinstance(t, list) else t
                    rows[layout] = {
                        'success_rate': get_success_rate(properties, t),
                        'energy': measured_energy(processor, cz, cx, layout),
                    }
                energies = [r['energy'] for r in rows.values()]
                srs = [r['success_rate'] for r in rows.values()]

                choices = {
                    'enumeration': max(rows, key=lambda l: rows[l]['success_rate']),
                    'oracle': min(rows, key=lambda l: rows[l]['energy']),
                }
                if not args.limit_layouts:
                    for level in (2, 3):
                        choices[f'qiskit_l{level}'] = tuple(
                            cb.qiskit_layout(cz, backend, level, 42)[0])
                    choices['mapomatic'] = tuple(cb.mapomatic_layout(cz, backend, 42)[0])
                    choices['tket'] = tuple(cb.tket_layout(cz, backend)[0])

                methods = {}
                for name, layout in choices.items():
                    r = rows[tuple(layout)]
                    methods[name] = {
                        'layout': list(layout),
                        'energy': r['energy'],
                        'energy_error': r['energy'] - e0,
                        'energy_rank': rank_lower_better(r['energy'], energies),
                        'success_rate': r['success_rate'],
                    }
                # Qiskit's free choice through QiskitProcessor: calibration-blind
                for level in (2, 3):
                    e = measured_energy(processor, cz, cx, None, level)
                    methods[f'qiskit_l{level}_free'] = {
                        'layout': None, 'energy': e, 'energy_error': e - e0,
                        'energy_rank_among_fixed': rank_lower_better(e, energies),
                    }
                from scipy.stats import spearmanr
                rho = spearmanr(srs, [-e for e in energies]).correlation

                rec = {
                    'backend': backend_name, 'n_blocks': n_blocks, 'seed': seed,
                    'n_params': n_params, 'sample_arch': sample,
                    'e0': e0, 'e_train_noise_free': e_train,
                    'n_layouts': len(rows), 'n_shots': N_SHOTS,
                    'random_layout_mean_energy': float(np.mean(energies)),
                    'energy_spread': float(max(energies) - min(energies)),
                    'spearman_success_rate_vs_neg_energy': float(rho),
                    'methods': methods,
                    'layouts': [{'layout': list(l), **r} for l, r in rows.items()],
                    'elapsed_seconds': time.time() - t1,
                }
                json.dump(rec, open(out_path, 'w'), indent=1)
                m = methods
                print(f"  {backend_name}: spread {rec['energy_spread']:.3f}, "
                      f"rho {rho:+.2f} | E enum {m['enumeration']['energy']:.3f} "
                      f"(rank {m['enumeration']['energy_rank']}), random mean "
                      f"{rec['random_layout_mean_energy']:.3f}, oracle "
                      f"{m['oracle']['energy']:.3f}, qiskit-free "
                      f"{m['qiskit_l2_free']['energy']:.3f} "
                      f"[{rec['elapsed_seconds']:.0f}s]", flush=True)


def _qiskit_hamiltonian():
    """Same H as a SparsePauliOp; Qiskit labels put qubit 0 rightmost."""
    from qiskit.quantum_info import SparsePauliOp

    def label(ops):
        return ''.join(ops.get(q, 'I') for q in reversed(range(N_WIRES)))

    terms = [(label({a: 'Z', b: 'Z'}), c) for c, (a, b) in ZZ_TERMS]
    terms += [(label({w: 'X'}), c) for c, w in X_TERMS]
    return SparsePauliOp.from_list(terms)


if __name__ == '__main__':
    main()

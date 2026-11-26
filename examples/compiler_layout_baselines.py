"""Initial layouts chosen by existing noise-aware compilers, scored the same way
as the success-rate enumeration (Access revision, Reviewer 1 #7 / Reviewer 2 #5).

For one trained run directory this script rebuilds the exact Qiskit circuit
that eval.py transpiles (encoder + sampled q_layer + measurement, taken from
one test sample through the model's own use_qiskit path), asks each mapping
tool for an initial layout, and writes

    <run_dir>/compiler_baselines/<backend>.json

with, per tool: the layout (virtual qubit i -> physical qubit layout[i]), its
estimated success rate and rank from layout_selection/<backend>.json, and its
accuracy and rank on the 300-sample layout sweep. Measured full-split accuracy
is a separate step: pass each layout back to eval.py as qiskit.initial_layout.

Tools:
  qiskit_l2   transpile(backend=<fake backend>, optimization_level=2) with no
              layout given, so VF2Layout/VF2PostLayout score candidates with
              the device's calibration errors: Qiskit's own noise-aware layout.
              NOT the paper's "Qiskit default" control -- that one goes through
              QiskitProcessor, whose Target (AerSimulator + copied coupling map
              and basis gates) has no error rates, so it is calibration-blind.
              run_compiler_baseline_evals.py measures both.
  qiskit_l3   the same at optimization_level=3.
  mapomatic   transpile at level 3, deflate, score every subgraph match
              against the backend's calibration (mapomatic.evaluate_layouts),
              map the best match back to the logical qubits.
  tket        pytket NoiseAwarePlacement on the backend's coupling map with
              per-qubit 1q/readout and per-edge 2q errors from its Target.

Why initial layouts and not each tool's own routed circuit: the success-rate
enumeration and the 120-layout sweep both hand Qiskit an initial_layout and let
level-2 routing do the rest, and with an initial_layout given Qiskit skips
VF2Layout/VF2PostLayout entirely. Feeding every tool's placement through that
same path compares placement decisions only, with routing held fixed.

Usage (from the repo root):
    python examples/compiler_layout_baselines.py \
        examples/configs/mnist/four0123/eval/quito/noise/opt2/all.yml \
        --run-dir runs/<run>
"""
import argparse
import json
import os
import sys
from itertools import permutations

import numpy as np
import torch
from torchpack.utils.config import configs

from core import builder

TOOLS = ('qiskit_l2', 'qiskit_l3', 'mapomatic', 'tket')


class _CaptureProcessor:
    """Stands in for QiskitProcessor: keeps the assembled circuits."""

    def __init__(self):
        self.circs = None

    def process_ready_circs(self, q_device, circs_all, parallel=True):
        self.circs = circs_all
        return torch.zeros(len(circs_all), q_device.n_wires)


def build_circuit(run_dir, eval_config):
    configs.load(os.path.join(run_dir, 'metainfo', 'configs.yaml'))
    configs.load(eval_config, recursive=True)
    configs.ckpt.weight_from_scratch = False
    state_dict = torch.load(os.path.join(run_dir, configs.ckpt.name),
                            map_location='cpu', weights_only=False)
    model = builder.make_model()
    model.load_state_dict(state_dict['model'], strict=False)
    solution = state_dict['solution']
    model.set_sample_arch(solution['arch'])
    model.eval()

    dataset = builder.make_dataset()
    # two samples: op_history2qiskit_expand_params asserts a per-sample
    # parameter list, which a batch of one collapses to a scalar
    split = dataset[configs.dataset.split]
    x = torch.stack([split[0]['image'], split[1]['image']])
    cap = _CaptureProcessor()
    model.set_qiskit_processor(cap)
    with torch.no_grad():
        model(x, use_qiskit=True)
    return cap.circs[0], solution, dict(configs.qiskit)


def get_backend(name):
    import qiskit_ibm_runtime.fake_provider as fp
    return getattr(fp, name)()


def target_errors(backend):
    """Per-qubit 1q gate / readout error and per-edge 2q error from the Target."""
    t = backend.target
    n = backend.num_qubits
    one_q, readout, two_q = {}, {}, {}
    for q in range(n):
        errs = [t[g][(q,)].error for g in ('sx', 'x')
                if g in t.operation_names and t[g].get((q,)) is not None
                and t[g][(q,)].error is not None]
        one_q[q] = float(np.mean(errs)) if errs else 0.0
        readout[q] = float(t['measure'][(q,)].error)
    for g in t.operation_names:
        for qargs, props in t[g].items():
            if qargs is not None and len(qargs) == 2 and props is not None \
                    and props.error is not None:
                two_q[tuple(qargs)] = float(props.error)
    return one_q, readout, two_q


def qiskit_layout(circ, backend, level, seed):
    passes, ps = [], {}

    def cb(**kw):
        passes.append(kw['pass_'].name())
        ps.update(dict(kw['property_set']))

    from qiskit import transpile
    t = transpile(circ, backend=backend, optimization_level=level,
                  seed_transpiler=seed, callback=cb)
    layout = t.layout.initial_index_layout(filter_ancillas=True)
    info = {
        'layout_passes': [p for p in dict.fromkeys(passes) if 'Layout' in p],
        'VF2Layout_stop_reason': str(ps.get('VF2Layout_stop_reason')),
        'VF2PostLayout_stop_reason': str(ps.get('VF2PostLayout_stop_reason')),
    }
    return [int(q) for q in layout[:circ.num_qubits]], t, info


def mapomatic_layout(circ, backend, seed):
    import mapomatic as mm
    _, trans, _ = qiskit_layout(circ, backend, 3, seed)
    small = mm.deflate_circuit(trans)
    matches = mm.matching_layouts(small, backend)
    scores = mm.evaluate_layouts(small, matches, backend)
    best = list(scores[0][0])
    # deflate_circuit keeps the active physical qubits in ascending order, so
    # deflated qubit k is active[k]; compose virtual -> initial physical ->
    # deflated index -> mapomatic's physical qubit.
    active = sorted({trans.find_bit(q).index for inst in trans.data
                     for q in inst.qubits
                     if inst.operation.name not in ('barrier', 'delay')})
    init = trans.layout.initial_index_layout(filter_ancillas=True)
    layout = [int(best[active.index(init[v])]) for v in range(circ.num_qubits)]
    return layout, {'n_matches': len(matches),
                    'best_score': float(scores[0][1])}


def tket_layout(circ, backend):
    from pytket.architecture import Architecture
    from pytket.circuit import Node, Qubit
    from pytket.extensions.qiskit import qiskit_to_tk
    from pytket.placement import NoiseAwarePlacement
    from qiskit import transpile

    one_q, readout, two_q = target_errors(backend)
    arch = Architecture([(Node(a), Node(b)) for a, b in two_q])
    placer = NoiseAwarePlacement(
        arch,
        node_errors={Node(q): e for q, e in one_q.items()},
        link_errors={(Node(a), Node(b)): e for (a, b), e in two_q.items()},
        readout_errors={Node(q): e for q, e in readout.items()},
    )
    logical = transpile(circ, basis_gates=['cx', 'u'], optimization_level=0)
    tkc = qiskit_to_tk(logical)
    pmap = placer.get_placement_map(tkc)
    layout, unplaced = [], []
    for v in range(circ.num_qubits):
        node = pmap.get(Qubit('q', v))
        if node is None or node.reg_name != 'node':
            layout.append(None)
            unplaced.append(v)
        else:
            layout.append(int(node.index[0]))
    # tket leaves qubits unplaced only when its pattern graph omits them;
    # fill those with the free qubit of lowest readout error, and say so.
    free = sorted((q for q in range(backend.num_qubits) if q not in layout),
                  key=lambda q: readout[q])
    for v in unplaced:
        layout[v] = free.pop(0)
    return layout, {'unplaced_filled': unplaced}


def rank_of(value, values, higher_is_better=True):
    """Competition rank (1 = best), same convention as the paper's tables."""
    if higher_is_better:
        return 1 + sum(v > value for v in values)
    return 1 + sum(v < value for v in values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()

    circ, solution, qcfg = build_circuit(args.run_dir, args.config)
    backend_name = qcfg['backend_name']
    backend = get_backend(backend_name)
    seed = qcfg.get('seed_transpiler', 42)

    with open(os.path.join(args.run_dir, 'layout_selection',
                           f'{backend_name}.json')) as f:
        selection = json.load(f)
    sr = {tuple(r['layout']): r['success_rate'] for r in selection['ranked']}
    sweep_path = [p for p in os.listdir(os.path.join(args.run_dir, 'layout_sweep'))
                  if p.endswith('.300.jsonl')]
    assert len(sweep_path) == 1, sweep_path
    sweep = {}
    with open(os.path.join(args.run_dir, 'layout_sweep', sweep_path[0])) as f:
        for line in f:
            rec = json.loads(line)
            sweep[tuple(rec['layout'])] = rec['accuracy_percent']
    n_avail = len(selection['ranked'][0]['layout']) and max(
        max(r['layout']) for r in selection['ranked']) + 1
    all_layouts = [tuple(p) for p in permutations(range(n_avail), circ.num_qubits)]
    assert set(all_layouts) == set(sr) == set(sweep), 'layout sets disagree'

    out = {
        'run_dir': args.run_dir,
        'eval_config': args.config,
        'backend': backend_name,
        'arch': solution['arch'],
        'searched_layout': list(solution['layout']),
        'enumerated_layout': selection['selected_layout'],
        'seed_transpiler': seed,
        'n_layouts': len(all_layouts),
        'tools': {},
    }
    choices = {
        'joint_search': (list(solution['layout']), {}),
        'enumeration': (selection['selected_layout'], {}),
    }
    for level, name in ((2, 'qiskit_l2'), (3, 'qiskit_l3')):
        layout, _, info = qiskit_layout(circ, backend, level, seed)
        choices[name] = (layout, info)
    choices['mapomatic'] = mapomatic_layout(circ, backend, seed)
    choices['tket'] = tket_layout(circ, backend)

    sr_vals, acc_vals = list(sr.values()), list(sweep.values())
    for name, (layout, info) in choices.items():
        key = tuple(layout)
        out['tools'][name] = {
            'layout': list(layout),
            'success_rate': sr[key],
            'success_rate_rank': rank_of(sr[key], sr_vals),
            'sweep300_accuracy_percent': sweep[key],
            'sweep300_rank': rank_of(sweep[key], acc_vals),
            **info,
        }

    out_dir = os.path.join(args.run_dir, 'compiler_baselines')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{backend_name}.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=1)
    for name, rec in out['tools'].items():
        print(f"{name:13s} layout={rec['layout']}  S_hat={rec['success_rate']:.4f} "
              f"(rank {rec['success_rate_rank']:3d})  sweep300={rec['sweep300_accuracy_percent']:.2f}% "
              f"(rank {rec['sweep300_rank']:3d})")
    print(f"-> {out_path}")


if __name__ == '__main__':
    sys.exit(main())

"""Sequential baseline for the NSGA-II joint search: take an architecture that
was chosen WITHOUT reference to the qubit mapping (e.g. an M2 expressibility-only
search result), then pick its layout by exhaustively scoring every physical-qubit
assignment with the analytic success_rate.

Why this exists
---------------
On a 5-qubit backend a 4-wire circuit has only 5P4 = 120 possible layouts, so
"search the architecture, then brute-force the mapping" is not merely feasible,
it is optimal by construction over that space -- no evolutionary search can beat
an exhaustive enumeration of 120 points. Any claim that joint (arch, layout)
search finds BETTER MAPPINGS than this is therefore false on this device, and a
reviewer will say so. The defensible claim for NSGA-II is different: that
choosing the architecture and the mapping *together* beats choosing the
architecture first and the mapping second, because the most expressible
architecture need not be the one that maps best. This script builds the
sequential arm of that comparison so the claim can actually be tested.

It scores layouts with the analytic success_rate rather than by running the
circuit, which is what makes the baseline cheap (~2 min for 120 layouts instead
of ~85 min of noisy execution) and training-free. That choice is validated
against measured accuracy: on FakeQuitoV2, where layout genuinely matters
(35 pp spread across the 120 layouts), Spearman(accuracy, success_rate) was
+0.82 / +0.76 across two seeds and argmax(success_rate) landed at rank 6/120 and
1/120 respectively. On Athens and Belem the whole spread is ~4 pp, at or below
the n=300 binomial noise floor, so no layout-selection rule can be distinguished
there -- do not read those rankings as signal.

Usage
-----
    python examples/select_layout_by_success_rate.py \
        --run-dir runs/<trained arch-only run> --backend FakeQuitoV2

Writes <run-dir>/layout_selection/<backend>.json and prints the ready-to-run
eval command, which pins the chosen layout via qiskit.initial_layout -- eval.py
gives that config key precedence over the layout stored in the checkpoint's
'solution', so the same trained checkpoint serves both arms of the comparison.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore")

import argparse
import itertools
import json

from torchpack.utils import io
from torchpack.utils.config import configs
from torchpack.utils.logging import logger
from qiskit_ibm_runtime.fake_provider import (
    FakeQuitoV2,
    FakeBelemV2,
    FakeAthensV2,
)

from torchquantum.plugin import tq2qiskit, tq2qiskit_parameterized
from torchquantum.util import get_success_rate

from core import builder

BACKENDS = {
    "FakeQuitoV2": FakeQuitoV2,
    "FakeBelemV2": FakeBelemV2,
    "FakeAthensV2": FakeAthensV2,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir', metavar='DIR', required=True,
                        help='trained run directory whose architecture is '
                             'taken as given (its checkpoint carries the arch)')
    parser.add_argument('--backend', required=True, choices=sorted(BACKENDS),
                        help='backend to select the layout for; overrides the '
                             'one named in the run config, so one trained '
                             'checkpoint can be targeted at every device')
    parser.add_argument('--ckpt', default='checkpoints/max-acc-valid.pt',
                        help="checkpoint to read the arch from (default "
                             "%(default)s). NOT the training config's "
                             "ckpt.name, which points at best_solution.pt")
    parser.add_argument('--arch', default=None,
                        help='comma-separated arch, overriding the one in the '
                             'checkpoint (for archs that never came from a '
                             'search)')
    parser.add_argument('--top-k', type=int, default=10,
                        help='how many layouts to print (default %(default)s)')
    parser.add_argument('--eval-config', default=None,
                        help='eval config to name in the printed command; '
                             'defaults to the 300-sample config for --backend')
    parser.add_argument('--sweep-jsonl', default=None,
                        help='an eval_layout_sweep.py jsonl for this same '
                             'run/backend. When given, the measured accuracy '
                             'of the selected layout is looked up instead of '
                             'having to re-run eval.py -- the sweep already '
                             'evaluated every layout, so the sequential '
                             'baseline needs no new noisy execution at all.')
    args = parser.parse_args()

    configs.load(os.path.join(args.run_dir, 'metainfo', 'configs.yaml'))
    # the run was trained against one device; we may be selecting for another
    configs.qiskit.backend_name = args.backend
    configs.qiskit.noise_model_name = args.backend
    # set_layout() is called explicitly per candidate below; make sure nothing
    # from the loaded config pins a layout behind our back
    configs.qiskit.initial_layout = None

    state_dict = io.load(os.path.join(args.run_dir, args.ckpt),
                         map_location='cpu', weights_only=False)

    if args.arch is not None:
        arch = [int(x) for x in args.arch.split(',')]
        arch_source = '--arch'
    elif 'solution' in state_dict:
        arch = list(state_dict['solution']['arch'])
        arch_source = f"{args.ckpt}['solution']['arch']"
    else:
        raise SystemExit(
            f"{args.ckpt} carries no 'solution'; pass --arch explicitly")

    backend = BACKENDS[args.backend]()
    properties = backend.properties()
    n_available_wires = len(properties.qubits)

    model = builder.make_model()
    model.eval()
    processor = builder.make_qiskit_processor()
    processor.properties = properties
    model.set_qiskit_processor(processor)
    model.set_sample_arch(arch)

    n_wires = model.q_device.n_wires
    logger.info(f"arch from {arch_source}: {arch}")
    logger.info(f"{args.backend}: {n_available_wires} physical qubits, "
                f"circuit uses {n_wires} wires -> "
                f"{n_available_wires}P{n_wires} layouts to enumerate")

    # the circuit is fixed across layouts; only the transpilation changes
    circ_fixed = tq2qiskit(model.q_device, model.q_layer)
    if configs.dataset.name == 'vqe':
        circ = circ_fixed
    else:
        circ_parameterized, _ = tq2qiskit_parameterized(
            model.q_device, model.encoder.func_list)
        circ = circ_parameterized.compose(circ_fixed)

    rows = []
    layouts = list(itertools.permutations(range(n_available_wires), n_wires))
    for i, layout in enumerate(layouts):
        processor.set_layout(list(layout))
        transpiled = processor.transpile(circ)
        if isinstance(transpiled, (list, tuple)):
            # QiskitProcessor.transpile() returns a list for a single circuit on
            # branches without the single-in/single-out fix (c12670ea, which
            # landed on es_expressibility_mapping only). Accept both so this
            # script runs unchanged wherever the trained checkpoints live.
            transpiled = transpiled[0]
        rows.append({
            'layout': list(layout),
            'success_rate': get_success_rate(properties, transpiled),
            'depth': transpiled.depth(),
        })
        if (i + 1) % 20 == 0:
            logger.info(f"  scored {i + 1}/{len(layouts)} layouts")

    rows.sort(key=lambda r: -r['success_rate'])
    best = rows[0]

    print(f"\nTop {args.top_k} layouts by success_rate "
          f"({args.backend}, {len(rows)} enumerated):")
    print(f"{'rank':>5}  {'layout':<16} {'success_rate':>14} {'depth':>7}")
    print('-' * 47)
    for i, r in enumerate(rows[:args.top_k]):
        print(f"{i + 1:>5}  {str(r['layout']):<16} "
              f"{r['success_rate']:>14.6f} {r['depth']:>7}")

    ckpt_layout = state_dict.get('solution', {}).get('layout')
    if ckpt_layout is not None:
        rank = next((i + 1 for i, r in enumerate(rows)
                     if r['layout'] == list(ckpt_layout)), None)
        print(f"\nlayout carried by the checkpoint: {list(ckpt_layout)} "
              f"-> rank {rank}/{len(rows)} by success_rate")

    out_dir = os.path.join(args.run_dir, 'layout_selection')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{args.backend}.json')
    with open(out_path, 'w') as f:
        json.dump({
            'run_dir': args.run_dir,
            'ckpt': args.ckpt,
            'backend': args.backend,
            'arch': arch,
            'arch_source': arch_source,
            'checkpoint_layout': list(ckpt_layout) if ckpt_layout else None,
            'n_layouts': len(rows),
            'selected_layout': best['layout'],
            'selected_success_rate': best['success_rate'],
            'ranked': rows,
        }, f, indent=1)
    print(f"\nsaved {out_path}")

    if args.sweep_jsonl is not None:
        measured = {}
        with open(args.sweep_jsonl) as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    measured[tuple(row['layout'])] = row['accuracy_percent']
        by_acc = sorted(measured.items(), key=lambda kv: -kv[1])
        sel = tuple(best['layout'])
        sel_acc = measured.get(sel)
        if sel_acc is None:
            print(f"\n!! selected layout {best['layout']} is absent from "
                  f"{args.sweep_jsonl}; cannot cross-reference")
        else:
            sel_rank = next(i + 1 for i, (lo, _) in enumerate(by_acc)
                            if lo == sel)
            top_acc = by_acc[0][1]
            ckpt_acc = measured.get(tuple(ckpt_layout)) if ckpt_layout else None
            print(f"\n=== cross-referenced against measured sweep "
                  f"({len(measured)} layouts) ===")
            print(f"  SEQUENTIAL BASELINE (argmax success_rate): "
                  f"{best['layout']}  acc {sel_acc:.1f}%  "
                  f"rank {sel_rank}/{len(measured)}")
            print(f"  exhaustive best (upper bound):            "
                  f"{list(by_acc[0][0])}  acc {top_acc:.1f}%")
            if ckpt_acc is not None:
                print(f"  layout the checkpoint carried:            "
                      f"{list(ckpt_layout)}  acc {ckpt_acc:.1f}%")
            print(f"  headroom left on the table: {top_acc - sel_acc:+.1f} pp "
                  f"(full spread across layouts: "
                  f"{top_acc - by_acc[-1][1]:.1f} pp)")

    eval_config = args.eval_config or (
        f"examples/configs/mnist/four0123/eval/"
        f"{args.backend.replace('Fake', '').replace('V2', '').lower()}"
        f"/noise/opt2/300.yml")
    layout_arg = '[' + ','.join(str(q) for q in best['layout']) + ']'
    print(f"\nEvaluate this arm with:\n"
          f"  python examples/eval.py {eval_config} \\\n"
          f"      --run-dir={args.run_dir} --jobs 8 \\\n"
          f"      ckpt.name={args.ckpt} \"qiskit.initial_layout={layout_arg}\"")


if __name__ == '__main__':
    main()

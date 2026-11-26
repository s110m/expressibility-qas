"""Evaluate one trained solution under EVERY qubit mapping, not just the one
the search happened to return.

Why this exists
---------------
`run_es.py`'s expressibility path scores a candidate with
`compute_expressibility_tq(expr_model, solution['arch'], ...)`, which is a
function of the architecture alone -- `set_layout` is only ever called in the
`loss_succ`/`acc_succ` branch (`run_es.py:171-172`). So for arms B and C the
layout gene is never read by the objective. Worse, the `solution_lib`
fingerprint keys on the whole solution *including* the layout, so same-arch
different-layout candidates are re-scored rather than deduped, and with
`es.expressibility.seed` pinned they all come back bit-identical; selection is
a strict `score < best_solution_score`, so the winner is whichever layout
happened to come first in the population. The layout in `best_solution.pt` is
therefore a tie-breaking artefact, not a search result.

The same problem shows up again, independently, when transferring a solution
searched on one device to another: the stored layout was (arbitrarily) fixed
under the search backend's coupling map, so carrying it to a different topology
is not a neutral choice. Reporting the distribution over all layouts removes
the arbitrary choice instead of defending it.

This script evaluates a run's trained checkpoint once per layout and records
min/max/mean/std plus the rank of the as-selected layout within that
distribution.

Relationship to eval.py
-----------------------
The load path is a faithful copy of `eval.py`'s (same config stacking, same
checkpoint handling, same inference loop), with two differences: the layout is
driven by this script rather than read from the checkpoint or
`configs.qiskit.initial_layout`, and results go to `layout_sweep/` as JSONL so
a long server run can be resumed after an interruption. Config paths that
`eval.py` supports but whose semantics under a layout sweep would be ambiguous
(`transpile_before_run`, `load_op_list`, VQE) are rejected up front rather than
silently mis-swept.
"""

import argparse
import datetime
import itertools
import json
import os
import random
import statistics

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.backends.cudnn
import torch.nn.functional as F
import tqdm

import torchquantum as tq
from torchpack.utils import io
from torchpack.utils.config import configs
from torchpack.utils.logging import logger

from qiskit_ibm_runtime import fake_provider

from core import builder
from torchquantum.algorithm.quantumnas.super_utils import get_named_sample_arch


def n_physical_qubits(name):
    """Qubit count of a fake backend, looked up by name.

    Same `getattr(fake_provider, ...)` lookup `QiskitProcessor.qiskit_init`
    uses, rather than a hardcoded table that would silently KeyError the first
    time a sweep is pointed at a device outside it.
    """
    backend_cls = getattr(fake_provider, name, None)
    if backend_cls is None:
        raise ValueError(
            f"No fake backend named {name!r} in qiskit_ibm_runtime."
            f"fake_provider; pass --n-available-wires to sweep anyway.")
    return backend_cls().num_qubits


def log_acc(output_all, target_all):
    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size
    loss = F.nll_loss(output_all, target_all).item()
    return accuracy, loss, size


def load_done(path):
    """Read already-completed layouts so an interrupted sweep can resume."""
    done = {}
    if not os.path.exists(path):
        return done
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # A crash mid-write can leave one truncated final line; the
                # layout it belonged to is simply re-run.
                logger.warning("Skipping malformed line in existing sweep file")
                continue
            done[tuple(rec['layout'])] = rec
    return done


def main():
    # Identical to eval.py's reproducibility block, so that the row this sweep
    # produces for the as-selected layout is directly comparable to (and should
    # reproduce) the number eval.py already reported for that run.
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE', help='eval config file')
    parser.add_argument('--run-dir', metavar='DIR', required=True,
                        help='trained run directory to evaluate')
    parser.add_argument('--gpu', type=str, default=None, help='gpu ids')
    parser.add_argument('--jobs', type=int, default=None,
                        help='max parallel jobs on qiskit')
    parser.add_argument('--n-available-wires', type=int, default=None,
                        help='physical qubit pool size; default: read from '
                             'the backend properties')
    parser.add_argument('--limit', type=int, default=None,
                        help='evaluate only the first N layouts (pilot runs)')
    parser.add_argument('--out-name', type=str, default=None,
                        help='basename under layout_sweep/; default derives '
                             'from the eval config path, matching eval.py')
    parser.add_argument('--restart', action='store_true',
                        help='discard an existing sweep file instead of '
                             'resuming from it')
    args, opts = parser.parse_known_args()

    configs.load(os.path.join(args.run_dir, 'metainfo', 'configs.yaml'))
    configs.load(args.config, recursive=True)
    configs.update(opts)

    configs.ckpt.weight_from_scratch = False
    configs.verbose = False
    configs.qiskit.hub = None

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.jobs is not None:
        configs.qiskit.max_jobs = args.jobs

    device = torch.device('cuda') if configs.run.device == 'gpu' \
        else torch.device('cpu')

    # Guards: these paths exist in eval.py but would make a layout sweep
    # meaningless or ambiguous, so fail loudly rather than sweep the wrong
    # thing for several hours.
    if not configs.qiskit.use_qiskit:
        raise ValueError(
            "qiskit.use_qiskit is False -- there is no layout to sweep. Pass "
            "one of the eval/<device>/noise/... configs.")
    if configs.model.transpile_before_run:
        raise ValueError(
            "model.transpile_before_run is True; eval.py forces "
            "set_layout([1,1,1,1]) on that path, so a layout sweep would not "
            "mean what it says.")
    if configs.model.load_op_list:
        raise ValueError("model.load_op_list is not supported by the sweep.")
    if configs.dataset.name == 'vqe':
        raise ValueError("VQE has no accuracy to sweep.")

    eval_config_dir = args.config.replace('/', '.').replace(
        'examples.', '').replace('.yml', '').replace('configs.', '')
    out_name = args.out_name or eval_config_dir
    out_dir = os.path.join(args.run_dir, 'layout_sweep')
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, f'{out_name}.jsonl')
    summary_path = os.path.join(out_dir, f'{out_name}.summary.json')

    if args.restart and os.path.exists(jsonl_path):
        os.remove(jsonl_path)

    # ---- data ----------------------------------------------------------
    dataset = builder.make_dataset()
    sampler = torch.utils.data.SequentialSampler(
        dataset[configs.dataset.split])
    dataflow = torch.utils.data.DataLoader(
        dataset[configs.dataset.split],
        sampler=sampler,
        batch_size=configs.run.bsz,
        num_workers=configs.run.workers_per_gpu,
        pin_memory=True)

    # ---- model ---------------------------------------------------------
    state_dict = io.load(os.path.join(args.run_dir, configs.ckpt.name),
                         map_location='cpu', weights_only=False)
    model_load = state_dict['model_arch']
    model = builder.make_model()
    for module_load, module in zip(model_load.modules(), model.modules()):
        if isinstance(module, tq.RandomLayer):
            module.rebuild_random_layer_from_op_list(
                n_ops_in=module_load.n_ops,
                wires_in=module_load.wires,
                op_list_in=module_load.op_list,
            )
    model.load_state_dict(state_dict['model'], strict=False)

    solution = state_dict.get('solution', None)
    if solution is None:
        raise ValueError(
            f"{args.run_dir} has no 'solution' in its checkpoint; nothing to "
            f"sweep against.")
    model.set_sample_arch(solution['arch'])
    selected_layout = list(solution['layout'])
    sweep_arch = list(solution['arch'])
    logger.info(f"Solution arch: {solution['arch']}")
    logger.info(f"As-selected layout: {selected_layout}")

    # eval.py lets a config override the checkpoint's arch (several leaves such
    # as eval/athens/noise/opt2/all_arch.yml do exactly that). Honour it here
    # too rather than sweeping a different circuit than the same config would
    # evaluate -- but keep `selected_layout` from the solution, since the
    # question being asked is still "where did the search's layout land?".
    config_arch = getattr(configs.model.arch, 'sample_arch', None)
    if config_arch is not None:
        if isinstance(config_arch, str):
            config_arch = get_named_sample_arch(model.arch_space, config_arch)
        logger.warning(f"Config overrides the checkpoint arch: {config_arch}")
        model.set_sample_arch(config_arch)
        sweep_arch = list(config_arch)

    if getattr(configs.qiskit, 'initial_layout', None) is not None:
        logger.warning(
            f"Ignoring qiskit.initial_layout="
            f"{configs.qiskit.initial_layout} from the config; this script "
            f"drives the layout itself.")

    if 'v_c_reg_mapping' in state_dict.keys():
        try:
            model.measure.set_v_c_reg_mapping(state_dict['v_c_reg_mapping'])
        except AttributeError:
            logger.warning("Cannot set v_c_reg_mapping.")

    if state_dict.get('noise_model_tq', None) is not None:
        model.set_noise_model_tq(state_dict['noise_model_tq'])
        model.noise_model_tq.mode = 'test'

    model.to(device)
    model.eval()

    qiskit_processor = builder.make_qiskit_processor()
    model.set_qiskit_processor(qiskit_processor)

    # ---- layouts -------------------------------------------------------
    n_wires = model.q_device.n_wires
    if args.n_available_wires is not None:
        n_available_wires = args.n_available_wires
    else:
        # `noise_model_name` is the device whose coupling map QiskitProcessor
        # actually transpiles against on the simulated path, so it -- not
        # `backend_name` (which is AerSimulator there) -- defines the pool.
        device_name = (getattr(configs.qiskit, 'noise_model_name', None)
                       or getattr(configs.qiskit, 'backend_name', None))
        if device_name is None:
            raise ValueError(
                "Neither qiskit.noise_model_name nor qiskit.backend_name is "
                "set; pass --n-available-wires.")
        n_available_wires = n_physical_qubits(device_name)
    layouts = [list(p) for p in
               itertools.permutations(range(n_available_wires), r=n_wires)]
    if args.limit is not None:
        layouts = layouts[:args.limit]
    logger.info(f"{n_wires} logical on {n_available_wires} physical qubits "
                f"-> {len(layouts)} layouts")

    done = load_done(jsonl_path)
    if done:
        logger.info(f"Resuming: {len(done)}/{len(layouts)} layouts already "
                    f"in {jsonl_path}")

    meta = {
        'eval_config': args.config,
        'run_dir': args.run_dir,
        'ckpt': configs.ckpt.name,
        'split': configs.dataset.split,
        'backend_name': getattr(configs.qiskit, 'backend_name', None),
        'noise_model_name': getattr(configs.qiskit, 'noise_model_name', None),
        'n_shots': getattr(configs.qiskit, 'n_shots', None),
        'optimization_level': getattr(configs.qiskit, 'optimization_level',
                                      None),
        'arch': sweep_arch,
        'solution_arch': list(solution['arch']),
        'selected_layout': selected_layout,
        'es_score': state_dict.get('score', None),
    }

    # ---- sweep ---------------------------------------------------------
    for i, layout in enumerate(layouts):
        if tuple(layout) in done:
            continue
        qiskit_processor.set_layout(layout)
        start = datetime.datetime.now()
        with torch.no_grad():
            target_all = None
            output_all = None
            for feed_dict in tqdm.tqdm(
                    dataflow, desc=f"[{i + 1}/{len(layouts)}] {layout}"):
                if configs.run.device == 'gpu':
                    inputs = feed_dict[configs.dataset.input_name].cuda(
                        non_blocking=True)
                    targets = feed_dict[configs.dataset.target_name].cuda(
                        non_blocking=True)
                else:
                    inputs = feed_dict[configs.dataset.input_name]
                    targets = feed_dict[configs.dataset.target_name]
                outputs = model(inputs, verbose=False, use_qiskit=True)
                if target_all is None:
                    target_all, output_all = targets, outputs
                else:
                    target_all = torch.cat([target_all, targets], dim=0)
                    output_all = torch.cat([output_all, outputs], dim=0)

        accuracy, loss, n_samples = log_acc(output_all, target_all)
        rec = {
            'layout': layout,
            'accuracy': accuracy,
            'accuracy_percent': accuracy * 100,
            'loss': loss,
            'n_samples': n_samples,
            'is_selected_layout': layout == selected_layout,
            'elapsed_seconds': (datetime.datetime.now() -
                                start).total_seconds(),
            'timestamp': datetime.datetime.now().isoformat(),
        }
        # Append-and-flush per layout: a sweep is hours long, and a crash at
        # layout 90 must not cost the first 89.
        with open(jsonl_path, 'a') as f:
            f.write(json.dumps(rec) + '\n')
            f.flush()
        done[tuple(layout)] = rec
        logger.info(f"[{i + 1}/{len(layouts)}] layout={layout} "
                    f"acc={accuracy * 100:.2f}%")

    # ---- summary -------------------------------------------------------
    records = [done[tuple(l)] for l in layouts if tuple(l) in done]
    if not records:
        raise RuntimeError(
            "No layouts were evaluated; nothing to summarize. (Was --limit 0 "
            "passed, or did every evaluation fail?)")
    accs = [r['accuracy_percent'] for r in records]
    order = sorted(records, key=lambda r: r['accuracy_percent'], reverse=True)

    selected = next((r for r in records
                     if r['layout'] == selected_layout), None)
    if selected is not None:
        rank = order.index(selected) + 1
        # Percentile of layouts this one beats; 100 = best available.
        pct = 100.0 * (len(order) - rank) / (len(order) - 1) \
            if len(order) > 1 else float('nan')
    else:
        rank, pct = None, None

    summary = {
        **meta,
        'timestamp': datetime.datetime.now().isoformat(),
        'n_layouts_evaluated': len(records),
        'n_layouts_total': len(layouts),
        'min_accuracy_percent': min(accs),
        'max_accuracy_percent': max(accs),
        'mean_accuracy_percent': statistics.mean(accs),
        'std_accuracy_percent': statistics.stdev(accs) if len(accs) > 1
        else 0.0,
        'median_accuracy_percent': statistics.median(accs),
        'range_accuracy_percent': max(accs) - min(accs),
        'best_layout': order[0]['layout'],
        'worst_layout': order[-1]['layout'],
        'selected_layout_accuracy_percent':
            selected['accuracy_percent'] if selected else None,
        'selected_layout_rank': rank,
        'selected_layout_percentile': pct,
        'accuracy_gap_to_best_percent':
            (order[0]['accuracy_percent'] - selected['accuracy_percent'])
            if selected else None,
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"--- layout sweep: {args.run_dir} @ "
                f"{configs.qiskit.backend_name} ({configs.dataset.split}, "
                f"n={records[0]['n_samples']}) ---")
    logger.info(f"  layouts      : {len(records)}/{len(layouts)}")
    logger.info(f"  min / max    : {min(accs):.2f}% / {max(accs):.2f}%  "
                f"(range {max(accs) - min(accs):.2f} pp)")
    logger.info(f"  mean +- std  : {statistics.mean(accs):.2f}% +- "
                f"{summary['std_accuracy_percent']:.2f}")
    if selected is not None:
        logger.info(f"  as-selected  : {selected['accuracy_percent']:.2f}% "
                    f"-> rank {rank}/{len(order)} "
                    f"({pct:.1f}th percentile), "
                    f"{summary['accuracy_gap_to_best_percent']:.2f} pp below "
                    f"best")
    logger.info(f"Wrote {summary_path}")


if __name__ == '__main__':
    main()

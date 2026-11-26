"""
Cheaply scores every solution on a saved NSGA-II Pareto front
(examples/run_nsga2.py's checkpoints/pareto_front.pt) by accuracy/loss,
reusing the SuperCircuit's already-trained shared weights -- no retraining,
just model.set_sample_arch(...) + a forward pass per distinct architecture on
the ideal (noiseless) torchquantum simulator.

Layout doesn't affect this ideal-simulator forward pass at all (only
depth/success_rate cared about physical qubits), so solutions are cached and
scored by architecture alone: multiple Pareto solutions sharing the same arch
but different layout reuse a single forward pass.

--noisy adds a SECOND score per solution, executed through the qiskit path
(qiskit_processor.set_layout(solution['layout']) + model(..., use_qiskit=True))
against the noise model named by the search run's own config -- i.e. the same
scoring run_es.py's `loss_succ` score_mode does for the single-objective ES.
That number is written as accuracy_noisy/loss_noisy alongside the ideal ones
and, unlike them, genuinely depends on the layout, so it can be used as the
export_pareto_solution.py selection criterion without the mapping half of the
search collapsing into a tie-break. It costs one noisy execution per DISTINCT
(arch, layout) pair -- minutes per solution, not seconds -- so it is off by
default.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore")

import argparse
import pdb
import random

import numpy as np
import torch
import torch.backends.cudnn
import torch.nn.functional as F
import torchquantum as tq
from torchpack.utils import io
from torchpack.utils.config import configs
from torchpack.utils.logging import logger

from core import builder
from core.tools import StepTimer


def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def fingerprint_arch(arch):
    n_layers_per_block = configs.model.arch.n_layers_per_block
    arch = list(arch)
    trimmed = arch[:arch[-1] * n_layers_per_block]
    trimmed.append(arch[-1])
    return str(trimmed)


def evaluate(model, dataflow, use_qiskit):
    """One pass over dataflow; returns (accuracy, loss)."""
    with torch.no_grad():
        target_all = None
        output_all = None
        for feed_dict in dataflow:
            if configs.run.device == 'gpu':
                inputs = feed_dict[configs.dataset.input_name].cuda(
                    non_blocking=True)
                targets = feed_dict[configs.dataset.target_name].cuda(
                    non_blocking=True)
            else:
                inputs = feed_dict[configs.dataset.input_name]
                targets = feed_dict[configs.dataset.target_name]

            outputs = model(inputs, use_qiskit=use_qiskit)

            if target_all is None:
                target_all = targets
                output_all = outputs
            else:
                target_all = torch.cat([target_all, targets], dim=0)
                output_all = torch.cat([output_all, outputs], dim=0)

    if configs.dataset.name == 'vqe':
        return -1, output_all[0].item()

    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    accuracy = masks.sum().item() / target_all.shape[0]
    loss = F.nll_loss(output_all, target_all).item()
    return accuracy, loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('pareto_front', metavar='FILE',
                        help='path to a pareto_front.pt saved by run_nsga2.py')
    parser.add_argument('--run-dir', metavar='DIR', required=True,
                        help='original SuperCircuit training run directory '
                             '(to load the trained checkpoint from)')
    parser.add_argument('--noisy', action='store_true',
                        help='additionally score every solution through the '
                             'qiskit noise model named by the search config, '
                             'with its own layout applied -- adds '
                             'accuracy_noisy/loss_noisy per entry. One noisy '
                             'execution per distinct (arch, layout) pair, so '
                             'expect ~1 min per solution, not seconds.')
    parser.add_argument('--jobs', type=int, default=None,
                        help='max parallel qiskit jobs (--noisy only); '
                             'overrides qiskit.max_jobs from the config')
    parser.add_argument('--out-name', type=str,
                        default='pareto_front_scored.pt',
                        help='output filename written next to the input front; '
                             'use a distinct name for a --noisy pass to keep '
                             'an existing ideal-only scored front intact')
    parser.add_argument('--pdb', action='store_true', help='pdb')
    parser.add_argument('--gpu', type=str, help='gpu ids', default=None)
    args, opts = parser.parse_known_args()

    checkpoints_dir = os.path.dirname(args.pareto_front)
    es_dir = os.path.dirname(checkpoints_dir)

    configs.load(os.path.join(es_dir, 'metainfo', 'configs.yaml'))
    configs.update(opts)
    set_seed(configs.debug.seed)

    if configs.debug.pdb or args.pdb:
        pdb.set_trace()

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if configs.run.device == 'gpu':
        device = torch.device('cuda')
    elif configs.run.device == 'cpu':
        device = torch.device('cpu')
    else:
        raise ValueError(configs.run.device)

    front_data = io.load(args.pareto_front, weights_only=False)
    pareto_front = front_data['pareto_front']
    objectives = front_data['objectives']

    logger.info(f"Scoring {len(pareto_front)} Pareto solutions from "
                f"{args.pareto_front} (objectives: {objectives})")

    if args.jobs is not None:
        configs.qiskit.max_jobs = args.jobs

    timing_name = 'pareto_scoring_timing.json' if not args.noisy \
        else 'pareto_scoring_noisy_timing.json'

    with StepTimer('pareto_front_scoring',
                   out_dir=os.path.join(es_dir, 'metainfo'),
                   filename=timing_name,
                   extra={'pareto_front': args.pareto_front,
                          'run_dir': args.run_dir,
                          'noisy': args.noisy,
                          'noise_model': configs.qiskit.noise_model_name
                          if args.noisy else None}):
        dataset = builder.make_dataset()
        sampler = torch.utils.data.SequentialSampler(dataset[configs.dataset.split])

        g = torch.Generator()
        g.manual_seed(configs.debug.seed)
        dataflow = torch.utils.data.DataLoader(
            dataset[configs.dataset.split],
            sampler=sampler,
            batch_size=configs.run.bsz,
            num_workers=configs.run.workers_per_gpu,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g)

        state_dict = io.load(
            os.path.join(args.run_dir, configs.ckpt.name),
            map_location=device, weights_only=False)

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
        model.to(device)
        model.eval()

        if args.noisy:
            # the search run's own config names the backend/noise model, so an
            # athens front is automatically scored under athens noise
            qiskit_processor = builder.make_qiskit_processor()
            model.set_qiskit_processor(qiskit_processor)
            logger.info(f"Noisy scoring enabled: backend "
                        f"{configs.qiskit.backend_name}, noise model "
                        f"{configs.qiskit.noise_model_name}, "
                        f"n_shots {configs.qiskit.n_shots}, "
                        f"max_jobs {configs.qiskit.max_jobs}")

        arch_cache = {}
        noisy_cache = {}
        scored_front = []

        for entry in pareto_front:
            solution = entry['solution']
            fingerprint = fingerprint_arch(solution['arch'])

            if fingerprint not in arch_cache:
                model.set_sample_arch(solution['arch'])
                accuracy, loss = evaluate(model, dataflow, use_qiskit=False)
                arch_cache[fingerprint] = {'accuracy': accuracy, 'loss': loss}

            scored_entry = {
                'solution': solution,
                'objectives': entry['objectives'],
                'accuracy': arch_cache[fingerprint]['accuracy'],
                'loss': arch_cache[fingerprint]['loss'],
            }

            if args.noisy:
                # unlike the ideal pass, this one depends on the layout, so the
                # cache key must carry it -- two solutions sharing an arch but
                # differing in layout are genuinely different measurements here
                noisy_key = (fingerprint, tuple(solution['layout']))
                if noisy_key not in noisy_cache:
                    model.qiskit_processor.set_layout(list(solution['layout']))
                    model.set_sample_arch(solution['arch'])
                    accuracy_noisy, loss_noisy = evaluate(
                        model, dataflow, use_qiskit=True)
                    noisy_cache[noisy_key] = {'accuracy_noisy': accuracy_noisy,
                                              'loss_noisy': loss_noisy}
                scored_entry.update(noisy_cache[noisy_key])

            scored_front.append(scored_entry)
            logger.info(f"Solution: {solution}, objectives: {entry['objectives']}, "
                        f"accuracy: {scored_entry['accuracy']:.5f}, "
                        f"loss: {scored_entry['loss']:.5f}" +
                        (f", accuracy_noisy: {scored_entry['accuracy_noisy']:.5f}"
                         f", loss_noisy: {scored_entry['loss_noisy']:.5f}"
                         if args.noisy else ''))

        out_path = os.path.join(checkpoints_dir, args.out_name)
        io.save(out_path, {'pareto_front': scored_front, 'objectives': objectives})
        noisy_note = ''
        if args.noisy:
            noisy_note = (f", {len(noisy_cache)} distinct (arch, layout) pairs "
                          f"evaluated under noise")
        logger.info(f"Scored {len(scored_front)} solutions "
                    f"({len(arch_cache)} distinct architectures evaluated"
                    f"{noisy_note}), saved to {out_path}")

        if args.noisy:
            # the pilot question in one line: does the noisy score actually
            # spread across the front, or is every candidate the same?
            noisy_accs = [e['accuracy_noisy'] for e in scored_front]
            ideal_accs = [e['accuracy'] for e in scored_front]
            logger.info(
                f"accuracy_noisy spread: min {min(noisy_accs):.5f}, "
                f"max {max(noisy_accs):.5f}, "
                f"range {max(noisy_accs) - min(noisy_accs):.5f}  |  "
                f"accuracy (ideal) spread: min {min(ideal_accs):.5f}, "
                f"max {max(ideal_accs):.5f}, "
                f"range {max(ideal_accs) - min(ideal_accs):.5f}")


if __name__ == '__main__':
    main()

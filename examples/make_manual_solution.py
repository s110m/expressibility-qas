"""
Build a (arch, layout) "solution" checkpoint without running run_es.py's
evolutionary search -- either a hand-picked (--mode manual) or a single
randomly-drawn (--mode random) point in the SuperCircuit's design space.

The output is a drop-in replacement for run_es.py's checkpoints/best_solution.pt
(same 'solution'/'score' keys), meant to be consumed by a
ckpt.weight_from_scratch: True training config (the train/*/scratch/ tree).
It stands in for the whole SuperCircuit-training + evolutionary-search half of
the pipeline, so it writes to es_runs/ next to real ES output, with the same
checkpoints/ + metainfo/ layout. (No tb/ subdirectory: there are no iterations
to plot -- a manual solution is a single fixed point, not a search.)

Full pipeline for "largest architecture, hand-picked layout, hybrid_full":

    python examples/make_manual_solution.py \
        examples/configs/mnist/four0123/train/searched/scratch/athens/noise/opt2/manual/u3cu3_s0/hybrid_full/blk8s1.1.1.yml \
        --mode manual --tag largest_layout0123_hybrid_full \
        --arch 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8 --layout 0,1,2,3

    python examples/train.py \
        examples/configs/mnist/four0123/train/searched/scratch/athens/noise/opt2/manual/u3cu3_s0/hybrid_full/blk8s1.1.1.yml \
        --ckpt-dir=es_runs/manual_solution_largest_layout0123_hybrid_full/ \
        debug.seed=42

    python examples/eval.py \
        examples/configs/mnist/four0123/eval/athens/noise/opt2/300.yml \
        --run-dir=runs/mnist.four0123.train.searched.scratch.athens.noise.opt2.manual.u3cu3_s0.hybrid_full.blk8s1.1.1_seed42/

IMPORTANT -- the `config` argument must be the SAME leaf you pass to train.py.
It is what decides whether the model is pure-quantum or hybrid: the
.../manual/u3cu3_s0/ tree builds a pure-quantum SuperQFCModel0, and the
.../manual/u3cu3_s0/hybrid_full/ tree adds model.arch.hybrid.mode: hybrid_full
plus model.arch.classical_in_dim. Passing the pure-quantum leaf here and the
hybrid leaf to train.py (or vice versa) still "works" -- the solution file only
carries arch+layout -- but you would be training a different model than the one
you scored. This script prints the resolved hybrid mode and a parameter
breakdown so the mismatch is visible immediately (a four0123 u3cu3_s0 largest
arch is 192 params pure-quantum vs 484 with hybrid_full's classical layers).

The --tag becomes part of the output path (es_runs/manual_solution_<tag>/), so
each manual/random solution gets its own directory that never collides with an
ES run's output. Put the variant in the tag when you build several.
"""
import argparse
import datetime
import os
import random

import numpy as np
import torch
import torch.utils.data

from torchpack.utils import io
from torchpack.utils.config import configs
from torchpack.utils.logging import logger
from core import builder
from torchquantum.algorithm.quantumnas.super_utils import ArchSampler


def parse_int_list(s):
    return [int(x) for x in s.split(',') if x.strip() != '']


def validate_arch(arch, arch_space):
    if len(arch) != len(arch_space):
        raise ValueError(
            f"arch has {len(arch)} entries but this design space expects "
            f"{len(arch_space)} (one gate-choice per layer, plus a "
            f"trailing block-count choice)")
    for i, (choice, space) in enumerate(zip(arch, arch_space)):
        if choice not in space:
            raise ValueError(
                f"arch[{i}]={choice} is not a valid choice at that "
                f"position; allowed values are {list(space)}")


def validate_layout(layout, n_wires, n_available_wires):
    if len(layout) != n_wires:
        raise ValueError(f"layout must have {n_wires} entries (one per "
                         f"logical wire), got {layout}")
    if len(set(layout)) != n_wires:
        raise ValueError(f"layout entries must map to distinct physical "
                         f"qubits, got {layout}")
    if any(q < 0 or q >= n_available_wires for q in layout):
        raise ValueError(f"layout entries must be in [0, "
                         f"{n_available_wires}), got {layout}")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def infer_backend_n_wires(default):
    """Physical-qubit count of the fake backend this config targets.

    A layout maps logical wires onto the *device's* qubits, so on a 5-qubit
    FakeAthensV2 a layout like [1, 2, 3, 4] is perfectly legal even though the
    circuit only has 4 wires. Defaulting the pool to the model's own n_wires
    would reject those, so read the real size off the backend when the config
    names one; fall back to `default` (a trivial 1-to-1 mapping) otherwise.
    """
    name = (getattr(configs.qiskit, 'backend_name', None)
            or getattr(configs.qiskit, 'noise_model_name', None))
    if name is None:
        return default, None
    try:
        from qiskit_ibm_runtime import fake_provider
        backend = getattr(fake_provider, name)()
        return backend.num_qubits, name
    except Exception as e:  # unknown name, real-QC name, import failure
        logger.warning(f"Could not resolve backend '{name}' to a qubit count "
                       f"({e}); falling back to {default} available wires. "
                       f"Pass --n-available-wires to override.")
        return default, None


def summarize_params(model, arch):
    """Parameter breakdown of the model this solution will actually train.

    `count_sample_params()` deliberately counts only the *sampled quantum*
    params, so on its own it looks identical for pure_quantum and hybrid runs
    -- which is exactly the confusion this function exists to prevent. The
    number train.py prints as "Model Size" is the total, classical layers
    included.
    """
    model.set_sample_arch(arch)
    total = sum(p.numel() for p in model.parameters())
    quantum_sampled = model.count_sample_params()

    classical = 0
    for attr in ('classical_encoder', 'classical_output'):
        module = getattr(model, attr, None)
        if module is not None:
            classical += sum(p.numel() for p in module.parameters())

    return {
        'hybrid_mode': getattr(model, 'hybrid_mode', 'pure_quantum'),
        'total_params': total,
        'quantum_sampled_params': quantum_sampled,
        'classical_params': classical,
        'trainable_params': sum(p.numel() for p in model.parameters()
                                if p.requires_grad),
    }


def evaluate_solution_tq(model, solution, device, seed):
    """Plain torchquantum-simulated (no noise/qiskit) accuracy+loss of
    `solution`, using whatever weights are currently loaded on `model`.
    Meant as a quick sanity number, not a rigorous noise-aware score --
    use run_es.py directly if you need the qiskit/noise-aware score."""
    import torch.nn.functional as F

    model.set_sample_arch(solution['arch'])
    model.to(device)
    model.eval()

    dataset = builder.make_dataset()
    sampler = torch.utils.data.SequentialSampler(dataset[configs.dataset.split])
    g = torch.Generator()
    g.manual_seed(seed)
    dataflow = torch.utils.data.DataLoader(
        dataset[configs.dataset.split],
        sampler=sampler,
        batch_size=configs.run.bsz,
        num_workers=configs.run.workers_per_gpu,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g)

    target_all, output_all = None, None
    with torch.no_grad():
        for feed_dict in dataflow:
            inputs = feed_dict[configs.dataset.input_name].to(device)
            targets = feed_dict[configs.dataset.target_name].to(device)
            outputs = model(inputs, use_qiskit=False)
            if target_all is None:
                target_all, output_all = targets, outputs
            else:
                target_all = torch.cat([target_all, targets], dim=0)
                output_all = torch.cat([output_all, outputs], dim=0)

    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    accuracy = masks.sum().item() / target_all.shape[0]
    loss = F.nll_loss(output_all, target_all).item()
    return accuracy, loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', metavar='FILE',
                        help='shape config used to build the SuperCircuit '
                             '(for arch_space/n_wires validation) -- '
                             'normally the same leaf you will pass to '
                             'train.py for the actual training run')
    parser.add_argument('--mode', choices=['manual', 'random'],
                        default='manual')
    parser.add_argument('--arch', type=str, default=None,
                        help='comma-separated arch gene, required for '
                             '--mode manual')
    parser.add_argument('--layout', type=str, default=None,
                        help='comma-separated physical-qubit layout, '
                             'required for --mode manual')
    parser.add_argument('--n-available-wires', type=int, default=None,
                        help='size of the physical-qubit pool the layout may '
                             'address (default: the qubit count of the fake '
                             'backend named in the config, e.g. 5 for '
                             'FakeAthensV2, else the model n_wires)')
    parser.add_argument('--tag', type=str, required=True,
                        help='short identifier for this solution; the '
                             'checkpoint is written to '
                             'es_runs/manual_solution_<tag>/checkpoints/'
                             'best_solution.pt so it never collides with '
                             'a real ES run\'s output')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='override the default '
                             'es_runs/manual_solution_<tag>/ location')
    parser.add_argument('--score', type=float, default=0.0,
                        help='placeholder score recorded in the checkpoint '
                             'when not passing --evaluate')
    parser.add_argument('--evaluate', action='store_true',
                        help='compute accuracy/loss for the solution using '
                             'weights inherited from a trained SuperCircuit '
                             '(requires --run-dir). Plain torchquantum '
                             'simulation only (no qiskit/noise); the score '
                             'is recorded for reporting only -- the '
                             'inherited weights are NOT saved into the '
                             'checkpoint, since this is meant to pair with '
                             'a ckpt.weight_from_scratch: True training '
                             'config')
    parser.add_argument('--run-dir', type=str, default=None,
                        help='trained SuperCircuit run dir to inherit '
                             'weights from; only used with --evaluate')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=str, default=None)
    parser.add_argument('--pdb', action='store_true')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    if args.pdb:
        import pdb
        pdb.set_trace()

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = builder.make_model()
    n_wires = model.q_device.n_wires
    arch_space = model.arch_space

    if args.n_available_wires is not None:
        n_available_wires = args.n_available_wires
        pool_source = '--n-available-wires'
    else:
        n_available_wires, backend_name = infer_backend_n_wires(n_wires)
        pool_source = (f"backend {backend_name}" if backend_name is not None
                       else "model n_wires; the training config names no "
                            "qiskit backend, so pass --n-available-wires to "
                            "address more physical qubits")
    logger.info(f"Layout may address physical qubits 0..{n_available_wires - 1} "
                f"({pool_source})")

    if args.mode == 'manual':
        if args.arch is None or args.layout is None:
            parser.error('--mode manual requires both --arch and --layout')
        arch = parse_int_list(args.arch)
        layout = parse_int_list(args.layout)
    else:
        # get_random_sample_arch() only uses self.arch_space, not
        # self.strategy -- this stand-in dict just needs a 'name' that
        # isn't 'progressive' (the one case __init__ inspects it), since
        # configs here (the train/searched/scratch/.../manual leaf) never
        # sets model.sampler.strategy the way super_* training configs do.
        sampler = ArchSampler(
            model,
            strategy={'name': 'random'},
            n_layers_per_block=configs.model.arch.n_layers_per_block)
        arch = list(sampler.get_random_sample_arch())
        layout = random.sample(range(n_available_wires), n_wires)
        logger.info(f"Randomly drew arch={arch}, layout={layout}")

    validate_arch(arch, arch_space)
    validate_layout(layout, n_wires, n_available_wires)

    solution = {'arch': arch, 'layout': layout}
    score = args.score

    if args.evaluate:
        if not args.run_dir:
            parser.error('--evaluate requires --run-dir pointing at a '
                        'trained SuperCircuit run')
        device = torch.device(
            'cuda' if configs.run.device == 'gpu' and torch.cuda.is_available()
            else 'cpu')
        state_dict = io.load(
            os.path.join(args.run_dir, configs.ckpt.name),
            map_location=device, weights_only=False)
        model.load_state_dict(state_dict['model'], strict=False)
        accuracy, loss = evaluate_solution_tq(model, solution, device, args.seed)
        score = loss
        logger.info(f"Inherited-weight tq evaluation: accuracy={accuracy:.5f} "
                   f"loss={loss:.5f}")

    # Report the model this solution actually selects, since the config
    # decides pure-quantum vs hybrid and a silent mismatch between this
    # script's config and train.py's is otherwise invisible until you notice
    # train.py's "Model Size" is the wrong number.
    params = summarize_params(model, arch)
    logger.info(f"Model this solution selects: "
                f"hybrid mode '{params['hybrid_mode']}', "
                f"{params['total_params']} total params "
                f"({params['quantum_sampled_params']} sampled quantum + "
                f"{params['classical_params']} classical), "
                f"{params['trainable_params']} trainable")
    if params['hybrid_mode'] == 'pure_quantum':
        logger.warning(
            f"'{args.config}' resolves to a PURE-QUANTUM model (no classical "
            f"layers). If you meant to train the hybrid model, pass the "
            f"hybrid_full leaf config instead -- to both this script and "
            f"train.py.")

    out_dir = args.out_dir or f"es_runs/manual_solution_{args.tag}"
    out_path = os.path.join(out_dir, 'checkpoints', 'best_solution.pt')
    if os.path.exists(out_path):
        logger.warning(f"Overwriting existing solution at {out_path}")

    io.save(out_path, {'solution': solution, 'score': score})

    # Mirror the metainfo/ that run_es.py writes, so a manual solution is as
    # self-describing as a searched one. (No tb/: nothing iterates here.)
    io.save(os.path.join(out_dir, 'metainfo', 'configs.yaml'), configs.dict())
    io.save(os.path.join(out_dir, 'metainfo', 'manual_solution.json'), {
        'timestamp': datetime.datetime.now().isoformat(),
        'mode': args.mode,
        'tag': args.tag,
        'config': args.config,
        'seed': args.seed,
        'arch': arch,
        'layout': layout,
        'n_available_wires': n_available_wires,
        'backend_name': getattr(configs.qiskit, 'backend_name', None),
        'score': score,
        'evaluated': bool(args.evaluate),
        'evaluate_run_dir': args.run_dir if args.evaluate else None,
        **params,
    })

    logger.info(f"Saved {args.mode} solution to {out_path}\n"
               f"  arch:   {arch}\n"
               f"  layout: {layout}\n"
               f"  score:  {score}\n"
               f"Train it with:\n"
               f"  python examples/train.py {args.config} "
               f"--ckpt-dir={out_dir}/ debug.seed={args.seed}")


if __name__ == '__main__':
    main()

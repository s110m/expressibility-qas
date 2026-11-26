import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore")

from qiskit_ibm_runtime.fake_provider import (
    FakeQuitoV2,
    FakeBelemV2,
    FakeAthensV2,
)

BACKENDS = {
    "FakeQuitoV2": FakeQuitoV2,
    "FakeBelemV2": FakeBelemV2,
    "FakeAthensV2": FakeAthensV2,
}

import argparse
import pdb
import random

import numpy as np
import torch
import torch.backends.cudnn
import torchquantum as tq
from torchpack.utils import io
from torchpack.utils.config import configs
from torchpack.utils.logging import logger
from torch.utils.tensorboard import SummaryWriter

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.mixed import (MixedVariableSampling, MixedVariableMating,
                              MixedVariableDuplicateElimination)
from pymoo.core.variable import Choice
from pymoo.operators.mutation.rm import ChoiceRandomMutation
from pymoo.optimize import minimize

from core import builder
from core.tools import (Converter, GenomeRepair, MultiObjectiveProblem,
                        ObjectiveEvaluator, StepTimer)
from torchquantum.util import legalize_unitary, get_cared_configs

ALL_OBJECTIVES = ('expressibility', 'depth', 'success_rate', 'swap_count')


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


def resolve_n_available_wires(model):
    """Size of the physical-qubit pool the layout gene may draw from.

    Kept byte-identical to `run_es.py`'s copy (this file already duplicates
    `BACKENDS` rather than importing from that module, which would pull in its
    TensorFlow import as a side effect).

    This is a property of the target device alone. It must NOT be conditioned on
    which objectives happen to need Qiskit: a search can legitimately skip Qiskit
    while still emitting a layout that `eval.py` applies to a real 5-qubit
    backend later.

    Falls back to the circuit width only when no device is named.
    """
    backend_name = getattr(configs.qiskit, 'backend_name', None)

    if backend_name == 'ibmq_qasm_simulator':
        if configs.qiskit.noise_model_name == 'ibmq_16_melbourne':
            return 15
        # Any other noise model on the simulator: no qubit count to read.
        return model.q_device.n_wires

    if backend_name in BACKENDS:
        n = len(BACKENDS[backend_name]().properties().qubits)
        if n < model.q_device.n_wires:
            raise ValueError(
                f'backend {backend_name} has {n} qubits but the circuit needs '
                f'{model.q_device.n_wires}')
        return n

    if backend_name is not None:
        logger.warning(
            f'backend_name {backend_name!r} is not in BACKENDS; falling back to '
            f'a {model.q_device.n_wires}-qubit layout pool. The layout gene will '
            f'not be able to reach any higher physical qubit.')

    return model.q_device.n_wires


class TensorBoardCallback(Callback):
    def __init__(self, writer, objectives):
        super().__init__()
        self.writer = writer
        self.objectives = objectives

    def notify(self, algorithm):
        f = algorithm.pop.get("F")
        for i, objective in enumerate(self.objectives):
            values = f[:, i]
            if objective == 'success_rate':
                values = -values
            self.writer.add_scalar(f'nsga2/{objective}_min', values.min(),
                                   algorithm.n_gen)
            self.writer.add_scalar(f'nsga2/{objective}_mean', values.mean(),
                                   algorithm.n_gen)
        logger.info(f"NSGA-II generation {algorithm.n_gen}: "
                    f"pop objective min/mean logged")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE', help='config file')
    parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    parser.add_argument('--pdb', action='store_true', help='pdb')
    parser.add_argument('--gpu', type=str, help='gpu ids', default=None)
    parser.add_argument('--print-configs', action='store_true',
                        help='print ALL configs')
    args, opts = parser.parse_known_args()

    configs.load(os.path.join(args.run_dir, 'metainfo', 'configs.yaml'))
    # capture the seed the SuperCircuit checkpoint at --run-dir was actually
    # trained with, since the recursive load below re-pulls in the es config
    # chain's own ancestor default.yml files (e.g. mnist/default.yml's
    # debug.seed: 42), which would otherwise silently overwrite it
    trained_seed = configs.debug.seed
    trained_set_seed = configs.debug.set_seed
    configs.load(args.config, recursive=True)
    configs.debug.seed = trained_seed
    configs.debug.set_seed = trained_set_seed
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

    if args.print_configs:
        print_conf = configs
    else:
        print_conf = get_cared_configs(configs, 'es')

    logger.info(f'NSGA-II multi-objective search started: "{args.run_dir}".' +
                '\n' + f'{print_conf}')

    nsga2_cfg = getattr(configs.es, 'nsga2', None)
    objectives = list(getattr(nsga2_cfg, 'objectives', ALL_OBJECTIVES)) \
        if nsga2_cfg is not None else list(ALL_OBJECTIVES)
    assert 2 <= len(objectives) <= len(ALL_OBJECTIVES), \
        "es.nsga2.objectives must contain 2 to " + str(len(ALL_OBJECTIVES)) \
        + " of: " + str(ALL_OBJECTIVES)
    assert all(o in ALL_OBJECTIVES for o in objectives)

    pop_size = getattr(nsga2_cfg, 'pop_size', 40) if nsga2_cfg is not None else 40
    n_gen = getattr(nsga2_cfg, 'n_gen', 40) if nsga2_cfg is not None else 40
    mutation_prob = getattr(nsga2_cfg, 'mutation_prob', 0.4) \
        if nsga2_cfg is not None else 0.4
    gene_mask = getattr(configs.es, 'gene_mask', None)

    expr_cfg = getattr(nsga2_cfg, 'expressibility', None) \
        if nsga2_cfg is not None else None
    expr_n_pairs = getattr(expr_cfg, 'n_pairs', 100) if expr_cfg is not None else 100
    expr_nbins = getattr(expr_cfg, 'nbins', 75) if expr_cfg is not None else 75
    expr_seed = getattr(expr_cfg, 'seed', None) if expr_cfg is not None else None

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

    if configs.legalization.legalize:
        legalize_unitary(model)
    model.to(device)
    model.eval()

    es_dir = 'es_runs/' + args.config.replace('/', '.').replace(
        'examples.', '').replace('.yml', '').replace('configs.', '')

    if configs.debug.set_seed:
        es_dir += f"_seed{configs.debug.seed}"

    io.save(os.path.join(es_dir, 'metainfo/configs.yaml'), configs.dict())

    writer = SummaryWriter(os.path.normpath(os.path.join(es_dir, 'tb')))

    with StepTimer('nsga2_search', out_dir=os.path.join(es_dir, 'metainfo'),
                   extra={'config': args.config, 'run_dir': args.run_dir,
                          'es_dir': es_dir, 'objectives': objectives}):
        needs_qiskit = ('depth' in objectives) or ('success_rate' in objectives) \
            or ('swap_count' in objectives)

        # The layout gene's physical-qubit pool is a property of the target
        # device, not of whether any objective happens to need Qiskit. Keeping the
        # two coupled was latently wrong here: an objective set without
        # depth/success_rate/swap_count (e.g. [expressibility, n_params]) would
        # silently collapse the pool to the circuit width, so on 5-qubit
        # Athens/Quito/Belem the search could not reach a layout touching qubit 4
        # -- 24 candidate layouts instead of P(5,4)=120. Every M3 run to date
        # carries success_rate and so took the wide path; this makes that
        # independent of the objective set. Same fix as run_es.py's
        # resolve_n_available_wires.
        n_available_wires = resolve_n_available_wires(model)

        qiskit_processor = None
        if needs_qiskit:
            backend = BACKENDS[configs.qiskit.backend_name]()
            qiskit_processor = builder.make_qiskit_processor()
            qiskit_processor.properties = backend.properties()
            model.set_qiskit_processor(qiskit_processor)

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f'Model Size: {total_params}')

        converter = Converter(model.q_device.n_wires, n_available_wires,
                             model.arch_space)

        objective_evaluator = ObjectiveEvaluator(
            model, objectives, qiskit_processor=qiskit_processor,
            expr_n_pairs=expr_n_pairs, expr_nbins=expr_nbins, expr_seed=expr_seed,
        )

        problem = MultiObjectiveProblem(converter, objective_evaluator,
                                        objectives, gene_mask=gene_mask)

        repair = GenomeRepair(converter.n_wires, converter.n_available_wires)
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=MixedVariableSampling(),
            mating=MixedVariableMating(
                mutation={Choice: ChoiceRandomMutation(prob_var=mutation_prob)},
                eliminate_duplicates=MixedVariableDuplicateElimination(),
                repair=repair,
            ),
            eliminate_duplicates=MixedVariableDuplicateElimination(),
            repair=repair,
        )

        logger.info(f"Start NSGA-II search over objectives: {objectives}")
        res = minimize(
            problem, algorithm, ('n_gen', n_gen),
            seed=configs.debug.seed,
            callback=TensorBoardCallback(writer, objectives),
            verbose=True,
        )

        pareto_front = []
        for x, f in zip(res.X, res.F):
            gene = [int(x[f'x{i}']) for i in range(problem.n_var)]
            solution = converter.gene2solution(gene)
            obj_values = {}
            for i, objective in enumerate(objectives):
                value = f[i]
                if objective == 'success_rate':
                    value = -value
                obj_values[objective] = float(value)
            pareto_front.append({'solution': solution, 'objectives': obj_values})
            logger.info(f"Pareto solution: {solution}, objectives: {obj_values}")

        io.save(os.path.join(es_dir, 'checkpoints/pareto_front.pt'),
               {'pareto_front': pareto_front, 'objectives': objectives})
        writer.add_text('nsga2/pareto_front', str(pareto_front), n_gen)

        logger.info(f"NSGA-II search finished. "
                    f"{len(pareto_front)} non-dominated solutions saved to "
                    f"{es_dir}/checkpoints/pareto_front.pt")


if __name__ == '__main__':
    main()

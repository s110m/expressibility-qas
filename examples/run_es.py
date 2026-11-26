import random
import numpy as np


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# from qiskit_ibm_runtime.fake_provider import FakeQuitoV2

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
import copy
import os
import pdb
import torch
import torch.backends.cudnn
import tqdm
import torch.nn.functional as F
import torchquantum as tq

from torchpack.utils import io
from torchpack.utils.config import configs
from torchpack.utils.logging import logger
from core import builder
from torchquantum.util import (legalize_unitary, build_module_op_list,
                                get_cared_configs, get_success_rate,
                                get_provider)
from torchquantum.plugin import tq2qiskit, tq2qiskit_parameterized
from torchquantum.algorithm.quantumnas.super_utils import compute_expressibility_tq
# from qiskit import IBMQ
from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService()
from core.tools import EvolutionEngine, StepTimer
from torch.utils.tensorboard import SummaryWriter


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

    This is a property of the target device alone. It must NOT be conditioned on
    `qiskit.use_qiskit` / `es.est_success_rate`: those say whether Qiskit runs
    circuits during scoring, and a search can legitimately skip Qiskit (as
    `score_mode: expressibility` does, for an ~8x speedup) while still emitting a
    layout that `eval.py` applies to a real 5-qubit backend later.

    Falls back to the circuit width only when no device is named -- the genuinely
    device-free case upstream's `else` branch was written for.
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


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Evaluator(object):
    def __init__(self):
        self.solution_lib = {}
        self.expr_model = None

    def _get_expr_model(self, model):
        """Dedicated CPU deep-copy used only for expressibility scoring.

        compute_expressibility_tq randomizes q_layer's parameters in place
        (it needs to sample many random-parameter states), so it must never
        be pointed at the live `model` -- that instance carries trained
        weights that are reused for the rest of the ES run and written to
        checkpoints. Mirrors the same deep-copy pattern already used by
        ArchSampler.build_expressibility_curriculum in super_utils.py.
        """
        if self.expr_model is None:
            self.expr_model = copy.deepcopy(model)
            self.expr_model.eval()
            self.expr_model.to(torch.device('cpu'))
        return self.expr_model

    def evaluate_all(self, model, dataflow, solutions, writer=None,
                     iter_n=None, population_size=None):
        scores = []

        best_solution = None
        best_solution_accuracy = 0
        best_solution_loss = 0
        best_solution_success_rate = 0
        best_solution_expressibility = 0
        best_solution_score = 999999

        use_expressibility = configs.es.score_mode == 'expressibility'

        for i, solution in tqdm.tqdm(enumerate(solutions)):
            fingerprint = solution.copy()
            arch = solution['arch']
            fingerprint['arch'] = arch[:arch[-1] *
                                       configs.model.arch.n_layers_per_block]
            fingerprint['arch'].append(arch[-1])
            fingerprint = str(fingerprint)
            if fingerprint in self.solution_lib.keys():
                """circuit has been simulated before"""
                logger.info(f"loaded from lib")
                loss = self.solution_lib[fingerprint]['loss']
                accuracy = self.solution_lib[fingerprint]['accuracy']
                success_rate = self.solution_lib[fingerprint][
                    'success_rate']
                expressibility = self.solution_lib[fingerprint][
                    'expressibility']
            elif use_expressibility:
                model.set_sample_arch(solution['arch'])
                expr_cfg = getattr(configs.es, 'expressibility', None)
                expr_n_pairs = getattr(expr_cfg, 'n_pairs', 100) \
                    if expr_cfg is not None else 100
                expr_nbins = getattr(expr_cfg, 'nbins', 75) \
                    if expr_cfg is not None else 75
                expr_seed = getattr(expr_cfg, 'seed', None) \
                    if expr_cfg is not None else None

                expr_model = self._get_expr_model(model)
                expressibility = compute_expressibility_tq(
                    expr_model, solution['arch'],
                    n_pairs=expr_n_pairs, nbins=expr_nbins, seed=expr_seed,
                )
                accuracy = -1
                loss = -1

                if configs.es.est_success_rate:
                    circ_fixed = tq2qiskit(model.q_device, model.q_layer)
                    if configs.dataset.name == 'vqe':
                        circ = circ_fixed
                    else:
                        circ_parameterized, params = tq2qiskit_parameterized(
                            model.q_device, model.encoder.func_list)
                        circ = circ_parameterized + circ_fixed

                    transpiled_circ = model.qiskit_processor.transpile(circ)

                    success_rate = get_success_rate(
                        model.qiskit_processor.properties,
                        transpiled_circ)
                else:
                    success_rate = 1

                self.solution_lib[fingerprint] = {
                    'loss': loss,
                    'accuracy': accuracy,
                    'success_rate': success_rate,
                    'expressibility': expressibility,
                }
            else:
                if model.qiskit_processor is not None:
                    model.qiskit_processor.set_layout(solution['layout'])
                model.set_sample_arch(solution['arch'])
                with torch.no_grad():
                    target_all = None
                    output_all = None
                    for feed_dict in dataflow:
                        if configs.run.device == 'gpu':
                            inputs = feed_dict[
                                configs.dataset.input_name].cuda(
                                non_blocking=True)
                            targets = feed_dict[
                                configs.dataset.target_name].cuda(
                                non_blocking=True)
                        else:
                            inputs = feed_dict[configs.dataset.input_name]
                            targets = feed_dict[configs.dataset.target_name]

                        outputs = model(
                            inputs, use_qiskit=configs.qiskit.use_qiskit)

                        if target_all is None:
                            target_all = targets
                            output_all = outputs
                        else:
                            target_all = torch.cat([target_all, targets],
                                                   dim=0)
                            output_all = torch.cat([output_all, outputs],
                                                   dim=0)

                if configs.dataset.name == 'vqe':
                    loss = output_all[0].item()
                    accuracy = -1
                else:
                    k = 1
                    _, indices = output_all.topk(k, dim=1)
                    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
                    size = target_all.shape[0]
                    corrects = masks.sum().item()
                    accuracy = corrects / size
                    loss = F.nll_loss(output_all, target_all).item()

                expressibility = -1

                if configs.es.est_success_rate:
                    circ_fixed = tq2qiskit(model.q_device, model.q_layer)
                    if configs.dataset.name == 'vqe':
                        circ = circ_fixed
                    else:
                        circ_parameterized, params = tq2qiskit_parameterized(
                            model.q_device, model.encoder.func_list)
                        circ = circ_parameterized + circ_fixed

                    transpiled_circ = model.qiskit_processor.transpile(circ)

                    success_rate = get_success_rate(
                        model.qiskit_processor.properties,
                        transpiled_circ)
                else:
                    success_rate = 1

                self.solution_lib[fingerprint] = {
                    'loss': loss,
                    'accuracy': accuracy,
                    'success_rate': success_rate,
                    'expressibility': expressibility,
                }
            if configs.es.score_mode == 'loss_succ':
                score = loss / success_rate
            elif configs.es.score_mode == 'vqe_loss_succ':
                score = loss * success_rate
            elif configs.es.score_mode == 'acc_succ':
                score = - accuracy * success_rate
            elif configs.es.score_mode == 'expressibility':
                score = expressibility
            else:
                raise NotImplementedError

            scores.append(score)
            logger.info(f"Current solution: {solution}\n"
                        f"Accuracy: {accuracy:.5f}, Loss: {loss:.5f}, "
                        f"Success Rate: {success_rate: .5f}, "
                        f"Expressibility: {expressibility:.5f}, "
                        f"Score: {score:.5f}")

            if score < best_solution_score:
                best_solution = solution
                best_solution_accuracy = accuracy
                best_solution_success_rate = success_rate
                best_solution_loss = loss
                best_solution_expressibility = expressibility
                best_solution_score = score

            logger.info(f"Best of iteration: "
                        f"Solution: {best_solution}\n"
                        f"Accuracy: {best_solution_accuracy:.5f}, "
                        f"Loss: {best_solution_loss:.5f}, "
                        f"Success Rate: {best_solution_success_rate: .5f}, "
                        f"Expressibility: {best_solution_expressibility:.5f}, "
                        f"Score: {best_solution_score:.5f}")

            if population_size is not None and writer is not None and \
                    population_size is not None:
                writer.add_scalar('es/accuracy', accuracy,
                                  iter_n * population_size + i)
                writer.add_scalar('es/loss', loss,
                                  iter_n * population_size + i)
                writer.add_scalar('es/success_rate', success_rate,
                                  iter_n * population_size + i)
                writer.add_scalar('es/expressibility', expressibility,
                                  iter_n * population_size + i)
                writer.add_scalar('es/score', score,
                                  iter_n * population_size + i)

        return scores, best_solution_accuracy, best_solution_loss, \
            best_solution_success_rate, best_solution_expressibility, \
            best_solution_score


def main() -> None:
    # torch.backends.cudnn.benchmark = True
    # set_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE', help='config file')
    parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    parser.add_argument('--pdb', action='store_true', help='pdb')
    parser.add_argument('--gpu', type=str, help='gpu ids', default=None)
    parser.add_argument('--jobs', type=int, default=None,
                        help='max parallel job on qiskit')
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

    if args.jobs is not None:
        configs.qiskit.max_jobs = args.jobs

    # if use qiskit, then not need to estimate success rate
    assert not (configs.es.est_success_rate and configs.qiskit.use_qiskit)

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

    logger.info(f'Evolutionary Search started: "{args.run_dir}".' + '\n' +
                f'{print_conf}')

    if configs.qiskit.use_qiskit:
        if configs.run.bsz == 'qiskit_max':
            # backend = service.backend(configs.qiskit.backend_name)
            # backend = FakeQuitoV2()
            backend = BACKENDS[configs.qiskit.backend_name]()
            configs.run.bsz = backend.max_circuits

    dataset = builder.make_dataset()
    sampler = torch.utils.data.SequentialSampler(dataset[
                                                     configs.dataset.split])
    
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
            # random layer, need to restore the architecture
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

    writer = SummaryWriter(os.path.normpath(os.path.join(es_dir,
                                                         'tb')))

    with StepTimer('evolutionary_search',
                   out_dir=os.path.join(es_dir, 'metainfo'),
                   extra={'config': args.config, 'run_dir': args.run_dir,
                          'es_dir': es_dir}):
        # How many physical qubits the layout gene may draw from is a property of
        # the *device*, not of whether this particular run happens to score with
        # Qiskit in the loop. Upstream coupled the two, which was harmless there
        # because `use_qiskit=False` meant noiseless simulation where the layout is
        # meaningless. It is not harmless for `score_mode: expressibility`, which
        # turns Qiskit off for speed but still writes a layout into
        # `best_solution.pt` that `eval.py` later applies: the pool silently
        # collapsed to the circuit width, so on 5-qubit Athens/Quito/Belem the
        # search could never propose a layout touching qubit 4 -- 24 candidate
        # layouts instead of P(5,4)=120.
        n_available_wires = resolve_n_available_wires(model)

        if configs.qiskit.use_qiskit or configs.es.est_success_rate:
            # backend = service.backend(configs.qiskit.backend_name)
            # backend = FakeQuitoV2()
            qiskit_processor = builder.make_qiskit_processor()
            model.set_qiskit_processor(qiskit_processor)

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f'Model Size: {total_params}')

        es_engine = EvolutionEngine(
            population_size=configs.es.population_size,
            parent_size=configs.es.parent_size,
            mutation_size=configs.es.mutation_size,
            mutation_prob=configs.es.mutation_prob,
            crossover_size=configs.es.crossover_size,
            n_wires=model.q_device.n_wires,
            n_available_wires=n_available_wires,
            arch_space=model.arch_space,
            gene_mask=configs.es.gene_mask,
            random_search=configs.es.random_search,
            legalize_layout=getattr(configs.es, 'legalize_layout', False)
        )

        evaluator = Evaluator()

        logger.info(f"Start Evolution Search")
        for k in range(configs.es.n_iterations):
            logger.info(f"ES iteration {k}:")
            solutions = es_engine.ask()
            scores, best_solution_accuracy, best_solution_loss, \
                best_solution_success_rate, best_solution_expressibility, \
                best_solution_score \
                = evaluator.evaluate_all(model, dataflow, solutions, writer, k,
                                         configs.es.population_size)
            es_engine.tell(scores)
            logger.info(f"Best solution: {es_engine.best_solution}")
            logger.info(f"Best score: {es_engine.best_score}")

            assert best_solution_score == es_engine.best_score
            writer.add_text('es/best_solution_arch',
                            str(es_engine.best_solution), k)
            writer.add_scalar('es/best_solution_accuracy',
                              best_solution_accuracy, k)
            writer.add_scalar('es/best_solution_loss', best_solution_loss, k)
            writer.add_scalar('es/best_solution_success_rate',
                              best_solution_success_rate, k)
            writer.add_scalar('es/best_solution_expressibility',
                              best_solution_expressibility, k)
            writer.add_scalar('es/best_solution_score', es_engine.best_score, k)

            # store the model and solution after every iteration
            state_dict = dict()
            if not configs.qiskit.use_real_qc:
                state_dict['model_arch'] = model
            state_dict['model'] = model.state_dict()
            state_dict['solution'] = es_engine.best_solution
            state_dict['score'] = es_engine.best_score
            if not configs.dataset.name == 'vqe':
                state_dict['encoder_func_list'] = model.encoder.func_list
            state_dict['q_layer_op_list'] = build_module_op_list(model.q_layer)
            io.save(os.path.join(es_dir, 'checkpoints/best_solution.pt'),
                    state_dict)

        logger.info(f"\n Best solution evaluation on tq:")
        # eval the best solution
        evaluator.evaluate_all(model, dataflow, [es_engine.best_solution])

    # eval with the noise model
    # if configs.es.eval.use_noise_model:
    #     logger.info(f"\n Best solution evaluation with noise model "
    #                 f"of {configs.qiskit.noise_model_name}:")
    #     configs.qiskit.use_qiskit = True
    #     evaluate_all(model, dataflow, [es_engine.best_solution])
    #
    # # eval on real QC
    # if configs.es.eval.use_real_qc:
    #     logger.info(f"\n Best solution evaluation on real "
    #                 f"QC {configs.qiskit.backend_name}:")
    #
    #     # need reset some parameters
    #     configs.qiskit.use_qiskit = True
    #     model.qiskit_processor.use_real_qc = True
    #     model.qiskit_processor.noise_model_name = None
    #     model.qiskit_processor.qiskit_init()
    #
    #     # if configs.es.eval.bsz == 'qiskit_max':
    #     #     configs.run.bsz = \
    #     #     model.qiskit_processor.backend.configuration().max_experiments
    #     # else:
    #     configs.run.bsz = configs.es.eval.bsz
    #
    #     configs.dataset.n_test_samples = configs.es.eval.n_test_samples
    #     dataset = builder.make_dataset()
    #     sampler = torch.utils.data.SequentialSampler(dataset['test'])
    #     dataflow = torch.utils.data.DataLoader(
    #         dataset['test'],
    #         sampler=sampler,
    #         batch_size=configs.run.bsz,
    #         num_workers=configs.run.workers_per_gpu,
    #         pin_memory=True)
    #
    #     evaluate_all(model, dataflow, [es_engine.best_solution])


if __name__ == '__main__':
    main()

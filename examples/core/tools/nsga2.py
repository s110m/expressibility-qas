import copy

import torch
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
from pymoo.core.variable import Choice
from torchpack.utils.config import configs
from torchpack.utils.logging import logger

from core.tools.es import legalize_layout_gene
from torchquantum.algorithm.quantumnas.super_utils import compute_expressibility_tq
from torchquantum.plugin import tq2qiskit, tq2qiskit_parameterized
from torchquantum.util import get_success_rate

__all__ = ['GenomeRepair', 'ObjectiveEvaluator', 'MultiObjectiveProblem',
           'build_problem_vars', 'gene_to_var_dict', 'var_dict_to_gene']


def gene_to_var_dict(gene):
    return {f'x{i}': v for i, v in enumerate(gene)}


def var_dict_to_gene(x, n_var):
    # cast off numpy scalar types from Choice.sample() so downstream code
    # (Converter, model.set_sample_arch, qiskit) sees plain python ints, same
    # as EvolutionEngine's random.choices()-based genome.
    return [int(x[f'x{i}']) for i in range(n_var)]


def build_problem_vars(gene_choice, gene_mask=None):
    """Same per-position choice sets as Converter.get_gene_choice(); a
    gene_mask entry != -1 pins that position to a single fixed value by
    giving it a Choice with only one option."""
    problem_vars = {}
    for i, choices in enumerate(gene_choice):
        if gene_mask is not None and gene_mask[i] != -1:
            options = [gene_mask[i]]
        else:
            options = list(choices)
        problem_vars[f'x{i}'] = Choice(options=options)
    return problem_vars


class GenomeRepair(Repair):
    """Enforces that the layout portion of every individual assigns distinct
    physical qubits, mirroring EvolutionEngine.satisfy_constraints's
    legalize_layout=True branch. Applied automatically by pymoo after both
    sampling and mating when passed as the algorithm's/mating's `repair=`."""

    def __init__(self, n_wires, n_available_wires=None):
        super().__init__()
        self.n_wires = n_wires
        # Device qubit count, so the repair draws replacements from the whole
        # device rather than from the circuit's own wire count.
        self.n_available_wires = n_available_wires

    def _do(self, problem, X, **kwargs):
        for ind in X:
            gene = var_dict_to_gene(ind, problem.n_var)
            legal_gene = legalize_layout_gene(gene, self.n_wires,
                                              self.n_available_wires)
            for i in range(self.n_wires):
                ind[f'x{i}'] = legal_gene[i]
        return X


def _count_two_qubit_gates(circ):
    return sum(1 for instr in circ.data if len(instr.qubits) == 2)


class ObjectiveEvaluator:
    """Computes a configurable subset of {expressibility, depth,
    success_rate, swap_count} for a solution, reusing the exact same
    circuit-construction and metric functions run_es.py's Evaluator uses for
    est_success_rate/expressibility. Caches by (trimmed-arch, layout)
    fingerprint, same trimming convention as run_es.py's Evaluator.evaluate_all.

    swap_count is NOT a literal count of 'swap'-named gate instructions: none
    of the fake backends used here have 'swap' in their basis gate set (e.g.
    FakeAthensV2's basis is ['delay','cx','rz','id','reset','x','sx','measure']),
    so the transpiler's routing pass always decomposes each inserted SWAP into
    ~3 CX gates before the circuit reaches its final form -- a literal 'swap'
    op never survives to be counted. Instead this counts the two-qubit-gate
    delta between the untranspiled and transpiled circuit (transpiled 2Q count
    minus original 2Q count), which is proportional to (roughly 3x) the actual
    number of SWAPs the router inserted for this specific layout -- a genuine,
    mapping-only signal, decoupled from both the architecture's own gate count
    and the backend's calibration data (unlike depth/success_rate, which both
    entangle mapping quality with other things)."""

    def __init__(self, model, objectives, qiskit_processor=None,
                 expr_n_pairs=100, expr_nbins=75, expr_seed=None):
        self.model = model
        self.objectives = objectives
        self.qiskit_processor = qiskit_processor
        self.expr_n_pairs = expr_n_pairs
        self.expr_nbins = expr_nbins
        self.expr_seed = expr_seed

        self.expr_model = None
        self.solution_lib = {}

    def _get_expr_model(self):
        # compute_expressibility_tq randomizes q_layer's parameters in place;
        # must never touch the live model whose trained weights matter for
        # depth/success_rate's circuit construction.
        if self.expr_model is None:
            self.expr_model = copy.deepcopy(self.model)
            self.expr_model.eval()
            self.expr_model.to(torch.device('cpu'))
        return self.expr_model

    def _fingerprint(self, solution):
        fingerprint = {'layout': list(solution['layout'])}
        arch = list(solution['arch'])
        n_layers_per_block = configs.model.arch.n_layers_per_block
        trimmed = arch[:arch[-1] * n_layers_per_block]
        trimmed.append(arch[-1])
        fingerprint['arch'] = trimmed
        return str(fingerprint)

    def _build_circuit(self, solution):
        self.model.set_sample_arch(solution['arch'])
        circ_fixed = tq2qiskit(self.model.q_device, self.model.q_layer)
        if configs.dataset.name == 'vqe':
            return circ_fixed
        circ_parameterized, _ = tq2qiskit_parameterized(
            self.model.q_device, self.model.encoder.func_list)
        return circ_parameterized.compose(circ_fixed)

    def compute(self, solution):
        fingerprint = self._fingerprint(solution)
        if fingerprint in self.solution_lib:
            return self.solution_lib[fingerprint]

        result = {}

        if 'expressibility' in self.objectives:
            expr_model = self._get_expr_model()
            result['expressibility'] = compute_expressibility_tq(
                expr_model, solution['arch'],
                n_pairs=self.expr_n_pairs, nbins=self.expr_nbins,
                seed=self.expr_seed,
            )

        needs_transpile = ('depth' in self.objectives
                          or 'success_rate' in self.objectives
                          or 'swap_count' in self.objectives)
        if needs_transpile:
            circ = self._build_circuit(solution)
            self.qiskit_processor.set_layout(list(solution['layout']))
            transpiled_circ = self.qiskit_processor.transpile(circ)

            if 'depth' in self.objectives:
                result['depth'] = transpiled_circ.depth()

            if 'success_rate' in self.objectives:
                result['success_rate'] = get_success_rate(
                    self.qiskit_processor.properties, transpiled_circ)

            if 'swap_count' in self.objectives:
                result['swap_count'] = (_count_two_qubit_gates(transpiled_circ)
                                        - _count_two_qubit_gates(circ))

        self.solution_lib[fingerprint] = result
        logger.info(f"Solution: {solution}, objectives: {result}")
        return result


class MultiObjectiveProblem(ElementwiseProblem):
    def __init__(self, converter, objective_evaluator, objectives, gene_mask=None):
        gene_choice = converter.get_gene_choice()
        problem_vars = build_problem_vars(gene_choice, gene_mask)
        super().__init__(vars=problem_vars, n_obj=len(objectives))
        self.converter = converter
        self.objective_evaluator = objective_evaluator
        self.objectives = objectives

    def _evaluate(self, x, out, *args, **kwargs):
        gene = var_dict_to_gene(x, self.n_var)
        solution = self.converter.gene2solution(gene)
        metrics = self.objective_evaluator.compute(solution)

        f = []
        for objective in self.objectives:
            value = metrics[objective]
            if objective == 'success_rate':
                # pymoo minimizes by convention; higher success_rate is better.
                value = -value
            f.append(value)

        out['F'] = f

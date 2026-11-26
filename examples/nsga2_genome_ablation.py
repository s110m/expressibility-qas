"""Genome ablation of the joint NSGA-II search
(Access revision, Reviewer 1 #3 / Reviewer 2 #3).

The paper attributes the joint search's middling layouts to the genome: an
improving mapping move has to change a layout gene while leaving all 17
architecture genes intact, which at the paper's p_m = 0.4 happens ~0.24 times
per run (Eq. (mutation)). This driver reruns the paper's two-objective search
(expressibility, success rate; population 40, 40 generations) under different
variation settings and *counts* those moves instead of predicting them.

Settings (--setting):
  p0.4        the paper's operator, per-gene resampling at 0.4 (re-run, so that
              every comparison is within one code version: the repair operator
              has changed since the original searches, see legalize_layout_gene)
  p0.2, p0.1, p0.05
              the same at lower per-gene rates
  split       per-group rates: architecture genes 0.1, layout genes 0.5
  staged      alternating generations. Odd generations are architecture
              phases (uniform crossover on all genes, mutation of architecture
              genes only, p = 0.4). Even generations are mapping phases (no
              crossover, mutation of layout genes only, p = 0.5), so every
              offspring is its parent's architecture on a new layout.
  tournament  p0.4 with binary tournament parent selection under the crowded
              comparison operator. run_nsga2.py passes its own MixedVariableMating,
              which bypasses NSGA2's tournament and selects parents uniformly at
              random, so every other setting (and the paper's runs) select
              randomly and leave all selection pressure to survival.

Per generation the callback records every evaluated offspring and whether the
combined parent+offspring pool that survival sorts contains another member
with the same expressed architecture on a different layout -- the event in
which the success rate can select between two mappings of one architecture
(counted per offspring), and whether that offspring strictly dominated such a
member (an improving mapping move).

At the end the deployed solution (argmin Expr over the first front, the
paper's rule) has all 120 layouts of its architecture scored by the success
rate, which gives the analytic rank of the layout the search returned. Output:
genome_ablation/<setting>/<backend>_s<seed>.json, plus es_runs-style
checkpoints/best_solution.pt for retraining with train.py.

Usage (repo root):
    python examples/nsga2_genome_ablation.py <nsga2 leaf config> \
        --run-dir <supercircuit run dir> --setting p0.1 debug.seed=100
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import random
import sqlite3
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import re
import ast
import time
from itertools import permutations

import numpy as np
import torch
import torchquantum as tq
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.crossover import Crossover
from pymoo.core.mixed import (MixedVariableDuplicateElimination,
                              MixedVariableMating, MixedVariableSampling)
from pymoo.core.mutation import Mutation
from pymoo.core.variable import Choice
from pymoo.operators.crossover.ux import UX
from pymoo.optimize import minimize
from torchpack.utils.config import configs
from torchpack.utils.logging import logger

from core import builder
from core.tools import Converter
from core.tools.nsga2 import (GenomeRepair, MultiObjectiveProblem,
                              ObjectiveEvaluator)
from pymoo.algorithms.moo.nsga2 import binary_tournament
from pymoo.core.population import Population
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.operators.selection.tournament import TournamentSelection
from run_nsga2 import BACKENDS, resolve_n_available_wires, set_seed

N_LAYOUT = 4
SETTINGS = {
    'p0.4': dict(arch=0.4, layout=0.4),
    'p0.2': dict(arch=0.2, layout=0.2),
    'p0.1': dict(arch=0.1, layout=0.1),
    'p0.05': dict(arch=0.05, layout=0.05),
    'split': dict(arch=0.1, layout=0.5),
    'staged': dict(arch=0.4, layout=0.5),
    'tournament': dict(arch=0.4, layout=0.4),
}


class Phase:
    """Shared by the operators and the callback: which generation is next."""

    def __init__(self, staged):
        self.staged = staged
        self.gen = 1  # generation the next mating produces offspring for

    @property
    def name(self):
        if not self.staged:
            return 'joint'
        return 'arch' if self.gen % 2 == 1 else 'layout'


def _is_layout(var_name):
    return int(var_name[1:]) < N_LAYOUT


class GroupedChoiceMutation(Mutation):
    """Per-gene random resampling with one rate per gene group. pymoo's
    MixedVariableMating calls the mutation once per Choice variable with a
    one-variable problem, so the variable's name says which group it is in."""

    def __init__(self, p_arch, p_layout, phase):
        super().__init__(prob=1.0)
        self.p_arch, self.p_layout, self.phase = p_arch, p_layout, phase

    def _do(self, problem, X, random_state=None, **kwargs):
        X = X.astype(object)
        (name, var), = problem.vars.items()
        layout = _is_layout(name)
        p = self.p_layout if layout else self.p_arch
        if self.phase.name == 'arch' and layout:
            p = 0.0
        if self.phase.name == 'layout' and not layout:
            p = 0.0
        mut = np.where(random_state.random(len(X)) < p)[0]
        X[mut, 0] = var.sample(len(mut), random_state=random_state)
        return X


class PhasedUX(Crossover):
    """pymoo's default uniform crossover, switched off in mapping phases so that
    every offspring keeps one parent's architecture exactly."""

    def __init__(self, phase):
        super().__init__(2, 2)
        self.ux, self.phase = UX(), phase

    def do(self, problem, pop, parents=None, **kwargs):
        if self.phase.name == 'layout':
            if parents is not None:
                pop = [pop[m] for m in parents]
            # same (offspring, mating) order as Crossover.do: offspring k of
            # mating j is a copy of parent k
            X = np.swapaxes(np.array([[p.get('X') for p in m] for m in pop],
                                     dtype=object), 0, 1)
            return Population.new('X', X.reshape(-1, problem.n_var))
        return self.ux.do(problem, pop, parents=parents, **kwargs)


class SharedExprCache:
    """Expressibility depends only on the expressed architecture and the pinned
    draw seed, so its value can be shared across every run of the ablation.
    sqlite so that concurrent runs can share it safely."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.db = sqlite3.connect(path, timeout=120)
        self.db.execute('CREATE TABLE IF NOT EXISTS expr (k TEXT PRIMARY KEY, v REAL)')
        self.db.commit()

    def get(self, key):
        row = self.db.execute('SELECT v FROM expr WHERE k=?', (key,)).fetchone()
        return None if row is None else row[0]

    def put(self, key, value):
        self.db.execute('INSERT OR IGNORE INTO expr VALUES (?, ?)', (key, value))
        self.db.commit()


class CachedEvaluator(ObjectiveEvaluator):
    def __init__(self, *a, expr_cache=None, expr_key_extra='', **k):
        super().__init__(*a, **k)
        self.expr_cache, self.expr_key_extra = expr_cache, expr_key_extra
        self.n_expr_computed = 0

    def compute(self, solution):
        fp = self._fingerprint(solution)
        if fp in self.solution_lib:
            return self.solution_lib[fp]
        arch_key = self.expr_key_extra + str(trimmed_arch(solution['arch']))
        cached = self.expr_cache.get(arch_key)
        objectives = self.objectives
        if cached is not None:
            self.objectives = [o for o in objectives if o != 'expressibility']
        try:
            result = super().compute(solution)
        finally:
            self.objectives = objectives
        if cached is not None:
            result['expressibility'] = cached
        else:
            self.n_expr_computed += 1
            self.expr_cache.put(arch_key, float(result['expressibility']))
        self.solution_lib[fp] = result
        return result


def trimmed_arch(arch):
    n = configs.model.arch.n_layers_per_block
    return list(arch[:arch[-1] * n]) + [arch[-1]]


class AblationLog(Callback):
    def __init__(self, phase, n_var):
        super().__init__()
        self.phase, self.n_var = phase, n_var
        self.prev_pop = None
        self.generations = []
        self.partial_path, self.converter, self.t0 = None, None, time.time()

    @staticmethod
    def _key(ind, n_var):
        g = [int(ind.X[f'x{i}']) for i in range(n_var)]
        return tuple(trimmed_arch(g[N_LAYOUT:])), tuple(g[:N_LAYOUT]), ind.F.copy()

    def notify(self, algorithm):
        off = algorithm.off
        rec = {'gen': int(algorithm.n_gen), 'phase': self.phase.name,
               'n_offspring': int(len(off)), 'same_arch_events': 0,
               'improving_moves': 0}
        if self.prev_pop is not None:
            pool = [self._key(i, self.n_var) for i in self.prev_pop] + \
                   [self._key(i, self.n_var) for i in off]
            n_prev = len(self.prev_pop)
            for j in range(n_prev, len(pool)):
                arch, lay, f = pool[j]
                partners = [pool[i] for i in range(len(pool))
                            if i != j and pool[i][0] == arch and pool[i][1] != lay]
                if partners:
                    rec['same_arch_events'] += 1
                    # same architecture => same Expr; lower -S_hat dominates
                    if any(f[1] < p[2][1] for p in partners):
                        rec['improving_moves'] += 1
        self.generations.append(rec)
        self.prev_pop = algorithm.pop.copy()
        self.phase.gen = int(algorithm.n_gen) + 1
        if self.partial_path is not None:
            try:
                self._dump_partial(algorithm)
            except Exception as e:  # a snapshot must never stop the search
                print('partial snapshot failed:', e)

    def _dump_partial(self, algorithm):
        F = np.array([ind.F for ind in algorithm.pop])
        nd = NonDominatedSorting().do(F, only_non_dominated_front=True)
        front = []
        for i in nd:
            ind = algorithm.pop[int(i)]
            gene = [int(ind.X[f'x{k}']) for k in range(self.n_var)]
            front.append({'solution': self.converter.gene2solution(gene),
                          'expressibility': float(ind.F[0]),
                          'success_rate': float(-ind.F[1])})
        gens = self.generations
        snap = {'partial': True, 'gen': int(algorithm.n_gen),
                'elapsed_seconds': time.time() - self.t0,
                'same_arch_events': int(sum(g['same_arch_events'] for g in gens)),
                'improving_moves': int(sum(g['improving_moves'] for g in gens)),
                'n_offspring_total': int(sum(g['n_offspring'] for g in gens[1:])),
                'generations': gens, 'front': front,
                'deployed': min(front, key=lambda r: r['expressibility'])}
        tmp = self.partial_path + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(snap, fh, indent=1)
        os.replace(tmp, self.partial_path)


def front_from_log(path, evaluator):
    """Non-dominated set of every evaluation logged by an unfinished search.

    The minimum-expressibility point is always non-dominated and NSGA-II is
    elitist, so the deployed solution recovered here is the one the search
    would have deployed had it stopped at the last logged evaluation."""
    pat = re.compile(r"Solution: (\{.*?\}), objectives: (\{.*?\})")
    recs = []
    for line in open(path, encoding='utf-8', errors='replace'):
        m = pat.search(line)
        if m:
            recs.append((ast.literal_eval(m.group(1)), ast.literal_eval(m.group(2))))
    # a finished log ends with the post-search layout enumeration of the
    # deployed architecture; it is not part of the search, so drop it
    tail = 0
    while (tail < len(recs) and trimmed_arch(recs[-1 - tail][0]['arch'])
           == trimmed_arch(recs[-1][0]['arch'])):
        tail += 1
    if tail >= 60:
        recs = recs[:-tail]
    seen = {}
    for sol, obj in recs:
        key = (tuple(sol['layout']), tuple(trimmed_arch(sol['arch'])))
        if key in seen:
            continue
        expr = obj.get('expressibility')
        if expr is None:
            expr = evaluator.expr_cache.get(
                evaluator.expr_key_extra + str(trimmed_arch(sol['arch'])))
        if expr is None:
            continue
        seen[key] = {'solution': sol, 'expressibility': float(expr),
                     'success_rate': float(obj['success_rate'])}
    pts = list(seen.values())
    F = np.array([[r['expressibility'], -r['success_rate']] for r in pts])
    nd = NonDominatedSorting().do(F, only_non_dominated_front=True)
    print(f'{len(pts)} logged evaluations, front of {len(nd)}')
    return [pts[int(i)] for i in nd]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--setting', required=True, choices=list(SETTINGS))
    parser.add_argument('--out-root', default='genome_ablation')
    parser.add_argument('--from-log', default=None,
                        help='finish an unfinished search from its log: the '
                             'front is rebuilt from every logged evaluation')
    args, opts = parser.parse_known_args()

    configs.load(os.path.join(args.run_dir, 'metainfo', 'configs.yaml'))
    trained_seed, trained_set_seed = configs.debug.seed, configs.debug.set_seed
    configs.load(args.config, recursive=True)
    configs.debug.seed, configs.debug.set_seed = trained_seed, trained_set_seed
    configs.update(opts)
    set_seed(configs.debug.seed)
    torch.set_num_threads(2)
    device = torch.device('cpu')

    nsga2_cfg = configs.es.nsga2
    objectives = list(nsga2_cfg.objectives)
    assert objectives == ['expressibility', 'success_rate'], objectives
    rates = SETTINGS[args.setting]
    backend_name = configs.qiskit.backend_name
    seed = configs.debug.seed
    out_dir = os.path.join(args.out_root, args.setting)
    out_json = os.path.join(out_dir, f'{backend_name}_s{seed}.json')
    if args.from_log:
        out_json = out_json[:-5] + '.fromlog.json'
    if os.path.exists(out_json):
        print('exists:', out_json)
        return
    os.makedirs(out_dir, exist_ok=True)

    state_dict = torch.load(os.path.join(args.run_dir, configs.ckpt.name),
                            map_location=device, weights_only=False)
    model_load = state_dict['model_arch']
    model = builder.make_model()
    for module_load, module in zip(model_load.modules(), model.modules()):
        if isinstance(module, tq.RandomLayer):
            module.rebuild_random_layer_from_op_list(
                n_ops_in=module_load.n_ops, wires_in=module_load.wires,
                op_list_in=module_load.op_list)
    model.load_state_dict(state_dict['model'], strict=False)
    model.to(device)
    model.eval()

    n_available_wires = resolve_n_available_wires(model)
    backend = BACKENDS[backend_name]()
    qiskit_processor = builder.make_qiskit_processor()
    qiskit_processor.properties = backend.properties()
    model.set_qiskit_processor(qiskit_processor)

    converter = Converter(model.q_device.n_wires, n_available_wires, model.arch_space)
    ex = nsga2_cfg.expressibility
    evaluator = CachedEvaluator(
        model, objectives, qiskit_processor=qiskit_processor,
        expr_n_pairs=ex.n_pairs, expr_nbins=ex.nbins, expr_seed=ex.seed,
        expr_cache=SharedExprCache(os.path.join(args.out_root, 'expr_cache.sqlite')),
        expr_key_extra=f'{ex.n_pairs}/{ex.nbins}/{ex.seed}:')
    problem = MultiObjectiveProblem(converter, evaluator, objectives,
                                    gene_mask=getattr(configs.es, 'gene_mask', None))

    phase = Phase(staged=(args.setting == 'staged'))
    repair = GenomeRepair(converter.n_wires, converter.n_available_wires)
    algorithm = NSGA2(
        pop_size=nsga2_cfg.pop_size,
        sampling=MixedVariableSampling(),
        mating=MixedVariableMating(
            selection=(TournamentSelection(func_comp=binary_tournament)
                       if args.setting == 'tournament' else RandomSelection()),
            crossover={Choice: PhasedUX(phase)},
            mutation={Choice: GroupedChoiceMutation(rates['arch'], rates['layout'], phase)},
            eliminate_duplicates=MixedVariableDuplicateElimination(),
            repair=repair),
        eliminate_duplicates=MixedVariableDuplicateElimination(),
        repair=repair)

    log = AblationLog(phase, problem.n_var)
    log.converter = converter
    log.partial_path = out_json[:-5] + '.partial.json'
    t0 = time.time()
    front = []
    if args.from_log:
        log.partial_path = None
        front = front_from_log(args.from_log, evaluator)
    else:
        res = minimize(problem, algorithm, ('n_gen', nsga2_cfg.n_gen), seed=seed,
                       callback=log, verbose=False, copy_algorithm=False)
        for x, f in zip(res.X, res.F):
            gene = [int(x[f'x{i}']) for i in range(problem.n_var)]
            sol = converter.gene2solution(gene)
            front.append({'solution': sol, 'expressibility': float(f[0]),
                          'success_rate': float(-f[1])})
    search_s = time.time() - t0
    deployed = min(front, key=lambda r: r['expressibility'])
    n_unique = len(evaluator.solution_lib)
    arch = deployed['solution']['arch']

    # analytic rank of the deployed layout among all P(5,4) layouts
    t1 = time.time()
    layouts = list(permutations(range(n_available_wires), converter.n_wires))
    srates = []
    for lay in layouts:
        r = evaluator.compute({'layout': list(lay), 'arch': arch})
        srates.append(r['success_rate'])
    srates = np.array(srates)
    dep_sr = evaluator.compute(deployed['solution'])['success_rate']
    sr_rank = 1 + int((srates > dep_sr + 1e-15).sum())
    best_layout = list(layouts[int(np.argmax(srates))])
    enum_s = time.time() - t1

    model.set_sample_arch(arch)
    n_params = int(sum(p.numel() for p in model.q_layer.parameters()
                       if p.requires_grad)) if hasattr(model, 'count_sample_params') is False \
        else int(model.count_sample_params())

    gens = log.generations
    out = {
        'partial': bool(args.from_log),
        'setting': args.setting, 'rates': rates, 'backend': backend_name,
        'seed': seed, 'config': args.config, 'run_dir': args.run_dir,
        'n_unique_evaluations': n_unique,
        'n_expr_computed': evaluator.n_expr_computed,
        'search_seconds': search_s, 'enumeration_seconds': enum_s,
        'same_arch_events': int(sum(g['same_arch_events'] for g in gens)),
        'improving_moves': int(sum(g['improving_moves'] for g in gens)),
        'n_offspring_total': int(sum(g['n_offspring'] for g in gens[1:])),
        'generations': gens,
        'front': front, 'front_size': len(front),
        'deployed': {**deployed, 'n_blocks': int(arch[-1]), 'n_params': n_params,
                     'success_rate_rank': sr_rank,
                     'success_rate_best': float(srates.max()),
                     'enumerated_layout': best_layout},
    }
    json.dump(out, open(out_json, 'w'), indent=1)

    ckpt_dir = os.path.join(out_dir, f'{backend_name}_s{seed}', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save({'solution': deployed['solution'],
                'score': deployed['expressibility']},
               os.path.join(ckpt_dir, 'best_solution.pt'))
    print(f"{args.setting} {backend_name} s{seed}: events {out['same_arch_events']} "
          f"improving {out['improving_moves']} blocks {arch[-1]} params {n_params} "
          f"Expr {deployed['expressibility']:.4f} S-rank {sr_rank} "
          f"({search_s/60:.1f} min)", flush=True)


if __name__ == '__main__':
    main()

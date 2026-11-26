"""Does layout enumeration survive a 127-qubit device?
(Access revision, Reviewer 1 #2 / Reviewer 2 #4.)

Two measurements on FakeWashingtonV2 (127 qubits, heavy-hex, CX basis -- the
same basis as the five-qubit devices of the paper):

1. Counting. The number of connected k-qubit subsets of the coupling graph,
   against the P(N, k) ordered assignments of brute force, for k = 2..8. A
   k-qubit circuit whose interaction graph is connected can only be placed
   without crossing the device on a connected subset, so the connected subsets
   times k! orderings are the candidate set worth scoring.

2. Scoring. For one of the paper's own four-qubit circuits (the three-block
   FakeQuitoV2 checkpoint of search seed 100), every connected 4-subset in
   every ordering is transpiled with that initial layout (optimization level
   2, seed 42) and scored with the paper's success-rate estimate, timing each
   step. Qiskit's own calibration-aware layout (levels 2 and 3) and mapomatic,
   both designed for large devices, are scored the same way and placed in the
   enumerated ranking.

Writes scaling_enumeration/FakeWashingtonV2.json.

Usage (repo root):
    python examples/scaling_enumeration.py runs/<quito s100 run> [--jobs 8]
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore")

import argparse
import itertools
import json
import math
import time
from multiprocessing import Pool

import numpy as np

BACKEND = 'FakeWashingtonV2'
EVAL_CFG = 'examples/configs/mnist/four0123/eval/quito/noise/opt2/all.yml'
SEED = 42


def connected_subsets(adj, k):
    """All connected k-vertex subsets (as sorted tuples), by extension."""
    level = {frozenset([v]) for v in adj}
    for _ in range(k - 1):
        nxt = set()
        for s in level:
            for v in s:
                for w in adj[v]:
                    if w not in s:
                        nxt.add(s | {w})
        level = nxt
    return sorted(tuple(sorted(s)) for s in level)


_G = {}


def _init(circ_qpy):
    import io
    from qiskit import qpy
    import qiskit_ibm_runtime.fake_provider as fp
    _G['circ'] = qpy.load(io.BytesIO(circ_qpy))[0]
    _G['backend'] = getattr(fp, BACKEND)()
    _G['props'] = _G['backend'].properties()


def _score(layout):
    from qiskit import transpile
    from torchquantum.util import get_success_rate
    t0 = time.perf_counter()
    t = transpile(_G['circ'], backend=_G['backend'], initial_layout=list(layout),
                  optimization_level=2, seed_transpiler=SEED)
    t1 = time.perf_counter()
    s = get_success_rate(_G['props'], t.remove_final_measurements(inplace=False))
    t2 = time.perf_counter()
    return layout, float(s), t1 - t0, t2 - t1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir')
    parser.add_argument('--jobs', type=int, default=8)
    args = parser.parse_args()

    import io
    from qiskit import qpy, transpile
    import compiler_layout_baselines as cb
    from torchquantum.util import get_success_rate

    backend = cb.get_backend(BACKEND)
    n = backend.num_qubits
    adj = {q: set() for q in range(n)}
    for a, b in backend.coupling_map.get_edges():
        adj[a].add(b)
        adj[b].add(a)

    counts = {}
    for k in range(2, 9):
        t0 = time.perf_counter()
        c = len(connected_subsets(adj, k))
        counts[k] = {'connected_subsets': c,
                     'candidate_layouts': c * math.factorial(k),
                     'ordered_assignments': math.perm(n, k),
                     'count_seconds': time.perf_counter() - t0}
        print(f'k={k}: {c} connected subsets, {c * math.factorial(k)} layouts '
              f'vs P({n},{k}) = {math.perm(n, k):.3e}', flush=True)

    circ, solution, _ = cb.build_circuit(args.run_dir, EVAL_CFG)
    buf = io.BytesIO()
    qpy.dump(circ, buf)
    subsets = connected_subsets(adj, circ.num_qubits)
    layouts = [p for s in subsets for p in itertools.permutations(s)]
    print(f'scoring {len(layouts)} layouts with {args.jobs} workers', flush=True)
    t0 = time.perf_counter()
    with Pool(args.jobs, initializer=_init, initargs=(buf.getvalue(),)) as pool:
        res = pool.map(_score, layouts, chunksize=16)
    wall = time.perf_counter() - t0
    sr = {tuple(l): s for l, s, _, _ in res}
    vals = np.array(list(sr.values()))
    best = max(sr, key=sr.get)
    rank = lambda s: 1 + int((vals > s).sum())

    tools = {}
    for level in (2, 3):
        layout, _, info = cb.qiskit_layout(circ, backend, level, SEED)
        t = transpile(circ, backend=backend, initial_layout=layout,
                      optimization_level=2, seed_transpiler=SEED)
        s = get_success_rate(backend.properties(),
                             t.remove_final_measurements(inplace=False))
        tools[f'qiskit_l{level}'] = {'layout': layout, 'success_rate': float(s),
                                     'rank': rank(s), 'in_candidates': tuple(layout) in sr,
                                     **info}
    try:
        layout, info = cb.mapomatic_layout(circ, backend, SEED)
        s = sr.get(tuple(layout))
        if s is None:
            t = transpile(circ, backend=backend, initial_layout=layout,
                          optimization_level=2, seed_transpiler=SEED)
            s = get_success_rate(backend.properties(),
                                 t.remove_final_measurements(inplace=False))
        tools['mapomatic'] = {'layout': layout, 'success_rate': float(s),
                              'rank': rank(s), **info}
    except Exception as e:  # report, do not hide, a tool failure
        tools['mapomatic'] = {'error': repr(e)}

    out = {
        'backend': BACKEND, 'n_qubits': n, 'run_dir': args.run_dir,
        'arch': solution['arch'], 'counts': counts,
        'n_layouts_scored': len(layouts),
        'wall_seconds': wall, 'jobs': args.jobs,
        'transpile_seconds_total': float(sum(r[2] for r in res)),
        'score_seconds_total': float(sum(r[3] for r in res)),
        'transpile_seconds_median': float(np.median([r[2] for r in res])),
        'score_seconds_median': float(np.median([r[3] for r in res])),
        'best_layout': list(best), 'best_success_rate': sr[best],
        'median_success_rate': float(np.median(vals)),
        'tools': tools,
    }
    os.makedirs('scaling_enumeration', exist_ok=True)
    path = os.path.join('scaling_enumeration', f'{BACKEND}.json')
    json.dump(out, open(path, 'w'), indent=1, default=list)
    print(json.dumps({k: v for k, v in out.items() if k != 'counts'}, indent=1,
                     default=list))


if __name__ == '__main__':
    main()

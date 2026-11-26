"""Expressibility against circuit size at several qubit counts.

Access revision, Reviewer 2 #4 (1) and #1: is the size coupling of
expressibility a property of the four-qubit setting, and where does it
saturate? The U3+CU3 design space of the paper (SuperU3CU3ShareFrontLayer0:
per block one U3 layer and one circular CU3 layer, each of 1..n gates, up to
eight blocks) is instantiated at n = 3..6 qubits, random architectures are
drawn stratified by block count, and each is scored with the paper's own
estimator (compute_expressibility_tq, 150 pairs, 75 bins, draw seed 1234).

For each width the script also estimates the estimator's floor: the KL that
exactly Haar-random states reach under the same finite sample and binning,
the value any circuit approaches as it becomes a state 2-design.

At n = 4 the construction is the paper's model; the script checks that it
reproduces expr32.json's values for the architectures listed there.

    python score_expressibility_widths.py --out expr_widths.json
"""
import argparse
import json
import math
import os
import random
import sys
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'examples'))

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402
from scipy.stats import spearmanr                               # noqa: E402
from torchquantum.algorithm.quantumnas.super_layers import (     # noqa: E402
    SuperU3CU3ShareFrontLayer0)
from torchquantum.algorithm.quantumnas.super_utils import (      # noqa: E402
    compute_expressibility_tq, isolated_rng)

N_BLOCKS, N_PAIRS, N_BINS, SEED = 8, 150, 75, 1234


class Circuit:
    """The attributes compute_expressibility_tq reads off a model."""

    def __init__(self, n):
        self.n_wires = n
        self.q_layer = SuperU3CU3ShareFrontLayer0(arch=dict(
            n_wires=n, n_blocks=N_BLOCKS, n_layers_per_block=2,
            n_front_share_blocks=1, n_front_share_wires=1, n_front_share_ops=1))

    def set_sample_arch(self, arch):
        self.q_layer.set_sample_arch(arch)

    def n_params(self, arch):
        self.set_sample_arch(arch)
        return int(self.q_layer.count_sample_params())


def kl_to_haar(fid, n):
    p, edges = np.histogram(fid, bins=N_BINS, range=(0, 1),
                            weights=np.ones_like(fid) / len(fid))
    d = 2 ** n
    q = (1 - (1 - edges[:-1]) ** (d - 1)) - (1 - (1 - edges[1:]) ** (d - 1))
    p = np.clip(p, 1e-12, 1); p /= p.sum()
    q = np.clip(q / q.sum(), 1e-12, 1); q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def haar_floor(n, reps=200, seed=0):
    rng = np.random.default_rng(seed)
    d = 2 ** n
    vals = []
    for _ in range(reps):
        z = rng.normal(size=(2, N_PAIRS, d)) + 1j * rng.normal(size=(2, N_PAIRS, d))
        z /= np.linalg.norm(z, axis=2, keepdims=True)
        fid = np.abs(np.sum(z[0].conj() * z[1], axis=1)) ** 2
        vals.append(kl_to_haar(fid, n))
    return float(np.mean(vals)), float(np.std(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='expr_widths.json')
    ap.add_argument('--widths', default='3,4,5,6')
    ap.add_argument('--per-block', type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(4)

    # n = 4 check against the paper's own scores
    c4 = Circuit(4)
    check = []
    for r in json.load(open(os.path.join(ROOT, 'expr32.json')))[:4]:
        v = compute_expressibility_tq(c4, r['arch'], n_pairs=N_PAIRS,
                                      nbins=N_BINS, seed=SEED)
        check.append({'arch': r['arch'], 'paper': r['expr_kl'], 'here': v})
        print(f"check n=4: paper {r['expr_kl']:.6f}  here {v:.6f}", flush=True)

    out = {'check_n4': check, 'widths': {}}
    for n in map(int, args.widths.split(',')):
        c = Circuit(n)
        rng = random.Random(1000 + n)
        rows = []
        for b in range(1, N_BLOCKS + 1):
            for _ in range(args.per_block):
                arch = [rng.randint(1, n) for _ in range(2 * N_BLOCKS)] + [b]
                kl = compute_expressibility_tq(c, arch, n_pairs=N_PAIRS,
                                               nbins=N_BINS, seed=SEED)
                rows.append({'arch': arch, 'blocks': b,
                             'params': c.n_params(arch), 'expr_kl': kl})
        floor_m, floor_s = haar_floor(n)
        kl = np.array([r['expr_kl'] for r in rows])
        lnp = np.log([r['params'] for r in rows])
        by_block = {b: float(np.mean([r['expr_kl'] for r in rows if r['blocks'] == b]))
                    for b in range(1, N_BLOCKS + 1)}
        rho = spearmanr(kl, lnp).correlation
        out['widths'][n] = {'rows': rows, 'spearman_expr_lnparams': float(rho),
                            'haar_floor_mean': floor_m, 'haar_floor_sd': floor_s,
                            'mean_expr_by_blocks': by_block,
                            'max_params': 2 * N_BLOCKS * n * 3}
        print(f"n={n}: rho(Expr, ln N) = {rho:+.3f}  floor {floor_m:.4f}+-{floor_s:.4f}  "
              + '  '.join(f'b{b}:{v:.3f}' for b, v in by_block.items()), flush=True)
    json.dump(out, open(os.path.join(ROOT, args.out), 'w'), indent=1)


if __name__ == '__main__':
    main()

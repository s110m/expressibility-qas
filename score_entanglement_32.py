"""Add entangling capability to the 32-architecture proxy comparison.

Access revision, Reviewer 1 #5: compare expressibility against another
mainstream training-free descriptor. Entangling capability (Sim et al. 2019) is
the mean Meyer--Wallach measure over random parameter draws,

    Ent = E_theta[ Q(|psi(theta)>) ],   Q = 2 (1 - (1/n) sum_k Tr rho_k^2),

where rho_k is the reduced state of qubit k. Q is 0 for product states and 1
at most; unlike expressibility it is MAXIMISED by an entanglement-seeking
search, so a POSITIVE rho against accuracy means alignment.

Same model construction and parameter randomizer as score_expressibility_32.py,
so the circuits scored are exactly the ones expr32.json scored. Draws are
pinned with the same isolated_rng seed.

    python score_entanglement_32.py --in expr32.json --out ent32.json
"""
import argparse
import json
import os
import sys
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'examples'))
os.chdir(os.path.join(ROOT, 'examples'))

import torch                                                    # noqa: E402
import torchquantum as tq                                       # noqa: E402
from torchpack.utils import io                                  # noqa: E402
from torchpack.utils.config import configs                      # noqa: E402
from core import builder                                        # noqa: E402
from torchquantum.algorithm.quantumnas.super_utils import (      # noqa: E402
    get_state_from_random_params, isolated_rng)

from score_expressibility_32 import SUPER                       # noqa: E402


def meyer_wallach(psi, n):
    t = psi.reshape([2] * n)
    purity = 0.0
    for k in range(n):
        m = torch.movedim(t, k, 0).reshape(2, -1)
        rho = m @ m.conj().T
        purity += torch.real(torch.trace(rho @ rho)).item()
    return 2 * (1 - purity / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ckpt', default='checkpoints/step-18400.pt')
    ap.add_argument('--n-samples', type=int, default=300)
    ap.add_argument('--seed', type=int, default=1234)
    args = ap.parse_args()

    archs = json.load(open(os.path.join(ROOT, args.inp)))
    run = os.path.join('..', SUPER)
    configs.load(os.path.join(run, 'metainfo', 'configs.yaml'))
    sd = io.load(os.path.join(run, args.ckpt), map_location='cpu',
                 weights_only=False)
    model = builder.make_model()
    for ml, m in zip(sd['model_arch'].modules(), model.modules()):
        if isinstance(m, tq.RandomLayer):
            m.rebuild_random_layer_from_op_list(
                n_ops_in=ml.n_ops, wires_in=ml.wires, op_list_in=ml.op_list)
    model.load_state_dict(sd['model'], strict=False)
    model.eval().to(torch.device('cpu'))

    n = model.n_wires
    out = []
    for i, r in enumerate(archs, 1):
        with isolated_rng(args.seed):
            qs = [meyer_wallach(get_state_from_random_params(model, r['arch']), n)
                  for _ in range(args.n_samples)]
        ent = sum(qs) / len(qs)
        out.append({**r, 'ent_mw': ent})
        print(f'  [{i:>2}/{len(archs)}] {r["source"]:22} par {r["params"]:>3}  '
              f'Ent {ent:.4f}  KL {r["expr_kl"]:.4f}  acc {r["acc_valid"]:5.2f}%')
    json.dump(out, open(os.path.join(ROOT, args.out), 'w'), indent=1)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()

"""Score a common architecture set by expressibility, for a like-for-like proxy test.

The four-arm comparison cannot separate proxy quality from circuit size: the two
search families select nearly disjoint parameter ranges (12-57 vs 57-114). This
sidesteps that by scoring ONE architecture set -- the 32 circuits spanning 6-192
params that already have from-scratch accuracy (fromscratch.json) and
SuperCircuit-inherited loss (inherited_pooled.json) -- with the second proxy too.
Then both proxies can be ranked against the same ground truth over the same
circuits, with size partialled out.

Uses the exact settings the 2xx/3xx searches ran with: n_pairs=150, nbins=75,
seed=1234. Weights are irrelevant here -- compute_expressibility_tq randomizes
q_layer in place -- but the model is built the same way regardless, so the op
list structure matches what those searches scored.

    python score_expressibility_32.py --in <inherited_pooled.json> --out expr32.json
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
    compute_expressibility_tq)

SUPER = ('runs/mnist.four0123.train.super.u3cu3_s0.ldiff_blkexpand'
         '.blk8s1.1.1_diff7_chu3_sta40_seed42')

# The six circuits the expressibility arms selected are themselves in the input
# set (sources exprES/* and exprRand/*), and all_arms.json records the KL their
# search wrote. Re-scoring them here must reproduce those numbers -- that is the
# validation, and it compares against recorded data rather than any constant
# copied into this file by hand.
VALIDATE_AGAINST = os.path.join(ROOT, 'all_arms.json')
ARM_SOURCE = {'B': 'exprES', 'C': 'exprRand'}


def validate(scored):
    if not os.path.exists(VALIDATE_AGAINST):
        print('  (all_arms.json absent -- skipped)')
        return True
    by_src = {r['source']: r for r in scored}
    ok, n = True, 0
    for r in json.load(open(VALIDATE_AGAINST)):
        if r['arm'] not in ARM_SOURCE:
            continue
        src = f"{ARM_SOURCE[r['arm']]}/athens/s{r['seed']}"
        e = by_src.get(src)
        if e is None:
            continue
        good = abs(e['expr_kl'] - r['score']) < 5e-5 and e['arch'] == r['arch']
        ok &= good
        n += 1
        print(f"  {src:24} recomputed {e['expr_kl']:.6f} vs recorded "
              f"{r['score']:.6f}   {'MATCH' if good else 'MISMATCH'}")
    return ok and n > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True,
                    help='inherited_pooled.json (32 archs w/ from-scratch acc)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--ckpt', default='checkpoints/step-18400.pt')
    ap.add_argument('--n-pairs', type=int, default=150)
    ap.add_argument('--nbins', type=int, default=75)
    ap.add_argument('--seed', type=int, default=1234)
    args = ap.parse_args()

    archs = json.load(open(args.inp))
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

    print(f'scoring {len(archs)} architectures  '
          f'(n_pairs={args.n_pairs}, nbins={args.nbins}, seed={args.seed})')
    out = []
    for i, r in enumerate(archs, 1):
        kl = compute_expressibility_tq(model, r['arch'], n_pairs=args.n_pairs,
                                       nbins=args.nbins, seed=args.seed)
        out.append({**r, 'expr_kl': kl})
        print(f'  [{i:>2}/{len(archs)}] {r["source"]:22} par {r["params"]:>3}  '
              f'KL {kl:.6f}  inh_loss {r["inh_loss"]:.4f}  '
              f'from-scratch {r["acc_valid"]:5.2f}%')

    print('\nvalidation -- KL recomputed here vs what the 2xx/3xx searches recorded:')
    ok = validate(out)
    if not ok:
        print('  !! scorer does not reproduce the search -- do not trust these KLs')

    json.dump(out, open(os.path.join(ROOT, args.out), 'w'), indent=1)
    print(f'\nwrote {args.out}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

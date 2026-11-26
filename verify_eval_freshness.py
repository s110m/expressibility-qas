"""Check every arm's eval results against the retrain that produced them.

An eval JSON is only trustworthy if (a) the architecture it records matches the
one in the search's best_solution.pt, and (b) it was written after the retrain
finished. Arms B/C silently failed both tests once already (results dated
2026-08-10 against a 2026-08-13 retrain, for a different circuit), which is why
this exists. Run from the repo root; every row should read OK.
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'examples'))
import torch  # noqa: E402  (needs examples/ on sys.path first)

FAM = "u3cu3_s0.ldiff_blkexpand.blk8s1.1.1_diff7_chu3_sta40"
ARMS = [(100, 'A'), (101, 'A'), (102, 'A'), (200, 'B'), (201, 'B'), (202, 'B'),
        (300, 'C'), (301, 'C'), (302, 'C'), (400, 'D'), (401, 'D'), (402, 'D')]


def nparams(arch):
    """Sampled quantum params: the first n_blocks layer-width pairs, x3 per gate."""
    n_blocks = arch[-1]
    return sum(arch[:2 * n_blocks]) * 3


def main():
    print(f"{'arm':3} {'seed':4} {'dev':7} {'es_arch(nb,par)':16} {'retrain_end':17} | "
          f"{'evalfile':10} {'eval_ts':17} {'eval(nb,par)':13} {'acc':7} STATUS")
    n_bad = 0
    for seed, arm in ARMS:
        # arms B/C only ever searched on athens; A/D searched on all three
        search_devs = ["athens", "belem", "quito"] if arm in "AD" else ["athens"]
        for dev in search_devs:
            es_dir = (f"{ROOT}/es_runs/mnist.four0123.es.{dev}.noise.opt2.setting0."
                      f"{FAM}_seed{seed}")
            sol = torch.load(f"{es_dir}/checkpoints/best_solution.pt",
                             map_location="cpu", weights_only=False)
            arch = sol["solution"]["arch"]
            es_id = (arch[-1], nparams(arch))

            # the retrain always lives under the *search* device's run dir
            train_dir = (f"{ROOT}/runs/mnist.four0123.train.searched.scratch.{dev}"
                         f".noise.opt2.setting0.{FAM}_seed42_from-search{seed}")
            retrain_end = json.load(
                open(f"{train_dir}/metainfo/timing.json"))["end_time"][:19]

            for path in sorted(glob.glob(f"{train_dir}/eval_results/*.json")):
                res = json.load(open(path))
                ev_arch = res["solution"]["arch"]
                ev_id = (ev_arch[-1], nparams(ev_arch))
                tag = (os.path.basename(path)
                       .replace("mnist.four0123.eval.", "")
                       .replace(".noise.opt2", "").replace(".json", ""))
                if ev_id != es_id:
                    status = "STALE-ARCH"
                elif res["timestamp"][:19] <= retrain_end:
                    status = "STALE-TIME"
                else:
                    status = "OK"
                n_bad += status != "OK"
                print(f"{arm:3} {seed:<4} {dev:7} {str(es_id):16} {retrain_end:17} | "
                      f"{tag:10} {res['timestamp'][:19]:17} {str(ev_id):13} "
                      f"{res['accuracy_percent']:7.2f} {status}")
    print(f"\n{n_bad} stale result(s).")
    return 1 if n_bad else 0


if __name__ == '__main__':
    sys.exit(main())

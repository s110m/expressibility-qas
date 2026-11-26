"""Plan the retrain/eval jobs for the three retrained settings of the ablation.

Cells (setting, device, seed) that deploy the same architecture share one
from-scratch retraining (the training configs differ only by name, and the
layout is inert during training); every cell is then evaluated on its own
device with its deployed and its enumerated layout passed explicitly.
Writes train3_jobs_<seeds>.txt ("setting device seed") and eval3_jobs_<seeds>.txt
("setting device seed canon_setting canon_device").
"""
import json, os, sys
GA = os.path.dirname(os.path.abspath(__file__))
SETTINGS = ('staged', 'p0.4', 'split')   # staged first: its s100 run already exists
DEVS = {'quito': 'FakeQuitoV2', 'athens': 'FakeAthensV2', 'belem': 'FakeBelemV2'}
seeds = [int(x) for x in sys.argv[1:]] or [100, 101, 102]
train, ev = [], []
for seed in seeds:
    canon = {}
    for s in SETTINGS:
        for dev, b in DEVS.items():
            f = os.path.join(GA, s, f'{b}_s{seed}.json')
            if not os.path.exists(f):
                continue
            arch = str(json.load(open(f))['deployed']['solution']['arch'])
            if arch not in canon:
                canon[arch] = (s, dev)
                train.append(f'{s} {dev} {seed}')
            ev.append(f'{s} {dev} {seed} {canon[arch][0]} {canon[arch][1]}')
tag = '_'.join(map(str, seeds))
open(os.path.join(GA, f'train3_jobs_{tag}.txt'), 'w', newline='\n').write('\n'.join(train) + '\n')
open(os.path.join(GA, f'eval3_jobs_{tag}.txt'), 'w', newline='\n').write('\n'.join(ev) + '\n')
print(len(train), 'trainings,', len(ev), 'evaluated cells')

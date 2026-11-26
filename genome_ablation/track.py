"""Status of the genome-ablation batch, and recovery of unfinished searches.

    python genome_ablation/track.py            # table + genome_ablation/status.json
    python genome_ablation/track.py --recover  # finish stalled/crashed searches from their logs

A search is 'done' when <setting>/<Backend>_s<seed>.json exists; 'running' when
its log was written in the last STALL_MIN minutes; otherwise 'stalled' (log
present, no result: crashed or killed) or 'pending'. Searches started after
2026-10-19 16:10 also leave a per-generation <...>.partial.json, which gives the
exact generation; older ones show the number of logged evaluations instead
(a finished search logs 1000-1600). --recover runs nsga2_genome_ablation.py
--from-log, which rebuilds the front from every logged evaluation and writes
<...>.fromlog.json with "partial": true and the same fields as a finished run.
"""
import json, os, re, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GA = os.path.join(ROOT, 'genome_ablation')
BK = {'athens': 'FakeAthensV2', 'quito': 'FakeQuitoV2', 'belem': 'FakeBelemV2'}
STALL_MIN = 15
N_GEN = 40
SUPER = ('/c/Neural-Networks-Files/torchquantum/runs/mnist.four0123.train.super.u3cu3_s0.'
         'ldiff_blkexpand.blk8s1.1.1_diff7_chu3_sta40_seed42_nopool')


def status():
    rows = []
    for line in open(os.path.join(GA, 'jobs.txt')):
        if not line.strip():
            continue
        s, dev, seed = line.split()
        b = BK[dev]
        stem = os.path.join(GA, s, f'{b}_s{seed}')
        log = os.path.join(GA, 'logs', f'{s}_{dev}_{seed}.log')
        r = {'setting': s, 'device': dev, 'seed': int(seed)}
        if os.path.exists(stem + '.json'):
            d = json.load(open(stem + '.json'))['deployed']
            r.update(state='done', s_rank=d['success_rate_rank'], blocks=d['n_blocks'])
        elif os.path.exists(log):
            age = (time.time() - os.path.getmtime(log)) / 60
            r['state'] = 'running' if age < STALL_MIN else 'stalled'
            r['log_age_min'] = round(age, 1)
            txt = open(log, encoding='utf-8', errors='replace').read()
            r['n_logged'] = txt.count('Solution:')
            if 'Traceback' in txt:
                r['state'] = 'crashed'
            if os.path.exists(stem + '.partial.json'):
                p = json.load(open(stem + '.partial.json'))
                r['gen'] = p['gen']
                r['partial_expr'] = p['deployed']['expressibility']
            if os.path.exists(stem + '.fromlog.json'):
                d = json.load(open(stem + '.fromlog.json'))['deployed']
                r.update(recovered=True, s_rank=d['success_rate_rank'], blocks=d['n_blocks'])
        else:
            r['state'] = 'pending'
        rows.append(r)
    return rows


def recover(rows):
    for r in rows:
        if r['state'] not in ('stalled', 'crashed') or r.get('recovered'):
            continue
        s, dev, seed = r['setting'], r['device'], r['seed']
        cfg = (f'examples/configs/mnist/four0123/es/{dev}/noise/opt2/setting0/u3cu3_s0/'
               'ldiff_blkexpand/nsga2/blk8s1.1.1_diff7_chu3_sta40.yml')
        cmd = [sys.executable, 'examples/nsga2_genome_ablation.py', cfg, '--run-dir', SUPER,
               '--setting', s, '--from-log', f'genome_ablation/logs/{s}_{dev}_{seed}.log',
               f'debug.seed={seed}', 'ckpt.name=checkpoints/step-18400.pt']
        print('recovering', s, dev, seed)
        out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                             env={**os.environ, 'OMP_NUM_THREADS': '2'})
        print((out.stdout.strip().splitlines() or ['(no output)'])[-1])


if __name__ == '__main__':
    rows = status()
    if '--recover' in sys.argv:
        recover(rows)
        rows = status()
    json.dump(rows, open(os.path.join(GA, 'status.json'), 'w'), indent=1)
    from collections import Counter
    print(dict(Counter(r['state'] for r in rows)))
    for r in rows:
        if r['state'] != 'done':
            prog = f"gen {r['gen']}/{N_GEN}" if 'gen' in r else f"{r.get('n_logged', 0)} evals logged"
            extra = f" -> recovered, S-rank {r['s_rank']}" if r.get('recovered') else ''
            print(f"{r['state']:8s} {r['setting']:10s} {r['device']:7s} {r['seed']}  "
                  + (prog if r['state'] != 'pending' else '') + extra)


def retrain_status():
    """Epoch reached by each from-scratch retraining (200 epochs = complete)."""
    out = []
    for log in sorted(os.listdir(os.path.join(GA, 'logs'))):
        if not (log.startswith('train_') and log.endswith('.log')):
            continue
        txt = open(os.path.join(GA, 'logs', log), encoding='utf-8', errors='replace').read()
        n_done = txt.count('Epoch finished in')
        out.append({'log': log, 'epoch': n_done, 'complete': n_done >= 200})
    return out


if __name__ == '__main__':
    for r in retrain_status():
        print(f"retrain  {r['log'][6:-4]:22s} epoch {r['epoch']}/200"
              + (' complete' if r['complete'] else ''))

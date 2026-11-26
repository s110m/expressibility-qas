"""Memory-gated driver for the remaining ablation trainings and evaluations.

Idempotent: train_cell.sh skips finished trainings (step-18400.pt) and
eval_cell.sh skips finished evaluations, so this can be re-run at any time.
A training counts as stalled when its log has not changed for STALL_MIN
minutes without reaching step-18400.pt; it is then restarted from scratch.
"""
import glob, os, subprocess, time
import psutil

ROOT = r'C:\Neural-Networks-Files\tq_mapping_wt'
GA = os.path.join(ROOT, 'genome_ablation')
BASH = r'C:\Program Files\Git\bin\bash.exe'
TRAIN_MIN_GB, EVAL_MIN_GB = 5.0, 10.0   # one eval with --jobs=4 peaks near 7.7 GB
MAX_EVALS, STALL_MIN = 2, 30


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def avail_gb():
    return psutil.virtual_memory().available / 2**30


def run_dir(s, dev, seed):
    return glob.glob(os.path.join(ROOT, 'runs', f'*.scratch.{dev}.*ga_{s.replace(".", "")}_s{seed}_seed42'))


def trained(s, dev, seed):
    r = run_dir(s, dev, seed)
    return bool(r) and os.path.exists(os.path.join(r[0], 'checkpoints', 'step-18400.pt'))


def train_log(s, dev, seed):
    return os.path.join(GA, 'logs', f'train_{s}_{dev}_{seed}.log')


def jobs(prefix):
    out = []
    for f in sorted(glob.glob(os.path.join(GA, f'{prefix}_jobs_*.txt'))):
        out += [tuple(l.split()) for l in open(f) if l.strip()]
    return out


def gpu_free_mb():
    """Free GPU memory; two trainings died of CUDA OOM on the 6 GB GPU (20:38, 21:08)."""
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=30).stdout
        return float(out.split()[0])
    except Exception:
        return 0.0


def running_evals():
    """Evaluations in progress on this machine, whoever started them."""
    sufs = set()
    for p in psutil.process_iter(['cmdline']):
        c = p.info['cmdline'] or []
        if any(x.endswith('eval.py') for x in c):
            sufs.update(x for x in c if x.startswith('--result-suffix'))
    return len(sufs)


def launch(script, args, tag):
    lf = open(os.path.join(GA, 'logs', f'orch_{tag}.txt'), 'a')
    return subprocess.Popen([BASH, f'genome_ablation/{script}', *args], cwd=ROOT,
                            stdout=lf, stderr=subprocess.STDOUT)


train_jobs = jobs('train3')
eval_jobs = jobs('eval3')
started = {}
evals = {}
while True:
    # trainings: start missing ones, restart stalled ones
    pending = 0
    for s, dev, seed in train_jobs:
        if trained(s, dev, seed):
            continue
        pending += 1
        key = (s, dev, seed)
        lf = train_log(s, dev, seed)
        age = (time.time() - os.path.getmtime(lf)) / 60 if os.path.exists(lf) else None
        alive = key in started and started[key].poll() is None
        never = age is None
        stalled = age is not None and age > STALL_MIN and not alive
        if (never or stalled) and not alive and avail_gb() >= TRAIN_MIN_GB and gpu_free_mb() >= 1200:
            log('start training', *key, '(stalled)' if stalled else '')
            started[key] = launch('train_cell.sh', key, f'train_{s}_{dev}_{seed}')
            time.sleep(120)   # stagger CUDA initialization
    # evaluations: only for cells whose canonical checkpoint is complete
    for k in [k for k, p in evals.items() if p is not None and p.poll() is not None]:
        log('eval finished', *k)
        evals[k] = None
    remaining = 0
    for s, dev, seed, cs, cdev in eval_jobs:
        key = (s, dev, seed)
        tag = f'{s.replace(".", "")}_{dev}_s{seed}'
        r = run_dir(cs, cdev, seed)
        done = r and len(glob.glob(os.path.join(r[0], 'eval_results', f'*.g[je]_{tag}.json'))) == 2
        if done:
            continue
        remaining += 1
        if key in evals or not trained(cs, cdev, seed):
            continue
        elog = os.path.join(GA, 'logs', f'eval_{tag}.log')
        if os.path.exists(elog) and time.time() - os.path.getmtime(elog) < 600:
            continue   # an evaluation from an earlier orchestrator is still running
        n_running = max(running_evals(), sum(1 for q in evals.values() if q is not None and q.poll() is None))
        if n_running < MAX_EVALS and avail_gb() >= EVAL_MIN_GB:
            log('start eval', *key)
            evals[key] = launch('eval_cell.sh', [s, dev, seed, cs, cdev], f'eval_{tag}')
    if pending == 0 and remaining == 0:
        log('all trainings and evaluations complete')
        break
    # an eval that exited without writing both results is retried once
    for key, p in list(evals.items()):
        if p is None:
            s, dev, seed = key
            tag = f'{s.replace(".", "")}_{dev}_s{seed}'
            if not glob.glob(os.path.join(ROOT, 'runs', '*', 'eval_results', f'*.ge_{tag}.json')):
                if not hasattr(log, 'retried'):
                    log.retried = set()
                if key not in log.retried:
                    log.retried.add(key)
                    del evals[key]
                    log('retry eval', *key)
    time.sleep(120)

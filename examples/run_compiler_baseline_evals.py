"""Full-split accuracy for the layouts in <run>/compiler_baselines/<backend>.json.

Companion to compiler_layout_baselines.py. For each run directory given, this
evaluates on the eval config's full split, all on the current machine:
  * every distinct initial layout among the joint search, the success-rate
    enumeration, the accuracy oracle, Qiskit's calibration-aware layout at
    levels 2 and 3 (qiskit_l2/qiskit_l3 in compiler_baselines, transpiled
    against the fake backend's own Target), mapomatic and tket, via
    qiskit.initial_layout -- the path the enumeration result was measured on;
  * Qiskit optimization levels 2 and 3 end to end through QiskitProcessor
    (free layout choice, own routing). Level 2 is the paper's existing
    "Qiskit default" control. This path is calibration-blind: QiskitProcessor
    transpiles against AerSimulator(noise_model=...) plus the device's coupling
    map and basis gates, a Target with no error rates.

Everything is re-measured rather than reused from the earlier VM results:
re-running the level-2 control here gave 78.01% against the VM's 77.63% for
the same checkpoint and config (athens seed 101), so numbers from the two
machines do not belong in one table. Results go to <run>/eval_results/ with a
"local_*" suffix, and a merged table to <run>/compiler_baselines/<backend>.full.json.

Usage (repo root; EVAL_JOBS overrides the qiskit worker count, default 8):
    python examples/run_compiler_baseline_evals.py <run_dir> [<run_dir> ...]
"""
import glob
import json
import os
import subprocess
import sys

JOBS = int(os.environ.get('EVAL_JOBS', 8))
LAYOUT_CONDITIONS = ('joint_search', 'enumeration', 'oracle', 'qiskit_l2',
                     'qiskit_l3', 'mapomatic', 'tket')


def result_path(run_dir, cfg, suffix):
    stem = cfg.replace('/', '.').replace('examples.', '').replace(
        '.yml', '').replace('configs.', '')
    return os.path.join(run_dir, 'eval_results', f'{stem}.{suffix}.json')


def run_eval(run_dir, cfg, suffix, overrides, out_suffix=None):
    out = result_path(run_dir, cfg, out_suffix or suffix)
    if os.path.exists(out):
        print(f'  exists, skipping: {os.path.basename(out)}')
        return out
    # --jobs/workers capped: each worker process holds ~1 GB (torch + tf
    # imports), and the config chain's 48 qiskit jobs exhaust a 32 GB machine
    cmd = [sys.executable, 'examples/eval.py', cfg, f'--run-dir={run_dir}',
           f'--result-suffix={suffix}', f'--jobs={JOBS}',
           'run.workers_per_gpu=0', *overrides]
    print('  $', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    assert os.path.exists(out), f'eval.py did not write {out}'
    return out


def main(run_dirs):
    for run_dir in run_dirs:
        (path,) = [p for p in glob.glob(os.path.join(
            run_dir, 'compiler_baselines', '*.json')) if not p.endswith('.full.json')]
        rec = json.load(open(path))
        cfg = rec['eval_config']  # compiler_layout_baselines.py ran on all.yml
        print(f'### {run_dir}', flush=True)

        layouts = {t: tuple(rec['tools'][t]['layout'])
                   for t in LAYOUT_CONDITIONS if t != 'oracle'}
        # the oracle is the arg-max of the 300-sample sweep, as in the paper
        oracle_vm = result_path(run_dir, cfg, 'oracle_layout')
        layouts['oracle'] = tuple(json.load(open(oracle_vm))['layout'])

        measured, table = {}, {}
        for cond in LAYOUT_CONDITIONS:
            layout = layouts[cond]
            if layout not in measured:
                arg = '[' + ','.join(map(str, layout)) + ']'
                measured[layout] = run_eval(
                    run_dir, cfg, f'local_{cond}_layout',
                    [f'qiskit.initial_layout={arg}'])
            table[cond] = measured[layout]
        # eval.py inserts .qiskit_layout before the suffix when the flag is set
        for level in (2, 3):
            table[f'qiskit_l{level}_free'] = run_eval(
                run_dir, cfg, f'local_l{level}',
                ['qiskit.ignore_ckpt_layout=True',
                 f'qiskit.optimization_level={level}'],
                out_suffix=f'qiskit_layout.local_l{level}')

        merged = {}
        for cond, p in table.items():
            r = json.load(open(p))
            merged[cond] = {'layout': r['layout'],
                            'accuracy_percent': r['accuracy_percent'],
                            'n_samples': r['n_samples'],
                            'optimization_level': r['optimization_level'],
                            'source': os.path.basename(p)}
        out = path.replace('.json', '.full.json')
        json.dump(merged, open(out, 'w'), indent=1)
        for cond, m in merged.items():
            print(f"  {cond:15s} {str(m['layout']):14s} {m['accuracy_percent']:.2f}%  "
                  f"(n={m['n_samples']}, opt{m['optimization_level']})", flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])

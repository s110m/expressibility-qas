"""Print the subset of `all_eval_commands_pure_m3.txt` that still needs running.

Why this exists
---------------
The command file was corrected after part of it had already been run on the
server (26 of the original 27 commands survived the correction unchanged), so
"just run the whole file again" wastes hours re-measuring results that are
already on disk and correct.

A result counts as DONE only if all of these hold:
  * `<run-dir>/eval_results/<name>.json` exists,
  * it has `n_samples == 4157` (the full split -- a leftover 300-sample file
    under the same name does not count),
  * `model_size == 192` (a 484 means a hybrid config leaked in), and
  * its recorded `layout` equals the layout the command pins via
    `qiskit.initial_layout`, when the command pins one.

That last check is the important one: quito s102's 300-sample
`exhaustive_layout.json` recorded the checkpoint layout rather than the
success-rate pick, i.e. it was produced by a command that silently did not
apply its override. Existence alone would have called that file done.

Usage
-----
    python examples/print_missing_all_evals.py                  # missing only
    python examples/print_missing_all_evals.py --show-done      # and why each is done
    python examples/print_missing_all_evals.py > todo.sh        # then: sh todo.sh

Run from the repo root, on whichever machine holds the results.
"""
import argparse
import ast
import json
import os
import re

CMD_FILE = 'examples/all_eval_commands_pure_m3.txt'
EXPECTED_SAMPLES = 4157
EXPECTED_MODEL_SIZE = 192


def result_path(cmd):
    """Mirror eval.py:168-178's eval_config_dir construction."""
    config = cmd.split()[2]
    run_dir = re.search(r'--run-dir=(\S+)', cmd).group(1)
    name = (config.replace('/', '.').replace('examples.', '')
            .replace('.yml', '').replace('configs.', ''))
    if re.search(r'qiskit\.ignore_ckpt_layout=True', cmd):
        name += '.qiskit_layout'
    suffix = re.search(r'--result-suffix (\S+)', cmd)
    if suffix:
        name += f'.{suffix.group(1)}'
    return os.path.join(run_dir, 'eval_results', f'{name}.json')


def wanted_layout(cmd):
    m = re.search(r'qiskit\.initial_layout=(\[[\d,\s]+\])', cmd)
    return ast.literal_eval(m.group(1)) if m else None


def status(cmd):
    """Return (done, reason)."""
    path = result_path(cmd)
    if not os.path.isfile(path):
        return False, 'no result file'
    try:
        res = json.load(open(path))
    except (json.JSONDecodeError, OSError) as exc:
        return False, f'unreadable ({exc.__class__.__name__})'

    if res.get('n_samples') != EXPECTED_SAMPLES:
        return False, f"n_samples {res.get('n_samples')} != {EXPECTED_SAMPLES}"
    if res.get('model_size') != EXPECTED_MODEL_SIZE:
        return False, (f"model_size {res.get('model_size')} != "
                       f"{EXPECTED_MODEL_SIZE} -- hybrid config leaked in")

    want = wanted_layout(cmd)
    if want is not None and res.get('layout') != want:
        return False, f"layout {res.get('layout')} != requested {want}"

    return True, f"acc {res.get('accuracy'):.4f}, layout {res.get('layout')}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--commands', default=CMD_FILE)
    parser.add_argument('--show-done', action='store_true',
                        help='also list completed commands, as comments')
    args = parser.parse_args()

    commands = [l.strip() for l in open(args.commands)
                if l.startswith('python')]
    missing = []
    for cmd in commands:
        done, reason = status(cmd)
        if done:
            if args.show_done:
                print(f'# DONE  {result_path(cmd)}  ({reason})')
        else:
            missing.append((cmd, reason))

    print(f'# {len(missing)} of {len(commands)} still to run')
    for cmd, reason in missing:
        print(f'# -> {reason}')
        print(cmd)


if __name__ == '__main__':
    main()

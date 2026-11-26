"""Pre-flight check for a SuperCircuit training run. Read-only; writes nothing.

Resolves the config exactly as train.py does (`configs.load(..., recursive=True)`
followed by the same CLI override and lr-eval steps), derives the same run
directory, then builds the model and pushes one real validation batch through it
to confirm what the encoder actually receives.

The last part is the point. A missing `model.arch.down_sample_kernel_size` does
not raise: SuperQFCModel0 skips pooling when the key is absent, and
GeneralEncoder slices `x[:, info["input_idx"]]` per gate without ever reading
`x.shape[1]`. The circuit is then handed the top-left 16 pixels of the unpooled
image -- eight of which are constant across the batch -- and trains to
completion at chance accuracy. That cost three SuperCircuits and eighteen
searches before it was noticed. See CLAUDE.md, "The `_nopool` quarantine".

Usage:
    python examples/preflight_supercircuit.py <config.yml> [debug.seed=42 ...]

Exits 0 if every check passes, 1 otherwise.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore")

import argparse
import sys

import torch
from torchpack.utils.config import configs

from core import builder

EXPECTED = [
    # (dotted key, expected value, section)
    ('model.arch.down_sample_kernel_size', 6, 'the bug that caused all this'),
    ('model.name', 'super_qfc0', 'design space'),
    ('model.arch.q_layer_name', 'u3cu3_s0', 'design space'),
    ('model.arch.encoder_op_list_name', '4x4_ryzxy', 'design space'),
    ('model.arch.n_blocks', 8, 'design space'),
    ('model.arch.n_layers_per_block', 2, 'design space'),
    ('model.arch.n_front_share_blocks', 1, 'design space'),
    ('model.arch.n_front_share_wires', 1, 'design space'),
    ('model.arch.n_front_share_ops', 1, 'design space'),
    ('model.arch.hybrid', None, 'design space'),
    ('model.sampler.strategy.name', 'ldiff_blkexpand', 'QuantumNAS sampling (paper Sec. VI)'),
    ('model.sampler.strategy.n_diffs', 7, 'QuantumNAS sampling (paper Sec. VI)'),
    ('model.sampler.strategy.n_chunks', 3, 'QuantumNAS sampling (paper Sec. VI)'),
    ('model.sampler.strategy.n_stages', 40, 'QuantumNAS sampling (paper Sec. VI)'),
    ('trainer.name', 'super_q_trainer', 'training recipe (paper Sec. VI)'),
    ('trainer.add_noise', False, 'training recipe (paper Sec. VI)'),
    ('run.n_epochs', 200, 'training recipe (paper Sec. VI)'),
    ('run.n_warm_epochs', 30, 'training recipe (paper Sec. VI)'),
    ('run.bsz', 256, 'training recipe (paper Sec. VI)'),
    ('optimizer.name', 'adam', 'training recipe (paper Sec. VI)'),
    ('optimizer.lr', 0.005, 'training recipe (paper Sec. VI)'),
    ('optimizer.weight_decay', 0.0001, 'training recipe (paper Sec. VI)'),
    ('scheduler.name', 'cosine_warm', 'training recipe (paper Sec. VI)'),
    ('criterion.name', 'nll', 'training recipe (paper Sec. VI)'),
]


def dotted(cfg, key):
    node = cfg
    for part in key.split('.'):
        if node is None:
            return None
        node = node.get(part) if hasattr(node, 'get') else getattr(node, part, None)
    return node


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)
    # train.py:126 -- the config stores lr as the string '5e-3'
    if isinstance(configs.optimizer.lr, str):
        configs.optimizer.lr = eval(configs.optimizer.lr)

    run_dir = 'runs/' + args.config.replace('/', '.').replace('examples.', '') \
                            .replace('.yml', '').replace('configs.', '')
    if configs.debug.set_seed:
        run_dir += f'_seed{configs.debug.seed}'

    print(f'config : {args.config}')
    print(f'run dir: {run_dir}')
    if os.path.isdir(run_dir):
        print('         !! ALREADY EXISTS -- training would overwrite it. '
              'Rename it first.')
    else:
        print('         (free)')

    ok = True
    section = None
    for key, want, sec in EXPECTED:
        if sec != section:
            print(f'\n--- {sec} ---')
            section = sec
        got = dotted(configs, key)
        good = got == want
        ok = ok and good
        print(f'  [{"OK " if good else "FAIL"}] {key:38} {got!r:<26} expected {want!r}')

    print('\n--- end-to-end: what the encoder actually receives ---')
    dataset = builder.make_dataset()
    dataflow = torch.utils.data.DataLoader(dataset['valid'], batch_size=512,
                                           shuffle=False, num_workers=0)
    image = next(iter(dataflow))['image']

    model = builder.make_model()
    model.eval()
    n_blocks = configs.model.arch.n_blocks
    model.set_sample_arch([4] * (n_blocks * configs.model.arch.n_layers_per_block) + [n_blocks])

    seen = {}
    handle = model.encoder.register_forward_pre_hook(
        lambda m, a: seen.__setitem__('x', a[1].detach()))
    with torch.no_grad():
        out = model(image)
    handle.remove()

    x = seen['x']
    per = x.std(0)
    n_zero = int((per < 1e-7).sum())
    n_params = sum(p.numel() for p in model.q_layer.parameters())

    for label, got, want in [
        ('encoder input width', tuple(x.shape[1:]), (len(model.encoder.func_list),)),
        ('encoded angles with zero variance', n_zero, 0),
        ('q_layer params (== "Model Size")', n_params, 192),
    ]:
        good = got == want
        ok = ok and good
        print(f'  [{"OK " if good else "FAIL"}] {label:38} {got!r:<26} expected {want!r}')

    print(f'  [    ] per-feature std of encoded angles  '
          + ' '.join(f'{v:.2f}' for v in per.tolist()))
    print(f'  [    ] logit std across batch             '
          + ' '.join(f'{v:.3f}' for v in out.std(0).tolist())
          + '   (~0.01 each means a dead input)')

    print('\n' + '=' * 68)
    print('  PRE-FLIGHT: ' + ('ALL CHECKS PASSED -- safe to train'
                              if ok else 'FAILURES ABOVE -- DO NOT RUN'))
    print('=' * 68)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

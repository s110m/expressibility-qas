# Expressibility Is Not Enough — code and results

Code, configuration files and trained checkpoints for

> S. Mousavi, S. Mousavi, and M. Daneshtalab, "Expressibility Is Not Enough:
> Circuit Size and Calibration-Aware Qubit Mapping in Training-Free Quantum
> Architecture Search," submitted to *IEEE Access*.

The code builds on [TorchQuantum](https://github.com/mit-han-lab/torchquantum)
and its QuantumNAS implementation (MIT License, see `LICENSE`). Noise is
simulated from the calibration data of Qiskit fake backends (`FakeAthensV2`,
`FakeBelemV2`, `FakeQuitoV2`, and `FakeLagosV2` and `FakeWashingtonV2` for the
scaling study); no job is sent to real hardware.

## Branches

The experiments were run on two branches, kept here as they ran.

| Branch | Paper sections | Contents |
|---|---|---|
| `es_expressibility_score` | VI-A to VI-F, VI-N | SuperCircuit training, architecture search scored by noisy loss or by expressibility (evolutionary and random search), retraining from scratch, the size-controlled proxy comparison, and the 120-layout sweeps with selection by estimated success rate |
| `es_expressibility_mapping` | VI-G to VI-M, VII-E | Joint NSGA-II search over architecture and layout, the success-rate ablations and drift study, the three-objective search, the genome ablation, the compiler baselines, the Hamiltonian (TFIM) task, and width and device scaling |

## Layout

- `examples/` — all code. Run scripts from the repository root as
  `python examples/<script>.py ...`; they import the TorchQuantum copy in
  `examples/torchquantum/`.
- `examples/configs/` — YAML configs, loaded recursively: each leaf file
  inherits every `default.yml` above it.
- `runs/` — training runs (SuperCircuit and retrained SubCircuits), with
  checkpoints, TensorBoard scalars and evaluation results.
- `es_runs/` — search runs, each with its `checkpoints/best_solution.pt`
  (selected architecture and layout).
- Branch `es_expressibility_mapping` also has `genome_ablation/`,
  `success_rate_ablation/`, `scaling_enumeration/` and `vqe_runs/`, the outputs
  of the corresponding experiments.

Every run directory contains `metainfo/configs.yaml`, the complete resolved
configuration it ran with, including its seed. That file is the authoritative
record of a run.

Search runs encode the search arm in the hundreds digit of the seed:
`1xx` noisy loss + evolutionary, `2xx` expressibility + evolutionary,
`3xx` expressibility + random search, `4xx` noisy loss + random search.

## Installation

```bash
pip install -r requirements.txt
pip install qiskit-aer qiskit-ibm-runtime pymoo
```

## Pipeline (example: FakeAthensV2, expressibility search, seed 200)

```bash
# 1. Train the SuperCircuit
python examples/train.py examples/configs/mnist/four0123/train/super/u3cu3_s0/ldiff_blkexpand/blk8s1.1.1_diff7_chu3_sta40.yml

# 2. Search, scoring candidates by expressibility
python examples/run_es.py examples/configs/mnist/four0123/es/athens/noise/opt2/setting0/u3cu3_s0/ldiff_blkexpand/blk8s1.1.1_diff7_chu3_sta40.yml \
    --run-dir=runs/mnist.four0123.train.super.u3cu3_s0.ldiff_blkexpand.blk8s1.1.1_diff7_chu3_sta40_seed42/ \
    es.score_mode=expressibility es.expressibility.n_pairs=150 es.expressibility.nbins=75 \
    es.expressibility.seed=1234 qiskit.use_qiskit=False debug.seed=200

# 3. Retrain the selected architecture from scratch
python examples/train.py examples/configs/mnist/four0123/train/searched/scratch/athens/noise/opt2/setting0/u3cu3_s0/ldiff_blkexpand/blk8s1.1.1_diff7_chu3_sta40.yml \
    --ckpt-dir=es_runs/mnist.four0123.es.athens.noise.opt2.setting0.u3cu3_s0.ldiff_blkexpand.blk8s1.1.1_diff7_chu3_sta40_seed200/ \
    --run-dir-suffix=from-search200 debug.seed=42

# 4. Evaluate under the device's noise model (full test split)
python examples/eval.py examples/configs/mnist/four0123/eval/athens/noise/opt2/all.yml \
    --run-dir=runs/mnist.four0123.train.searched.scratch.athens.noise.opt2.setting0.u3cu3_s0.ldiff_blkexpand.blk8s1.1.1_diff7_chu3_sta40_seed42_from-search200/

# 5. Evaluate the same checkpoint under all 120 layouts
python examples/eval_layout_sweep.py examples/configs/mnist/four0123/eval/athens/noise/opt2/300.yml \
    --run-dir=runs/mnist.four0123.train.searched.scratch.athens.noise.opt2.setting0.u3cu3_s0.ldiff_blkexpand.blk8s1.1.1_diff7_chu3_sta40_seed42_from-search200/
```

Any extra `key=value` argument overrides the corresponding config entry. Run
directories are derived from the config path and seed, so rerunning a config
overwrites its directory: rename an existing run first. Layouts are selected by
estimated success rate with `examples/select_layout_by_success_rate.py`; the
joint search is `examples/run_nsga2.py` (branch `es_expressibility_mapping`).

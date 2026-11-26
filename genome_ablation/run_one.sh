#!/bin/bash
s=$1; b=$2; seed=$3
cd /c/Neural-Networks-Files/tq_mapping_wt
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python examples/nsga2_genome_ablation.py examples/configs/mnist/four0123/es/$b/noise/opt2/setting0/u3cu3_s0/ldiff_blkexpand/nsga2/blk8s1.1.1_diff7_chu3_sta40.yml --run-dir /c/Neural-Networks-Files/torchquantum/runs/mnist.four0123.train.super.u3cu3_s0.ldiff_blkexpand.blk8s1.1.1_diff7_chu3_sta40_seed42_nopool --setting $s debug.seed=$seed ckpt.name=checkpoints/step-18400.pt > genome_ablation/logs/${s}_${b}_${seed}.log 2>&1
tail -1 genome_ablation/logs/${s}_${b}_${seed}.log

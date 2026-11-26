#!/bin/bash
# From-scratch retraining (train seed 42) of one canonical ablation cell.
# Finished only at step-18400.pt (epoch 200 x 92 steps); an unfinished run is
# restarted from scratch.
s=$1; dev=$2; seed=$3
cd /c/Neural-Networks-Files/tq_mapping_wt
declare -A BK=([athens]=FakeAthensV2 [quito]=FakeQuitoV2 [belem]=FakeBelemV2)
b=${BK[$dev]}
leaf=examples/configs/mnist/four0123/train/searched/scratch/$dev/noise/opt2/setting0/u3cu3_s0/ldiff_blkexpand/ga_${s/./}_s$seed.yml
run=runs/$(echo ${leaf#examples/configs/} | sed 's#/#.#g; s#\.yml$##')_seed42
[ -f $run/checkpoints/step-18400.pt ] && { echo "trained $s $dev $seed"; exit 0; }
rm -rf $run
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python examples/train.py $leaf --ckpt-dir=genome_ablation/$s/${b}_s$seed debug.seed=42 run.workers_per_gpu=0 > genome_ablation/logs/train_${s}_${dev}_${seed}.log 2>&1
[ -f $run/checkpoints/step-18400.pt ] && echo "trained $s $dev $seed" || echo "TRAIN FAILED $s $dev $seed"

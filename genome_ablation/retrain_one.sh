#!/bin/bash
# Retrain one genome-ablation solution from scratch (train seed 42) and evaluate
# it on the full split at the searched layout and at the enumerated layout.
s=$1; dev=$2; seed=$3
cd /c/Neural-Networks-Files/tq_mapping_wt
declare -A BK=([athens]=FakeAthensV2 [quito]=FakeQuitoV2 [belem]=FakeBelemV2)
b=${BK[$dev]}
J=genome_ablation/$s/${b}_s$seed.json
[ -f $J ] || { echo "no search result $J"; exit 1; }
leaf=examples/configs/mnist/four0123/train/searched/scratch/$dev/noise/opt2/setting0/u3cu3_s0/ldiff_blkexpand/ga_${s/./}_s$seed.yml
run=runs/$(echo ${leaf#examples/configs/} | sed 's#/#.#g; s#\.yml$##')_seed42
log=genome_ablation/logs/train_${s}_${dev}_${seed}.log
if [ ! -f $run/checkpoints/max-acc-valid.pt ]; then
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python examples/train.py $leaf --ckpt-dir=genome_ablation/$s/${b}_s$seed debug.seed=42 run.workers_per_gpu=0 > $log 2>&1 || { echo "train failed $s $dev $seed"; exit 1; }
fi
enum=$(python -c "import json;print(json.load(open('$J'))['deployed']['enumerated_layout'])" | tr -d ' ')
cfg=examples/configs/mnist/four0123/eval/$dev/noise/opt2/all.yml
for pair in "ga_joint|" "ga_enum|qiskit.initial_layout=$enum"; do
  suf=${pair%%|*}; ov=${pair#*|}
  ls $run/eval_results/*.$suf.json >/dev/null 2>&1 && continue
  OMP_NUM_THREADS=2 python examples/eval.py $cfg --run-dir=$run --result-suffix=$suf --jobs=4 run.workers_per_gpu=0 $ov >> $log 2>&1 || echo "eval failed $s $dev $seed $suf"
done
python -c "
import json,glob
r={s:json.load(open(glob.glob('$run/eval_results/*.'+s+'.json')[0]))['accuracy_percent'] for s in ('ga_joint','ga_enum')}
print('$s $dev $seed', r)"

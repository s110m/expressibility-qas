#!/bin/bash
# Full-split noisy evaluation of one ablation cell on its own device, at its
# deployed layout (gj_) and its enumerated layout (ge_), using the checkpoint
# of the canonical cell that trained the same architecture.
s=$1; dev=$2; seed=$3; cs=$4; cdev=$5
cd /c/Neural-Networks-Files/tq_mapping_wt
declare -A BK=([athens]=FakeAthensV2 [quito]=FakeQuitoV2 [belem]=FakeBelemV2)
J=genome_ablation/$s/${BK[$dev]}_s$seed.json
leaf=examples/configs/mnist/four0123/train/searched/scratch/$cdev/noise/opt2/setting0/u3cu3_s0/ldiff_blkexpand/ga_${cs/./}_s$seed.yml
run=runs/$(echo ${leaf#examples/configs/} | sed 's#/#.#g; s#\.yml$##')_seed42
[ -f $run/checkpoints/step-18400.pt ] || { echo "NOT TRAINED $cs $cdev $seed"; exit 1; }
dep=$(python -c "import json;print(json.load(open('$J'))['deployed']['solution']['layout'])" | tr -d ' ')
enum=$(python -c "import json;print(json.load(open('$J'))['deployed']['enumerated_layout'])" | tr -d ' ')
cfg=examples/configs/mnist/four0123/eval/$dev/noise/opt2/all.yml
tag=${s/./}_${dev}_s$seed
log=genome_ablation/logs/eval_${tag}.log
for pair in "gj_$tag|$dep" "ge_$tag|$enum"; do
  suf=${pair%%|*}; lay=${pair#*|}
  ls $run/eval_results/*.$suf.json >/dev/null 2>&1 && continue
  OMP_NUM_THREADS=2 python examples/eval.py $cfg --run-dir=$run --result-suffix=$suf --jobs=4 run.workers_per_gpu=0 qiskit.initial_layout=$lay >> $log 2>&1 || echo "EVAL FAILED $suf"
done
python -c "
import json,glob
r={k:json.load(open(glob.glob('$run/eval_results/*.'+k+'_$tag.json')[0]))['accuracy_percent'] for k in ('gj','ge')}
print('$s $dev $seed', r)"

#!/bin/bash
# Retrain the p0.4 / split / staged solutions as their searches finish, <=3 at a time.
cd /c/Neural-Networks-Files/tq_mapping_wt
declare -A BK=([athens]=FakeAthensV2 [quito]=FakeQuitoV2 [belem]=FakeBelemV2)
jobs=()
for seed in 100 101 102; do for s in staged split p0.4; do for dev in quito athens belem; do jobs+=("$s $dev $seed"); done; done; done
for j in "${jobs[@]}"; do
  set -- $j
  until [ -f genome_ablation/$1/${BK[$2]}_s$3.json ]; do sleep 60; done
  while [ $(jobs -rp | wc -l) -ge 3 ]; do sleep 30; done
  genome_ablation/retrain_one.sh $1 $2 $3 >> genome_ablation/retrain.log 2>&1 &
done
wait
echo "ALL RETRAINS DONE" >> genome_ablation/retrain.log

#!/bin/bash
# Retrain + evaluate the three retrained settings for the given search seeds.
cd /c/Neural-Networks-Files/tq_mapping_wt
P=${P:-6}
python genome_ablation/plan3.py "$@"
tag=$(echo "$@" | tr " " _)
xargs -P $P -L 1 genome_ablation/train_cell_staggered.sh < genome_ablation/train3_jobs_$tag.txt
xargs -P 3 -L 1 genome_ablation/eval_cell.sh < genome_ablation/eval3_jobs_$tag.txt
echo "run3 finished $*"

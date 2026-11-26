#!/usr/bin/env bash
# =============================================================================
#  M4 -- three-objective NSGA-II, two families
#  branch: es_expressibility_mapping
#
#  TWO FAMILIES, ATTACKING DIFFERENT HALVES OF THE PROBLEM
#
#    esd  [expressibility, success_rate, depth]
#         Depth is an ARCHITECTURE counterweight. Expressibility improves
#         monotonically with depth while noisy accuracy peaks at interior
#         depth, so Expr alone has no brake on over-deepening. M3 selected a
#         mean of 7.7 blocks on Belem, at the very top of the design space.
#
#    ess  [expressibility, success_rate, swap_count]
#         Routing overhead is a MAPPING signal -- the two-qubit-gate delta the
#         router adds for THIS layout, decoupled from both the architecture's
#         gate count and the calibration tables. It targets the finding that
#         M3's returned layout ranks 59.8 of 120 against a 60.5 null.
#
#  TWO DEPLOYMENT RULES PER FAMILY
#    arm A "best"  argmin expressibility                 -- the paper's rule
#    arm B "cap"   argmin expressibility, 3rd obj <= p50 -- rule restricted to
#                                                          the better half
#
#  Why both: argmin expressibility reads ONLY Expr, so it ignores the new
#  objective at selection time, and since Expr grows with depth it tends to pick
#  the deepest member of whatever front it gets. Arm A alone could show zero
#  effect by construction. A vs M3 isolates the objective; B vs A isolates the
#  rule. Confirmed on a smoke front: argmin Expr took depth 180 / success_rate
#  0.018, the same front capped took depth 81 / 0.121 for 0.010 of Expr.
#
#  BASELINE. M3 = [expressibility, success_rate] at the SAME seeds 100/101/102,
#  mean 76.60% full-split. Nothing here overwrites it: M3 lives under
#  es_runs/*.nsga2.* and runs/*.nsga2_best_s*, M4 under *.nsga2_esd|ess.* and
#  runs/*.nsga2esd_*|nsga2ess_*. Both run_nsga2.py and train.py derive their
#  output directory from the config path. Verified: 72 targets, 0 collisions,
#  and the preflight below re-checks it wherever this runs.
#
#  READ LAYOUT RANK FOR THE ess FAMILY. Whether a mapping signal helps is a
#  claim about the SELECTED LAYOUT's rank among all 120, which needs the sweep
#  stage, not the single noisy eval. The sweep is the last stage on purpose:
#  if it does not finish, everything else is still complete.
#
#  NOTE ON THE SUPERNET. --run-dir is the pre-pooling-fix supernet (blob
#  552bd1a4, superacc/largest_valid 28.92%). Deliberate and harmless:
#  run_nsga2.py:192-199 loads it for RandomLayer op-list STRUCTURE only and
#  never reads a trained weight. Same supernet as M3, which is what keeps the
#  comparison paired. Do not quote supernet accuracy from these runs.
#
#  USAGE
#      bash run_m4_three_objective.sh                # everything, -j 3
#      bash run_m4_three_objective.sh -j 8           # 8 jobs at a time
#      bash run_m4_three_objective.sh -f esd         # one family only
#      bash run_m4_three_objective.sh -s "search export retrain eval"
#      bash run_m4_three_objective.sh -n             # dry run
#
#  Every stage is resume-safe: a step whose output exists is skipped, so
#  re-running after a crash continues rather than restarting.
#
#  TIMING, measured from the existing M3 runs in this repo
#      search   67-88 min   x18 = 22 h serial
#      export   ~1 s        x36 = negligible
#      retrain  36-57 min   x36 = 29 h serial
#      eval     1.7-4.5 min x36 = 2 h serial
#      sweep    ~85 min     x18 = 25 h serial      (LAST, droppable)
#      TOTAL  ~78 h serial | ~13 h at -j 6 | ~9 h at -j 9 | ~6.5 h at -j 12
#      Without the sweep stage: ~53 h serial | ~9 h at -j 6 | ~6 h at -j 9
# =============================================================================
set -u -o pipefail

JOBS=3
STAGES="search export retrain eval sweep summary"
FAMILIES="esd ess"
DRY=0

while getopts "j:s:f:nh" opt; do
    case "$opt" in
        j) JOBS="$OPTARG" ;;
        s) STAGES="$OPTARG" ;;
        f) FAMILIES="$OPTARG" ;;
        n) DRY=1 ;;
        h) sed -n '2,68p' "$0"; exit 0 ;;
        *) echo "see -h"; exit 2 ;;
    esac
done

DEVICES="athens quito belem"
SEEDS="100 101 102"
SUPERNET="runs/mnist.four0123.train.super.u3cu3_s0.ldiff_blkexpand.blk8s1.1.1_diff7_chu3_sta40_seed42"
LOGDIR="m4_logs"
mkdir -p "$LOGDIR"

# third objective, per family -- also the metric the cap is applied to
cap_metric() { case "$1" in esd) echo depth ;; ess) echo swap_count ;; esac; }

es_dir()     { echo "es_runs/mnist.four0123.es.$2.noise.opt2.setting0.u3cu3_s0.ldiff_blkexpand.nsga2_$1.blk8s1.1.1_diff7_chu3_sta40_seed$3"; }
run_dir()    { echo "runs/mnist.four0123.train.searched.scratch.$2.noise.opt2.setting0.u3cu3_s0.ldiff_blkexpand.nsga2$1_$4_s$3_seed42"; }
search_cfg() { echo "examples/configs/mnist/four0123/es/$2/noise/opt2/setting0/u3cu3_s0/ldiff_blkexpand/nsga2_$1/blk8s1.1.1_diff7_chu3_sta40.yml"; }
train_cfg()  { echo "examples/configs/mnist/four0123/train/searched/scratch/$2/noise/opt2/setting0/u3cu3_s0/ldiff_blkexpand/nsga2$1_$4_s$3.yml"; }
eval_cfg()   { echo "examples/configs/mnist/four0123/eval/$1/noise/opt2/all.yml"; }
sweep_cfg()  { echo "examples/configs/mnist/four0123/eval/$1/noise/opt2/300.yml"; }

# ---- tiny job pool ----------------------------------------------------------
QUEUE_CMD=(); QUEUE_LOG=(); QUEUE_TAG=()
enqueue() { QUEUE_TAG+=("$1"); QUEUE_LOG+=("$2"); QUEUE_CMD+=("$3"); }

drain() {
    local stage="$1" n="${#QUEUE_CMD[@]}" i=0 running=0 fails=0
    [ "$n" -eq 0 ] && { echo "[$stage] nothing to do (all outputs present)"; return 0; }
    echo "[$stage] $n job(s), $JOBS at a time"
    if [ "$DRY" -eq 1 ]; then
        for ((i=0; i<n; i++)); do echo "  ${QUEUE_CMD[$i]}"; done
        QUEUE_CMD=(); QUEUE_LOG=(); QUEUE_TAG=(); return 0
    fi
    for ((i=0; i<n; i++)); do
        echo "  -> ${QUEUE_TAG[$i]}  (log: ${QUEUE_LOG[$i]})"
        ( eval "${QUEUE_CMD[$i]}" ) > "${QUEUE_LOG[$i]}" 2>&1 &
        running=$((running+1))
        if [ "$running" -ge "$JOBS" ]; then wait -n 2>/dev/null || wait; running=$((running-1)); fi
    done
    wait
    for ((i=0; i<n; i++)); do
        if grep -qiE "^(Traceback|.*Error:)" "${QUEUE_LOG[$i]}" 2>/dev/null; then
            echo "  !! possible failure in ${QUEUE_TAG[$i]} -- check ${QUEUE_LOG[$i]}"
            fails=$((fails+1))
        fi
    done
    echo "[$stage] done${fails:+ ($fails suspicious log(s))}"
    QUEUE_CMD=(); QUEUE_LOG=(); QUEUE_TAG=()
}

has() { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }

# ---- preflight --------------------------------------------------------------
echo "=== preflight ==="
[ -d "$SUPERNET/checkpoints" ] || { echo "MISSING supernet: $SUPERNET"; exit 1; }
missing=0
for f in $FAMILIES; do for d in $DEVICES; do
    [ -f "$(search_cfg "$f" "$d")" ] || { echo "MISSING $(search_cfg "$f" "$d")"; missing=1; }
    for s in $SEEDS; do for arm in best cap; do
        [ -f "$(train_cfg "$f" "$d" "$s" "$arm")" ] || { echo "MISSING $(train_cfg "$f" "$d" "$s" "$arm")"; missing=1; }
    done; done
done; done
[ "$missing" -eq 0 ] || exit 1
echo "supernet and all configs present for families: $FAMILIES"

# No M4 target may be a directory this script did not create. M3 lives under
# es_runs/*.nsga2.* and runs/*.nsga2_best_s* and is never named here.
collisions=0
for f in $FAMILIES; do for d in $DEVICES; do for s in $SEEDS; do
    for t in "$(es_dir "$f" "$d" "$s")" "$(es_dir "$f" "$d" "$s")_cap" \
             "$(run_dir "$f" "$d" "$s" best)" "$(run_dir "$f" "$d" "$s" cap)"; do
        [ -e "$t" ] && { echo "  pre-existing: $t"; collisions=$((collisions+1)); }
    done
done; done; done
if [ "$collisions" -gt 0 ]; then
    echo "$collisions M4 output director(ies) already exist (listed above)."
    echo "Finished steps are SKIPPED, never overwritten. Delete any left over"
    echo "from an interrupted run. Nothing under es_runs/*.nsga2.* or"
    echo "runs/*.nsga2_best_s* (M3) is ever touched by this script."
    if [ "$DRY" -eq 0 ]; then
        printf "Continue? [y/N] "; read -r ans
        case "$ans" in [yY]*) ;; *) echo "aborted"; exit 1 ;; esac
    fi
else
    echo "no output directory collisions: all M4 targets are new"
fi
echo "branch: $(git rev-parse --abbrev-ref HEAD)"
echo

# ---- STAGE 1: search --------------------------------------------------------
if has search; then
    for f in $FAMILIES; do for d in $DEVICES; do for s in $SEEDS; do
        [ -f "$(es_dir "$f" "$d" "$s")/checkpoints/pareto_front.pt" ] && continue
        enqueue "search $f $d seed$s" "$LOGDIR/search_${f}_${d}_$s.log" \
          "python examples/run_nsga2.py $(search_cfg "$f" "$d") --run-dir=$SUPERNET debug.seed=$s"
    done; done; done
    drain search
fi

# ---- STAGE 2: export both deployment rules ----------------------------------
if has export; then
    for f in $FAMILIES; do
        m="$(cap_metric "$f")"
        for d in $DEVICES; do for s in $SEEDS; do
            src="$(es_dir "$f" "$d" "$s")/checkpoints/pareto_front.pt"
            [ -f "$src" ] || { echo "  skip export $f $d/$s -- no front yet"; continue; }
            [ -f "$(es_dir "$f" "$d" "$s")/checkpoints/best_solution.pt" ] || \
              enqueue "export-A $f $d seed$s" "$LOGDIR/export_best_${f}_${d}_$s.log" \
                "python examples/export_pareto_solution.py $src --select expressibility"
            [ -f "$(es_dir "$f" "$d" "$s")_cap/checkpoints/best_solution.pt" ] || \
              enqueue "export-B $f $d seed$s" "$LOGDIR/export_cap_${f}_${d}_$s.log" \
                "python examples/export_pareto_solution.py $src --select expressibility --cap $m=p50 --out-dir $(es_dir "$f" "$d" "$s")_cap"
        done; done
    done
    drain export
fi

# ---- STAGE 3: retrain from scratch ------------------------------------------
# Confirm in each log: "Model Size: 192" and "Loading the solution".
if has retrain; then
    for f in $FAMILIES; do for d in $DEVICES; do for s in $SEEDS; do
        for arm in best cap; do
            ck="$(es_dir "$f" "$d" "$s")"; [ "$arm" = cap ] && ck="${ck}_cap"
            [ -f "$ck/checkpoints/best_solution.pt" ] || { echo "  skip retrain $f/$arm $d/$s -- no solution"; continue; }
            [ -f "$(run_dir "$f" "$d" "$s" "$arm")/checkpoints/max-acc-valid.pt" ] && continue
            enqueue "retrain $f/$arm $d seed$s" "$LOGDIR/retrain_${f}_${arm}_${d}_$s.log" \
              "python examples/train.py $(train_cfg "$f" "$d" "$s" "$arm") --ckpt-dir=$ck debug.seed=42"
        done
    done; done; done
    drain retrain
fi

# ---- STAGE 4: noisy eval, full split (n = 4157) -----------------------------
if has eval; then
    for f in $FAMILIES; do for d in $DEVICES; do for s in $SEEDS; do
        for arm in best cap; do
            out="$(run_dir "$f" "$d" "$s" "$arm")"
            [ -f "$out/checkpoints/max-acc-valid.pt" ] || { echo "  skip eval $f/$arm $d/$s -- not trained"; continue; }
            [ -f "$out/eval_results/mnist.four0123.eval.$d.noise.opt2.all.m4_layout.json" ] && continue
            enqueue "eval $f/$arm $d seed$s" "$LOGDIR/eval_${f}_${arm}_${d}_$s.log" \
              "python examples/eval.py $(eval_cfg "$d") --run-dir=$out --jobs 8 --result-suffix m4_layout ckpt.name=checkpoints/max-acc-valid.pt"
        done
    done; done; done
    drain eval
fi

# ---- STAGE 5: 120-layout sweep, n = 300 -------------------------------------
# This is what answers "did the mapping actually get better?" -- it gives the
# rank of the SELECTED layout among all 120, comparable to M3's 59.8 mean
# against the 60.5 uniform-draw null. Slowest stage; drop it with -s if short
# on time, the rest of the results stand without it.
if has sweep; then
    for f in $FAMILIES; do for d in $DEVICES; do for s in $SEEDS; do
        for arm in best cap; do
            out="$(run_dir "$f" "$d" "$s" "$arm")"
            [ -f "$out/checkpoints/max-acc-valid.pt" ] || continue
            [ -f "$out/layout_sweep/mnist.four0123.eval.$d.noise.opt2.300.summary.json" ] && continue
            enqueue "sweep $f/$arm $d seed$s" "$LOGDIR/sweep_${f}_${arm}_${d}_$s.log" \
              "python examples/eval_layout_sweep.py $(sweep_cfg "$d") --run-dir=$out --jobs 8 ckpt.name=checkpoints/max-acc-valid.pt"
        done
    done; done; done
    drain sweep
fi

# ---- STAGE 6: collect -------------------------------------------------------
if has summary; then
    echo "=== summary ==="
    python collect_m4_results.py | tee "$LOGDIR/summary.txt"
fi

echo
echo "All requested stages finished. Logs in $LOGDIR/."
echo "Read the table with:  python collect_m4_results.py"

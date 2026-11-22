#!/usr/bin/env bash

set -uo pipefail

GPU="${1:-${CUDA_VISIBLE_DEVICES:-0}}"
export CUDA_VISIBLE_DEVICES="$GPU"
SERIAL="${SERIAL:-0}"

PY=$(which python)
cd "$(dirname "$0")"
CK=../ckpts
LOGS=../results/logs

N_LANES=4
NCPU=$(nproc)

_hms() {
    printf '%dh%02dm%02ds' $(($1 / 3600)) $((($1 % 3600) / 60)) $(($1 % 60))
}

run_step() {
    local name="$1"; shift
    local log="$LOGS/${name}_${RUN_STAMP}.log"
    local t0 t1 rc dur
    t0=$(date +%s)

    echo "=== [$name] START $(date '+%F %T') ===" | tee "$log"
    echo "    cmd: $*" >>"$log"

    if [ "$SERIAL" = 1 ]; then
        "$@" 2>&1 | tee -a "$log"
        rc=${PIPESTATUS[0]}
    else
        "$@" >>"$log" 2>&1
        rc=$?
    fi

    t1=$(date +%s); dur=$((t1 - t0))
    if [ "$rc" -eq 0 ]; then
        echo "=== [$name] OK  elapsed $(_hms $dur) ===" | tee -a "$log"
        printf '%-32s %-8s %s\n' "$name" "OK" "$(_hms $dur)" >>"$SUMMARY_FILE"
    else
        echo "=== [$name] FAILED (exit $rc) elapsed $(_hms $dur)" \
             "-- continuing with next step ($log) ===" | tee -a "$log"
        printf '%-32s %-8s %s\n' "$name" "FAIL($rc)" "$(_hms $dur)" >>"$SUMMARY_FILE"
    fi
}

# --- lanes: steps within a lane run in order, lanes run concurrently --------
lane_depth() {
    run_step depth_mono \
      $PY eval_depth.py --weights $CK/mono_640x192 --model-name monodepth2
    run_step depth_monostereo \
      $PY eval_depth.py --weights $CK/mono+stereo_640x192 --model-name mono_stereo
}

lane_det_smoke() { run_step det_smoke $PY eval_det.py --resume --model smoke; }
lane_det_pgd()   { run_step det_pgd   $PY eval_det.py --resume --model pgd; }

lane_seg() {
    run_step seg_deeplabv3plus \
      $PY eval_seg.py --resume --dataset cityscapes \
      --config $CK/deeplabv3plus_r101-d8_4xb2-80k_cityscapes-512x1024.py \
      --ckpt   $CK/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_20200606_114143-068fcfe9.pth \
      --model-name deeplabv3plus_r101

    run_step seg_mask2former \
      $PY eval_seg.py --resume --dataset cityscapes \
      --config $CK/mask2former_swin-l-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024.py \
      --ckpt   $CK/mask2former_swin-l-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024_20221202_141901-28ad20f1.pth \
      --model-name mask2former_swinl

    run_step seg_kitti200 \
      $PY eval_seg.py --resume --dataset kitti \
      --config $CK/deeplabv3plus_r101-d8_4xb2-80k_cityscapes-512x1024.py \
      --ckpt   $CK/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_20200606_114143-068fcfe9.pth \
      --model-name deeplabv3plus_r101_kitti200
}

LANES=(lane_depth lane_det_smoke lane_det_pgd lane_seg)

mkdir -p "$LOGS"
exec 200>"$LOGS/.run_tasks.lock"
if ! flock -n 200; then
    echo "[run_tasks] another instance is already running" \
         "($LOGS/.run_tasks.lock) -- exiting" >&2
    exit 1
fi

RUN_STAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="$LOGS/.summary_$RUN_STAMP"
: >"$SUMMARY_FILE"
T0=$(date +%s)

if [ "$SERIAL" = 1 ]; then
    THREADS=$NCPU
else
    THREADS=$(( NCPU / N_LANES > 0 ? NCPU / N_LANES : 1 ))
fi
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
       OPENBLAS_NUM_THREADS=$THREADS NUMBA_NUM_THREADS=$THREADS

echo "###############################################################"
echo "# run_tasks  start $(date '+%F %T')  GPU=$CUDA_VISIBLE_DEVICES"
if [ "$SERIAL" = 1 ]; then
    echo "# mode: serial, $THREADS CPU threads"
else
    echo "# mode: ${#LANES[@]} parallel lanes, $THREADS CPU threads each" \
         "(${LANES[*]})"
fi
echo "# run stamp: $RUN_STAMP   logs: $LOGS/<step>_$RUN_STAMP.log"
echo "###############################################################"

$PY - <<'EOF'
import config as C, eval_det
for dg, sev in C.all_conditions():
    try:
        eval_det.build_cond_root(dg, sev)
    except FileNotFoundError as e:
        print(f"[run_tasks] cond root {dg}_{sev}: {e}")
EOF

if [ "$SERIAL" = 1 ]; then
    for lane in "${LANES[@]}"; do "$lane"; done
else
    # Ctrl-C / kill tears down every lane and its python child.
    trap 'trap - INT TERM; echo "[run_tasks] interrupted"; kill 0' INT TERM
    for lane in "${LANES[@]}"; do "$lane" & done
    wait
    trap - INT TERM
fi

DUR=$(( $(date +%s) - T0 ))
N_STEPS=$(wc -l <"$SUMMARY_FILE")
N_FAILED=$(grep -c FAIL "$SUMMARY_FILE")
echo
echo "###############################################################"
echo "# run_tasks finished $(date '+%F %T')  total $(_hms $DUR)"
echo "#---------------------------------------------------------------"
sed 's/^/# /' "$SUMMARY_FILE"
echo "#---------------------------------------------------------------"
if [ "$N_FAILED" -eq 0 ]; then
    echo "# all $N_STEPS steps succeeded"
else
    echo "# $N_FAILED/$N_STEPS steps FAILED -- see table above and"
    echo "# the matching $LOGS/<step>_$RUN_STAMP.log"
fi
echo "###############################################################"
rm -f "$SUMMARY_FILE"
exit "$N_FAILED"

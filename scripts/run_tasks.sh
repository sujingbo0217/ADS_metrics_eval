#!/usr/bin/env bash
# Usage:
#   ./run_tasks.sh
#   ./run_tasks.sh 1        # GPU 1
#   CUDA_VISIBLE_DEVICES=0 ./run_tasks.sh
set -uo pipefail

GPU="${1:-${CUDA_VISIBLE_DEVICES:-0}}"
export CUDA_VISIBLE_DEVICES="$GPU"

PY=$(which python)
cd "$(dirname "$0")"
CK=../ckpts
LOGS=../results/logs

SUMMARY=()
N_FAILED=0

_hms() {
    printf '%dh%02dm%02ds' $(($1 / 3600)) $((($1 % 3600) / 60)) $(($1 % 60))
}

run_step() {
    local name="$1"; shift
    local log="$LOGS/${name}_${RUN_STAMP}.log"
    local t0 t1 rc dur
    t0=$(date +%s)

    echo "=== [$name] START $(date '+%F %T') ===" | tee "$log"
    echo "    cmd: $*" | tee -a "$log"

    "$@" 2>&1 | tee -a "$log"
    rc=${PIPESTATUS[0]}

    t1=$(date +%s); dur=$((t1 - t0))
    if [ "$rc" -eq 0 ]; then
        echo "=== [$name] OK  elapsed $(_hms $dur) ===" | tee -a "$log"
        SUMMARY+=("$(printf '%-32s %-8s %s' "$name" "OK" "$(_hms $dur)")")
    else
        echo "=== [$name] FAILED (exit $rc) elapsed $(_hms $dur)" \
             "-- continuing with next step ===" | tee -a "$log"
        SUMMARY+=("$(printf '%-32s %-8s %s' "$name" "FAIL($rc)" "$(_hms $dur)")")
        N_FAILED=$((N_FAILED + 1))
    fi
}

mkdir -p "$LOGS"
exec 200>"$LOGS/.run_tasks.lock"
if ! flock -n 200; then
    echo "[run_tasks] another instance is already running" \
         "($LOGS/.run_tasks.lock) -- exiting" >&2
    exit 1
fi

RUN_STAMP=$(date +%Y%m%d_%H%M%S)
T0=$(date +%s)
echo "###############################################################"
echo "# run_tasks  start $(date '+%F %T')  GPU=$CUDA_VISIBLE_DEVICES"
echo "# run stamp: $RUN_STAMP   logs: $LOGS/<step>_$RUN_STAMP.log"
echo "###############################################################"

# Depth first: cheapest (~0.5 h) and a quick sanity check that the env works.
run_step depth_mono \
  $PY eval_depth.py --weights $CK/mono_640x192 --model-name monodepth2
run_step depth_monostereo \
  $PY eval_depth.py --weights $CK/mono+stereo_640x192 --model-name mono_stereo

# Detection: one process covers all 91 conditions, so the ~23 s numba CUDA
# kernel compile is paid once. Keep both models back-to-back so they share
# that backend (see the note in run_smoke.sh).
run_step det_smoke $PY eval_det.py --resume --model smoke
run_step det_pgd   $PY eval_det.py --resume --model pgd

# Segmentation (primary): Cityscapes val 500.
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

# Segmentation (secondary): KITTI-200 zero-shot, only for RQ4.
run_step seg_kitti200 \
  $PY eval_seg.py --resume --dataset kitti \
  --config $CK/deeplabv3plus_r101-d8_4xb2-80k_cityscapes-512x1024.py \
  --ckpt   $CK/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_20200606_114143-068fcfe9.pth \
  --model-name deeplabv3plus_r101_kitti200

DUR=$(( $(date +%s) - T0 ))
echo
echo "###############################################################"
echo "# run_tasks finished $(date '+%F %T')  total $(_hms $DUR)"
echo "#---------------------------------------------------------------"
printf '# %s\n' "${SUMMARY[@]}"
echo "#---------------------------------------------------------------"
if [ "$N_FAILED" -eq 0 ]; then
    echo "# all ${#SUMMARY[@]} steps succeeded"
else
    echo "# $N_FAILED/${#SUMMARY[@]} steps FAILED -- see table above and"
    echo "# the matching $LOGS/<step>_$RUN_STAMP.log"
fi
echo "###############################################################"
exit "$N_FAILED"

#!/bin/bash
# Run the geometric analysis pipeline against multiple synthetic / degraded
# image sets, in three independent passes:
#
#   1. kitti_real  vs  kitti_sim                          -> result_sim.json
#   2. kitti_real  vs  kitti_gaussian_blur  level_{1..3}  -> result_blur.json
#   3. kitti_real  vs  kitti_gaussian_noise level_{1..3}  -> result_noise.json
#
# Embedding layout expected under $EMB_DIR:
#   kitti_real/<space>.fbin
#   kitti_sim/<space>.fbin
#   kitti_gaussian_blur/level_{1,2,3}/<space>.fbin
#   kitti_gaussian_noise/level_{1,2,3}/<space>.fbin

set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'
step() { echo -e "\n${GREEN}══════ $* ══════${NC}\n"; }
warn() { echo -e "${YELLOW}$*${NC}"; }
err()  { echo -e "${RED}$*${NC}" >&2; }

# ─── Paths / config ────────────────────────────────────────────────────────
PROJECT_ROOT="/home/jsu02/Documents/ADS_metrics_eval"
EMB_DIR="/scratch/jsu02/sim-real-embedding"
RESULTS_DIR="$PROJECT_ROOT/results"
LOG_FILE="$RESULTS_DIR/output.log"

REAL="kitti_real"
SYN="kitti_sim"
BLUR="kitti_gaussian_blur"
NOISE="kitti_gaussian_noise"
JITTER="kitti_color_jitter"
BRIGHT="kitti_bright"
CONTRAST="kitti_contrast"
BLUR_LEVELS=( "$BLUR/level_1"  "$BLUR/level_2"  "$BLUR/level_3"  )
NOISE_LEVELS=( "$NOISE/level_1" "$NOISE/level_2" "$NOISE/level_3" )
JITTER_LEVELS=( "$JITTER/level_1" "$JITTER/level_2" "$JITTER/level_3" )
BRIGHT_LEVELS=( "$BRIGHT/level_1" "$BRIGHT/level_2" "$BRIGHT/level_3" )
CONTRAST_LEVELS=( "$CONTRAST/level_1" "$CONTRAST/level_2" "$CONTRAST/level_3" )

ALL_SPACES=( inception_v3 clip_vit_b32 resnet50 lpips_vgg pixel segformer )
K_VALUES=( 10 20 50 )

mkdir -p "$RESULTS_DIR"

# ─── Helpers ───────────────────────────────────────────────────────────────

# Echo the embedding spaces that have a .fbin for $REAL AND every level passed
# in. Skips spaces missing in any of the requested tags so each experiment can
# proceed with whatever subset is fully available.
collect_spaces() {
    local out=""
    for s in "${ALL_SPACES[@]}"; do
        local ok=1
        [ -f "$EMB_DIR/$REAL/${s}.fbin" ] || ok=0
        for tag in "$@"; do
            [ -f "$EMB_DIR/$tag/${s}.fbin" ] || { ok=0; break; }
        done
        [ "$ok" = "1" ] && out+=" $s"
    done
    echo "$out" | xargs
}

run_pipeline() {
    local label="$1"; shift
    local out_file="$1"; shift
    local levels=( "$@" )

    local spaces
    spaces=$(collect_spaces "${levels[@]}")
    if [ -z "$spaces" ]; then
        err "[$label] no embedding spaces shared by '$REAL' and ${levels[*]}; skipping"
        return 0
    fi

    step "$label"
    echo "  real   : $REAL"
    echo "  levels : ${levels[*]}"
    echo "  spaces : $spaces"
    echo "  k      : ${K_VALUES[*]}"
    echo "  output : $out_file"

    python "$PROJECT_ROOT/python/pipeline.py" \
        --embeddings-dir "$EMB_DIR" \
        --real "$REAL" \
        --levels "${levels[@]}" \
        --spaces $spaces \
        --k "${K_VALUES[@]}" \
        --output-dir "$out_file" 2>&1 | tee -a "$LOG_FILE"
}

# ─── Run ───────────────────────────────────────────────────────────────────
{
    echo "===== ADS Metrics Eval ====="
    echo "Started : $(date -Is)"
    echo "EMB_DIR : $EMB_DIR"
} | tee -a "$LOG_FILE"

run_pipeline "real only (baseline)"   "$RESULTS_DIR/result_real.json"  "$REAL"
run_pipeline "real vs sim"            "$RESULTS_DIR/result_sim.json"   "$SYN"
run_pipeline "real vs gaussian blur"  "$RESULTS_DIR/result_blur.json"  "${BLUR_LEVELS[@]}"
run_pipeline "real vs gaussian noise" "$RESULTS_DIR/result_noise.json" "${NOISE_LEVELS[@]}"
run_pipeline "real vs color jitter" "$RESULTS_DIR/result_color_jitter.json" "${JITTER_LEVELS[@]}"
run_pipeline "real vs bright" "$RESULTS_DIR/result_bright.json" "${BRIGHT_LEVELS[@]}"
run_pipeline "real vs contrast" "$RESULTS_DIR/result_contrast.json" "${CONTRAST_LEVELS[@]}"

step "Done!"
echo "Results saved in: $RESULTS_DIR"

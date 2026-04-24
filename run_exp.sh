#!/bin/bash
set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'
step() { echo -e "\n${GREEN}══════ $1 ══════${NC}\n"; }

PROJECT_ROOT="/home/jsu02/Documents/ADS_metrics_eval"
EMB_DIR="/scratch/jsu02/sim-real-embedding"
REAL_TAG="kitti_real_images"
SYN_TAG="kitti_sim_images"
TEST_TAG="kitti_pool_images"

SPACES=""
for s in inception_v3 clip_vit_b32 resnet50 lpips_vgg pixel segformer; do
    if [ -f "$EMB_DIR/$REAL_TAG/${s}.fbin" ] && [ -f "$EMB_DIR/$SYN_TAG/${s}.fbin" ]; then
        SPACES="$SPACES $s"
    fi
done
SPACES=$(echo $SPACES | xargs)  # trim

if [ -z "$SPACES" ]; then
    echo -e "${RED}No embedding .fbin files found in $EMB_DIR${NC}"
    exit 1
fi

echo "Spaces found: $SPACES"
echo "Real tag: $REAL_TAG"
echo "Syn tag:  $SYN_TAG"

python "$PROJECT_ROOT/python/pipeline.py" \
    --embeddings-dir "$EMB_DIR" \
    --real "$REAL_TAG" \
    --levels "$SYN_TAG" "$TEST_TAG" \
    --spaces $SPACES \
    --k 10 20 50 \
    --output-dir "$PROJECT_ROOT/results" | tee -a "$PROJECT_ROOT/results/output.log" &&

step "Done!"
echo "Results saved to: $PROJECT_ROOT/results/results.json"
echo ""
echo "To view results:"
echo "  python -c \"import json; r=json.load(open('$PROJECT_ROOT/results/results.json')); print(json.dumps(r['sweep'], indent=2))\""

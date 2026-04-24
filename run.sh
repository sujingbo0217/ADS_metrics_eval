#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# Full build + test + experiment pipeline
# Run from project root: /home/jsu02/Documents/ADS_metrics_eval
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

PROJECT_ROOT="/home/jsu02/Documents/ADS_metrics_eval"
cd "$PROJECT_ROOT"

# ─── Colors for output ───
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

step() { echo -e "\n${GREEN}══════ $1 ══════${NC}\n"; }
warn() { echo -e "${YELLOW}⚠ $1${NC}"; }

# ═══════════════════════════════════════════════════════════════════
# Phase 0: Check prerequisites
# ═══════════════════════════════════════════════════════════════════
step "Phase 0: Checking prerequisites"

echo "CUDA:"
nvcc --version | tail -1

echo "Python:"
python --version

echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Check Python packages
python -c "
missing = []
for pkg in ['numpy', 'torch', 'pybind11', 'networkx', 'community', 'sklearn', 'scipy', 'matplotlib']:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)
if missing:
    print(f'Missing Python packages: {missing}')
    print('Install with:')
    print(f'  pip install {\" \".join(missing)}')
    # community is actually python-louvain
    if 'community' in missing:
        print('  Note: \"community\" is installed via: pip install python-louvain')
    exit(1)
else:
    print('All Python packages OK')
"

# ═══════════════════════════════════════════════════════════════════
# Phase 1: Build
# ═══════════════════════════════════════════════════════════════════
step "Phase 1: Build CUDA library + Python bindings"

mkdir -p build
cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=$(which nvcc) \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -Dpybind11_DIR=$(python -m pybind11 --cmakedir)

make -j$(nproc)

cd "$PROJECT_ROOT"

# Verify the Python module loads
echo ""
python -c "
import sys
sys.path.insert(0, 'python')
import knn_ext
print(f'knn_ext loaded: {knn_ext.__file__}')
print(f'  Functions: {[x for x in dir(knn_ext) if not x.startswith(\"_\")]}')
print(f'  DistanceType: L2={knn_ext.DistanceType.L2}, InnerProduct={knn_ext.DistanceType.InnerProduct}')
" && echo -e "${GREEN}Python binding OK${NC}" || { echo -e "${RED}Python binding FAILED${NC}"; exit 1; }

# ═══════════════════════════════════════════════════════════════════
# Phase 2: C++ unit test
# ═══════════════════════════════════════════════════════════════════
step "Phase 2: C++ unit tests"

./build/test_knn && echo -e "${GREEN}C++ tests PASSED${NC}" || { echo -e "${RED}C++ tests FAILED${NC}"; exit 1; }

# ═══════════════════════════════════════════════════════════════════
# Phase 3: Python smoke test (small synthetic data)
# ═══════════════════════════════════════════════════════════════════
step "Phase 3: Python smoke test"

python -c "
import sys, numpy as np
sys.path.insert(0, 'python')
import knn_ext

np.random.seed(42)
N_r, N_q, D, k = 200, 100, 128, 10

real = np.random.randn(N_r, D).astype(np.float32)
syn  = np.random.randn(N_q, D).astype(np.float32) + 0.5  # shifted

# Cross-set k-NN
idx, dist = knn_ext.cross_set_knn(real, syn, k)
assert idx.shape == (N_q, k), f'cross_set shape: {idx.shape}'
assert dist.shape == (N_q, k)
print(f'cross_set_knn: OK  shape={idx.shape}')

# Pooled k-NN
mixed = np.vstack([real, syn])
pidx, pdist = knn_ext.pooled_knn(mixed, k)
assert pidx.shape == (N_r + N_q, k)
print(f'pooled_knn:    OK  shape={pidx.shape}')

# Barycenter shift
shifts = knn_ext.barycenter_shift(real, syn, idx, k)
assert shifts.shape == (N_q,)
print(f'bary_shift:    OK  mean={shifts.mean():.4f}')

# LID
lid = knn_ext.compute_lid(dist, k)
assert lid.shape == (N_q,)
print(f'LID:           OK  mean={lid.mean():.2f}')

# Overlap (need real-only k-NN first)
ridx, _ = knn_ext.pooled_knn(real, k)
ovlp = knn_ext.neighbor_overlap(ridx, pidx, k)
assert ovlp.shape == (N_r,)
print(f'overlap:       OK  mean={ovlp.mean():.4f}')

# In-degree
indeg = knn_ext.compute_indegree(pidx, k)
assert indeg.shape == (N_r + N_q,)
print(f'indegree:      OK  mean_real={indeg[:N_r].mean():.1f}  mean_syn={indeg[N_r:].mean():.1f}')

# L2 normalize
normed = knn_ext.l2_normalize_rows(real)
norms = np.linalg.norm(normed, axis=1)
assert np.allclose(norms, 1.0, atol=1e-5)
print(f'l2_normalize:  OK  norms≈1.0')

print()
print('ALL SMOKE TESTS PASSED')
" && echo -e "${GREEN}Smoke tests PASSED${NC}" || { echo -e "${RED}Smoke tests FAILED${NC}"; exit 1; }

# ═══════════════════════════════════════════════════════════════════
# Phase 4: Extract embeddings (if not already done)
# ═══════════════════════════════════════════════════════════════════
step "Phase 4: Extract embeddings"

EMB_DIR="$PROJECT_ROOT/embeddings"
REAL_DIR="$PROJECT_ROOT/data/kitti_real_images"
SIM_DIR="$PROJECT_ROOT/data/kitti_sim_images"

# Check if image directories exist
if [ ! -d "$REAL_DIR" ] || [ ! -d "$SIM_DIR" ]; then
    warn "Image directories not found:"
    [ ! -d "$REAL_DIR" ] && echo "  Missing: $REAL_DIR"
    [ ! -d "$SIM_DIR" ] && echo "  Missing: $SIM_DIR"
    echo "Skipping embedding extraction. Place images and re-run."
    echo "Or run manually:"
    echo "  cd python/embedding"
    echo "  python embed.py --folders $REAL_DIR $SIM_DIR --output-dir $EMB_DIR --extractors inception_v3 clip_vit_b32 resnet50 lpips_vgg pixel"
    SKIP_EXPERIMENT=1
else
    cd "$PROJECT_ROOT/python/embedding"

    # Extract per-folder embeddings
    # Using a subset of extractors for the first run; add segformer later
    EXTRACTORS="inception_v3 clip_vit_b32 resnet50 lpips_vgg pixel"

    if [ -f "$EMB_DIR/kitti_real_images/inception_v3.fbin" ] && [ -f "$EMB_DIR/kitti_sim_images/inception_v3.fbin" ]; then
        echo "Embeddings already exist in $EMB_DIR, skipping extraction."
        echo "(Use --overwrite flag in embed.py to re-extract)"
    else
        python embed.py \
            --folders "$REAL_DIR" "$SIM_DIR" \
            --output-dir "$EMB_DIR" \
            --extractors $EXTRACTORS \
            --batch-size 32 \
            --device cuda
    fi

    cd "$PROJECT_ROOT"
    SKIP_EXPERIMENT=0
fi

# ═══════════════════════════════════════════════════════════════════
# Phase 5: Run the experiment
# ═══════════════════════════════════════════════════════════════════
if [ "${SKIP_EXPERIMENT:-0}" = "0" ]; then
    step "Phase 5: Run k-NN geometric analysis"

    cd "$PROJECT_ROOT/python"

    # Determine which embedding folders exist
    REAL_TAG="kitti_real_images"
    SYN_TAG="kitti_sim_images"

    # Determine which spaces were extracted
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

    python pipeline.py \
        --embeddings-dir "$EMB_DIR" \
        --real "$REAL_TAG" \
        --levels "$SYN_TAG" \
        --spaces $SPACES \
        --k 10 20 50 \
        --output-dir "$PROJECT_ROOT/results"

    step "Done!"
    echo "Results saved to: $PROJECT_ROOT/results/results.json"
    echo ""
    echo "To view results:"
    echo "  python -c \"import json; r=json.load(open('$PROJECT_ROOT/results/results.json')); print(json.dumps(r['sweep'], indent=2))\""
else
    warn "Experiment skipped (missing image data). Fix Phase 4 and re-run."
fi

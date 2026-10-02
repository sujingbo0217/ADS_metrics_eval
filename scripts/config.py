from pathlib import Path

ROOT = Path("/scratch/jsu02/kitti")         # change on your machine
REPO_ROOT = Path(__file__).resolve().parent.parent / "repos"

# detection/images/training/image_2, detection/labels/training/label_2, ImageSets/
KITTI_OBJ = ROOT / "detection"
# segmentation/training/image_2, training/semantic
KITTI_SEG = ROOT / "segmentation"
KITTI_RAW = ROOT / "raw_data"              # raw drives for Eigen split
MONODEPTH2_REPO = REPO_ROOT / "monodepth2"
ROBODEPTH_REPO = REPO_ROOT / "RoboDepth"
KITTI_EVAL_REPO = REPO_ROOT / "kitti-object-eval-python"
MMDET3D_REPO = REPO_ROOT / "mmdetection3d"

# mmdet3d-standard KITTI root (symlink farm + kitti_infos_*.pkl from
# tools/create_data.py). Its training/image_2 points at the CLEAN images; the
# per-condition roots built by eval_det.py reuse everything else from here.
KITTI_STD_ROOT = KITTI_OBJ / "working"
# Per-condition symlink roots for mono3D (one per degradation/severity).
COND_ROOTS = ROOT / "cond_roots"

CITYSCAPES = Path("/scratch/jsu02/cityscapes")      # leftImg8bit/ and gtFine/ (change on your machine)
CITYSCAPES_SPLIT = "val"                            # official val 500

# depth: pre-generated RoboDepth KITTI-C (downloaded): <degradation>/<severity>/
#   kitti_data/... plus a bonus clean/ (18*5*697 + 697 clean = 63,427 pngs).
ROBODEPTH_KITTI_C = ROOT / "depth" / "kitti_c"
# det/seg: generated degraded sets live UNDER each task's own directory
# (KITTI_OBJ/"degraded", KITTI_SEG/"degraded"); no separate top-level store.
DEGRADED_SUBDIR = "degraded"
# Embeddings stay on scratch: they run to several GB and are a derived cache.
EMB_ROOT = ROOT / "embeddings"
# Task-performance outputs and mmengine work dirs also stay on scratch
RESULTS = ROOT / "results"

REPO_RESULTS = Path(__file__).resolve().parent.parent / "results"
ANALYSIS_DIR = REPO_RESULTS / "analysis"
LOG_DIR = REPO_RESULTS / "logs"

GT_DEPTHS_NPZ = MONODEPTH2_REPO / \
    "splits/eigen/gt_depths.npz"  # export_gt_depth.py output
EIGEN_TEST_LIST = MONODEPTH2_REPO / "splits/eigen/test_files.txt"

# -----------------------------------------------------------------------------
# CONDITIONS
# -----------------------------------------------------------------------------
# RoboDepth 18-degradation suite (ImageNet-C convention), 5 severities each.
DEGRADATIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise", "iso_noise",          # noise
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",              # blur
    "snow", "frost", "fog",                                                # weather
    "brightness", "dark", "contrast", "color_quant",                       # photometric
    "elastic_transform", "pixelate", "jpeg_compression",                   # digital
]
SEVERITIES = [1, 2, 3, 4, 5]
CLEAN = ("clean", 0)

# Mapping to the old paper's 5 transformation families (for the narrative table)
OLD_FAMILY_MAP = {
    "gaussian_noise": "noise", "shot_noise": "noise", "impulse_noise": "noise",
    "iso_noise": "noise",
    "defocus_blur": "blur", "glass_blur": "blur", "motion_blur": "blur",
    "zoom_blur": "blur",
    "brightness": "brightness", "dark": "brightness",
    "contrast": "contrast",
    "color_quant": "colorjitter", "hue": "colorjitter",
    "snow": "weather", "frost": "weather", "fog": "weather",
    "elastic_transform": "digital", "pixelate": "digital",
    "jpeg_compression": "digital",
}


def all_conditions():
    dgs = DEGRADATIONS
    return [CLEAN] + [(c, s) for c in dgs for s in SEVERITIES]


# -----------------------------------------------------------------------------
# TASKS & IMAGE SETS
# -----------------------------------------------------------------------------
TASKS = ["det", "seg", "depth", "cityscapes"]
METRIC_POOL_TASK = "det"


def split_ids(task):
    """Return list of image ids (strings) for the task's evaluation set."""
    if task == "det":
        return [l.strip() for l in open(KITTI_OBJ / "ImageSets/val.txt") if l.strip()]
    if task == "seg":
        return sorted(p.stem for p in (KITTI_SEG / "training/image_2").glob("*.png"))
    if task == "depth":
        # "drive rel_path side"
        return [l.strip() for l in open(EIGEN_TEST_LIST) if l.strip()]
    if task == "cityscapes":
        return sorted(_cs_index())
    raise ValueError(task)


def clean_image_path(task, image_id):
    if task == "det":
        return KITTI_OBJ / f"images/training/image_2/{image_id}.png"
    if task == "seg":
        return KITTI_SEG / f"training/image_2/{image_id}.png"
    if task == "depth":
        folder, frame, side = image_id.split()
        cam = {"l": "image_02", "r": "image_03"}[side]
        return KITTI_RAW / folder / cam / "data" / f"{int(frame):010d}.png"
    if task == "cityscapes":
        return _cs_index()[image_id]
    raise ValueError(task)


def degraded_image_path(task, image_id, degradation, severity):
    """Path of a degraded image for (task, image, degradation, severity).

    depth  -> RoboDepth pre-generated KITTI-C (robodepth_path), INCLUDING the
              clean condition (see the v1.1 routing decision in robodepth_path).
    det/seg -> generated under the task's own dir:
        <KITTI_OBJ|KITTI_SEG>/degraded/<degradation>/<severity>/<image_id>.png
    """
    if task == "depth":
        return robodepth_path(image_id, degradation, severity)
    if degradation == "clean":
        return clean_image_path(task, image_id)
    root = {"det": KITTI_OBJ, "seg": KITTI_SEG, "cityscapes": CITYSCAPES}[task]
    return root / DEGRADED_SUBDIR / degradation / str(severity) / f"{image_id}.png"


def robodepth_path(image_id, degradation, severity):
    """Path inside the downloaded RoboDepth KITTI-C.

    Verified layout (kitti_c.zip, 63,427 pngs): images are pre-resized to
    640x192 and stored as
        kitti_c/<degradation>/<severity>/kitti_data/<folder>/image_02/data/<frame>.png
    with the bonus clean set one level shallower (NO severity component):
        kitti_c/clean/kitti_data/<folder>/image_02/data/<frame>.png
    where <folder> = "<date>/<date>_drive_XXXX_sync"; only the left cam
    (image_02) is provided (all 697 eigen-test entries are side "l").

    The depth CLEAN condition is routed here not KITTI_RAW, 
    so clean and degraded share the same 640x192 resize pipeline. 
    Pairing full-res clean against 640x192 degraded images
    would fold a resize difference into every paired metric and every
    embedding. KITTI_RAW is used only by monodepth2/export_gt_depth.py.
    """
    folder, frame, side = image_id.split()
    cam = {"l": "image_02", "r": "image_03"}[side]
    stem = ROBODEPTH_KITTI_C / degradation
    if degradation != "clean":
        stem = stem / str(severity)
    return stem / "kitti_data" / folder / cam / "data" / f"{int(frame):010d}.png"


_CS_INDEX = None
def _cs_index():
    """
    image_id -> clean (val = 500)
    image_id = '<city>_<seq>_<frame>'
    """
    global _CS_INDEX
    if _CS_INDEX is None:
        d = CITYSCAPES / "leftImg8bit" / CITYSCAPES_SPLIT
        suf = "_leftImg8bit.png"
        _CS_INDEX = {p.name[:-len(suf)]: p for p in sorted(d.glob(f"*/*{suf}"))}
    return _CS_INDEX

def cityscapes_gt_path(image_id, kind="labelTrainIds"):
    city = _cs_index()[image_id].parent.name
    return (CITYSCAPES / "gtFine" / CITYSCAPES_SPLIT / city /
            f"{image_id}_gtFine_{kind}.png")


# -----------------------------------------------------------------------------
# EMBEDDING SPACES / GEOMETRY
# -----------------------------------------------------------------------------
SPACES = ["inception_v3", "clip_vitb32", "resnet50", "vgg_lpips", 
            "alexnet_lpips", "pixel", "segformer", "dinov2_vitl14"]

# Which training objective each space was optimised for. This, not the count of
# spaces, is what controls how much independent evidence the pool carries: four
# ImageNet-supervised CNNs are largely redundant with one another, whereas CLIP,
# DINOv2 and SegFormer each bring a genuinely different objective. Used to keep
# the confirmatory metric family objective-diverse rather than merely large.
SPACE_OBJECTIVE = {
    "inception_v3":  "imagenet-supervised",
    "resnet50":      "imagenet-supervised",
    "vgg_lpips":     "imagenet-supervised",   # backbone shared with LPIPS-vgg
    "alexnet_lpips": "imagenet-supervised",   # backbone shared with LPIPS-alex
    "clip_vitb32":   "image-text-contrastive",
    "segformer":     "cityscapes-segmentation",
    "dinov2_vitl14": "self-supervised",       # no labels anywhere in training
    "pixel":         "none",                  # network-free control
}

K_LIST = [10, 20, 50]
K_DEFAULT = 10
EXCLUDE_SELF = True                       # leave-one-out per-point reference
SEED = 2026
# (H, W) downsample for pixel space
PIXEL_SIZE = (64, 212)

# -----------------------------------------------------------------------------
# SIGN CONVENTIONS
# Expected direction of each measure as degradation severity increases.
# +1: increases with degradation; -1: decreases; 0: degradation-dependent sign
# (use |x - clean_baseline| for correlation, discuss raw sign separately).
# -----------------------------------------------------------------------------
DIRECTION = {
    "fid": +1, "kid": +1, "frechet": +1, "mmd": +1,
    "ssim": -1, "psnr": -1, "mse": +1, "lpips": +1,
    "clip_sim": -1, "cos_sim": -1,
    "baryshift": +1, "lid": 0, "overlap": +1, "indeg_delta": 0,
    "nmi": +1, "ari": +1,
}
# Task columns converted to relative performance drop in correlate.py; direction of
# the RAW column (before conversion): -1 = higher is better, +1 = error metric.
TASK_METRIC_DIRECTION = {
    "ap_mod_r40": -1, "ap_easy_r40": -1, "ap_hard_r40": -1,
    "ap3d_mod_r40": -1, "apbev_mod_r40": -1,
    "q": -1, "conf": -1, "miss": +1, "fp": +1,
    "miou": -1, "pixacc": -1, "entropy": +1,
    "abs_rel": +1, "sq_rel": +1, "rmse": +1, "rmse_log": +1, "silog": +1,
    "a1": -1, "a2": -1, "a3": -1,
}

# det/seg degraded dirs are created on demand by gen_degradation.py; only the
# shared output roots are pre-created here.
for d in (EMB_ROOT, RESULTS, ANALYSIS_DIR, LOG_DIR,
          RESULTS / "perimage_metrics", RESULTS / "task_perf",
          RESULTS / "correlations", RESULTS / "figs", RESULTS / "logs"):
    d.mkdir(parents=True, exist_ok=True)

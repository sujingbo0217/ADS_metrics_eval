"""Generate the 18 x 5 degraded version of KITTI det-val (3,769) and seg (200)
image sets. Depth reuses RoboDepth's pre-generated KITTI-C download.

Outputs are written under each task's own directory (config.degraded_image_path):
    detection/degraded/<degradation>/<severity>/<id: 1-5>.png
    segmentation/degraded/<degradation>/<severity>/<id: 1-5>.png

Degradation implementations (see RoboDepth/corruptions/utils.py):
    1. RoboDepth's 3 custom extras: dark (low_light), color_quant, iso_noise,
        imported directly from RoboDepth/corruptions/utils.py.
    2. `imagecorruptions.corrupt` for the other 15 degradations.
        REQUIRED: `pip install imagecorruptions`.

Every (image, degradation, severity) triple gets a fixed RNG seed so stochastic
degradations (noise family, glass_blur, snow, ...) are reproducible. Outputs are
PNG to avoid double-compression interactions (jpeg_compression severity is part
of the degradation itself, not the storage format).

Usage:
    python gen_degradation.py --task [det/seg] [--workers 16]
"""
import argparse
import hashlib
from multiprocessing import Pool

import numpy as np
from PIL import Image

import config as C

# --- degradation backends -----------------------------------------------------
def _patch_skimage_gaussian():
    """imagecorruptions (ImageNet-C code) calls skimage `gaussian(..., multichannel=)`
    and `random_noise`, whose `multichannel` kwarg was removed in scikit-image
    >=0.19 (replaced by `channel_axis`). Shim skimage.filters.gaussian to accept
    the old kwarg so the reference degradations run unchanged on modern skimage.
    Must run BEFORE importing imagecorruptions (it binds `gaussian` at import)."""
    import inspect
    import skimage.filters as _skf
    _orig = _skf.gaussian
    if "multichannel" in inspect.signature(_orig).parameters:
        return  # old skimage still supports it
    def _gaussian(*args, **kw):
        if "multichannel" in kw:
            mc = kw.pop("multichannel")
            kw.setdefault("channel_axis", -1 if mc else None)
        return _orig(*args, **kw)
    _skf.gaussian = _gaussian


try:
    _patch_skimage_gaussian()
    from imagecorruptions import corrupt as _ic_degrade
    from imagecorruptions import get_corruption_names as _ic_names
    _IC_AVAILABLE = set(_ic_names("all"))
except Exception as e:  # imagecorruptions is REQUIRED (RoboDepth depends on it)
    _ic_degrade, _IC_AVAILABLE = None, set()
    print(f"[gen_degradation] WARNING: imagecorruptions unavailable ({e}); "
          f"15 standard degradations will not work -- `pip install imagecorruptions`")

# RoboDepth's 3 custom extras, loaded straight from its repo file (exact parity),
# under a unique module name to avoid clashing with any other `utils` on sys.path.
_RD_EXTRAS = {}
_rd_utils_path = C.ROBODEPTH_REPO / "corruptions" / "utils.py"
if _rd_utils_path.exists() and _ic_degrade is not None:
    try:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            "robodepth_corruptions_utils", _rd_utils_path)
        _rd = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_rd)
        # signatures: fn(PIL|ndarray, severity) -> PIL.Image
        _RD_EXTRAS = {"dark": _rd.low_light,
                      "color_quant": _rd.color_quant,
                      "iso_noise": _rd.iso_noise}
        print("[gen_degradation] using RoboDepth extras: dark/color_quant/iso_noise")
    except Exception as e:  # pragma: no cover
        print(f"[gen_degradation] RoboDepth extras import failed ({e}); "
              f"falling back to local re-implementations")


# --- local fallbacks for RoboDepth extras ------------------------------------
def _dark(x: np.ndarray, severity: int) -> np.ndarray:
    """Low-light: gamma curve + intensity scale + Poisson shot noise."""
    gamma = [1.5, 1.8, 2.2, 2.6, 3.0][severity - 1]
    scale = [0.60, 0.50, 0.42, 0.34, 0.26][severity - 1]
    x = (x / 255.0) ** gamma * scale
    lam = np.clip(x * 255.0 * 0.8, 1e-6, None)
    x = np.random.poisson(lam) / (255.0 * 0.8)
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def _color_quant(x: np.ndarray, severity: int) -> np.ndarray:
    bits = [6, 5, 4, 3, 2][severity - 1]
    shift = 8 - bits
    return ((x.astype(np.uint8) >> shift) << shift).astype(np.uint8)


def _iso_noise(x: np.ndarray, severity: int) -> np.ndarray:
    """Poisson-Gaussian sensor noise (ISO gain)."""
    g_sigma = [0.03, 0.06, 0.10, 0.15, 0.20][severity - 1] * 255.0
    p_scale = [60, 25, 12, 5, 3][severity - 1]
    x = x.astype(np.float32)
    x = np.random.poisson(np.clip(x, 0, 255) / 255.0 *
                          p_scale) / p_scale * 255.0
    x = x + np.random.normal(0.0, g_sigma, size=x.shape)
    return np.clip(x, 0, 255).astype(np.uint8)


def _hue(x: np.ndarray, severity: int) -> np.ndarray:
    """Deterministic hue rotation (replaces the old paper's stochastic
    ColorJitter hue). torchvision functional adjust_hue with fixed factor."""
    import torch
    import torchvision.transforms.functional as F
    t = torch.from_numpy(np.ascontiguousarray(x).copy()).permute(2, 0, 1)
    t = F.adjust_hue(t, C.HUE_FACTORS[severity - 1])
    return t.permute(1, 2, 0).numpy().astype(np.uint8)


_LOCAL = {"dark": _dark, "color_quant": _color_quant,
          "iso_noise": _iso_noise, "hue": _hue}


def apply_degradation(img: np.ndarray, degradation: str, severity: int) -> np.ndarray:
    # 1) RoboDepth's 3 custom extras (exact parity with the depth KITTI-C).
    #    color_quant uses PIL.ImageOps.posterize, so it needs a PIL input.
    if degradation in _RD_EXTRAS:
        x = Image.fromarray(img) if degradation == "color_quant" else img
        return np.asarray(_RD_EXTRAS[degradation](x, severity), dtype=np.uint8)
    # 2) the 15 standard degradations via imagecorruptions (RoboDepth uses this too)
    if degradation in _IC_AVAILABLE and _ic_degrade is not None:
        return _ic_degrade(img, corruption_name=degradation, severity=severity)
    # 3) local fallbacks: deterministic `hue`, plus the 3 extras if (and only if)
    #    the RoboDepth repo is unavailable.
    # if degradation in _LOCAL:
    #     return _LOCAL[degradation](img, severity)
    raise KeyError(f"no implementation for degradation '{degradation}'")


def _seed_for(image_id: str, degradation: str, severity: int) -> int:
    h = hashlib.md5(
        f"{C.SEED}|{image_id}|{degradation}|{severity}".encode()).hexdigest()
    return int(h[:8], 16)


def _job(args):
    task, image_id, degradation, severity = args
    out = C.degraded_image_path(task, image_id, degradation, severity)
    if out.exists():
        return 0  # resumable
    out.parent.mkdir(parents=True, exist_ok=True)
    np.random.seed(_seed_for(image_id, degradation, severity))
    img = np.asarray(Image.open(
        C.clean_image_path(task, image_id)).convert("RGB"))
    Image.fromarray(apply_degradation(img, degradation, severity)).save(out)
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["det", "seg", "cityscapes"], required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    ids = C.split_ids(a.task)
    dgs = C.DEGRADATIONS
    jobs = [(a.task, i, c, s)
            for c in dgs for s in C.SEVERITIES for i in ids]
    print(f"[gen_degradation] {a.task}: {len(ids)} images x {len(dgs)}x5 "
          f"= {len(jobs)} outputs")
    with Pool(a.workers) as p:
        done = sum(p.imap_unordered(_job, jobs, chunksize=64))
    print(f"[gen_degradation] wrote {done} new files "
          f"(skipped {len(jobs) - done} existing)")
    # verification hint
    task_root = {"det": C.KITTI_OBJ, "seg": C.KITTI_SEG,
             "cityscapes": C.CITYSCAPES}[a.task] / C.DEGRADED_SUBDIR
    print("verify counts:  find", task_root, "-name '*.png' | wc -l")


if __name__ == "__main__":
    main()

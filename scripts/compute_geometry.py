"""Five geometric properties over the full 90-condition grid.

Produces tables shaped exactly like the metric tables, so that everything
downstream reuses unchanged code:

    ANALYSIS_DIR/geometry.csv
        one row per (dataset, degradation, severity, space, k), the 13
        property columns of data_utils.GEOMETRY_COLS. A geometric property at
        a fixed (space, k) is therefore a drop-in replacement for a
        condition-level metric column, and correlate_geometry.py exploits
        that. Read one dataset back with data_utils.geometry(dataset).

    ANALYSIS_DIR/image_geometry_<dataset>.parquet
        one row per (degradation, severity, image_id, space, k):
            lid_real, lid_gen      P3, per-point LID of each copy
            trust, cont            P4, the per-point penalties T and C average
            mix_real, mix_gen      P5, that image's own cross-set neighbor share
        This is the tier-2 input, mirroring image_metrics_<dataset>.parquet.

        THREE of the five properties have a per-image form, not five. P1 is a
        centroid displacement and P2 a spectral entropy: both are properties of
        a SET, and no single image has a share of them -- exactly as FID and KID
        have no per-image value on the metric side. The scope structure of the
        geometry therefore mirrors the scope structure of the metrics, which is
        what lets tier 2 treat the two alike.

Usage:
    python compute_geometry.py --dataset all
    python compute_geometry.py --dataset kitti_seg --spaces inception_v3 --k 10
"""
import argparse
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

import config as C
import data_utils as D
from geometry_lib import all_properties, prepare, set_stats

# Spaces that exist as cached .npy today. The LPIPS-coordinate and SSIM spaces
# need a new extractor and are deliberately absent -- see
# reports/geometry_rebuild.html for the open decisions.
#
# dinov2_vitl14 joined on 2026-09-28, after Sec. 5.4d added FD/KD in that space.
# It is the only self-supervised phi in the pool, so it is the one place where
# "is this geometry a property of the images or of ImageNet supervision" can be
# asked directly.
#
# pixel joined on 2026-10-05 so that SSIM and MSE, which have no learned phi,
# can be compared with geometry in their own space.
#
# All five are cached by extract_metrics.py --level distribution.
SPACES = ["inception_v3", "clip_vitb32", "vgg_lpips", "dinov2_vitl14", "pixel"]
KS = [5, 10, 20, 50, 100]
# Per-image geometry is written only where a per-image TASK score exists to
# correlate it against -- C.IMAGE_DATASETS, for the same reason as the image
# metrics: AP is an integral over a dataset-wide PR curve and has no per-image
# form, so kitti_det per-image geometry would have nothing to pair with.
# Per-image rows are written at EVERY k, not just one.
#
# P5's per-image value is the count of cross-set neighbours over k, so it can
# only take k+1 values. At k = 10 that is 11 levels, and in one Cityscapes
# condition 500 images land on 9 distinct values with 41% of them sharing one --
# too coarse to regress against. Widening k is the only way to give it
# resolution, and since the same sweep also hands tier 2 a k ablation for P3 and
# P4 at no extra compute, all of them are kept.
#
# Cost is storage, not time: the per-point arrays are produced by the kernels
# regardless, so this only writes down what was already computed. Rows go from
# 377k to 1.9M across the three pools.
# The six per-point arrays, and the column each becomes.
PT_COLS = {"p3_lid_pt_real": "lid_real", "p3_lid_pt_gen": "lid_gen",
           "p4_trust_pt": "trust", "p4_cont_pt": "cont",
           "p5_mix_pt_real": "mix_real", "p5_mix_pt_gen": "mix_gen"}


def _load(dataset, tag, space):
    p = C.EMB_ROOT / C.DATASETS[dataset] / f"{tag}__{space}.npy"
    if not p.exists():
        return None
    return np.load(p).astype(np.float32)


def run_dataset(dataset, spaces, ks, perimage, limit=0):
    perimage = perimage and dataset in C.IMAGE_DATASETS
    save = not limit
    rows, pt_rows = [], []
    ids = C.split_ids(C.DATASETS[dataset])
    conditions = [c for c in C.all_conditions() if c[0] != "clean"]
    if limit:
        conditions = conditions[:limit]

    t0 = time.time()
    for space in spaces:
        R = _load(dataset, "clean_0", space)
        if R is None:
            print(f"  [{dataset}/{space}] no clean embedding, skipping", flush=True)
            continue
        print(f"  [{dataset}/{space}] R = {R.shape}", flush=True)
        # P1 and P2 are per-SET quantities: the clean set's do not depend on the
        # condition or on k, and the condition's do not depend on k. Hoisting
        # them out of the two loops is the one piece of sharing the (R, G, k)
        # API cannot do for itself.
        R = prepare(R, space)
        sR = set_stats(R)
        for n, (dg, sev) in enumerate(conditions):
            G = _load(dataset, f"{dg}_{sev}", space)
            if G is None:
                print(f"    missing {dg}_{sev}__{space}.npy, skipping", flush=True)
                continue
            G = prepare(G, space)
            sG = set_stats(G)
            for k in ks:
                r = all_properties(R, G, k=k, space=space, keep_pointwise=perimage,
                                   sR=sR, sG=sG)
                pt = r.pop("_pointwise", None)
                rows.append(dict(degradation=dg, severity=sev,
                                 space=space, k=k, **r))
                if pt is not None:
                    # Embeddings are stored in split_ids order, so row i of every
                    # per-point array is image ids[i]. Keeping the id makes this
                    # table joinable to image_metrics_<dataset>.parquet and to the
                    # per-image task tables on the same keys.
                    blk = pd.DataFrame({c: np.asarray(pt[a], dtype=np.float32)
                                        for a, c in PT_COLS.items()})
                    blk.insert(0, "k", np.int16(k))
                    blk.insert(0, "space", space)
                    blk.insert(0, "image_id", ids)
                    blk.insert(0, "severity", np.int16(sev))
                    blk.insert(0, "degradation", dg)
                    pt_rows.append(blk)
            if n % 10 == 0:
                print(f"    [{dataset}/{space}] {n + 1}/{len(conditions)} "
                      f"{dg}/{sev} ({time.time() - t0:.0f}s)", flush=True)

    if save and rows:
        D.write_geometry(dataset, rows)
    print(f"  [{dataset}] {len(rows)} rows -> "
          f"{D.GEOMETRY_CSV.name}{'' if save else ' (smoke test, not saved)'} "
          f"({time.time() - t0:.0f}s)", flush=True)
    if save and perimage and pt_rows:
        D.write_image_geometry(dataset, pd.concat(pt_rows, ignore_index=True))
        n_pt = sum(len(b) for b in pt_rows)
        print(f"  [{dataset}] {n_pt} per-image rows -> "
              f"{D.image_geometry_path(dataset).name}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", choices=["all", *C.DATASETS])
    ap.add_argument("--spaces", nargs="*", default=SPACES)
    ap.add_argument("--k", nargs="*", type=int, default=KS)
    ap.add_argument("--no-perimage", action="store_true",
                    help="skip the P3/P4/P5 per-image table. It is written by "
                         "default for " + ", ".join(C.IMAGE_DATASETS) + ".")
    ap.add_argument("--limit", type=int, default=0,
                    help="first N conditions only, for a smoke test")
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()

    if not a.no_log:
        C.LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logf = C.LOG_DIR / f"geometry_{a.dataset}_{stamp}.log"

        class _Tee:
            def __init__(self, *streams): self.streams = streams
            def write(self, d):
                for s_ in self.streams:
                    s_.write(d); s_.flush()
            def flush(self):
                for s_ in self.streams: s_.flush()

        _fh = open(logf, "w")
        sys.stdout = _Tee(sys.__stdout__, _fh)
        sys.stderr = _Tee(sys.__stderr__, _fh)
        print(f"[log] {logf}", flush=True)

    C.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[geometry] spaces={a.spaces} k={a.k} "
          f"perimage={'off' if a.no_perimage else C.IMAGE_DATASETS}", flush=True)
    for d in (C.DATASETS if a.dataset == "all" else [a.dataset]):
        run_dataset(d, a.spaces, a.k, not a.no_perimage, a.limit)


if __name__ == "__main__":
    main()

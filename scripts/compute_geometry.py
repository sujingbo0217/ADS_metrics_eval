"""Compute geometric properties

Usage:
    python compute_geometry.py
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
from geometry_lib import (POINT_COLS, POINT_SELECTED, clean_reference,
                          condition_properties, neighborhood_quality, prepare,
                          select_construction)

SPACES = [
    "inception_v3", "clip_vitb32", "vgg_lpips", "dinov2_vitl14", "pixel",
    "glcm", "density", "density_sorted", "rgb_hist",
]
KS = [5, 10, 20, 50, 100]
SELECTION_CSV = C.ANALYSIS_DIR / "geometry_selection.csv"


def _load(dataset, tag, space):
    p = C.EMB_ROOT / C.DATASETS[dataset] / f"{tag}__{space}.npy"
    if not p.exists():
        return None
    return np.load(p).astype(np.float32)


def _clean_task_scores(dataset, ids):
    out = {}
    for model in C.models_on(dataset):
        if C.MODELS[model]["task"] == "detection":
            continue
        try:
            t = D.task_score(model, "perimage", clean=True)
        except FileNotFoundError:
            continue
        t = t.xs(C.CLEAN, level=["degradation", "severity"])
        out[model] = t.reindex(ids).to_numpy(dtype=np.float64)
    return out


def _write_selection(dataset, frames):
    new = pd.concat(frames, ignore_index=True).assign(dataset=dataset)
    if SELECTION_CSV.exists():
        old = pd.read_csv(SELECTION_CSV)
        # replace only the spaces just computed, so a run on new spaces keeps the others
        stale = (old.dataset == dataset) & old.space.isin(new.space.unique())
        new = pd.concat([old[~stale], new], ignore_index=True)
    first = ["dataset", "space", "candidate", "k"]
    new[first + [c for c in new.columns if c not in first]].to_csv(SELECTION_CSV, index=False)


def run_dataset(dataset, spaces, ks, perimage, select, limit=0):
    perimage = perimage and dataset in C.IMAGE_DATASETS
    save = not limit
    rows, pt_rows, sel_frames = [], [], []
    ids = C.split_ids(C.DATASETS[dataset])
    conditions = [c for c in C.all_conditions() if c[0] != "clean"]
    if limit:
        conditions = conditions[:limit]
    task_scores = _clean_task_scores(dataset, ids) if select else {}

    t0 = time.time()
    for space in spaces:
        R = _load(dataset, "clean_0", space)
        if R is None:
            print(f"  [{dataset}/{space}] no clean embedding, skipping", flush=True)
            continue
        R = prepare(R, space)
        print(f"  [{dataset}/{space}] R = {R.shape}", flush=True)

        # The construction is chosen on the clean set alone, once per dataset
        # and representation, and frozen for all conditions.
        selected = "primary"
        if select:
            chosen, q = select_construction(neighborhood_quality(R, ks, space, task_scores))
            selected = chosen
            sel_frames.append(q.assign(space=space))
            print(f"    construction: {selected}  (mean ranks "
                  + ", ".join(f"{c} {r:.2f}" for c, r in
                              q.groupby("candidate").mean_rank.first().dropna()
                               .sort_values().items())
                  + f")  ({time.time() - t0:.0f}s)", flush=True)

        # Everything that depends on R alone, for every k.
        ref = clean_reference(R, space, ks, selected)
        print(f"    clean reference: ER {ref['er']:.1f}, TwoNN {ref['id']:.1f}, "
              f"d_t {ref['d_t']} ({time.time() - t0:.0f}s)", flush=True)

        for n, (dg, sev) in enumerate(conditions):
            G = _load(dataset, f"{dg}_{sev}", space)
            if G is None:
                print(f"    missing {dg}_{sev}__{space}.npy, skipping", flush=True)
                continue
            G = prepare(G, space)
            for k, (r, pt) in zip(ks, condition_properties(R, G, ref, perimage)):
                rows.append(dict(degradation=dg, severity=sev, space=space, k=k, **r))
                if pt is not None:
                    # Embeddings are stored in split_ids order, so row i of every
                    # per-point array is image ids[i]. Keeping the id makes this
                    # table joinable to image_metrics_<dataset>.parquet and to the
                    # per-image task tables on the same keys.
                    cols = POINT_COLS + [c + "_sel" for c in POINT_SELECTED]
                    blk = pd.DataFrame({c: np.asarray(pt[c], dtype=np.float32)
                                        for c in cols if c in pt})
                    blk.insert(0, "k", np.int16(k))
                    blk.insert(0, "space", space)
                    blk.insert(0, "image_id", ids)
                    blk.insert(0, "severity", np.int16(sev))
                    blk.insert(0, "degradation", dg)
                    pt_rows.append(blk)
            if n % 10 == 0:
                print(f"    [{dataset}/{space}] {n + 1}/{len(conditions)} "
                      f"{dg}/{sev} ({time.time() - t0:.0f}s)", flush=True)

    if save and sel_frames:
        _write_selection(dataset, sel_frames)
        print(f"  [{dataset}] selection -> {SELECTION_CSV.name}", flush=True)
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
                    help="skip the per-image table. It is written by "
                         "default for " + ", ".join(C.IMAGE_DATASETS) + ".")
    ap.add_argument("--no-select", action="store_true",
                    help="skip the construction selection; the _sel columns "
                         "then repeat the primary ones")
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
          f"select={'off' if a.no_select else 'on'} "
          f"perimage={'off' if a.no_perimage else C.IMAGE_DATASETS}", flush=True)
    for d in (C.DATASETS if a.dataset == "all" else [a.dataset]):
        run_dataset(d, a.spaces, a.k, not a.no_perimage, not a.no_select, a.limit)


if __name__ == "__main__":
    main()

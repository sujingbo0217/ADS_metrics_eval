"""Realism metrics for the two experiments, selected by --level.

- distribution level: FID, KID, FD@DINOv2, KD@DINOv2.
    Native set functionals: the clean embedding set of a dataset is compared
    with the embedding set of one corruption condition as wholes; there is no
    per-image value. One row per (dataset, condition); the clean condition is
    evaluated as clean against clean. The same pass caches every embedding
    space (inception_v3, clip_vitb32, dinov2_vitl14, vgg_lpips, pixel) as
    float16 under EMB_ROOT/<dataset>/<condition>__<space>.npy, which the image
    level and the geometry analysis read -- so run this level first.
    -> ANALYSIS_DIR/distribution_metrics.csv
         dataset  degradation  severity  FID  KID  FD@DINOv2  KD@DINOv2

- image level: the 12 metrics on kitti_seg, cityscapes, kitti_depth.
    kitti_det is excluded because AP has no per-image form. The mean of a
    column over one condition's images is that metric's condition value.

Usage
    python extract_metrics.py --level [distribution/image] [--dataset all]
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from PIL import Image

import config as C
import data_utils as D
import metrics_lib as M

# dataset -> batch size. Batches differ only to bound memory at 2048x1024.
BATCH = {"kitti_det": 32, "kitti_seg": 32, "cityscapes": 8, "kitti_depth": 64}
DEV_T = "cuda" if torch.cuda.is_available() else "cpu"
LOAD_THREADS = 24
CPU_WORKERS = 48
CPU_EX = None

SPACES = ["inception_v3", "clip_vitb32", "vgg_lpips", "dinov2_vitl14", "pixel"]
DIST_OUT = D.DISTRIBUTION_METRICS_CSV
CPU_COLS = {"ssim": "SSIM", "tsi": "TSI", "wd": "WD", "kl": "KL", "histi": "HistI"}
EMB_COLS = {"inception_v3": "Cos@Inc", "clip_vitb32": "CLIP-sim"}


# ---------------------------------------------------------------------------
# image I/O
# ---------------------------------------------------------------------------
def emb_dir(dataset):
    return C.EMB_ROOT / C.DATASETS[dataset]


def load_batch(dataset, ids, dg, sev, pool_ex):
    """A LIST of native-resolution [3, H, W] tensors in [0, 1], in ids order."""
    key = C.DATASETS[dataset]

    def one(i):
        im = Image.open(C.degraded_image_path(key, i, dg, sev)).convert("RGB")
        return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.
    # PNG decode releases the GIL, so threads give near-linear speedup.
    return list(pool_ex.map(one, ids))


def size_groups(imgs):
    """Indices bucketed by image shape (KITTI has four distinct sizes)."""
    g = {}
    for j, x in enumerate(imgs):
        g.setdefault(tuple(x.shape), []).append(j)
    return g


def stack_for(imgs, size):
    """Bilinear resize to a backbone's input size (no antialiasing)."""
    out = [None] * len(imgs)
    for _, idxs in size_groups(imgs).items():
        r = torch.nn.functional.interpolate(
            torch.stack([imgs[j] for j in idxs]).to(DEV_T),
            size=size, mode="bilinear", align_corners=False)
        for k, j in enumerate(idxs):
            out[j] = r[k]
    return torch.stack(out)


def embed_grouped(bb, imgs, batch):
    """bb.pooled over a ragged list, batched within each distinct shape."""
    z = torch.zeros(len(imgs), bb.pooled_dim)
    for _, idxs in size_groups(imgs).items():
        for i in range(0, len(idxs), batch):
            sl = idxs[i:i + batch]
            z[sl] = bb.pooled(torch.stack([imgs[j] for j in sl])).cpu()
    return z.numpy()


def _pixel_worker(job):
    """Module-level so ProcessPoolExecutor can pickle it. Touches no CUDA."""
    a, b, names = job
    return [M.cpu_pixel(n, a, b) for n in names]


def cpu_paired(ref, imgs, names):
    """Per-image CPU pixel metrics of imgs[j] against ref[j], over every worker."""
    jobs = [(M.to_u8(ref[j]), M.to_u8(imgs[j]), names) for j in range(len(imgs))]
    v = np.array(list(CPU_EX.map(_pixel_worker, jobs, chunksize=2)))
    return {n: v[:, i] for i, n in enumerate(names)}


# ---------------------------------------------------------------------------
# distribution level
# ---------------------------------------------------------------------------
def embed_all(bbs, imgs, batch):
    """One decode -> every embedding space, as float32 numpy arrays."""
    z_inc, z_clip = [], []
    for i in range(0, len(imgs), batch):
        b = imgs[i:i + batch]
        z_inc.append(bbs["inception_v3"].pooled(stack_for(b, (299, 299))).cpu())
        z_clip.append(bbs["clip_vitb32"].pooled(stack_for(b, (224, 224))).cpu())
    return {
        "inception_v3": torch.cat(z_inc).numpy(),
        "clip_vitb32": torch.cat(z_clip).numpy(),
        # DINOv2 resizes inside .pooled() (bicubic + antialias, as dgm-eval);
        # stack_for would impose a bilinear resize first and change the answer.
        "dinov2_vitl14": embed_grouped(bbs["dinov2_vitl14"], imgs, batch),
        # GAP of relu5_3 at native resolution, for the geometry side.
        "vgg_lpips": embed_grouped(bbs["vgg"], imgs, batch),
        "pixel": embed_grouped(bbs["pixel"], imgs, batch),
    }


def set_metrics(ref, z, n):
    """The four distribution-level metrics of one condition against clean."""
    ri, zi = ref["inception_v3"].astype(np.float64), z["inception_v3"].astype(np.float64)
    rd, zd = ref["dinov2_vitl14"].astype(np.float64), z["dinov2_vitl14"].astype(np.float64)
    s = min(1000, n)
    return {"FID": M.frechet(ri, zi),
            "KID": M.kid(ri, zi, n_subsets=50, subset_size=s)[0],
            "FD@DINOv2": M.frechet(rd, zd),
            "KD@DINOv2": M.kid(rd, zd, n_subsets=50, subset_size=s)[0]}


def run_distribution(dataset, bbs, ids, conditions, save, LD):
    batch = BATCH[dataset]
    edir = emb_dir(dataset)
    edir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ref = embed_all(bbs, load_batch(dataset, ids, "clean", 0, LD), batch)
    rows = []
    for k, (dg, sev) in enumerate(conditions):
        z = ref if dg == "clean" else embed_all(
            bbs, load_batch(dataset, ids, dg, sev, LD), batch)
        if save:
            for sp in SPACES:
                np.save(edir / f"{dg}_{sev}__{sp}.npy", z[sp].astype(np.float16))
        rows.append(dict(dataset=dataset, degradation=dg, severity=sev,
                         **set_metrics(ref, z, len(ids))))
        progress(dataset, k, len(conditions), dg, sev, t0)

    new = pd.DataFrame(rows)[["dataset", "degradation", "severity",
                              *C.DISTRIBUTION_METRICS]]
    if save:
        if DIST_OUT.exists():
            old = pd.read_csv(DIST_OUT)
            new = pd.concat([old[old.dataset != dataset], new], ignore_index=True)
        order = {d: i for i, d in enumerate(C.DATASETS)}
        cond = {c: i for i, c in enumerate(C.all_conditions())}
        new = (new.assign(_d=new.dataset.map(order),
                          _c=[cond[c] for c in zip(new.degradation, new.severity)])
                  .sort_values(["_d", "_c"]).drop(columns=["_d", "_c"]))
        new.to_csv(DIST_OUT, index=False)
    return DIST_OUT.name


# ---------------------------------------------------------------------------
# image level
# ---------------------------------------------------------------------------
def emb_cosines(dataset, tag, n):
    d, out = emb_dir(dataset), {}
    for space, col in EMB_COLS.items():
        a = np.load(d / f"clean_0__{space}.npy").astype(np.float32)[:n]
        b = np.load(d / f"{tag}__{space}.npy").astype(np.float32)[:n]
        out[col] = M.cosine(torch.from_numpy(a), torch.from_numpy(b)).numpy()
    return out


def run_image(dataset, bbs, ids, conditions, save, LD):
    batch = BATCH[dataset]
    t0 = time.time()
    clean = load_batch(dataset, ids, "clean", 0, LD)
    conditions = [c for c in conditions if c[0] != "clean"]
    frames = []
    for k, (dg, sev) in enumerate(conditions):
        imgs = load_batch(dataset, ids, dg, sev, LD)
        vp = {c: np.zeros(len(imgs)) for c in ("LPIPS", "MSE", "Cos@VGG", "CPL", "SSS")}
        for _, idxs in size_groups(imgs).items():
            for i in range(0, len(idxs), batch):
                sel = idxs[i:i + batch]
                a = torch.stack([clean[j] for j in sel])
                b = torch.stack([imgs[j] for j in sel])
                vp["LPIPS"][sel] = M.metric("lpips", a, b, bbs["lpips"])
                vp["MSE"][sel] = M.metric("mse", a, b)
                vp["Cos@VGG"][sel] = M.metric("cossim", a, b, bbs["vgg16"])
                vp["CPL"][sel] = M.metric("cpl", a, b, bbs["vgg16"])
                vp["SSS"][sel] = M.metric("sss", a, b, bbs["vgg16_raw"])
        cpu = cpu_paired(clean, imgs, list(CPU_COLS))
        frames.append(pd.DataFrame(dict(
            degradation=dg, severity=sev, image_id=ids,
            **emb_cosines(dataset, f"{dg}_{sev}", len(ids)), **vp,
            **{CPU_COLS[n]: v for n, v in cpu.items()})))
        progress(dataset, k, len(conditions), dg, sev, t0)

    df = pd.concat(frames, ignore_index=True)[
        ["degradation", "severity", "image_id", *C.IMAGE_METRICS]]
    if save:
        df.to_parquet(D.image_metrics_path(dataset), index=False)
    return f"{D.image_metrics_path(dataset).name} ({len(df):,} rows)"


# ---------------------------------------------------------------------------
def progress(dataset, k, total, dg, sev, t0):
    if k % 10 == 0 or k == total - 1:
        print(f"  [{dataset}] {k + 1}/{total} {dg}/{sev} ({time.time() - t0:.0f}s)",
              flush=True)


def tee_log(name):
    """Mirror stdout/stderr into LOG_DIR/<name>_<stamp>.log."""
    C.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logf = C.LOG_DIR / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log"

    class _Tee:
        def __init__(self, *s): self.s = s
        def write(self, d):
            for x in self.s:
                x.write(d); x.flush()
        def flush(self):
            for x in self.s:
                x.flush()
    fh = open(logf, "w")
    sys.stdout = _Tee(sys.__stdout__, fh)
    sys.stderr = _Tee(sys.__stderr__, fh)
    print(f"[log] {logf}", flush=True)


def main():
    global CPU_EX
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", required=True, choices=["distribution", "image"])
    ap.add_argument("--dataset", default="all",
                    help="'all', or one of " + ", ".join(C.DATASETS))
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke test: first N images, 3 corrupted conditions, writes nothing")
    a = ap.parse_args()

    allowed = list(C.DATASETS) if a.level == "distribution" else C.IMAGE_DATASETS
    datasets = allowed if a.dataset == "all" else [a.dataset]
    if any(d not in allowed for d in datasets):
        ap.error(f"--level {a.level} supports: {', '.join(allowed)}")
    if not a.limit:
        tee_log(f"extract_{a.level}_{a.dataset}")

    if a.level == "distribution":
        model = {
            "inception_v3": M.InceptionBackbone(),
            "clip_vitb32": M.ClipBackbone(),
            "vgg": M.LpipsBackbone("vgg"),
            "dinov2_vitl14": M.DinoV2Backbone("vitl14"),
            "pixel": M.PixelBackbone(C.PIXEL_SIZE),
        }
        run = run_distribution
    elif a.level == "image":
        CPU_EX = ProcessPoolExecutor(max_workers=CPU_WORKERS)
        model = {"lpips": M.LpipsBackbone("vgg"),
                 "vgg16": M.KerasVGG16("block5_conv3", preprocess=True),
                 "vgg16_raw": M.KerasVGG16("block5_pool", preprocess=False)}
        run = run_image
    else:
        raise ValueError(a.level)

    LD = ThreadPoolExecutor(max_workers=LOAD_THREADS)
    try:
        for d in datasets:
            ids = C.split_ids(C.DATASETS[d])
            if a.limit:
                ids = ids[:a.limit]
            conditions = C.all_conditions()[:4] if a.limit else C.all_conditions()
            print(f"[{a.level}/{d}] {len(ids)} images x {len(conditions)} conditions",
                  flush=True)
            t0 = time.time()
            out = run(d, model, ids, conditions, not a.limit, LD)
            print(f"[{a.level}/{d}] done in {time.time() - t0:.0f}s -> {out}"
                  f"{' (smoke test, not saved)' if a.limit else ''}", flush=True)
    finally:
        LD.shutdown()
        if CPU_EX is not None:
            CPU_EX.shutdown()


if __name__ == "__main__":
    main()

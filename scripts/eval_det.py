"""Monocular 3D detection evaluation over 18x5 (+clean) conditions.

Models: SMOKE (DLA-34) and PGD (ResNet-101), both official mmdet3d 
weights trained on the Chen split (3712 train / 3769 val). 

MECHANISM. kitti_infos_val.pkl stores image paths RELATIVE to data_root
('training/image_2/000001.png'), so one pkl serves all 91 conditions. For each
condition we build a throwaway root whose training/image_2 symlinks to that
condition's degraded images and whose calib/label_2/ImageSets/pkl symlink back
to the standard root:

    <COND_ROOTS>/<dg>_<sev>/
        training/image_2  -> <KITTI_OBJ>/degraded/<dg>/<sev>
        training/calib    -> <KITTI_STD_ROOT>/training/calib
        training/label_2  -> <KITTI_STD_ROOT>/training/label_2
        ImageSets         -> <KITTI_STD_ROOT>/ImageSets
        kitti_infos_val.pkl -> <KITTI_STD_ROOT>/kitti_infos_val.pkl

Usage:
    python eval_det.py --model [smoke/pgd]
"""
import argparse
import os

import pandas as pd
from tqdm import tqdm

import config as C
import kitti_iou_cpu

MODELS = {
    "smoke": dict(
        cfg="smoke_dla34_dlaneck_gn-all_4xb8-6x_kitti-mono3d.py",
        ckpt="smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d_"
             "20210929_015553-d46d9bb0.pth", anchor_ap3d_mod_r11=13.8),
    "pgd": dict(
        cfg="pgd_r101-caffe_fpn_head-gn_4xb3-4x_kitti-mono3d.py",
        ckpt="pgd_r101_caffe_fpn_gn-head_3x4_4x_kitti-mono3d_"
             "20211022_102608-8a97533b.pth",
        anchor_ap3d_mod_r11=18.33),
}
CKPT_DIR = C.REPO_ROOT.parent / "ckpts"


def build_cond_root(degradation, severity):
    """Create (idempotently) the symlink root for one condition."""
    name = f"{degradation}_{severity}"
    root = C.COND_ROOTS / name
    (root / "training").mkdir(parents=True, exist_ok=True)

    if degradation == "clean":
        img_src = C.KITTI_STD_ROOT / "training/image_2"
    else:
        img_src = C.KITTI_OBJ / C.DEGRADED_SUBDIR / degradation / str(severity)
    if not img_src.exists():
        raise FileNotFoundError(f"missing degraded images: {img_src}")
    # A directory that merely EXISTS is not enough: while data is still being
    # copied in, a condition can be half-populated, and the dataloader would
    # then die partway through a multi-hour run. Fail fast and say which
    # condition is short instead.
    n_have = sum(1 for _ in os.scandir(img_src))
    n_want = len(C.split_ids("det"))
    if n_have < n_want:
        raise FileNotFoundError(
            f"{degradation}/{severity} is incomplete: {n_have}/{n_want} images "
            f"in {img_src} -- still transferring? (--resume will pick this "
            f"condition up once it is complete)")

    links = {
        root / "training/image_2": img_src,
        root / "training/calib": C.KITTI_STD_ROOT / "training/calib",
        root / "training/label_2": C.KITTI_STD_ROOT / "training/label_2",
        root / "ImageSets": C.KITTI_STD_ROOT / "ImageSets",
        root / "kitti_infos_val.pkl": C.KITTI_STD_ROOT / "kitti_infos_val.pkl",
    }
    for dst, src in links.items():
        if dst.is_symlink() or dst.exists():
            if dst.is_symlink() and os.readlink(dst) == str(src):
                continue
            dst.unlink()
        dst.symlink_to(src)
    return root


def run_condition(model_key, degradation, severity, work_dir):
    """Run one condition in-process and return mmengine's metric dict."""
    from mmengine.config import Config
    from mmengine.runner import Runner

    spec = MODELS[model_key]
    root = build_cond_root(degradation, severity)
    tag = f"{model_key}_{degradation}_{severity}"

    cfg = Config.fromfile(str(CKPT_DIR / spec["cfg"]))
    cfg.launcher = "none"
    cfg.work_dir = str(work_dir / tag)
    cfg.load_from = str(CKPT_DIR / spec["ckpt"])

    cfg.test_dataloader.dataset.data_root = str(root) + "/"
    cfg.test_dataloader.dataset.ann_file = "kitti_infos_val.pkl"
    # MUST be set separately -- see the GOTCHA note in the module docstring.
    cfg.test_evaluator.ann_file = str(root / "kitti_infos_val.pkl")
    preds = C.RESULTS / "task_perf" / "preds"
    preds.mkdir(parents=True, exist_ok=True)
    cfg.test_evaluator.pklfile_prefix = str(preds / tag)

    runner = Runner.from_cfg(cfg)
    return runner.test()


def flatten(metrics, model_key, degradation, severity):
    """Metric dict -> one flat row, plus short aliases for the headline APs.

    Raw keys look like
        'Kitti metric/pred_instances_3d/KITTI/Car_3D_AP40_moderate_strict'
    ('strict' = the 0.7 IoU threshold KITTI ranks Car on).
    """
    row = dict(model=model_key, degradation=degradation, severity=severity)
    for k, v in metrics.items():
        try:
            row[k] = float(v)
        except (TypeError, ValueError):
            continue
    alias = {
        "ap3d_easy_r40": "Car_3D_AP40_easy_strict",
        "ap3d_mod_r40": "Car_3D_AP40_moderate_strict",
        "ap3d_hard_r40": "Car_3D_AP40_hard_strict",
        "apbev_easy_r40": "Car_BEV_AP40_easy_strict",
        "apbev_mod_r40": "Car_BEV_AP40_moderate_strict",
        "apbev_hard_r40": "Car_BEV_AP40_hard_strict",
        "ap2d_mod_r40": "Car_2D_AP40_moderate_strict",
        # AP11 kept only to check against the metafile anchor (see MODELS);
        # KITTI has ranked on R40 since 2019 -- do not report AP11.
        "ap3d_mod_r11": "Car_3D_AP11_moderate_strict",
    }
    for short, suffix in alias.items():
        hit = [v for k, v in row.items() if k.endswith(suffix)]
        if hit:
            row[short] = hit[0]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--resume", action="store_true",
                    help="skip conditions already present in the output file")
    ap.add_argument("--iou", choices=["auto", "gpu", "cpu"], default="auto",
                    help="rotated-IoU backend for KITTI AP (default auto: "
                         "mmdet3d's GPU kernel if numba can compile it here, "
                         "else the CPU port)")
    a = ap.parse_args()

    backend = kitti_iou_cpu.patch(mode=a.iou, verbose=True)
    print(f"[eval_det] rotated-IoU backend: {backend}", flush=True)

    out = C.RESULTS / "task_perf" / f"mono3d_{a.model}_condition.parquet"
    work_dir = C.RESULTS / "logs" / f"mono3d_{a.model}"

    conditions = C.all_conditions()
    rows = []
    if a.resume and out.exists():
        prev = pd.read_parquet(out)
        rows = prev.to_dict("records")
        done = {(r["degradation"], r["severity"]) for r in rows}
        conditions = [c for c in conditions if c not in done]
        print(f"[eval_det] resuming: {len(done)} done, "
              f"{len(conditions)} to go")

    for dg, sev in tqdm(conditions, desc=f"mono3d {a.model}"):
        metrics = run_condition(a.model, dg, sev, work_dir)
        row = flatten(metrics, a.model, dg, sev)
        rows.append(row)
        pd.DataFrame(rows).to_parquet(out, index=False)
        print(f"[eval_det] {a.model} {dg}_{sev}: "
              f"3D_mod={row.get('ap3d_mod_r40', float('nan')):.4f} "
              f"BEV_mod={row.get('apbev_mod_r40', float('nan')):.4f}")

    print(f"[eval_det] wrote {out}")


if __name__ == "__main__":
    main()

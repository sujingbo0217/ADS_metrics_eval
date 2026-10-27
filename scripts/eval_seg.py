"""Semantic segmentation evaluation over the 18x5 (+clean) conditions.

Two datasets
  --dataset cityscapes : PRIMARY. Official val 500, in-domain models (trained on
        Cityscapes train 2,975). GT read directly as *_gtFine_labelTrainIds.png.
  --dataset kitti      : SECONDARY. KITTI-200 zero-shot, GT is Cityscapes labelIds

Outputs:
  RESULTS/task_perf/seg_<model>_perimage.parquet   (per-image mIoU/pixacc/entropy)
  RESULTS/task_perf/seg_<model>_condition.parquet  (global-confusion-matrix mIoU)

Usage:
    python eval_seg.py --dataset cityscapes --config <cfg.py> --ckpt <w.pth> \
                       --model-name deeplabv3plus_r101
"""
import argparse

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import config as C

N_CLS = 19
LABEL2TRAIN = np.full(256, 255, dtype=np.uint8)
for lid, tid in {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8,
                 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16,
                 32: 17, 33: 18}.items():
    LABEL2TRAIN[lid] = tid

# Clean-mIoU sanity anchors from the mmseg model-zoo READMEs (guidebook 3.2e).
CLEAN_ANCHOR = {"deeplabv3plus_r101": 0.805, "mask2former_swinl": 0.832}


def confusion(gt, pred):
    m = gt != 255
    idx = gt[m].astype(np.int64) * N_CLS + pred[m].astype(np.int64)
    return np.bincount(idx, minlength=N_CLS * N_CLS).reshape(N_CLS, N_CLS)


def miou_from_cm(cm, present_only=False):
    tp = np.diag(cm).astype(np.float64)
    denom = cm.sum(1) + cm.sum(0) - tp
    iou = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
    if present_only:
        keep = cm.sum(1) > 0            # classes present in this image's GT
        iou = iou[keep]
    return float(np.nanmean(iou)), iou


def load_gt(dataset, ids):
    """image_id -> trainId map (uint8, 255 = ignore)."""
    gts = {}
    for iid in tqdm(ids, desc="load GT", leave=False):
        if dataset == "cityscapes":
            # already trainIds (csCreateTrainIdLabelImgs output)
            g = np.asarray(Image.open(C.cityscapes_gt_path(iid)))
            gts[iid] = g.astype(np.uint8)
        else:
            g = np.asarray(Image.open(
                C.KITTI_SEG / f"training/semantic/{iid}.png"))
            gts[iid] = LABEL2TRAIN[g]
    return gts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cityscapes", "kitti"],
                    default="cityscapes")
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke test: use only the first N images")
    ap.add_argument("--resume", action="store_true",
                    help="skip conditions already present in the output files")
    a = ap.parse_args()

    from mmengine import Config
    from mmseg.apis import init_model, inference_model

    cfg = Config.fromfile(a.config)
    # --- protocol lock: whole-image inference for every condition/model -------
    cfg.model.test_cfg = dict(mode="whole")
    model = init_model(cfg, a.ckpt, device="cuda")

    task = "cityscapes" if a.dataset == "cityscapes" else "seg"
    ids = C.split_ids(task)
    if a.limit:
        ids = ids[:a.limit]
    print(f"[eval_seg] {a.dataset}: {len(ids)} images, model={a.model_name}")
    gts = load_gt(a.dataset, ids)

    conditions = C.all_conditions()
    out = C.RESULTS / "task_perf"
    per_path = out / f"seg_{a.model_name}_perimage.parquet"
    cond_path = out / f"seg_{a.model_name}_condition.parquet"

    per_rows, cond_rows = [], []
    if a.resume and cond_path.exists() and per_path.exists():
        cond_rows = pd.read_parquet(cond_path).to_dict("records")
        per_rows = pd.read_parquet(per_path).to_dict("records")
        done = {(r["degradation"], r["severity"]) for r in cond_rows}
        conditions = [c for c in conditions if c not in done]
        print(f"[eval_seg] resuming: {len(done)} conditions done, "
              f"{len(conditions)} to go")

    for dg, sev in conditions:
        cm_global = np.zeros((N_CLS, N_CLS), dtype=np.int64)
        for iid in tqdm(ids, desc=f"seg {dg}_{sev}", leave=False):
            p = str(C.degraded_image_path(task, iid, dg, sev))
            res = inference_model(model, p)
            pred = res.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.uint8)
            # softmax entropy over classes as a confidence proxy (mean/pixel)
            try:
                logits = res.seg_logits.data.float()
                pr = logits.softmax(0)
                ent = float(-(pr * (pr + 1e-12).log()).sum(0).mean())
            except Exception:
                ent = np.nan
            gt = gts[iid]
            if pred.shape != gt.shape:
                pred = np.asarray(Image.fromarray(pred).resize(
                    (gt.shape[1], gt.shape[0]), Image.NEAREST))
            cm = confusion(gt, pred)
            cm_global += cm
            miou_i, _ = miou_from_cm(cm, present_only=True)
            m = gt != 255
            per_rows.append(dict(task=task, model=a.model_name, degradation=dg,
                                 severity=sev, image_id=iid, miou=miou_i,
                                 pixacc=float((pred[m] == gt[m]).mean()),
                                 entropy=ent))
        miou_g, iou_vec = miou_from_cm(cm_global)
        cond_rows.append(dict(model=a.model_name, degradation=dg,
                              severity=sev, miou=miou_g,
                              **{f"iou_{c}": float(v)
                                 for c, v in enumerate(iou_vec)}))
        pd.DataFrame(per_rows).to_parquet(per_path, index=False)
        pd.DataFrame(cond_rows).to_parquet(cond_path, index=False)
        print(f"seg {dg}_{sev}: mIoU={miou_g:.4f}")

    clean_miou = cond_rows[0]["miou"]

    print(f"[eval_seg] clean mIoU = {clean_miou:.4f}; wrote {per_path.name} "
          f"and {cond_path.name} ({len(cond_rows)} conditions)")


if __name__ == "__main__":
    main()

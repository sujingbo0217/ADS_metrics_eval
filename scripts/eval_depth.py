"""Monocular depth evaluation on degraded Eigen test split.

Monodepth2 protocol:
  - predict disparity at 640x192, resize to full res, depth = 1/disp
  - per-image metrics: abs_rel, sq_rel, rmse, rmse_log, a1..a3, silog(x100)

Usage:
    python eval_depth.py --weights ckpts/mono_640x192
"""
import argparse
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

import config as C

sys.path.insert(0, str(C.MONODEPTH2_REPO))
from layers import disp_to_depth            # noqa: E402  (monodepth2 repo)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MIN_D, MAX_D = 1e-3, 80.0
MONO_MIN_D, MONO_MAX_D = 0.1, 100.0


def garg_crop_mask(h, w):
    m = np.zeros((h, w), dtype=bool)
    m[int(0.40810811 * h):int(0.99189189 * h),
      int(0.03594771 * w):int(0.96405229 * w)] = True
    return m


def metrics(gt, pred):
    thresh = np.maximum(gt / pred, pred / gt)
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    rmse = np.sqrt(((gt - pred) ** 2).mean())
    rmse_log = np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())
    abs_rel = (np.abs(gt - pred) / gt).mean()
    sq_rel = (((gt - pred) ** 2) / gt).mean()
    e = np.log(pred) - np.log(gt)
    silog = (np.mean(e ** 2) - np.mean(e) ** 2) * 100.0
    return dict(abs_rel=abs_rel, sq_rel=sq_rel, rmse=rmse, rmse_log=rmse_log,
                a1=a1, a2=a2, a3=a3, silog=silog)


def load_model(weights_dir):
    import networks  # monodepth2
    enc_p, dec_p = f"{weights_dir}/encoder.pth", f"{weights_dir}/depth.pth"
    enc_ckpt = torch.load(enc_p, map_location="cpu")
    feed_h, feed_w = enc_ckpt["height"], enc_ckpt["width"]
    encoder = networks.ResnetEncoder(18, False)
    encoder.load_state_dict({k: v for k, v in enc_ckpt.items()
                             if k in encoder.state_dict()})
    decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))
    decoder.load_state_dict(torch.load(dec_p, map_location="cpu"))
    encoder.eval().to(DEV)
    decoder.eval().to(DEV)
    return encoder, decoder, feed_h, feed_w


@torch.no_grad()
def predict_depth(encoder, decoder, pil, feed_h, feed_w, out_hw):
    """
    The decoder's ("disp", 0) head is a raw SIGMOID, not a disparity. monodepth2
    maps it through disp_to_depth(d, 0.1, 100):
        scaled_disp = 1/100 + (1/0.1 - 1/100) * d ;  depth = 1/scaled_disp
    """
    x = pil.resize((feed_w, feed_h), Image.LANCZOS)
    x = torch.from_numpy(np.asarray(x)).permute(
        2, 0, 1).float().unsqueeze(0) / 255.0
    sigmoid_disp = decoder(encoder(x.to(DEV)))[("disp", 0)]
    scaled_disp, _ = disp_to_depth(sigmoid_disp, MONO_MIN_D, MONO_MAX_D)
    scaled_disp = torch.nn.functional.interpolate(
        scaled_disp, out_hw, mode="bilinear", align_corners=False)
    return 1.0 / scaled_disp.squeeze().cpu().numpy().astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--model-name", default="monodepth2")
    ap.add_argument("--no-median-scaling", action="store_true")
    a = ap.parse_args()

    encoder, decoder, fh, fw = load_model(a.weights)
    gt_depths = np.load(C.GT_DEPTHS_NPZ, allow_pickle=True,
                        fix_imports=True)["data"]
    ids = C.split_ids("depth")
    assert len(ids) == len(gt_depths) == 697, \
        f"Eigen split mismatch: {len(ids)} ids vs {len(gt_depths)} gts"

    conditions = C.all_conditions()
    rows = []
    for dg, sev in conditions:
        missing = 0
        for j, iid in enumerate(tqdm(ids, desc=f"depth {dg}_{sev}", leave=False)):
            p = C.degraded_image_path("depth", iid, dg, sev)
            if not p.exists():
                missing += 1
                continue
            gt = gt_depths[j]
            h, w = gt.shape
            # predict straight at GT resolution (resized disparity)
            pred = predict_depth(encoder, decoder, Image.open(p).convert("RGB"),
                                 fh, fw, (h, w))
            mask = (gt > MIN_D) & (gt < MAX_D) & garg_crop_mask(h, w)
            g, pr = gt[mask], pred[mask]
            if not a.no_median_scaling:
                pr = pr * np.median(g) / np.median(pr)
            pr = np.clip(pr, MIN_D, MAX_D)
            rows.append(dict(task="depth", model=a.model_name, degradation=dg,
                             severity=sev, image_id=iid, **metrics(g, pr)))
        if missing:
            print(f"[tasks_depth] WARNING {dg}_{sev}: {missing} images missing "
                  f"(check robodepth_path layout in config.py)")
        else:
            done = [r for r in rows if r["degradation"] == dg
                    and r["severity"] == sev]
            print(f"depth {dg}_{sev}: abs_rel="
                  f"{np.mean([r['abs_rel'] for r in done]):.4f} "
                  f"a1={np.mean([r['a1'] for r in done]):.4f}")

    df = pd.DataFrame(rows)
    per = C.RESULTS / "task_perf" / f"depth_{a.model_name}_perimage.parquet"
    df.to_parquet(per)
    cond = (df.groupby(["degradation", "severity"])
              .mean(numeric_only=True).reset_index())
    cond.to_parquet(C.RESULTS / "task_perf" /
                    f"depth_{a.model_name}_condition.parquet")
    print(df.groupby(["degradation", "severity"])["abs_rel"].mean().head(12))
    print(f"[tasks_depth] wrote {per}")


if __name__ == "__main__":
    main()

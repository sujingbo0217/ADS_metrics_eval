"""Do realism metrics track task damage?

Metrics and tasks: larger = worse, rho > 0 means "the metric agrees with the
damage". The clean condition is excluded: drops are defined relative to it.

--level distribution
    FID, KID, FD@DINOv2, KD@DINOv2:
        D_c = s (T_c - T_clean) / |T_clean|,   s = -1 for AP/mIoU, +1 for AbsRel
        rho, p = Spearman(metric_c, D_c) over 90 degradation conditions

--level image
    The 12 image-level metrics against each image's drop
        y_is  = s (T_is - T_i,clean)     image i at severity s vs its clean copy
        rho_d = Spearman(metric_is, y_is) over all N images x 5 severities of
                corruption type d
        rho   = tanh( mean_d atanh(rho_d) ) over the 18 types

Usage
    python correlate_metric_task.py --level [distribution/image]
"""
import argparse
import time

import numpy as np
import pandas as pd
from scipy import stats

import config as C
import data_utils as D

CLIP = 0.999999


# ---------------------------------------------------------------------------
# distribution level
# ---------------------------------------------------------------------------
def run_distribution():
    rows = []
    for model, spec in C.MODELS.items():
        met, drop = D.distribution_metrics(spec["dataset"]), D.condition_drop(model)
        for name, direction in C.DISTRIBUTION_METRICS.items():
            j = pd.concat([(direction * met[name]).rename("x"), drop.rename("y")],
                          axis=1, join="inner").dropna()
            r, p = stats.spearmanr(j.x, j.y)
            rows.append(dict(dataset=spec["dataset"], task=spec["task"], model=model,
                             metric=name, n=len(j), rho=r, p=p))
    df = pd.DataFrame(rows)
    out = C.ANALYSIS_DIR / "exp_distribution.csv"
    df.to_csv(out, index=False)

    print("=" * 78)
    print("EXPERIMENT A  Spearman(metric, relative task drop) over 90 conditions")
    print("=" * 78)
    print(df.pivot(index="model", columns="metric", values="rho")
            .reindex(index=list(C.MODELS), columns=list(C.DISTRIBUTION_METRICS))
            .round(3).to_string())
    print(f"-> {out}")


# ---------------------------------------------------------------------------
# per-image level
# ---------------------------------------------------------------------------
def grid(s, ids):
    """Series indexed (severity, image_id) -> [N images, 5 severities] array."""
    return s.unstack("severity").reindex(index=ids, columns=C.SEVERITIES).to_numpy(float)


def fisher(r):
    return np.tanh(np.nanmean(np.arctanh(np.clip(r, -CLIP, CLIP)), axis=0))


def image_model(model, met, metrics=C.IMAGE_METRICS):
    spec = C.MODELS[model]
    j = met.join(D.image_drop(model), how="inner")
    ids = j.index.get_level_values("image_id").unique()
    n = len(ids)
    degs = [d for d in C.DEGRADATIONS if d in j.index.get_level_values("degradation")]

    point = {m: {} for m in metrics}                 # metric -> {type: rho_d}
    for d in degs:
        sub = j.xs(d, level="degradation")
        Y = grid(sub["drop"], ids)
        if np.isnan(Y).any():
            raise ValueError(f"{model}/{d}: some images miss a severity")
        for m, direction in metrics.items():
            X = direction * grid(sub[m], ids)
            if np.nanstd(X) == 0:
                continue                             # constant metric: undefined rho
            point[m][d] = stats.spearmanr(X.ravel(), Y.ravel()).statistic

    rows, rows_d = [], []
    for m in metrics:
        ds = list(point[m])
        r_d = np.array([point[m][d] for d in ds])
        z = np.arctanh(np.clip(r_d, -CLIP, CLIP))
        rows.append(dict(dataset=spec["dataset"], task=spec["task"], model=model,
                         metric=m, n_images=n, n_types=len(ds), rho=fisher(r_d),
                         z_sd=z.std(ddof=1), n_pos=int((r_d > 0).sum()),
                         rho_min=r_d.min(), rho_max=r_d.max()))
        for d, r in zip(ds, r_d):
            rows_d.append(dict(dataset=spec["dataset"], model=model, metric=m,
                               degradation=d, n=5 * n, rho=r))
    return rows, rows_d


def run_image():
    rows, rows_d = [], []
    for dataset in C.IMAGE_DATASETS:
        met = D.image_metrics(dataset)
        for model in C.models_on(dataset):
            t0 = time.time()
            r, rd = image_model(model, met)
            rows += r
            rows_d += rd
            print(f"  {model:12s} {dataset:12s} done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    out, out_d = C.ANALYSIS_DIR / "exp_image.csv", C.ANALYSIS_DIR / "exp_image_by_degradation.csv"
    df.to_csv(out, index=False)
    pd.DataFrame(rows_d).to_csv(out_d, index=False)

    models = [m for d in C.IMAGE_DATASETS for m in C.models_on(d)]
    print("=" * 78)
    print("EXPERIMENT B  per-image Spearman(metric, image drop) per corruption type,")
    print("              Fisher-z over 18 types")
    print("=" * 78)
    p = df.pivot(index="metric", columns="model", values="rho").reindex(columns=models)
    print(p.reindex(p.mean(1).sort_values(ascending=False).index).round(3).to_string())
    print(f"-> {out}\n-> {out_d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", required=True, choices=["distribution", "image"])
    a = ap.parse_args()
    if a.level == "distribution":
        run_distribution()
    else:
        run_image()


if __name__ == "__main__":
    main()

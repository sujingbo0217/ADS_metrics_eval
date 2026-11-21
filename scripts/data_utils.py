import pandas as pd

import config as C

CLIP = 0.999999
COND = ["degradation", "severity"]
IMAGE = ["degradation", "severity", "image_id"]
GEOMETRY_COLS = [
    "construction",
    "gp1_e", "gp2_rho", "gp2_delta", "gp3_h", "gp3_kappa", "gp4_tau",
    "gp5_m", "gp5_chi", "gp6_T", "gp6_C", "gp6_b", "gp7_o",
    "gp3_h_sel", "gp4_tau_sel", "gp5_m_sel", "gp6_T_sel", "gp6_C_sel", "gp6_b_sel",
    "gp1_er_real", "gp1_er_gen", "gp2_id_real", "gp2_id_gen",
    "gp5_m_real", "gp5_m_gen", "gp7_nu", "gp7_nu0", "gp7_d_t",
]

DISTRIBUTION_METRICS_CSV = C.ANALYSIS_DIR / "distribution_metrics.csv"
GEOMETRY_CSV = C.ANALYSIS_DIR / "geometry.csv"


def image_metrics_path(dataset):
    return C.ANALYSIS_DIR / f"image_metrics_{dataset}.parquet"


def image_geometry_path(dataset):
    return C.ANALYSIS_DIR / f"image_geometry_{dataset}.parquet"


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def distribution_metrics(dataset):
    """The four set metrics of extract_metrics.py."""
    d = pd.read_csv(DISTRIBUTION_METRICS_CSV)
    d = d[(d.dataset == dataset) & (d.degradation != "clean")]
    return d.drop(columns="dataset").set_index(COND)


def image_metrics(dataset):
    d = pd.read_parquet(image_metrics_path(dataset))
    return d[d.degradation != "clean"].set_index(IMAGE).sort_index()


def condition_metrics(dataset):
    """Every metric available for dataset at condition level, in C.METRICS order.

    Image-level metrics only on C.IMAGE_DATASETS, whose tasks score every image;
    detection (kitti_det) has condition-level scores only, so it gets only the
    distribution-level columns, even where an image-level file exists."""
    d = distribution_metrics(dataset)
    if dataset in C.IMAGE_DATASETS:
        d = d.join(image_metrics(dataset).groupby(level=COND).mean(), how="inner")
    return d[[m for m in C.METRICS if m in d.columns]]


# ---------------------------------------------------------------------------
# task performance
# ---------------------------------------------------------------------------
def task_score(model, level="condition", oriented=False, clean=False):
    """Raw task score of model, indexed by COND (level="condition") or IMAGE
    (level="perimage"). oriented=True multiplies by s so larger = worse."""
    m = C.MODELS[model]
    if level == "perimage" and m["task"] == "detection":
        raise ValueError(f"{model}: AP has no per-image form")
    t = pd.read_parquet(C.RESULTS / "task_perf" / f"{m['perf']}_{level}.parquet")
    if not clean:
        t = t[t.degradation != "clean"]
    s = t.set_index(COND if level == "condition" else IMAGE)[m["score"]].sort_index()
    return (sign(model) * s if oriented else s).rename(m["score"])


def sign(model):
    return -1.0 if C.MODELS[model]["higher_better"] else 1.0


def condition_drop(model):
    t = task_score(model, clean=True)
    clean = float(t.loc[C.CLEAN])
    t = t.drop(C.CLEAN)
    return (sign(model) * (t - clean) / abs(clean)).rename("drop")


def image_drop(model):
    t = task_score(model, "perimage", clean=True).reset_index()
    clean = t[t.degradation == "clean"].set_index("image_id")[C.MODELS[model]["score"]]
    t = t[t.degradation != "clean"]
    y = sign(model) * (t[C.MODELS[model]["score"]].to_numpy()
                       - clean.loc[t.image_id].to_numpy())
    return pd.Series(y, index=pd.MultiIndex.from_frame(t[IMAGE]), name="drop").sort_index()


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def geometry(dataset):
    """degradation, severity, space, k, GEOMETRY_COLS; None before compute_geometry.py."""
    if not GEOMETRY_CSV.exists():
        return None
    g = pd.read_csv(GEOMETRY_CSV)
    g = g[g.dataset == dataset].drop(columns="dataset")
    return g if len(g) else None


def image_geometry(dataset):
    p = image_geometry_path(dataset)
    return pd.read_parquet(p) if p.exists() else None


def write_geometry(dataset, rows):
    """Replace dataset's rows of geometry.csv, conditions in config order."""
    new = pd.DataFrame(rows).assign(dataset=dataset)
    new = new[["dataset", *COND, "space", "k", *GEOMETRY_COLS]]
    if GEOMETRY_CSV.exists():
        old = pd.read_csv(GEOMETRY_CSV)
        if list(old.columns) != list(new.columns):
            print(f"  {GEOMETRY_CSV.name}: dropping {old.dataset.nunique()} "
                  "dataset(s) written with the previous columns")
            old = old.iloc[0:0]
        new = pd.concat([old[old.dataset != dataset], new], ignore_index=True)
    order = {c: i for i, c in enumerate(C.all_conditions())}
    new = (new.assign(_d=new.dataset.map({d: i for i, d in enumerate(C.DATASETS)}),
                      _c=[order[c] for c in zip(new.degradation, new.severity)])
              .sort_values(["_d", "_c", "space", "k"], kind="stable")
              .drop(columns=["_d", "_c"]))
    new.to_csv(GEOMETRY_CSV, index=False)


def write_image_geometry(dataset, frame):
    frame.to_parquet(image_geometry_path(dataset), index=False)

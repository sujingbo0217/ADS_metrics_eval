"""Geometric properties computed on (R, G) embedding pairs via knn_ext.

R is the clean set, G the corrupted set of one condition. 
The two are PAIRED:
row i of G is the corruption of row i of R,

    GP1 (*) spectral shape    e     = log ER(G) / ER(R)
    GP2 (*) local scale       rho   = med log r_k(g_i) - med log r_k(r_i)
            and dimension     delta = log TwoNN(G) / TwoNN(R)
    GP3     hubness           h     = skew O_k(G) - skew O_k(R)
            concentration     kappa = log RC(G) / RC(R)
    GP4     connectivity      tau   = mean_{k in 1,2,3,5} log beta0_G(k) / beta0_R(k)
    GP5 (*) mixing            m     = other-set share over its random expectation 1/2,
                                        each point's counterpart excluded
            support           chi   = log med LID_k(g_i; R \ r_i) / med LID_k(r_i; R \ r_i)
    GP6     preservation      T, C  = Venna & Kaski trustworthiness and continuity
            drift             b     = med_i |bary_R(A_i) - bary_R(B_i)| / r_k(r_i)
    GP7     off-manifold      o     = med nu_i - med nu0_i
"""
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
try:
    import knn_ext
except ModuleNotFoundError as e:                       # pragma: no cover
    raise ModuleNotFoundError(
        "knn_ext is not importable. It is a CUDA pybind11 module and must be "
        "built against the SAME Python that imports it -- a module built for "
        "another version is silently invisible here, it does not raise.\n"
        f"  this interpreter : {sys.version.split()[0]} at {sys.executable}\n"
        "  expected module  : "
        f"{Path(__file__).resolve().parent.parent / 'python'}/"
        f"knn_ext.cpython-{sys.version_info.major}{sys.version_info.minor}-*.so\n"
        "Build it with:\n"
        "  cd <repo> && rm -rf build && cmake -S . -B build "
        "-DPython3_EXECUTABLE=$(which python) "
        "-Dpybind11_DIR=$(python -m pybind11 --cmakedir) && "
        "cmake --build build -j\n"
        "See README, Build."
    ) from e

CN = knn_ext.Construction


COSINE_SPACES = {"clip_vitb32", "clip_vit_b32", "vgg_lpips_cosine"}

HEADLINE_COLS = ["gp1_e", "gp2_rho", "gp2_delta", "gp3_h", "gp3_kappa", "gp4_tau",
                 "gp5_m", "gp5_chi", "gp6_T", "gp6_C", "gp6_b", "gp7_o"]

RELATIONAL_COLS = ["gp3_h", "gp4_tau", "gp5_m", "gp6_T", "gp6_C", "gp6_b"]
POINT_COLS = ["lid_real", "lid_gen", "rk_real", "rk_gen",      # GP2
              "occ_gen", "contrast_gen",                       # GP3
              "mix_real", "mix_gen", "xlid_gen",               # GP5
              "trust", "cont", "drift",                        # GP6
              "nu", "nu0"]                                     # GP7
POINT_SELECTED = ["occ_gen", "mix_real", "mix_gen", "trust", "cont", "drift"]

K_COMPONENTS = (1, 2, 3, 5)     # the sparse graphs GP4 compares
K_T = 50                        # GP7 local-PCA size, more points than d_t

CANDIDATES = {
    "primary":       lambda k: (CN.Primary, 0),
    "snn_k":         lambda k: (CN.SharedNeighbors, k),
    "snn_2k":        lambda k: (CN.SharedNeighbors, 2 * k),
    "local_scaling": lambda k: (CN.LocalScaling, k),
    "geodesic_k/2":  lambda k: (CN.Geodesic, math.ceil(k / 2)),
    "geodesic_k":    lambda k: (CN.Geodesic, k),
}


def dist_type(space):
    return (knn_ext.DistanceType.InnerProduct if space in COSINE_SPACES
            else knn_ext.DistanceType.L2)


def prepare(X, space):
    """float32 and L2-normalized where the space is cosine.

    The bindings declare every array c_style | forcecast, so dtype and layout
    are numpy's problem now, not ours.
    """
    if space in COSINE_SPACES:
        return knn_ext.l2_normalize_rows(X)
    return np.ascontiguousarray(X, dtype=np.float32)


def _nb(construction, k):
    kind, param = CANDIDATES[construction](k)
    return dict(construction=kind, param=param)


def _log_ratio(a, b):
    return float(np.log(max(a, 1e-12) / max(b, 1e-12)))


def _nanmedian(x):
    """Median over the finite entries; NaN, without a warning, if there are none
    (nu at G = R, where every displacement is zero)."""
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else np.nan


def neighborhood_quality(R, ks, space, task_scores=None):
    """Q1--Q4 of every candidate at every k, on the clean set only.

    task_scores maps a model name to its clean per-image task score in row
    order of R. Where given, Q5 -- the Spearman correlation between an image's
    score and the mean score of its neighbors -- is added for each model. It
    is reported and never selected on (Sec. sec:neighborhoods).
    """
    dt = dist_type(space)
    n = R.shape[0]
    rows = []
    for name in CANDIDATES:
        for k in ks:
            kind, param = CANDIDATES[name](k)
            row = dict(candidate=name, k=k, param=param)
            if k > n - 1 or param > n - 1:
                rows.append(dict(row, feasible=False))
                continue
            try:
                q = knn_ext.neighborhood_quality(R, k, dt, kind, param)
            except ValueError:
                # e.g. a subsample in Q4 too small for k_s
                rows.append(dict(row, feasible=False))
                continue
            row.update(feasible=True, **q)
            if task_scores:
                idx = knn_ext.neighbor_lists(R, k, dt, kind, param)
                for model, y in task_scores.items():
                    row[f"q5_{model}"] = _q5(idx, y)
            rows.append(row)
    return rows


def _q5(idx, y):
    """Spearman of y_i against the mean y over i's neighbors (-1 = none)."""
    y = np.asarray(y, dtype=np.float64)
    valid = idx >= 0
    pred = np.where(valid, y[np.where(valid, idx, 0)], 0.0).sum(1)
    cnt = valid.sum(1)
    ok = (cnt > 0) & np.isfinite(y)
    pred = pred[ok] / cnt[ok]
    return float(stats.spearmanr(pred, y[ok]).statistic) if ok.sum() > 2 else np.nan


def select_construction(quality):
    """The selection rule of Sec. sec:neighborhoods, on neighborhood_quality rows.

    A candidate is admissible if it is feasible at every k of the sweep and
    passes Q2 there: no more components than the primary construction, and one
    component for a geodesic. Among the admissible ones it is the lowest mean
    rank over Q1 (lower better), Q3 (lower better) and Q4 (higher better), the
    ranks taken at each k and then averaged over k. Ties go to primary.
    Returns (name, per-candidate table).
    """
    import pandas as pd
    q = pd.DataFrame(quality)
    prim = q[q.candidate == "primary"].set_index("k").components
    ok = q.feasible & (q.components <= q.k.map(prim))
    ok &= ~q.candidate.str.startswith("geodesic") | (q.components == 1)
    q["admissible"] = ok
    adm = q.groupby("candidate").admissible.all()
    adm = adm[adm].index
    a = q[q.candidate.isin(adm)].copy()
    a["r1"] = a.groupby("k").hub_skew.rank()
    a["r3"] = a.groupby("k").lid_cv.rank()
    a["r4"] = a.groupby("k").stability.rank(ascending=False)
    a["rank"] = a[["r1", "r3", "r4"]].mean(axis=1)
    mean_rank = a.groupby("candidate")["rank"].mean()
    order = sorted(mean_rank.index, key=lambda c: (mean_rank[c], c != "primary"))
    chosen = order[0] if order else "primary"
    # admissible is per k; a candidate enters the ranking only if it holds at all k
    q["admissible_all"] = q.candidate.isin(adm)
    q["mean_rank"] = q.candidate.map(mean_rank)
    q["selected"] = q.candidate == chosen
    return chosen, q


def clean_reference(R, space, ks, selected="primary", k_t=K_T):
    dt = dist_type(space)
    n = R.shape[0]
    ref = dict(space=space, ks=list(ks), selected=selected, k_t=min(k_t, n - 1))
    ref["er"] = knn_ext.effective_rank(R)
    mean_d, nn_d = knn_ext.relative_contrast(R, dt)
    ref["rc"] = float(mean_d.sum() / nn_d.sum())
    cons = ["primary"] + ([selected] if selected != "primary" else [])
    ref["beta0"] = {}
    for c in cons:
        for k in ks:
            ref["beta0"][c, k] = knn_ext.component_counts(R, list(K_COMPONENTS), dt,
                                                          **_nb(c, k))
    ref["per_k"] = {}
    for k in ks:
        lid, rk, twonn = knn_ext.lid_profile(R, k, dt)
        d = dict(lid=lid, rk=rk, twonn=float(twonn),
                 xlid=knn_ext.cross_lid(R, R, k, dt, True))
        for c in cons:
            d["occ_skew", c] = float(stats.skew(knn_ext.k_occurrence(R, k, dt, **_nb(c, k))))
        ref["per_k"][k] = d
    ref["id"] = ref["per_k"][ks[0]]["twonn"]
    ref["d_t"] = int(np.clip(round(ref["id"]), 1, ref["k_t"] - 1))
    k_ref = min(max(ks), ref["k_t"])
    ref["nu0_all"] = knn_ext.off_manifold_reference(R, k_ref, ref["d_t"], ref["k_t"], dt)
    return ref


def condition_properties(R, G, ref, keep_pointwise=False):
    """All twelve columns, plus the _sel and extra columns, for every k of ref.

    R and G must be prepared and row-aligned. Returns a list of (row, pointwise)
    with one entry per k; pointwise is None unless keep_pointwise.
    """
    space, ks, sel = ref["space"], ref["ks"], ref["selected"]
    dt = dist_type(space)
    cons = ["primary"] + ([sel] if sel != "primary" else [])

    er_g = knn_ext.effective_rank(G)
    mean_d, nn_d = knn_ext.relative_contrast(G, dt)
    rc_g = float(mean_d.sum() / nn_d.sum())
    nu = knn_ext.off_manifold(R, G, ref["d_t"], ref["k_t"], dt)
    base = dict(gp1_e=_log_ratio(er_g, ref["er"]), gp1_er_real=ref["er"], gp1_er_gen=er_g,
                gp3_kappa=_log_ratio(rc_g, ref["rc"]),
                gp7_nu=_nanmedian(nu), gp7_d_t=ref["d_t"])

    out = []
    for k in ks:
        rf = ref["per_k"][k]
        lid_g, rk_g, twonn_g = knn_ext.lid_profile(G, k, dt)
        xlid_g = knn_ext.cross_lid(R, G, k, dt, True)
        nu0 = np.median(ref["nu0_all"][:, :min(k, ref["nu0_all"].shape[1])], axis=1)
        row = dict(base, construction=sel,
                   gp2_rho=float(np.median(np.log(np.maximum(rk_g, 1e-30)))
                                 - np.median(np.log(np.maximum(rf["rk"], 1e-30)))),
                   gp2_delta=_log_ratio(twonn_g, rf["twonn"]),
                   gp2_id_real=rf["twonn"], gp2_id_gen=float(twonn_g),
                   gp5_chi=_log_ratio(_nanmedian(xlid_g), _nanmedian(rf["xlid"])),
                   gp7_nu0=float(np.median(nu0)))
        row["gp7_o"] = row["gp7_nu"] - row["gp7_nu0"]
        pt = dict(lid_real=rf["lid"], lid_gen=lid_g, rk_real=rf["rk"], rk_gen=rk_g,
                  contrast_gen=mean_d / np.maximum(nn_d, 1e-30), xlid_gen=xlid_g,
                  nu=nu, nu0=nu0)

        for c in cons:
            sfx = "" if c == "primary" else "_sel"
            nb = _nb(c, k)
            occ = knn_ext.k_occurrence(G, k, dt, **nb)
            b0 = knn_ext.component_counts(G, list(K_COMPONENTS), dt, **nb)
            m_r, m_g = knn_ext.cross_set_mixing(R, G, k, dt, exclude_counterpart=True, **nb)
            t, cont = knn_ext.neighborhood_preservation(R, G, k, dt, **nb)
            drift = knn_ext.neighborhood_drift(R, G, k, dt, **nb)
            row["gp3_h" + sfx] = float(stats.skew(occ)) - rf["occ_skew", c]
            row["gp4_tau" + sfx] = float(np.mean(np.log(b0 / ref["beta0"][c, k])))
            row["gp5_m" + sfx] = 0.5 * (float(m_r.mean()) + float(m_g.mean()))
            row["gp6_T" + sfx] = float(t.mean())
            row["gp6_C" + sfx] = float(cont.mean())
            row["gp6_b" + sfx] = float(np.median(drift))
            if c == "primary":
                row["gp5_m_real"], row["gp5_m_gen"] = float(m_r.mean()), float(m_g.mean())
            pt.update({"occ_gen" + sfx: occ / k, "mix_real" + sfx: m_r,
                       "mix_gen" + sfx: m_g, "trust" + sfx: t, "cont" + sfx: cont,
                       "drift" + sfx: drift})
        if sel == "primary":
            for c in RELATIONAL_COLS:
                row[c + "_sel"] = row[c]
        out.append((row, pt if keep_pointwise else None))
    return out

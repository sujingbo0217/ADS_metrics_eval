"""Stage 7b: do the geometric properties predict task performance drop?

Mirrors correlate_metrics.py exactly, so that geometry and
the realism metrics are judged by the same instrument and can be put in one
table. A geometric property at a fixed (space, k) occupies the same slot a
metric column occupies.

    tier 1   one property value per condition, Spearman over the 90 conditions
             against relative task performance drop. -> geometry_tier1.csv
    tier 2   the per-image form of P3, P4 and P5 within each condition against
             per-image task error, aggregated by Fisher z over the 90
             conditions, at the requested k. -> geometry_tier2.csv

             P5's per-image value is a count of cross-set neighbours over k, so
             it takes only k+1 distinct values: 11 at k = 10, where 41% of one
             Cityscapes condition's 500 images share a single value. Its tier-2
             coefficient is therefore resolution-limited rather than merely
             small, and --k 50 is the honest setting to read it at. P3 and P4
             are continuous at every k.

             Only three of the five appear: P1 is a centroid displacement and P2
             a spectral entropy, both properties of a SET with no per-image
             share -- the same reason FID and KID are absent from the metric
             tier 2. This is a structural fact about the properties, not a gap
             in the implementation.

CONFIRMATORY FAMILY. Six headline properties at k = 10 only, across the task
runs their pool supports. The k ablation is reported as a sensitivity check and
is NOT folded into the multiple-comparison family: reusing the same conditions
at five k values yields five highly dependent tests, and correcting them as if
independent would be neither conservative nor meaningful.

TWO PROPERTIES HAVE NO EXPECTED SIGN, and this is a statement about the physics
rather than caution. P1, P4 and P5 are directional by construction: damage moves
the cloud away, breaks neighborhoods, and separates the two sets. P2 and P3 are
not. A degradation can either COLLAPSE the representation (blur and smoothing
remove directions and lower the local dimension) or INFLATE it (noise and
compression artefacts add them). Both are real damage, so a two-sided test is
the honest one, and a "wrong sign" flag would be meaningless. They are reported
with |rho| and no sign expectation.

Usage:
    python correlate_geometry.py
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

import config as C
import data_utils as D

# property -> expected sign of its correlation with damage, or None when the
# direction is genuinely not determined a priori (see the module docstring).
HEADLINE = {
    "p1_dmu":            +1,    # damage moves the centroid away
    "p2_log_er_ratio":   None,  # collapse or inflation, both are damage
    "p3_log_lid_ratio":  None,  # ditto
    "p4_trustworthiness": -1,   # damage invents neighbors
    "p4_continuity":      -1,   # damage loses neighbors
    "p5_rho":             -1,   # damage separates the two sets
}
EXTRA = ["p1_dmu_cloud", "p1_radial", "p1_cos", "p5_rho_scaled", "p5_asymmetry",
         "p2_er_real", "p2_er_gen", "p3_lid_real", "p3_lid_gen",
         "p5_rho_real", "p5_rho_gen"]


def tier1(k):
    rows = []
    for model, spec in C.MODELS.items():
        g = D.geometry(spec["dataset"])
        if g is None:
            continue
        pdrop = D.condition_drop(model)
        for space, gs in g[g.k == k].groupby("space"):
            gs = gs.set_index(["degradation", "severity"])
            for prop in list(HEADLINE) + EXTRA:
                if prop not in gs.columns:
                    continue
                j = pd.concat([gs[prop].rename("x"), pdrop.rename("y")],
                              axis=1, join="inner").dropna()
                if len(j) < 10 or j.x.nunique() < 3:
                    continue
                r, p = stats.spearmanr(j.x, j.y)
                tau, p_tau = stats.kendalltau(j.x, j.y, variant="b")
                exp = HEADLINE.get(prop, "n/a")
                rows.append(dict(
                    dataset=spec["dataset"], task=spec["task"], model=model,
                    space=space, k=k, prop=prop,
                    headline=prop in HEADLINE, n=len(j),
                    rho=r, abs_rho=abs(r), p=p, tau_b=tau, p_tau=p_tau,
                    expected_sign=exp,
                    wrong_sign=(bool(np.sign(r) != exp)
                                if isinstance(exp, int) else False)))
    if not rows:
        return None
    df = pd.DataFrame(rows)
    # Correct only within the confirmatory family; the extras are descriptive.
    m = df.headline
    df.loc[m, "p_fdr"] = multipletests(df.loc[m, "p"], method="fdr_bh")[1]
    df["sig"] = df.p_fdr < 0.05
    return df


def tier2(k=10):
    rows = []
    for dataset in C.IMAGE_DATASETS:
        g = D.image_geometry(dataset)
        if g is None:
            continue
        g = g[g.k == k]              # the table holds every k
        for model in C.models_on(dataset):
            # within one condition, the image's own score oriented so larger = worse
            err = D.task_score(model, "perimage", oriented=True).rename("err")
            j = g.merge(err.reset_index(), on=D.IMAGE, how="inner")
            rows += _tier2_rows(j, dataset, model, k)
    return pd.DataFrame(rows) if rows else None


def _tier2_rows(j, dataset, model, k):
    rows = []
    if not len(j):
        return rows
    # All six per-point columns: P3's two LIDs, P4's two penalties, P5's
    # two mixing shares. P1 and P2 are absent because they have no
    # per-image form, not because they were skipped.
    props = [c for c in ("lid_gen", "lid_real", "trust", "cont",
                         "mix_gen", "mix_real") if c in j.columns]
    for space, js in j.groupby("space"):
        for prop in props:
            rs = []
            for _, grp in js.groupby(["degradation", "severity"], sort=False):
                if grp.err.nunique() < 3 or grp[prop].nunique() < 3:
                    continue
                rs.append(stats.spearmanr(grp[prop], grp.err).statistic)
            rs = np.clip([x for x in rs if np.isfinite(x)], -0.999999, 0.999999)
            if len(rs) < 10:
                continue
            z = np.arctanh(rs)
            rows.append(dict(dataset=dataset, task=C.MODELS[model]["task"],
                             model=model, space=space, k=k,
                             prop=prop, n_cond=len(rs),
                             rho_z=float(np.tanh(z.mean())),
                             abs_rho=float(abs(np.tanh(z.mean())))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()

    t1 = tier1(a.k)
    if t1 is None:
        print(f"no {D.GEOMETRY_CSV.name} -- run compute_geometry.py first")
        return
    t1.to_csv(C.ANALYSIS_DIR / "geometry_tier1.csv", index=False)

    h = t1[t1.headline]
    print("=" * 84)
    print(f"GEOMETRY TIER 1  |Spearman rho| with relative task performance drop  (k = {a.k})")
    print("=" * 84)
    for space, hs in h.groupby("space"):
        print(f"\n--- {space} ---")
        print(hs.pivot_table(index="prop", columns="model",
                             values="abs_rho").round(3).to_string())
    print(f"\nsignificant after BH-FDR: {int(h.sig.sum())}/{len(h)}")
    ws = h[h.wrong_sign & h.sig]
    print(f"significant WITH WRONG SIGN: {len(ws)}"
          + (f"  <- {sorted(set(ws.prop))}" if len(ws) else ""))
    print("  (p2, p3 excluded from the sign check by design -- see the docstring)")
    print(f"\n-> {C.ANALYSIS_DIR / 'geometry_tier1.csv'}")

    t2 = tier2(a.k)
    if t2 is None:
        print("\nno image_geometry_<dataset>.parquet -- "
              "rerun compute_geometry.py without --no-perimage for tier 2")
        return
    t2.to_csv(C.ANALYSIS_DIR / "geometry_tier2.csv", index=False)
    print("\n" + "=" * 84)
    print("GEOMETRY TIER 2  Fisher-z mean |rho| of per-image LID within condition")
    print("=" * 84)
    print(t2.pivot_table(index=["space", "prop"], columns="model",
                         values="abs_rho").round(3).to_string())
    print(f"\n-> {C.ANALYSIS_DIR / 'geometry_tier2.csv'}")


if __name__ == "__main__":
    main()

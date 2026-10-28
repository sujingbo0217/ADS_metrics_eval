"""The five geometric properties, computed on (R, G) embedding pairs via knn_ext.

R is the clean set, G the degraded set of one condition. The two are PAIRED:
row i of G is the degradation of row i of R, which P4 requires.

    P1  position   dmu = ||mu_G - mu_R|| / sigma_R,  sigma_R = sqrt(tr(S_R)/d)
                   baseline 0, with a radial/angular split since a cosine metric
                   is blind to the radial half by construction.
    P2  shape      log( ER(G) / ER(R) ),  ER(X) = exp(H(p)), p_i = lam_i/sum lam
                   baseline 0, negative = directional diversity collapsed.
    P3  local      log( LID(G) / LID(R) ), TwoNN globally; per-point MLE for the
                   image-level regressions. baseline 0.
    P4  neighbors continuity and trustworthiness (Venna & Kaski rank-weighted),
                   baseline 1. Needs the pairing.
    P5  mixing     cross-set neighbor fraction over its random-mixing
                   expectation, baseline 1 (see the ceiling note in p5_mixing).

WHICH KERNEL DOES WHAT
    set_stats                   P1 and P2, one pass per set
    lid_profile                 P3, one pooled search feeding both estimators
    neighborhood_preservation  P4
    cross_set_mixing            P5
"""
import sys
from pathlib import Path

import numpy as np

# The CUDA module is built into build/ and symlinked into python/ by the
# CMake POST_BUILD rule, neither of which is on sys.path when this is imported
# from scripts/. Same pattern npy_to_fbin.py uses.
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
        "-DCMAKE_BUILD_TYPE=Release -DKNN_CUDA_ARCH=<80|86|89|90> "
        "-DPython3_EXECUTABLE=$(which python) "
        "-Dpybind11_DIR=$(python -m pybind11 --cmakedir) && "
        "cmake --build build -j\n"
        "See README section 1b."
    ) from e

# Spaces whose metric is cosine: rows are L2-normalised and InnerProduct used.
#
# The rule is that the geometry's distance matches the functional the metric
# family applies to that same space, so that a property and the metric it is
# compared against are measuring in the same units. CLIP is here because the
# metric we hold it against, CLIP-sim, is a cosine. dinov2_vitl14 is
# deliberately NOT here: what we hold it against is FD / KD, which are Frechet
# and polynomial-MMD on raw unnormalised features, matching dgm-eval. Keeping
# DINOv2 in L2 also preserves the identity that makes P1 legible -- P1 is the
# barycenter term of the Frechet distance in the same space, which is only true
# without the normalisation.
COSINE_SPACES = {"clip_vitb32", "clip_vit_b32", "vgg_lpips_cosine"}


def dist_type(space):
    return (knn_ext.DistanceType.InnerProduct if space in COSINE_SPACES
            else knn_ext.DistanceType.L2)


def prepare(X, space):
    """float32 and L2-normalised where the space is cosine.

    The bindings declare every array c_style | forcecast, so dtype and layout
    are numpy's problem now, not ours.
    """
    if space in COSINE_SPACES:
        return knn_ext.l2_normalize_rows(X)
    return np.ascontiguousarray(X, dtype=np.float32)


def set_stats(X):
    """(mu, sigma, effective rank) for one set. P1 and P2 both need it."""
    mu, sigma, eff_rank = knn_ext.set_stats(X)
    return dict(mu=mu, sigma=sigma, eff_rank=eff_rank)


# ---------------------------------------------------------------------------
# P1  position
# ---------------------------------------------------------------------------
def p1_position(R, G, sR=None, sG=None):
    """Centroid displacement in units of the real set's own rms dispersion.

    sigma_R is the PER-AXIS rms spread, so dmu answers "how many typical axis
    radii did the cloud move". Dividing by sqrt(d) restates it in units of the
    cloud's total radius; both are reported so the scale is never ambiguous.

    The two centroids and sigma_R come from set_stats; everything below is
    O(d) arithmetic on a pair of mean vectors, which is not worth a kernel.
    """
    sR = sR or set_stats(R)
    sG = sG or set_stats(G)
    mu_r, mu_g, sig = sR["mu"], sG["mu"], sR["sigma"]
    d = R.shape[1]
    shift = np.linalg.norm(mu_g - mu_r)
    nr, ng = np.linalg.norm(mu_r), np.linalg.norm(mu_g)
    return dict(
        p1_dmu=float(shift / max(sig, 1e-12)),
        p1_dmu_cloud=float(shift / max(sig * np.sqrt(d), 1e-12)),
        p1_radial=float(ng / max(nr, 1e-12)),
        p1_cos=float(mu_r @ mu_g / max(nr * ng, 1e-12)))


# ---------------------------------------------------------------------------
# P2  shape
# ---------------------------------------------------------------------------
def p2_shape(R, G, sR=None, sG=None):
    """Ratio of effective ranks, Roy & Vetterli (2007).

    ER = exp of the Shannon entropy of the normalised covariance spectrum. Note
    the ceiling: ER <= min(n-1, d), so at n = 500 no space can report more than
    499 however wide it is.
    """
    sR = sR or set_stats(R)
    sG = sG or set_stats(G)
    er_r, er_g = sR["eff_rank"], sG["eff_rank"]
    return dict(p2_log_er_ratio=float(np.log(er_g / er_r)),
                p2_er_real=er_r, p2_er_gen=er_g)


# ---------------------------------------------------------------------------
# P3  local intrinsic dimension
# ---------------------------------------------------------------------------
def p3_local(R, G, k, space):
    """Per-point Levina-Bickel MLE and the global TwoNN estimate.

    lid_profile runs one pooled search per set at max(k, 3) neighbors and
    derives both from it: TwoNN always uses the first two neighbors whatever k
    is, and the bias-corrected MLE needs k >= 3, so it returns NaN per point on
    the k = 1 corner of the ablation grid while the global figure stands.
    """
    dt = dist_type(space)
    lid_r, gr = knn_ext.lid_profile(R, k, dt)
    lid_g, gg = knn_ext.lid_profile(G, k, dt)
    return dict(p3_log_lid_ratio=float(np.log(max(gg, 1e-9) / max(gr, 1e-9))),
                p3_lid_real=gr, p3_lid_gen=gg,
                p3_lid_pt_real=lid_r, p3_lid_pt_gen=lid_g)


# ---------------------------------------------------------------------------
# P4  neighborhood preservation
# ---------------------------------------------------------------------------
def p4_neighborhood(R, G, k, space):
    """Venna & Kaski continuity and trustworthiness. Baseline 1.

    Trustworthiness penalises neighbors that appear in G but were far apart in
    R (false friends); continuity penalises neighbors that were close in R and
    got pushed away in G (lost friends). They are not redundant: a degradation
    that merges distinct scenes scores badly on one, and one that scatters a
    tight cluster badly on the other.

    Both statistics are sums of a PER-POINT penalty, so the kernel returns that
    per-point form and the set-level number is its mean. "Was this image's
    neighborhood preserved" is what tier 2 regresses against.
    """
    t_pt, c_pt = knn_ext.neighborhood_preservation(R, G, k, dist_type(space))
    return dict(p4_trustworthiness=float(t_pt.mean()),
                p4_continuity=float(c_pt.mean()),
                p4_trust_pt=t_pt, p4_cont_pt=c_pt)


# ---------------------------------------------------------------------------
# P5  cross-set mixing
# ---------------------------------------------------------------------------
def p5_mixing(R, G, k, space):
    """kNN form of the Friedman-Rafsky two-sample statistic.

    Each point's k neighbors in R u G are counted by which set they came from
    and divided by what random mixing would give. 0 means perfectly separable.
    Used descriptively; no null distribution is fitted.

    CEILING. With PAIRED sets -- ours, where g_i is the degradation of r_i -- the
    upper end is NOT 1. Each point's own twin is a guaranteed cross-set
    neighbor, so a perfectly mixed paired sample reads

        rho_max = (1 + (k-1)/2) / (k/2) = (k+1)/k

    which is 1.10 at k = 10 and 1.02 at k = 50, and is reproduced to four
    decimals on duplicate sets. Reading rho > 1 as "more than fully mixed" would
    be an artefact of the pairing, so p5_rho_scaled divides it out.
    """
    pt_real, pt_gen = knn_ext.cross_set_mixing(R, G, k, dist_type(space))
    rho_r, rho_g = float(pt_real.mean()), float(pt_gen.mean())
    rho = 0.5 * (rho_r + rho_g)
    return dict(p5_rho_real=rho_r, p5_rho_gen=rho_g, p5_rho=rho,
                p5_rho_scaled=rho / ((k + 1) / k), p5_asymmetry=rho_r - rho_g,
                p5_mix_pt_real=pt_real, p5_mix_pt_gen=pt_gen)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def all_properties(R, G, k=10, space="inception_v3", keep_pointwise=False,
                   sR=None, sG=None):
    """All five for one (R, G, k, space). R and G must be row-aligned.

    sR and sG are set_stats of the PREPARED sets. Neither depends on k, and the
    clean set's does not depend on the condition either, so a driver sweeping
    both can compute them once -- see compute_geometry.py. Left out, they are
    computed here.
    """
    R, G = prepare(R, space), prepare(G, space)
    # P1 and P2 read the same per-set pass, so it happens once per set and not
    # once per property.
    sR = sR or set_stats(R)
    sG = sG or set_stats(G)
    out = {}
    out.update(p1_position(R, G, sR, sG))
    out.update(p2_shape(R, G, sR, sG))
    p3 = p3_local(R, G, k, space)
    pt_r, pt_g = p3.pop("p3_lid_pt_real"), p3.pop("p3_lid_pt_gen")
    out.update(p3)
    out.update(p4_neighborhood(R, G, k, space))
    out.update(p5_mixing(R, G, k, space))

    # Three of the five have a per-image form; P1 and P2 do not, exactly as
    # FID and KID do not on the metric side. A centroid displacement and a
    # spectral entropy are properties of a SET -- there is no such thing as one
    # image's share of them. So the scope structure of the geometry mirrors the
    # scope structure of the metrics, which is what lets tier 2 treat them alike.
    pt = {"p3_lid_pt_real": pt_r, "p3_lid_pt_gen": pt_g}
    for key in ("p4_trust_pt", "p4_cont_pt", "p5_mix_pt_real", "p5_mix_pt_gen"):
        pt[key] = out.pop(key)
    if keep_pointwise:
        out["_pointwise"] = pt
    return out


# ---------------------------------------------------------------------------
# self-test: every property against a case whose answer is known in advance,
# and every kernel against the numpy reference in geometry_ref.py
# ---------------------------------------------------------------------------
def _selftest():
    import geometry_ref as ref

    rng = np.random.default_rng(0)
    n, d = 600, 40
    R = rng.normal(size=(n, d)).astype(np.float32)
    sp = "inception_v3"

    print("P1  baseline and a known shift")
    print(f"    G = R              dmu = {p1_position(R, R)['p1_dmu']:.4f}   (want 0)")
    G = R + np.eye(d, dtype=np.float32)[0] * 2      # sigma_R = 1 for N(0, I)
    print(f"    G = R + 2*e0       dmu = {p1_position(R, G)['p1_dmu']:.4f}   (want 2)")

    print("P2  baseline and a known collapse")
    print(f"    G = R              log ER = {p2_shape(R, R)['p2_log_er_ratio']:+.4f}   (want 0)")
    G = R.copy(); G[:, 20:] *= 0.01
    r = p2_shape(R, G)
    print(f"    half axes crushed  log ER = {r['p2_log_er_ratio']:+.4f}   "
          f"(ER {r['p2_er_real']:.1f} -> {r['p2_er_gen']:.1f}, want ~-0.69)")

    print("P3  TwoNN recovers a known dimension (mild low bias is its boundary effect)")
    for dim in (2, 5, 10):
        X = rng.uniform(size=(1500, dim)).astype(np.float32)
        print(f"    uniform in {dim:2d}-d     TwoNN = "
              f"{knn_ext.lid_profile(X, 3, knn_ext.DistanceType.L2)[1]:5.2f}")

    print("P4  baseline and a known scramble")
    r = p4_neighborhood(R, R, 10, sp)
    print(f"    G = R              T = {r['p4_trustworthiness']:.4f}  "
          f"C = {r['p4_continuity']:.4f}   (want 1, 1)")
    r = p4_neighborhood(R, R[rng.permutation(n)], 10, sp)
    print(f"    G = shuffled R     T = {r['p4_trustworthiness']:.4f}  "
          f"C = {r['p4_continuity']:.4f}   (want ~0.5, chance)")

    print("P5  paired ceiling and a known separation")
    for k in (5, 10, 20, 50):
        r = p5_mixing(R, R + 1e-4, k, sp)
        print(f"    G = R, k = {k:2d}      rho = {r['p5_rho']:.4f}   "
              f"(want (k+1)/k = {(k+1)/k:.4f})   scaled = {r['p5_rho_scaled']:.4f}")
    print(f"    G far from R       rho = {p5_mixing(R, R + 500, 10, sp)['p5_rho']:.4f}   (want 0)")

    print("\nkernels against the numpy reference (both metrics, a degradation "
          "that is neither a no-op nor a scramble)")
    ok = True
    for space in ("inception_v3", "clip_vitb32"):
        A = prepare(R, space)
        B = prepare(R + 0.4 * rng.normal(size=(n, d)).astype(np.float32), space)
        for k in (1, 5, 10, 50):
            got = all_properties(A, B, k=k, space=space, keep_pointwise=True)
            want = ref.all_properties(A, B, k=k, space=space, keep_pointwise=True)
            ok &= _compare(got, want, f"{space} k={k}")
    print("  all within tolerance" if ok else "  MISMATCH -- see above")
    return ok


def _compare(got, want, label, atol=2e-4, rtol=2e-4):
    """Scalars and per-point arrays, GPU against numpy."""
    ok = True
    for key in sorted(want):
        if key == "_pointwise":
            continue
        a, b = got[key], want[key]
        if not np.isclose(a, b, atol=atol, rtol=rtol, equal_nan=True):
            print(f"  {label:22s} {key:20s} gpu {a!r} != numpy {b!r}")
            ok = False
    for key in sorted(want["_pointwise"]):
        a = np.asarray(got["_pointwise"][key], dtype=np.float64)
        b = np.asarray(want["_pointwise"][key], dtype=np.float64)
        bad = ~np.isclose(a, b, atol=atol, rtol=rtol, equal_nan=True)
        if bad.any():
            i = int(np.argmax(bad))
            print(f"  {label:22s} {key:20s} {bad.sum()}/{len(a)} points differ, "
                  f"worst at {i}: gpu {a[i]!r} != numpy {b[i]!r}")
            ok = False
    return ok


if __name__ == "__main__":
    sys.exit(0 if _selftest() else 1)

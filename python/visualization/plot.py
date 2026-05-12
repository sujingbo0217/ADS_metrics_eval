#!/usr/bin/env python3
"""plot.py — UMAP visualization of pre-computed image embeddings.

Per embedding space (extractor) the script produces one figure per
scenario (file extension follows ``--format``: png by default, also pdf
/ svg):

1. ``real_vs_sim.<ext>`` — 1x2 panels
       [0] Real only
       [1] Real + Sim   (sim drawn on top of real)

2. ``<degradation>.<ext>`` — 2x2 panels (progressive overlay), one
   figure per degradation type under ``DEGRADATION_TYPES``
   (gaussian_blur, gaussian_noise, brightness, contrast, color_jitter):
       [0] Real only
       [1] Real + Level 1
       [2] Real + Level 1 + Level 2
       [3] Real + Level 1 + Level 2 + Level 3

A single UMAP is fit per figure on the *union* of all points so that the
spatial layout is consistent across panels and the user can see how the
distribution drifts level by level.

Optional ``--sweep`` mode performs a grid search over ``n_neighbors`` x
``min_dist`` and saves one sweep figure per scenario per extractor
(``sweep_real_vs_sim.<ext>``, ``sweep_<degradation>.<ext>``). Each cell
shows the fully-overlaid view (real + sim, or real + all degradation
levels) under one (n_neighbors, min_dist) combo — useful for parameter
tuning.

Optional ``--compare-extractors`` mode produces a single S x N overview
figure (rows = scenario, cols = extractor) where each cell shows only the
fully-covered panel — Real + Sim, and Real + <degradation> (L1+L2+L3)
for every degradation type — so all scenarios across all extractors fit
in one image. With the default scenario / extractor sets this is a 6x6
grid. Each cell uses its own UMAP fit / axis limits since different
extractors live in different vector spaces — this is *not* a joint
projection across extractors, just a side-by-side layout for visual
comparison. Output goes to ``<output-dir>/compare/overview.<ext>``.

UMAP projections are cached transparently to ``<output-dir>/.umap_cache``
keyed on (extractor, scenario, n_neighbors, min_dist, subsample). Entries
are auto-invalidated by an input fingerprint, so re-running with different
colors / dpi / format / extractor list still hits the cache and re-renders
in seconds. Crucially this also means a sweep run pre-warms the cache for
follow-up ``--compare-extractors`` runs that pick the chosen
``(n_neighbors, min_dist)`` from the sweep.

Layout assumed under ``--base-dir`` (matches ``embed.py`` outputs):

    <base_dir>/
        kitti_real/<extractor>.fbin
        kitti_sim/<extractor>.fbin
        kitti_gaussian_blur/level_{1,2,3}/<extractor>.fbin
        kitti_gaussian_noise/level_{1,2,3}/<extractor>.fbin
        kitti_bright/level_{1,2,3}/<extractor>.fbin
        kitti_contrast/level_{1,2,3}/<extractor>.fbin
        kitti_color_jitter/level_{1,2,3}/<extractor>.fbin

Recommended workflow — sweep first, then compare-extractors:

    python python/visualization/plot.py \
        --base-dir /scratch/jsu02/sim-real-embedding \
        --output-dir figures/tuning \
        --extractors inception_v3 clip_vit_b32 resnet50 lpips_vgg pixel segformer \
        --sweep \
        --sweep-n-neighbors 10 20 50 100 \
        --sweep-min-dist 0.0 0.1 0.3 0.5 \
        --subsample 4000 \
        --skip-per-extractor

    python python/visualization/plot.py \
        --base-dir /scratch/jsu02/sim-real-embedding \
        --output-dir figures/virtualization \
        --extractors inception_v3 clip_vit_b32 resnet50 lpips_vgg pixel segformer \
        --n-neighbors 20 --min-dist 0.3 \
        --subsample 4000 \
        --compare-extractors \
        --skip-per-extractor \
        --format pdf --dpi 600

To get a cache hit on step 2, ``--base-dir``, ``--output-dir``,
``--subsample``, and the chosen ``(--n-neighbors, --min-dist)`` must all
match values used during the sweep.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import umap

# Make ``from embedding.fbin_io import read_fbin`` work when the script is
# launched directly from the repo root.
HERE = Path(__file__).resolve().parent
PY_ROOT = HERE.parent  # python/
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from embedding.fbin_io import read_fbin  # noqa: E402

# ─── Constants ─────────────────────────────────────────────────────────────

EXTRACTORS = [
    "inception_v3",
    "clip_vit_b32",
    "resnet50",
    "lpips_vgg",
    "segformer",
    "pixel",
]

# Spaces that use cosine / inner-product distance (match pipeline.py).
COSINE_EXTRACTORS = {"clip_vit_b32"}

# Real is drawn first as a neutral dark-gray backdrop so colored overlays
# (sim / level 1-3) pop visually against it. Each overlay uses a distinct
# saturated hue (blue / amber / teal / magenta) rather than a luminance
# ramp — this stays readable both on screen and in print, even where
# overlay clouds heavily intersect.
PALETTE = {
    "Real":    "#b7b7b7",  # gray
    "Sim":     "#1f77b4",  # blue
    "Level 1": "#fcdf03",  # yellow
    "Level 2": "#fcad03",  # orange
    "Level 3": "#fc4103",  # red
}

ALPHA = 0.5
POINT_SIZE = 8
SEED = 42

# (subdir, human-readable label, short label for compare overview row).
# Order here drives the row order of the cross-extractor compare grid
# (real_vs_sim is always the first row).
DEGRADATION_TYPES = [
    ("kitti_gaussian_blur",  "Gaussian Blur",  "Blur"),
    ("kitti_gaussian_noise", "Gaussian Noise", "Noise"),
    ("kitti_bright",         "Brightness",     "Bright"),
    ("kitti_contrast",       "Contrast",       "Contrast"),
    ("kitti_color_jitter",   "Color Jitter",   "ColorJitter"),
]


# ─── Helpers ───────────────────────────────────────────────────────────────

def _metric_for(extractor: str) -> str:
    return "cosine" if extractor in COSINE_EXTRACTORS else "euclidean"


def _maybe_subsample(
    x: np.ndarray, n_max: Optional[int], rng: np.random.Generator
) -> np.ndarray:
    if n_max is None or n_max <= 0 or x.shape[0] <= n_max:
        return x
    idx = rng.choice(x.shape[0], size=n_max, replace=False)
    idx.sort()
    return x[idx]


def _load_fbin(
    base_dir: str | os.PathLike,
    rel: str,
    extractor: str,
    *,
    subsample: Optional[int],
    rng: np.random.Generator,
) -> np.ndarray:
    path = Path(base_dir) / rel / f"{extractor}.fbin"
    if not path.is_file():
        raise FileNotFoundError(path)
    arr = read_fbin(str(path))
    arr = _maybe_subsample(arr, subsample, rng)
    return np.ascontiguousarray(arr, dtype=np.float32)


def _fit_umap(
    arrays: Sequence[np.ndarray],
    *,
    metric: str,
    n_neighbors: int,
    min_dist: float,
) -> List[np.ndarray]:
    """Fit one UMAP on the concatenation; return per-array 2-D slices."""
    sizes = [int(a.shape[0]) for a in arrays]
    combined = np.ascontiguousarray(np.vstack(arrays), dtype=np.float32)
    eff_neighbors = max(2, min(n_neighbors, combined.shape[0] - 1))

    with warnings.catch_warnings():
        # umap-learn warns when random_state forces n_jobs=1; we accept that.
        warnings.simplefilter("ignore", category=UserWarning)
        reducer = umap.UMAP(
            n_neighbors=eff_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=SEED,
            n_components=2,
            verbose=False,
        )
        proj = np.asarray(reducer.fit_transform(combined), dtype=np.float32)

    out: List[np.ndarray] = []
    idx = 0
    for n in sizes:
        out.append(proj[idx:idx + n])
        idx += n
    return out


# ─── UMAP projection cache ─────────────────────────────────────────────────
#
# UMAP projections only depend on the high-dimensional inputs and the UMAP
# knobs (n_neighbors, min_dist, metric). Re-rendering with a different
# palette / alpha / dpi / output format does *not* invalidate them, so we
# persist them to ``<output-dir>/.umap_cache/<extractor>/*.npz`` and load
# on hit. Saves ~50 s per scenario per extractor on recolor, and lets a
# follow-up ``--compare-extractors`` run reuse projections from a prior
# ``--sweep`` run for free.

def _array_fingerprint(arrs: Sequence[np.ndarray]) -> str:
    """Cheap content fingerprint of a list of arrays.

    Uses shape + dtype + a deterministically subsampled byte slice so cost
    stays O(constant) regardless of input size while still detecting any
    change to the underlying .fbin or to the random subsampling pattern.
    """
    h = hashlib.blake2s(digest_size=12)
    for a in arrs:
        h.update(repr(a.shape).encode())
        h.update(repr(a.dtype).encode())
        view = np.ascontiguousarray(a).reshape(-1)
        if view.size == 0:
            continue
        idx = np.linspace(0, view.size - 1, num=min(256, view.size),
                          dtype=np.int64)
        h.update(view[idx].tobytes())
    return h.hexdigest()


def _cache_path(
    cache_root: Optional[Path],
    *,
    scenario_key: str,
    n_neighbors: int,
    min_dist: float,
    subsample: Optional[int],
) -> Optional[Path]:
    if cache_root is None:
        return None
    md_str = f"{min_dist:g}".replace(".", "p").replace("-", "m")
    sub_str = str(subsample) if subsample else "all"
    name = f"{scenario_key}__nn{n_neighbors}_md{md_str}_sub{sub_str}.npz"
    return cache_root / name


def _fit_umap_cached(
    arrays: Sequence[np.ndarray],
    *,
    metric: str,
    n_neighbors: int,
    min_dist: float,
    cache_path: Optional[Path],
) -> Tuple[List[np.ndarray], str]:
    """Like ``_fit_umap`` but read/write a .npz cache at ``cache_path``.

    Returns ``(projections, status)`` where ``status`` is one of ``"hit"``,
    ``"miss"``, or ``"stale"`` (existed but fingerprint mismatched).
    """
    status = "miss"
    if cache_path is not None and cache_path.is_file():
        try:
            npz = np.load(str(cache_path))
            n_arrays = int(npz["n_arrays"])
            if n_arrays == len(arrays):
                proj = [np.ascontiguousarray(npz[f"a{i}"], dtype=np.float32)
                        for i in range(n_arrays)]
                shapes_ok = all(
                    p.shape == (a.shape[0], 2) for p, a in zip(proj, arrays)
                )
                fp_disk = str(npz["fingerprint"])
                fp_now = _array_fingerprint(arrays)
                if shapes_ok and fp_disk == fp_now:
                    return proj, "hit"
            status = "stale"
        except Exception:
            status = "stale"

    proj = _fit_umap(
        arrays,
        metric=metric, n_neighbors=n_neighbors, min_dist=min_dist,
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(cache_path),
            n_arrays=np.int32(len(proj)),
            fingerprint=_array_fingerprint(arrays),
            **{f"a{i}": p for i, p in enumerate(proj)},
        )
    return proj, status


def _shared_limits(
    layers: Sequence[Tuple[str, np.ndarray]], pad_frac: float = 0.04
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    all_xy = np.vstack([xy for _, xy in layers])
    xmin, ymin = all_xy.min(axis=0)
    xmax, ymax = all_xy.max(axis=0)
    px = pad_frac * (xmax - xmin + 1e-9)
    py = pad_frac * (ymax - ymin + 1e-9)
    return (xmin - px, xmax + px), (ymin - py, ymax + py)


def _draw_panel(
    ax,
    layers: Sequence[Tuple[str, np.ndarray]],
    *,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    title: str,
) -> None:
    for label, xy in layers:
        ax.scatter(
            xy[:, 0], xy[:, 1],
            s=POINT_SIZE,
            alpha=ALPHA,
            c=PALETTE.get(label, "#34495e"),
            label=f"{label} (n={xy.shape[0]})",
            edgecolors="none",
            linewidths=0,
            rasterized=True,
        )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11)

    leg = ax.legend(
        loc="best",
        fontsize=8,
        markerscale=2.0,
        framealpha=0.85,
        handletextpad=0.3,
        borderpad=0.3,
    )
    # Make legend markers fully opaque regardless of scatter alpha.
    for h in leg.legend_handles:
        h.set_alpha(1.0)


def _grid_shape(n: int) -> Tuple[int, int]:
    if n <= 1:
        return 1, 1
    if n == 2:
        return 1, 2
    if n <= 4:
        return 2, 2
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    return nrows, ncols


def _make_figure(
    panels: Sequence[Tuple[str, Sequence[Tuple[str, np.ndarray]]]],
    *,
    suptitle: str,
    panel_size: float = 5.0,
):
    n = len(panels)
    nrows, ncols = _grid_shape(n)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(panel_size * ncols, panel_size * nrows),
        squeeze=False,
    )
    flat_axes = [ax for row in axes for ax in row]

    # Shared extents across all panels keep relative drift visually obvious.
    every_layer: List[Tuple[str, np.ndarray]] = []
    for _, layers in panels:
        every_layer.extend(layers)
    xlim, ylim = _shared_limits(every_layer)

    for ax, (subtitle, layers) in zip(flat_axes, panels):
        _draw_panel(ax, layers, xlim=xlim, ylim=ylim, title=subtitle)
    for ax in flat_axes[n:]:
        ax.set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=1.0)
    fig.tight_layout()
    return fig


# ─── Panel builders ────────────────────────────────────────────────────────
#
# A "panels" list is the canonical intermediate representation:
#   panels = [(subtitle, [(label, xy_2d), ...]), ...]
# It can be cheaply cached (just 2-D coords) and replayed into either a
# per-extractor figure or a multi-extractor compare grid without re-fitting
# UMAP.

Panels = List[Tuple[str, List[Tuple[str, np.ndarray]]]]


def _panels_real_vs_sim(
    real: np.ndarray,
    sim: np.ndarray,
    *,
    metric: str,
    n_neighbors: int,
    min_dist: float,
    cache_path: Optional[Path] = None,
) -> Tuple[Panels, str]:
    proj, status = _fit_umap_cached(
        [real, sim],
        metric=metric, n_neighbors=n_neighbors, min_dist=min_dist,
        cache_path=cache_path,
    )
    r, s = proj
    panels: Panels = [
        ("Real only",  [("Real", r)]),
        ("Real + Sim", [("Real", r), ("Sim", s)]),
    ]
    return panels, status


def _panels_progressive(
    real: np.ndarray,
    levels: Sequence[np.ndarray],
    *,
    metric: str,
    n_neighbors: int,
    min_dist: float,
    cache_path: Optional[Path] = None,
) -> Tuple[Panels, str]:
    proj, status = _fit_umap_cached(
        [real, *levels],
        metric=metric, n_neighbors=n_neighbors, min_dist=min_dist,
        cache_path=cache_path,
    )
    r = proj[0]
    L = proj[1:]
    layer_names = [f"Level {i + 1}" for i in range(len(L))]

    panels: Panels = [("Real only", [("Real", r)])]
    for k in range(1, len(L) + 1):
        sub = "Real + " + " + ".join(layer_names[:k])
        layers: List[Tuple[str, np.ndarray]] = [("Real", r)]
        for i in range(k):
            layers.append((layer_names[i], L[i]))
        panels.append((sub, layers))
    return panels, status


def _suptitle_for(
    extractor: str, scenario_label: str, *,
    metric: str, n_neighbors: int, min_dist: float,
) -> str:
    return (
        f"{extractor} — {scenario_label}   "
        f"(metric={metric}, n_neighbors={n_neighbors}, min_dist={min_dist})"
    )


# ─── Render entry points ───────────────────────────────────────────────────

def render_real_vs_sim(
    real: np.ndarray,
    sim: np.ndarray,
    *,
    extractor: str,
    out_dir: Path,
    n_neighbors: int,
    min_dist: float,
    fmt: str,
    dpi: int,
    cache_path: Optional[Path] = None,
) -> Tuple[Path, Panels, str]:
    metric = _metric_for(extractor)
    panels, status = _panels_real_vs_sim(
        real, sim,
        metric=metric, n_neighbors=n_neighbors, min_dist=min_dist,
        cache_path=cache_path,
    )
    fig = _make_figure(
        panels,
        suptitle=_suptitle_for(
            extractor, "Real vs Sim",
            metric=metric, n_neighbors=n_neighbors, min_dist=min_dist,
        ),
    )
    out_path = out_dir / extractor / f"real_vs_sim.{fmt}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path, panels, status


def render_progressive_levels(
    real: np.ndarray,
    levels: Sequence[np.ndarray],
    *,
    extractor: str,
    deg_label: str,
    out_name: str,
    out_dir: Path,
    n_neighbors: int,
    min_dist: float,
    fmt: str,
    dpi: int,
    cache_path: Optional[Path] = None,
) -> Tuple[Path, Panels, str]:
    metric = _metric_for(extractor)
    panels, status = _panels_progressive(
        real, levels,
        metric=metric, n_neighbors=n_neighbors, min_dist=min_dist,
        cache_path=cache_path,
    )
    fig = _make_figure(
        panels,
        suptitle=_suptitle_for(
            extractor, f"Real vs {deg_label} progressive degradation",
            metric=metric, n_neighbors=n_neighbors, min_dist=min_dist,
        ),
    )
    out_path = out_dir / extractor / f"{out_name}.{fmt}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path, panels, status


def render_compare_overview(
    panels_cache: "dict[str, dict[str, Panels]]",
    *,
    extractors: Sequence[str],
    out_path: Path,
    n_neighbors: int,
    min_dist: float,
    dpi: int,
    panel_w: float = 4.0,
    panel_h: float = 4.0,
) -> Optional[Path]:
    """Single overview grid: rows = scenarios, cols = extractors.

    Each cell shows only the fully-covered panel of its scenario:
    Real + Sim for ``real_vs_sim``, and Real + L1 + L2 + L3 for every
    progressive degradation in ``DEGRADATION_TYPES``. With the default
    6 scenarios (sim + 5 degradations) and 6 extractors this produces a
    6x6 grid that captures every distribution-shift view in a single
    figure.

    Each cell uses its own xlim/ylim because every extractor's UMAP fit
    lives in its own coordinate system; this is a side-by-side layout,
    not a joint projection across extractors.
    """
    scenarios: List[Tuple[str, str]] = [
        ("real_vs_sim", "Real + Sim"),
        *[
            (subdir, f"Real + {short}\n(L1+L2+L3)")
            for subdir, _label, short in DEGRADATION_TYPES
        ],
    ]

    available = [
        e for e in extractors
        if all(e in panels_cache.get(key, {}) and panels_cache[key][e]
               for key, _ in scenarios)
    ]
    if not available:
        return None

    nrows = len(scenarios)
    ncols = len(available)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(panel_w * ncols, panel_h * nrows),
        squeeze=False,
    )

    for i, (scenario_key, scenario_label) in enumerate(scenarios):
        for j, extractor in enumerate(available):
            ax = axes[i][j]
            panels = panels_cache[scenario_key][extractor]
            # Last panel = fully-overlaid view (Real + everything).
            _, layers = panels[-1]
            xlim, ylim = _shared_limits(layers)
            metric = _metric_for(extractor)
            title = f"{extractor}\n({metric})" if i == 0 else ""
            _draw_panel(ax, layers, xlim=xlim, ylim=ylim, title=title)
            if j == 0:
                ax.set_ylabel(
                    scenario_label,
                    fontsize=12, fontweight="bold", labelpad=10,
                )

    fig.suptitle(
        f"Cross-extractor overview   "
        f"(n_neighbors={n_neighbors}, min_dist={min_dist})",
        fontsize=14, y=1.0,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_param_sweep(
    real: np.ndarray,
    overlays: Sequence[Tuple[str, np.ndarray]],
    *,
    extractor: str,
    scenario_label: str,
    scenario_key: str,
    out_filename: str,
    out_dir: Path,
    dpi: int,
    n_neighbors_list: Sequence[int],
    min_dist_list: Sequence[float],
    cache_root: Optional[Path] = None,
    subsample: Optional[int] = None,
) -> Tuple[Path, int, int]:
    """Render an n_neighbors x min_dist grid for a single scenario.

    ``overlays`` is a list of ``(label, array)`` pairs drawn on top of
    ``real`` in every cell — e.g. ``[("Sim", sim)]`` for the real-vs-sim
    case, or ``[("Level 1", l1), ("Level 2", l2), ("Level 3", l3)]`` for
    the fully-degraded blur / noise case.

    Returns ``(out_path, n_hits, n_total)`` so the caller can report cache
    effectiveness for the sweep.
    """
    metric = _metric_for(extractor)
    nrows = len(min_dist_list)
    ncols = len(n_neighbors_list)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.2 * ncols, 4.0 * nrows),
        squeeze=False,
    )

    overlay_labels = [lbl for lbl, _ in overlays]
    overlay_arrays = [a for _, a in overlays]

    n_hits = 0
    n_total = 0
    for i, md in enumerate(min_dist_list):
        for j, nn in enumerate(n_neighbors_list):
            ax = axes[i][j]
            n_total += 1
            try:
                cache_path = _cache_path(
                    cache_root,
                    scenario_key=scenario_key,
                    n_neighbors=int(nn), min_dist=float(md),
                    subsample=subsample,
                )
                proj, status = _fit_umap_cached(
                    [real, *overlay_arrays],
                    metric=metric, n_neighbors=int(nn), min_dist=float(md),
                    cache_path=cache_path,
                )
                if status == "hit":
                    n_hits += 1
            except Exception as exc:  # noqa: BLE001 — diagnostic panel
                ax.text(0.5, 0.5, f"failed:\n{exc}",
                        ha="center", va="center", fontsize=9, wrap=True)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            layers: List[Tuple[str, np.ndarray]] = [("Real", proj[0])]
            for lbl, xy in zip(overlay_labels, proj[1:]):
                layers.append((lbl, xy))
            xlim, ylim = _shared_limits(layers)
            _draw_panel(
                ax,
                layers,
                xlim=xlim, ylim=ylim,
                title=f"n_neighbors={nn}, min_dist={md}",
            )

    fig.suptitle(
        f"{extractor} — UMAP parameter sweep "
        f"({scenario_label}, metric={metric})",
        fontsize=13, y=1.0,
    )
    fig.tight_layout()
    out_path = out_dir / extractor / out_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path, n_hits, n_total


# ─── CLI ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--base-dir", default="/scratch/jsu02/sim-real-embedding",
        help="root that contains kitti_real, kitti_sim, "
             "kitti_gaussian_blur/level_*, kitti_gaussian_noise/level_*.",
    )
    p.add_argument(
        "--output-dir", default=str(HERE / "figures"),
        help="directory where rendered figures are written. The UMAP "
             "projection cache lives under <output-dir>/.umap_cache, so "
             "keep this consistent across sweep / compare runs to reuse "
             "projections.",
    )
    p.add_argument(
        "--extractors", nargs="+", default=EXTRACTORS, choices=EXTRACTORS,
        help="which embedding spaces to visualize.",
    )

    # UMAP knobs. n_neighbors=30 + min_dist=0.1 follows the README guidance
    # for visualizing distribution drift across degradation levels.
    p.add_argument("--n-neighbors", type=int, default=30,
                   help="UMAP n_neighbors (5-15 local, 30 mid, 100 global).")
    p.add_argument("--min-dist", type=float, default=0.1,
                   help="UMAP min_dist (0.0-0.1 tight, 0.3-0.5 spread).")

    p.add_argument("--subsample", type=int, default=None,
                   help="cap each set at this many points (random subsample).")

    p.add_argument(
        "--format", default="png",
        choices=["png", "pdf", "svg"],
        help="output file format (default: png). Scatter layers are "
             "rasterized internally, so for pdf/svg the --dpi flag still "
             "controls point-cloud resolution.",
    )
    p.add_argument(
        "--dpi", type=int, default=160,
        help="resolution in DPI used by fig.savefig (default: 160). "
             "Use e.g. 600 for publication-grade pdf.",
    )

    p.add_argument("--sweep", action="store_true",
                   help="also dump n_neighbors x min_dist parameter-sweep "
                        "figures per extractor — one per scenario "
                        "(real+sim, real+blur all levels, real+noise all "
                        "levels).")
    p.add_argument("--sweep-n-neighbors", nargs="+", type=int,
                   default=[5, 15, 30, 100])
    p.add_argument("--sweep-min-dist", nargs="+", type=float,
                   default=[0.0, 0.1, 0.3, 0.5])

    p.add_argument(
        "--compare-extractors", action="store_true",
        help="also dump a single cross-extractor overview figure to "
             "<output-dir>/compare/overview.<fmt>: rows = scenario "
             "(real+sim, real+blur all levels, real+noise all levels), "
             "cols = extractor. Each cell keeps its own UMAP fit; this is "
             "a side-by-side layout, not a joint projection.",
    )

    p.add_argument("--skip-real-vs-sim", action="store_true")
    p.add_argument("--skip-degradation", action="store_true")
    p.add_argument(
        "--skip-per-extractor", action="store_true",
        help="skip per-extractor figures; only useful with "
             "--compare-extractors to render comparison-only output.",
    )
    return p.parse_args()


def _gather_levels(
    base_dir: str | os.PathLike,
    subdir: str,
    extractor: str,
    *,
    subsample: Optional[int],
    rng: np.random.Generator,
) -> Optional[List[np.ndarray]]:
    """Load all level_* fbins under ``base_dir/subdir`` for ``extractor``.

    Returns ``None`` if the directory is missing or any level is missing the
    requested extractor.
    """
    deg_root = Path(base_dir) / subdir
    if not deg_root.is_dir():
        print(f"  ! {deg_root} missing — skipping {subdir}")
        return None

    level_dirs = sorted(
        d for d in deg_root.glob("level_*") if d.is_dir()
    )
    if not level_dirs:
        print(f"  ! no level_* subdirs under {deg_root} — skipping {subdir}")
        return None

    levels: List[np.ndarray] = []
    for d in level_dirs:
        fbin = d / f"{extractor}.fbin"
        if not fbin.is_file():
            print(f"  ! {fbin} missing — skipping {subdir}")
            return None
        arr = read_fbin(str(fbin))
        arr = _maybe_subsample(arr, subsample, rng)
        levels.append(np.ascontiguousarray(arr, dtype=np.float32))
    return levels


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(SEED)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    cache_root = out_root / ".umap_cache"

    print(
        f"base-dir   : {args.base_dir}\n"
        f"output-dir : {out_root}\n"
        f"UMAP       : n_neighbors={args.n_neighbors}, "
        f"min_dist={args.min_dist}\n"
        f"subsample  : {args.subsample}\n"
        f"extractors : {args.extractors}\n"
        f"compare    : {args.compare_extractors}\n"
        f"cache      : {cache_root}"
    )

    # scenario_key -> extractor -> panels (used for cross-extractor compare).
    panels_cache: "dict[str, dict[str, Panels]]" = {
        "real_vs_sim": {},
        **{subdir: {} for subdir, _label, _short in DEGRADATION_TYPES},
    }

    for extractor in args.extractors:
        print(f"\n=== {extractor} ===")
        try:
            real = _load_fbin(
                args.base_dir, "kitti_real", extractor,
                subsample=args.subsample, rng=rng,
            )
        except FileNotFoundError as exc:
            print(f"  ! missing real data ({exc}); skipping {extractor}")
            continue
        print(f"  real : N={real.shape[0]} D={real.shape[1]}")

        metric = _metric_for(extractor)
        ext_cache_root = cache_root / extractor

        sim: Optional[np.ndarray] = None
        if not args.skip_real_vs_sim:
            try:
                sim = _load_fbin(
                    args.base_dir, "kitti_sim", extractor,
                    subsample=args.subsample, rng=rng,
                )
                print(f"  sim  : N={sim.shape[0]} D={sim.shape[1]}")
            except FileNotFoundError as exc:
                print(f"  ! missing sim ({exc}); skipping real-vs-sim")
                sim = None

            if sim is not None:
                # Only fit UMAP for the per-extractor real-vs-sim figure
                # if we'll actually use the result (figure on disk and/or
                # cross-extractor comparison grid).
                if args.skip_per_extractor and not args.compare_extractors:
                    print(f"  · real_vs_sim    skipped (no figure requested)")
                else:
                    rvs_cache = _cache_path(
                        ext_cache_root,
                        scenario_key="real_vs_sim",
                        n_neighbors=args.n_neighbors,
                        min_dist=args.min_dist,
                        subsample=args.subsample,
                    )
                    t0 = time.time()
                    if args.skip_per_extractor:
                        panels, status = _panels_real_vs_sim(
                            real, sim,
                            metric=metric, n_neighbors=args.n_neighbors,
                            min_dist=args.min_dist,
                            cache_path=rvs_cache,
                        )
                        print(f"  · real_vs_sim   {time.time() - t0:5.1f}s "
                              f"[{status}] (panels cached for compare, "
                              f"no per-extractor figure)")
                    else:
                        path, panels, status = render_real_vs_sim(
                            real, sim,
                            extractor=extractor,
                            out_dir=out_root,
                            n_neighbors=args.n_neighbors,
                            min_dist=args.min_dist,
                            fmt=args.format,
                            dpi=args.dpi,
                            cache_path=rvs_cache,
                        )
                        print(f"  · real_vs_sim   {time.time() - t0:5.1f}s "
                              f"[{status}] -> {path}")
                    panels_cache["real_vs_sim"][extractor] = panels

        # Pre-load degradation level stacks once: we may need them for both
        # the per-extractor progressive figure and the parameter sweep.
        degradation_levels: "dict[str, List[np.ndarray]]" = {}
        if not args.skip_degradation or args.sweep:
            for subdir, _label, _short in DEGRADATION_TYPES:
                levels = _gather_levels(
                    args.base_dir, subdir, extractor,
                    subsample=args.subsample, rng=rng,
                )
                if levels is not None:
                    degradation_levels[subdir] = levels

        if not args.skip_degradation:
            for subdir, label, _short in DEGRADATION_TYPES:
                levels = degradation_levels.get(subdir)
                if levels is None:
                    continue
                shapes = ", ".join(
                    f"L{i+1} N={a.shape[0]}" for i, a in enumerate(levels)
                )
                print(f"  {subdir:<22}: {shapes}")

                if args.skip_per_extractor and not args.compare_extractors:
                    print(f"  · {subdir:<22}  skipped (no figure requested)")
                    continue

                deg_cache = _cache_path(
                    ext_cache_root,
                    scenario_key=subdir,
                    n_neighbors=args.n_neighbors,
                    min_dist=args.min_dist,
                    subsample=args.subsample,
                )
                t0 = time.time()
                if args.skip_per_extractor:
                    panels, status = _panels_progressive(
                        real, levels,
                        metric=metric, n_neighbors=args.n_neighbors,
                        min_dist=args.min_dist,
                        cache_path=deg_cache,
                    )
                    print(f"  · {subdir:<22} {time.time() - t0:5.1f}s "
                          f"[{status}] (panels cached for compare, "
                          f"no per-extractor figure)")
                else:
                    path, panels, status = render_progressive_levels(
                        real, levels,
                        extractor=extractor,
                        deg_label=label,
                        out_name=subdir,
                        out_dir=out_root,
                        n_neighbors=args.n_neighbors,
                        min_dist=args.min_dist,
                        fmt=args.format,
                        dpi=args.dpi,
                        cache_path=deg_cache,
                    )
                    print(f"  · {subdir:<22} {time.time() - t0:5.1f}s "
                          f"[{status}] -> {path}")
                panels_cache[subdir][extractor] = panels

        if args.sweep:
            sweep_specs: List[Tuple[str, str, str, List[Tuple[str, np.ndarray]]]] = []
            if sim is not None:
                sweep_specs.append((
                    "Real + Sim",
                    "real_vs_sim",
                    f"sweep_real_vs_sim.{args.format}",
                    [("Sim", sim)],
                ))
            for subdir, label, _short in DEGRADATION_TYPES:
                levels = degradation_levels.get(subdir)
                if levels is None:
                    continue
                overlays = [
                    (f"Level {i+1}", arr) for i, arr in enumerate(levels)
                ]
                sweep_specs.append((
                    f"Real + {label} (Lvl 1+2+3)",
                    subdir,
                    f"sweep_{subdir}.{args.format}",
                    overlays,
                ))

            for scenario_label, scenario_key, fname, overlays in sweep_specs:
                t0 = time.time()
                path, n_hits, n_total = render_param_sweep(
                    real, overlays,
                    extractor=extractor,
                    scenario_label=scenario_label,
                    scenario_key=scenario_key,
                    out_filename=fname,
                    out_dir=out_root,
                    dpi=args.dpi,
                    n_neighbors_list=args.sweep_n_neighbors,
                    min_dist_list=args.sweep_min_dist,
                    cache_root=ext_cache_root,
                    subsample=args.subsample,
                )
                print(f"  · sweep [{scenario_label:<32}] "
                      f"{time.time() - t0:5.1f}s "
                      f"[cache {n_hits}/{n_total} hit] -> {path}")

    if args.compare_extractors:
        print("\n=== Cross-extractor compare ===")
        compare_dir = out_root / "compare"
        t0 = time.time()
        path = render_compare_overview(
            panels_cache,
            extractors=args.extractors,
            out_path=compare_dir / f"overview.{args.format}",
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            dpi=args.dpi,
        )
        if path is None:
            print("  ! overview: no extractors had panels for all "
                  "three scenarios (real_vs_sim, blur, noise); skipping")
        else:
            print(f"  · overview {time.time() - t0:5.1f}s -> {path}")

    print("\nall done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

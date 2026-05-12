#!/usr/bin/env python3
"""
Geometric-property visualization for ADS V&V realism evaluation.

Usage:
    python plot_metrics.py --data-dir ./results --out-dir ./figures

To add a new transformation, just drop its result_<name>.json into --data-dir
and append one entry to TRANSFORMS below.
"""

from __future__ import annotations
import numpy as np
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

# ──────────────────────────────────────────────────────────────────────
#  Registry: add a new transformation by appending ONE entry here.
# ──────────────────────────────────────────────────────────────────────


@dataclass
class TransformSpec:
    """Descriptor for one image-transformation experiment."""
    key: str                          # internal id, also the dict key
    label: str                        # display name on plots
    filename: str                     # JSON filename (inside --data-dir)
    color: str                        # matplotlib colour
    marker: str                       # matplotlib marker
    linestyle: str = "-"
    level_labels: List[str] = field(  # x-tick labels (auto-derived if empty)
        default_factory=list
    )
    is_baseline: bool = False         # True for real-vs-real
    is_reference: bool = False        # True for sim (horizontal line)

    # convenience: filled at load time
    _data: Optional[dict] = field(default=None, repr=False, init=False)


# ──── CURRENT TRANSFORMS (edit / extend this list) ────────────────────
TRANSFORMS: List[TransformSpec] = [
    # ── baselines / references ──
    TransformSpec(
        key="real", label="Real Baseline",
        filename="result_real.json",
        color="#7f8c8d", marker="", linestyle=":",
        is_baseline=True,
    ),
    TransformSpec(
        key="sim", label="vKITTI Sim",
        filename="result_sim-b577a7cd.json",
        color="#2c3e50", marker="", linestyle="--",
        is_reference=True,
    ),
    # ── controlled transforms ──
    TransformSpec(
        key="noise", label="Gaussian Noise",
        filename="result_noise-476c68e5.json",
        color="#e74c3c", marker="o",
        level_labels=["σ=0.02", "σ=0.04", "σ=0.08"],
    ),
    TransformSpec(
        key="blur", label="Gaussian Blur",
        filename="result_blur-33507ab9.json",
        color="#3498db", marker="s",
        level_labels=["σ=2", "σ=4", "σ=8"],
    ),
    TransformSpec(
        key="color_jitter", label="Color Jitter (Hue)",
        filename="result_color_jitter.json",
        color="#2ecc71", marker="^",
        level_labels=["h=0.1", "h=0.3", "h=0.5"],
    ),
    TransformSpec(
        key="brightness", label="Brightness",
        filename="result_bright.json",
        color="#f39c12", marker="D",
        level_labels=["b=2", "b=4", "b=8"],
    ),
    TransformSpec(
        key="contrast", label="Contrast",
        filename="result_contrast.json",
        color="#9b59b6", marker="v",
        level_labels=["c=0.2", "c=0.5", "c=1.0"],
    ),
    # ── TEMPLATE: uncomment & fill to add a new transform ──
    # TransformSpec(
    #     key="rain",  label="Rain Simulation",
    #     filename="result_rain.json",
    #     color="#1abc9c",  marker="P",
    #     level_labels=["light", "moderate", "heavy"],
    # ),
]

# ──────────────────────────────────────────────────────────────────────
#  Geometric metrics we extract from each JSON record.
# ──────────────────────────────────────────────────────────────────────


@dataclass
class MetricSpec:
    key: str          # internal id
    label: str        # y-axis / title label
    extract: str      # 'direct' field name  OR  'computed'
    higher_is_worse: bool = True  # arrow semantics for the paper


METRICS: List[MetricSpec] = [
    MetricSpec("baryshift",   "BaryShift",      "baryshift_mean"),
    MetricSpec("lid",         "LID",            "lid_mean"),
    MetricSpec("overlap",     "Overlap",        "overlap_mean"),
    MetricSpec("nmi",         "NMI",            "nmi"),
    MetricSpec("ari",         "ARI",            "ari"),
    MetricSpec("indeg_delta", "In-degree Δ",    "computed"),
]

SPACES = [
    ("inception_v3",  "Inception-v3"),
    ("clip_vit_b32",  "CLIP ViT-B/32"),
    ("resnet50",      "ResNet-50"),
    ("lpips_vgg",     "LPIPS / VGG"),
    ("pixel",         "Pixel"),
    ("segformer",     "SegFormer"),
]

K_VALUES = ["10", "20", "50"]

# ──────────────────────────────────────────────────────────────────────
#  Data loading
# ──────────────────────────────────────────────────────────────────────


def load_transforms(data_dir: str, specs: List[TransformSpec]) -> List[TransformSpec]:
    """Load JSON for every spec whose file exists; warn & skip missing."""
    loaded = []
    for spec in specs:
        path = os.path.join(data_dir, spec.filename)
        if not os.path.isfile(path):
            print(
                f"[WARN] {spec.filename} not found in {data_dir}, skipping '{spec.key}'")
            continue
        with open(path) as f:
            spec._data = json.load(f)
        loaded.append(spec)
    return loaded


def _extract_metric(record: dict, metric: MetricSpec) -> float:
    if metric.extract == "computed":
        # In-degree Δ = real_mean - syn_mean
        return record["indeg_real_mean"] - record["indeg_syn_mean"]
    return record[metric.extract]


def get_values(
    spec: TransformSpec,
    space: str,
    k: str,
    metric: MetricSpec,
) -> List[float]:
    """Return a list of metric values, one per level."""
    levels = spec._data["config"]["level_tags"]
    sweep = spec._data["sweep"]
    return [_extract_metric(sweep[space][k][lvl], metric) for lvl in levels]

# ──────────────────────────────────────────────────────────────────────
#  Plotting helpers
# ──────────────────────────────────────────────────────────────────────


def _leveled(specs: List[TransformSpec]) -> List[TransformSpec]:
    """Return only multi-level (non-baseline, non-reference) transforms."""
    return [s for s in specs if not s.is_baseline and not s.is_reference]


def _baseline(specs: List[TransformSpec]) -> Optional[TransformSpec]:
    return next((s for s in specs if s.is_baseline), None)


def _reference(specs: List[TransformSpec]) -> Optional[TransformSpec]:
    return next((s for s in specs if s.is_reference), None)


def _add_ref_lines(ax, specs, space, k, metric):
    """Draw horizontal reference lines for baseline & sim."""
    bl = _baseline(specs)
    if bl and bl._data:
        val = get_values(bl, space, k, metric)[0]
        ax.axhline(val, color=bl.color, ls=bl.linestyle, lw=1.2,
                   alpha=0.7, label=bl.label, zorder=1)
    ref = _reference(specs)
    if ref and ref._data:
        val = get_values(ref, space, k, metric)[0]
        ax.axhline(val, color=ref.color, ls=ref.linestyle, lw=1.5,
                   alpha=0.8, label=ref.label, zorder=1)


def _style_ax(ax, title="", xlabel="Severity Level", ylabel=""):
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=8)

# ──────────────────────────────────────────────────────────────────────
#  Plot functions  (each generates one figure file)
# ──────────────────────────────────────────────────────────────────────


def plot_single_space_all_metrics(
    specs: List[TransformSpec],
    space: str,
    space_label: str,
    k: str,
    out_dir: str,
):
    """
    Fig 1-type: one subplot per metric, all transforms overlaid.
    Fixed to one (space, k) pair.
    """
    mlist = [m for m in METRICS if m.key != "ari"]  # skip ARI to save space
    ncols = 3
    nrows = (len(mlist) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    fig.suptitle(
        f"All Metrics  —  {space_label},  k = {k}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]

    leveled = _leveled(specs)
    for idx, metric in enumerate(mlist):
        ax = axes_flat[idx]
        for spec in leveled:
            vals = get_values(spec, space, k, metric)
            xs = list(range(1, len(vals) + 1))
            ax.plot(xs, vals, marker=spec.marker, color=spec.color,
                    label=spec.label, lw=2, markersize=7, zorder=3)
        _add_ref_lines(ax, specs, space, k, metric)
        _style_ax(ax, title=metric.label, ylabel=metric.label)
        ax.set_xticks(range(1, 4))
        ax.set_xticklabels(["L1", "L2", "L3"])
        if idx == 0:
            ax.legend(fontsize=7, loc="best")

    # hide leftover axes
    for j in range(len(mlist), nrows * ncols):
        axes_flat[j].set_visible(False)

    _save(fig, out_dir, f"metrics_{space}_k{k}")


def plot_metric_across_spaces(
    specs: List[TransformSpec],
    metric: MetricSpec,
    k: str,
    out_dir: str,
):
    """
    Fig 2-type: one subplot per embedding space, single metric,
    all transforms overlaid.
    """
    ncols = 3
    nrows = (len(SPACES) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    fig.suptitle(
        f"{metric.label}  across all embedding spaces  (k = {k})",
        fontsize=13, fontweight="bold", y=1.01,
    )
    axes_flat = axes.flat

    leveled = _leveled(specs)
    for s_idx, (space, space_label) in enumerate(SPACES):
        ax = axes_flat[s_idx]
        for spec in leveled:
            vals = get_values(spec, space, k, metric)
            xs = list(range(1, len(vals) + 1))
            ax.plot(xs, vals, marker=spec.marker, color=spec.color,
                    label=spec.label, lw=2, markersize=7, zorder=3)
        _add_ref_lines(ax, specs, space, k, metric)
        _style_ax(ax, title=space_label, ylabel=metric.label)
        ax.set_xticks(range(1, 4))
        ax.set_xticklabels(["L1", "L2", "L3"])
        if s_idx == 0:
            ax.legend(fontsize=6, loc="best")

    for j in range(len(SPACES), nrows * ncols):
        axes_flat[j].set_visible(False)

    _save(fig, out_dir, f"{metric.key}_all_spaces_k{k}")


def plot_lpips_behavior(
    specs: List[TransformSpec],
    k: str,
    out_dir: str,
):
    """
    Fig 3-type: LPIPS space only, BaryShift + LID + NMI side-by-side,
    highlighting inverse vs normal trends.
    """
    show = [m for m in METRICS if m.key in ("baryshift", "lid", "nmi")]
    fig, axes = plt.subplots(1, len(show), figsize=(5 * len(show), 4.5))
    fig.suptitle(
        f"LPIPS / VGG  —  Inverse vs Normal trends  (k = {k})",
        fontsize=12, fontweight="bold",
    )

    leveled = _leveled(specs)
    for idx, metric in enumerate(show):
        ax = axes[idx]
        for spec in leveled:
            vals = get_values(spec, "lpips_vgg", k, metric)
            trend = "↓" if vals[-1] < vals[0] else "↑"
            ax.plot(
                [1, 2, 3], vals,
                marker=spec.marker, color=spec.color,
                label=f"{spec.label} {trend}",
                lw=2.2, markersize=8, zorder=3,
            )
        _add_ref_lines(ax, specs, "lpips_vgg", k, metric)
        _style_ax(ax, title=f"LPIPS: {metric.label}", ylabel=metric.label)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["L1", "L2", "L3"])
        ax.legend(fontsize=7)

    _save(fig, out_dir, f"lpips_inverse_k{k}")


def plot_signatures_heatmap(
    specs: List[TransformSpec],
    space: str,
    space_label: str,
    k: str,
    out_dir: str,
):
    """
    Fig 4-type: normalised heatmap of (condition × metric).
    Rows = metrics, columns = conditions.
    """
    show_metrics = [m for m in METRICS if m.key in
                    ("baryshift", "lid", "overlap", "nmi", "indeg_delta")]

    # Build condition list: baseline, L1+L3 per transform, sim
    conditions: List[Tuple[str, float, float, float, float, float]] = []
    cond_labels: List[str] = []

    bl = _baseline(specs)
    if bl and bl._data:
        vals = [get_values(bl, space, k, m)[0] for m in show_metrics]
        conditions.append(vals)
        cond_labels.append("Real")

    for spec in _leveled(specs):
        for li in [0, -1]:                       # first and last level
            n_levels = len(spec._data["config"]["level_tags"])
            level_idx = li if li >= 0 else n_levels + li
            vals = [get_values(spec, space, k, m)[level_idx]
                    for m in show_metrics]
            tag = "L1" if li == 0 else f"L{n_levels}"
            conditions.append(vals)
            short = spec.label.split("(")[0].strip()
            if len(short) > 10:
                short = short[:10] + "."
            cond_labels.append(f"{short} {tag}")

    ref = _reference(specs)
    if ref and ref._data:
        vals = [get_values(ref, space, k, m)[0] for m in show_metrics]
        conditions.append(vals)
        cond_labels.append(ref.label)

    mat = np.array(conditions)
    # min-max normalise per metric (column)
    vmin = mat.min(axis=0, keepdims=True)
    vmax = mat.max(axis=0, keepdims=True)
    normed = (mat - vmin) / (vmax - vmin + 1e-12)

    fig, ax = plt.subplots(figsize=(max(10, 1.1 * len(cond_labels)), 4.5))
    im = ax.imshow(normed.T, cmap="YlOrRd", aspect="auto")
    ax.set_yticks(range(len(show_metrics)))
    ax.set_yticklabels([m.label for m in show_metrics], fontsize=9)
    ax.set_xticks(range(len(cond_labels)))
    ax.set_xticklabels(cond_labels, rotation=45, ha="right", fontsize=8)
    ax.set_title(
        f"Geometric Signatures  —  {space_label},  k = {k}",
        fontsize=11, fontweight="bold",
    )
    plt.colorbar(im, ax=ax, label="Normalised (0 = min, 1 = max)", shrink=0.8)

    _save(fig, out_dir, f"signatures_{space}_k{k}")


def plot_k_sensitivity(
    specs: List[TransformSpec],
    space: str,
    space_label: str,
    metric: MetricSpec,
    out_dir: str,
):
    """
    Fig 5-type: one subplot per k value, showing how a single metric
    behaves across severity levels.
    """
    fig, axes = plt.subplots(1, len(K_VALUES),
                             figsize=(5 * len(K_VALUES), 4.5))
    fig.suptitle(
        f"k-Sensitivity: {metric.label}  —  {space_label}",
        fontsize=12, fontweight="bold",
    )

    leveled = _leveled(specs)
    for ki, k in enumerate(K_VALUES):
        ax = axes[ki]
        for spec in leveled:
            vals = get_values(spec, space, k, metric)
            ax.plot(
                range(1, len(vals) + 1), vals,
                marker=spec.marker, color=spec.color,
                label=spec.label, lw=2, markersize=7, zorder=3,
            )
        _add_ref_lines(ax, specs, space, k, metric)
        _style_ax(ax, title=f"k = {k}", ylabel=metric.label)
        ax.set_xticks(range(1, 4))
        ax.set_xticklabels(["L1", "L2", "L3"])
        if ki == 0:
            ax.legend(fontsize=7)

    _save(fig, out_dir, f"k_sens_{metric.key}_{space}")


def plot_radar(
    specs: List[TransformSpec],
    space: str,
    space_label: str,
    k: str,
    out_dir: str,
):
    """
    Radar / spider chart: each transform's L3 (or last level) normalised
    metric profile, useful for comparing geometric signatures at a glance.
    """
    show_metrics = [m for m in METRICS if m.key in
                    ("baryshift", "lid", "overlap", "nmi")]
    labels = [m.label for m in show_metrics]
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_title(
        f"Geometric Profiles (L3)  —  {space_label},  k = {k}",
        fontsize=11, fontweight="bold", pad=20,
    )

    # Collect L3 values for normalisation
    leveled = _leveled(specs)
    raw_rows = []
    for spec in leveled:
        row = [get_values(spec, space, k, m)[-1] for m in show_metrics]
        raw_rows.append(row)
    raw = np.array(raw_rows)
    vmin = raw.min(axis=0)
    vmax = raw.max(axis=0)
    rng = vmax - vmin + 1e-12

    for i, spec in enumerate(leveled):
        normed = ((raw[i] - vmin) / rng).tolist()
        normed += normed[:1]
        ax.plot(angles, normed, marker=spec.marker, color=spec.color,
                label=spec.label, lw=2, markersize=6)
        ax.fill(angles, normed, color=spec.color, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.3, 1.1))

    _save(fig, out_dir, f"radar_{space}_k{k}")

# ──────────────────────────────────────────────────────────────────────
#  I/O
# ──────────────────────────────────────────────────────────────────────


def _save(fig, out_dir: str, stem: str, dpi: int = 200):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{stem}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")

# ──────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Generate geometric-property plots for ADS V&V evaluation."
    )
    parser.add_argument(
        "--data-dir", default=".",
        help="Directory containing result_*.json files",
    )
    parser.add_argument(
        "--out-dir", default="./figures",
        help="Directory for output PNGs",
    )
    parser.add_argument(
        "--k", default="10",
        choices=K_VALUES,
        help="Default k value for non-sensitivity plots (default: 10)",
    )
    parser.add_argument(
        "--dpi", type=int, default=200,
        help="Output image DPI (default: 200)",
    )
    parser.add_argument(
        "--only", default=None,
        help="Comma-separated list of plot names to generate "
             "(metrics, cross_space, lpips, heatmap, k_sens, radar). "
             "Default: all.",
    )
    args = parser.parse_args()

    # Load data
    specs = load_transforms(args.data_dir, TRANSFORMS)
    if not specs:
        print("[ERROR] No result JSON files found. Check --data-dir.")
        sys.exit(1)
    print(f"Loaded {len(specs)} experiment(s) from {args.data_dir}\n")

    which = set(args.only.split(",")) if args.only else {
        "metrics", "cross_space", "lpips", "heatmap", "k_sens", "radar",
    }
    k = args.k

    # 1. All metrics in one space
    if "metrics" in which:
        print("[1/6] All metrics per space …")
        for space, label in SPACES:
            plot_single_space_all_metrics(specs, space, label, k, args.out_dir)

    # 2. Single metric across all spaces
    if "cross_space" in which:
        print("[2/6] Cross-space comparison …")
        for m in METRICS:
            if m.key == "ari":
                continue
            plot_metric_across_spaces(specs, m, k, args.out_dir)

    # 3. LPIPS inverse-trend analysis
    if "lpips" in which:
        print("[3/6] LPIPS inverse-trend …")
        plot_lpips_behavior(specs, k, args.out_dir)

    # 4. Geometric-signatures heatmap
    if "heatmap" in which:
        print("[4/6] Signatures heatmap …")
        for space, label in SPACES:
            plot_signatures_heatmap(specs, space, label, k, args.out_dir)

    # 5. k-sensitivity
    if "k_sens" in which:
        print("[5/6] k-sensitivity …")
        bs = next(m for m in METRICS if m.key == "baryshift")
        lid = next(m for m in METRICS if m.key == "lid")
        for space, label in SPACES:
            plot_k_sensitivity(specs, space, label, bs, args.out_dir)
            plot_k_sensitivity(specs, space, label, lid, args.out_dir)

    # 6. Radar chart
    if "radar" in which:
        print("[6/6] Radar charts …")
        for space, label in SPACES:
            plot_radar(specs, space, label, k, args.out_dir)

    print(f"\nDone — all figures saved to {args.out_dir}/")


if __name__ == "__main__":
    main()

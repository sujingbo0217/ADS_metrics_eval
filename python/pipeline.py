#!/usr/bin/env python3
"""
pipeline.py — k-NN geometric analysis over pre-computed .fbin embeddings.

Usage
-----
    # Per-folder embeddings (default mode of embed.py)
    python pipeline.py \
        --embeddings-dir embeddings \
        --real real \
        --levels deg_l0 deg_l1 deg_l2 deg_l3 deg_l4 \
        --spaces inception_v3 clip_vit_b32 resnet50 lpips_vgg segformer pixel \
        --k 10 20 50 \
        --output-dir results

    # Single space, single k (quick test)
    python pipeline.py \
        --embeddings-dir embeddings \
        --real real --levels deg_l0 \
        --spaces inception_v3 --k 20 \
        --output-dir results/test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from community import community_louvain  # python-louvain
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import knn_ext  # compiled CUDA pybind11 module

from embedding.fbin_io import read_fbin

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

# Spaces that use cosine similarity (InnerProduct after L2-normalization).
# All others default to L2.
COSINE_SPACES = {"clip_vit_b32"}


def _dist_type(space: str) -> knn_ext.DistanceType:
    if space in COSINE_SPACES:
        return knn_ext.DistanceType.InnerProduct
    return knn_ext.DistanceType.L2


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _maybe_normalize(emb: np.ndarray, space: str) -> np.ndarray:
    """L2-normalize rows when the space uses cosine / inner-product distance."""
    if space in COSINE_SPACES:
        return knn_ext.l2_normalize_rows(np.ascontiguousarray(emb, dtype=np.float32))
    return np.ascontiguousarray(emb, dtype=np.float32)


def _build_mutual_knn_graph(pooled_idx: np.ndarray) -> nx.Graph:
    """Build an undirected mutual k-NN graph from pooled k-NN indices.

    An edge (i, j) exists iff j ∈ kNN(i) AND i ∈ kNN(j).
    Uses set-based lookup for O(N·k) total time.
    """
    n, k = pooled_idx.shape
    neighbor_sets = [set(pooled_idx[i]) - {i} for i in range(n)]

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in neighbor_sets[i]:
            # j > i avoids duplicate checks
            if j > i and i in neighbor_sets[j]:
                G.add_edge(i, j)
    return G


def _community_analysis(
    pooled_idx: np.ndarray,
    n_real: int,
    n_total: int,
) -> dict:
    """Run Louvain community detection on the mutual k-NN graph and compute
    NMI / ARI against the ground-truth real/fake labels."""
    G = _build_mutual_knn_graph(pooled_idx)

    partition = community_louvain.best_partition(G, random_state=42)
    pred_labels = np.array([partition.get(i, -1) for i in range(n_total)])

    # Ground-truth: 0 = real (first n_real rows), 1 = synthetic
    gt_labels = np.array([0] * n_real + [1] * (n_total - n_real))

    nmi = normalized_mutual_info_score(gt_labels, pred_labels)
    ari = adjusted_rand_score(gt_labels, pred_labels)
    modularity = community_louvain.modularity(partition, G)
    n_communities = len(set(partition.values()))

    return {
        "nmi": float(nmi),
        "ari": float(ari),
        "modularity": float(modularity),
        "n_communities": int(n_communities),
        "n_edges": G.number_of_edges(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Core: single (space, level, k) run
# ═══════════════════════════════════════════════════════════════════════════

def run_single(
    emb_real: np.ndarray,       # (N_r, d)  already normalized if needed
    emb_syn: np.ndarray,        # (N_q, d)  already normalized if needed
    real_only_idx: np.ndarray,  # (N_r, k)  precomputed real-only k-NN
    k: int,
    dist_type: knn_ext.DistanceType,
) -> dict:
    """Compute all five geometric properties for one (R, Q_L, k) triple.

    k-NN directions
    ----------------
    Cross-set (Q → R):  base = real,  query = synthetic.
        For each synthetic point, find its k nearest real neighbors.
        Used for: Barycenter Shift, LID.

    Pooled (R ∪ Q):  self-query on the concatenated set.
        Used for: Neighborhood Overlap, In-degree, Community.
    """
    N_r = emb_real.shape[0]
    N_q = emb_syn.shape[0]
    N_p = N_r + N_q

    # ── 1. Cross-set k-NN: Q → R ──
    # For each synthetic point, find k nearest neighbors in real set.
    cross_idx, cross_dist = knn_ext.cross_set_knn(
        emb_real, emb_syn, k, dist_type
    )
    # cross_idx:  (N_q, k)  indices into emb_real
    # cross_dist: (N_q, k)  distances, ascending

    # ── 2. Pooled k-NN: R ∪ Q ──
    # rows [0, N_r) = real, [N_r, N_p) = syn
    emb_pooled = np.vstack([emb_real, emb_syn])
    pooled_idx, pooled_dist = knn_ext.pooled_knn(
        np.ascontiguousarray(emb_pooled, dtype=np.float32), k, dist_type
    )
    # pooled_idx: (N_p, k)  indices into emb_pooled

    # ── 3. Geometric properties ──

    # 3a. Barycenter Shift (from cross-set)
    #     For each synthetic point: distance from itself to the centroid
    #     of its k nearest real neighbors.
    shifts = knn_ext.barycenter_shift(emb_real, emb_syn, cross_idx, k)
    # shifts: (N_q,)

    # 3b. LID (from cross-set distances)
    #     Local Intrinsic Dimensionality estimated from the distance profile
    #     of each synthetic point to its real neighbors.
    lid = knn_ext.compute_lid(cross_dist, k)
    # lid: (N_q,)

    # 3c. Neighborhood Overlap (pooled vs real-only)
    #     For each real point: fraction of its real-only k-NN that survive
    #     in the pooled k-NN (where synthetic points may displace real ones).
    overlap = knn_ext.neighbor_overlap(real_only_idx, pooled_idx, k)
    # overlap: (N_r,)

    # 3d. In-degree (pooled graph)
    #     How many times each point is chosen as a neighbor.
    indegree = knn_ext.compute_indegree(pooled_idx, k)
    # indegree: (N_p,)  int array
    indeg_real = indegree[:N_r].astype(np.float64)
    indeg_syn = indegree[N_r:].astype(np.float64)

    # 3e. Community structure (mutual k-NN → Louvain)
    community = _community_analysis(pooled_idx, N_r, N_p)

    # ── 4. Aggregate statistics ──
    result = {
        # Barycenter Shift
        "baryshift_mean": float(np.mean(shifts)),
        "baryshift_std": float(np.std(shifts)),
        "baryshift_median": float(np.median(shifts)),

        # LID
        "lid_mean": float(np.mean(lid)),
        "lid_std": float(np.std(lid)),
        "lid_median": float(np.median(lid)),

        # Overlap
        "overlap_mean": float(np.mean(overlap)),
        "overlap_std": float(np.std(overlap)),

        # In-degree
        "indeg_real_mean": float(np.mean(indeg_real)),
        "indeg_real_std": float(np.std(indeg_real)),
        "indeg_syn_mean": float(np.mean(indeg_syn)),
        "indeg_syn_std": float(np.std(indeg_syn)),

        # Community
        **community,
    }
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Sweep across degradation levels for a fixed (space, k)
# ═══════════════════════════════════════════════════════════════════════════

def run_degradation_sweep(
    emb_real: np.ndarray,
    emb_levels: Dict[str, np.ndarray],
    k: int,
    dist_type: knn_ext.DistanceType,
) -> Dict[str, dict]:
    """Run all geometric analyses for each degradation level."""
    N_r = emb_real.shape[0]

    # Precompute real-only k-NN (used for overlap at every level)
    print(f"    real-only k-NN (N={N_r}, k={k}) ...", end=" ", flush=True)
    t0 = time.time()
    real_only_idx, _ = knn_ext.pooled_knn(emb_real, k, dist_type)
    print(f"{time.time() - t0:.3f}s")

    results: Dict[str, dict] = {}
    for level_tag in sorted(emb_levels.keys()):
        emb_syn = emb_levels[level_tag]
        N_q = emb_syn.shape[0]
        print(f"    level={level_tag}  N_q={N_q} ...", end=" ", flush=True)
        t0 = time.time()

        res = run_single(emb_real, emb_syn, real_only_idx, k, dist_type)
        dt = time.time() - t0
        res["time_seconds"] = round(dt, 2)
        results[level_tag] = res

        print(
            f"{dt:.1f}s  "
            f"Bary={res['baryshift_mean']:.4f}  "
            f"LID={res['lid_mean']:.2f}  "
            f"Ovlp={res['overlap_mean']:.4f}  "
            f"NMI={res['nmi']:.4f}  "
            f"ARI={res['ari']:.4f}"
        )

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 5: Cross-Space Redundancy Analysis
# ═══════════════════════════════════════════════════════════════════════════

def cross_space_redundancy(
    all_results: Dict[str, Dict[int, Dict[str, dict]]],
) -> dict:
    """Compute pairwise Spearman correlations of geometric properties
    across embedding spaces.

    For each pair of spaces (A, B), for each property, correlate the
    property values across degradation levels.  High correlation means
    the two spaces carry redundant information for that property.

    Parameters
    ----------
    all_results : {space: {k: {level: {prop: value}}}}

    Returns
    -------
    dict with structure:
        {
            "spearman": {prop: {(spaceA, spaceB): (rho, p)}},
            "k_used": k,
        }
    """
    spaces = sorted(all_results.keys())
    if len(spaces) < 2:
        return {}

    # Use the first k value available
    first_space = spaces[0]
    k_vals = sorted(all_results[first_space].keys())
    k = k_vals[0]

    properties = [
        "baryshift_mean", "lid_mean", "overlap_mean",
        "indeg_real_mean", "indeg_syn_mean", "nmi", "ari",
    ]

    # Build per-space property vectors (one value per level)
    def _prop_vector(space: str, prop: str) -> np.ndarray:
        levels = sorted(all_results[space][k].keys())
        return np.array([all_results[space][k][lv][prop] for lv in levels])

    spearman = {}
    for prop in properties:
        spearman[prop] = {}
        for i, sA in enumerate(spaces):
            for sB in spaces[i + 1:]:
                vA = _prop_vector(sA, prop)
                vB = _prop_vector(sB, prop)
                if len(vA) < 3:
                    continue
                rho, p = spearmanr(vA, vB)
                spearman[prop][(sA, sB)] = {
                    "rho": float(rho),
                    "p_value": float(p),
                }

    return {"spearman": spearman, "k_used": k}


# ═══════════════════════════════════════════════════════════════════════════
# Experiment: k-Sensitivity Analysis
# ═══════════════════════════════════════════════════════════════════════════

def k_sensitivity_summary(
    all_results: Dict[str, Dict[int, Dict[str, dict]]],
) -> dict:
    """For each space, compare how properties change across k values
    at the same degradation level.  Useful for selecting robust k."""
    summary = {}
    properties = ["baryshift_mean", "lid_mean", "overlap_mean", "nmi"]

    for space, per_k in all_results.items():
        k_vals = sorted(per_k.keys())
        levels = sorted(next(iter(per_k.values())).keys())
        space_summary = {}
        for prop in properties:
            # Matrix: rows = levels, cols = k values
            mat = np.array([
                [per_k[k][lv][prop] for k in k_vals]
                for lv in levels
            ])
            # Coefficient of variation across k, averaged over levels
            means = mat.mean(axis=1, keepdims=True)
            means = np.where(np.abs(means) < 1e-10, 1.0, means)
            cv = (mat.std(axis=1) / np.abs(means)).mean()
            space_summary[prop] = {
                "k_values": k_vals,
                "cv_across_k": float(cv),
            }
        summary[space] = space_summary
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# Driver
# ═══════════════════════════════════════════════════════════════════════════

def run_from_fbins(
    embeddings_dir: str,
    real_tag: str,
    level_tags: List[str],
    spaces: List[str],
    k_values: List[int],
    output_dir: str,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    all_results: Dict[str, Dict[int, Dict[str, dict]]] = {}

    for space in spaces:
        real_path = os.path.join(embeddings_dir, real_tag, f"{space}.fbin")
        if not os.path.isfile(real_path):
            print(f"[skip] missing {real_path}", file=sys.stderr)
            continue

        print(f"\n{'=' * 60}")
        print(f"Space: {space}")
        print(f"{'=' * 60}")

        emb_real_raw = read_fbin(real_path)
        dist = _dist_type(space)
        emb_real = _maybe_normalize(emb_real_raw, space)
        print(
            f"  real: N={emb_real.shape[0]}  D={emb_real.shape[1]}  dist={dist}")

        emb_levels: Dict[str, np.ndarray] = {}
        for tag in level_tags:
            p = os.path.join(embeddings_dir, tag, f"{space}.fbin")
            if not os.path.isfile(p):
                print(f"  [skip level] missing {p}", file=sys.stderr)
                continue
            emb_levels[tag] = _maybe_normalize(read_fbin(p), space)

        if not emb_levels:
            print("  no levels found, skipping space", file=sys.stderr)
            continue

        per_k: Dict[int, Dict[str, dict]] = {}
        for k in k_values:
            print(f"\n  k = {k}")
            per_k[k] = run_degradation_sweep(emb_real, emb_levels, k, dist)
        all_results[space] = per_k

    if not all_results:
        print("no results produced", file=sys.stderr)
        return {}

    # ── Post-sweep analyses ──
    print(f"\n{'=' * 60}")
    print("Post-sweep analyses")
    print(f"{'=' * 60}")

    redundancy = cross_space_redundancy(all_results)
    k_sens = k_sensitivity_summary(all_results)

    output = {
        "config": {
            "embeddings_dir": os.path.abspath(embeddings_dir),
            "real_tag": real_tag,
            "level_tags": level_tags,
            "spaces": spaces,
            "k_values": k_values,
        },
        "sweep": _serialize(all_results),
        "cross_space_redundancy": _serialize(redundancy),
        "k_sensitivity": k_sens,
    }

    out_path = os.path.join(output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nresults -> {out_path}")

    # Also dump a compact summary table to stdout
    _print_summary_table(all_results)

    return output


def _serialize(obj):
    """Make an object JSON-serializable (convert tuple keys, numpy types)."""
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _print_summary_table(all_results: dict) -> None:
    """Print a compact summary table across spaces, levels, and k values."""
    print(f"\n{'─' * 80}")
    print("SUMMARY")
    print(f"{'─' * 80}")
    header = (
        f"{'Space':<14} {'k':>3} {'Level':<10} "
        f"{'Bary':>8} {'LID':>8} {'Ovlp':>8} "
        f"{'NMI':>8} {'ARI':>8}"
    )
    print(header)
    print("─" * len(header))

    for space in sorted(all_results.keys()):
        for k in sorted(all_results[space].keys()):
            for level in sorted(all_results[space][k].keys()):
                r = all_results[space][k][level]
                print(
                    f"{space:<14} {k:>3} {level:<10} "
                    f"{r['baryshift_mean']:>8.4f} "
                    f"{r['lid_mean']:>8.2f} "
                    f"{r['overlap_mean']:>8.4f} "
                    f"{r['nmi']:>8.4f} "
                    f"{r['ari']:>8.4f}"
                )


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run k-NN geometric analysis on .fbin embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--embeddings-dir", required=True,
        help="Root directory containing per-folder .fbin files (from embed.py).",
    )
    p.add_argument(
        "--real", required=True,
        help="Subfolder tag for the real image set.",
    )
    p.add_argument(
        "--levels", nargs="+", required=True,
        help="Subfolder tags for degradation levels (e.g. deg_l0 deg_l1 ...).",
    )
    p.add_argument(
        "--spaces", nargs="+", required=True,
        help="Embedding spaces to evaluate (must exist as <tag>/<space>.fbin).",
    )
    p.add_argument(
        "--k", nargs="+", type=int, default=[10, 20, 50],
        help="k values for k-NN.",
    )
    p.add_argument(
        "--output-dir", default="results",
        help="Directory to write results.json.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_from_fbins(
        embeddings_dir=args.embeddings_dir,
        real_tag=args.real,
        level_tags=args.levels,
        spaces=args.spaces,
        k_values=args.k,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

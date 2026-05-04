#!/usr/bin/env python3
"""
embed.py — extract image embeddings and dump them to ``.fbin`` files.

File format (little-endian):
    [uint32 N][uint32 D][float32 * N * D]   (row-major)

Two output modes
----------------

1. Per-folder (default): one ``.fbin`` per extractor written directly into
   ``--output-dir``.  Run once per source folder; ``--folders`` accepts a
   single entry in this mode.

    python embed.py \
        --folders /scratch/sim-real/gaussian_blur_level_1 \
        --output-dir /scratch/jsu02/sim-real-embedding/kitti_gaussian_blur/level_1 \
        --extractors inception_v3 clip_vit_b32 pixel

    .../kitti_gaussian_blur/level_1/
        inception_v3.fbin
        clip_vit_b32.fbin
        pixel.fbin
        manifest.json

2. Pooled (``--pool``): images from all folders are concatenated in the
   order given and a single ``.fbin`` per extractor is written to
   ``--output-dir``.  ``paths.txt`` lists the source path of each row
   and ``manifest.json`` records the [start, end) row range for each
   input folder so you can slice back by origin.

    python embed.py --pool \
        --folders /scratch/kitti_real /scratch/kitti_sim \
        --output-dir ../../data \
        --extractors inception_v3 clip_vit_b32 resnet50

    ../../data/
        inception_v3.fbin
        clip_vit_b32.fbin
        resnet50.fbin
        paths.txt
        manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Callable, Dict, List, Tuple

import numpy as np

from fbin_io import write_fbin
from embedding_extractors import (
    extract_clip,
    extract_inception,
    extract_lpips_embedding,
    extract_pixel,
    extract_resnet,
    extract_segformer,
)

# name -> (callable, expected_dim_or_None)
EXTRACTORS: Dict[str, Tuple[Callable, int | None]] = {
    "inception_v3": (extract_inception, 2048),
    "clip_vit_b32": (extract_clip, 512),
    "resnet50":     (extract_resnet, 2048),
    "lpips_vgg":    (extract_lpips_embedding, 1024),  # after PCA
    "segformer":    (extract_segformer, None),        # probed at runtime
    "pixel":        (extract_pixel, None),            # depends on --pixel-size
}


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(folder: str) -> List[str]:
    """Return all image paths under ``folder`` (recursive), sorted."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"not a directory: {folder}")
    out: List[str] = []
    for root, _, files in os.walk(folder):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                out.append(os.path.join(root, fn))
    out.sort()
    return out


def run_extractors(
    paths: List[str],
    extractors: List[str],
    out_dir: str,
    output_root: str,
    device: str,
    batch_size: int,
    overwrite: bool,
    label: str = "",
) -> Dict[str, dict]:
    """Run every requested extractor on ``paths`` and save one .fbin per extractor
    into ``out_dir``.  Manifest paths are recorded relative to ``output_root``.
    """
    os.makedirs(out_dir, exist_ok=True)
    results: Dict[str, dict] = {}

    for name in extractors:
        if name not in EXTRACTORS:
            raise ValueError(
                f"unknown extractor '{name}'. available: {sorted(EXTRACTORS)}"
            )
        out_path = os.path.join(out_dir, f"{name}.fbin")
        if os.path.exists(out_path) and not overwrite:
            print(f"  · {name}: {out_path} exists, skipping (--overwrite to redo)")
            continue

        fn, expected_dim = EXTRACTORS[name]
        t0 = time.time()
        emb = fn(paths, device=device, batch_size=batch_size)
        if emb.ndim != 2:
            raise RuntimeError(f"{name} returned shape {emb.shape}, expected 2-D")
        if expected_dim is not None and emb.shape[1] != expected_dim:
            print(
                f"  ! {name}: got dim {emb.shape[1]}, expected {expected_dim}",
                file=sys.stderr,
            )
        emb = np.ascontiguousarray(emb, dtype=np.float32)
        write_fbin(out_path, emb)
        dt = time.time() - t0
        tag = f"[{label}] " if label else ""
        print(
            f"  · {tag}{name}: N={emb.shape[0]} D={emb.shape[1]}  "
            f"({dt:.1f}s)  ->  {out_path}"
        )
        results[name] = {
            "path": os.path.relpath(out_path, output_root),
            "N": int(emb.shape[0]),
            "D": int(emb.shape[1]),
            "seconds": round(dt, 3),
        }

    return results


def gather_pooled(folders: List[str]) -> Tuple[List[str], Dict[str, dict]]:
    """Concatenate images across folders (folders in the given order, images
    sorted within each folder).  Returns (all_paths, per_folder_ranges)."""
    all_paths: List[str] = []
    ranges: Dict[str, dict] = {}
    for folder in folders:
        paths = list_images(folder)
        start = len(all_paths)
        all_paths.extend(paths)
        end = len(all_paths)
        tag = os.path.basename(os.path.normpath(folder))
        # disambiguate duplicate basenames
        if tag in ranges:
            tag = f"{tag}#{sum(1 for k in ranges if k.split('#')[0] == tag)}"
        ranges[tag] = {
            "source": os.path.abspath(folder),
            "count": end - start,
            "start": start,
            "end": end,
        }
        print(f"[{tag}] {end - start} images (rows {start}..{end})")
    return all_paths, ranges


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract image embeddings into .fbin files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--folders", nargs="+", required=True,
        help="one or more image folders (scanned recursively).",
    )
    p.add_argument(
        "--output-dir", required=True,
        help="root output directory.",
    )
    p.add_argument(
        "--extractors", nargs="+",
        default=["inception_v3", "clip_vit_b32", "resnet50", "pixel"],
        choices=sorted(EXTRACTORS.keys()),
        help="which extractors to run.",
    )
    p.add_argument(
        "--pool", action="store_true",
        help=(
            "pool all input folders into a single embedding set; writes "
            "{output_dir}/{extractor}.fbin (+ paths.txt + manifest.json). "
            "Without this flag, --folders must be a single folder and "
            ".fbin files are written directly into --output-dir."
        ),
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--overwrite", action="store_true",
        help="re-compute and overwrite existing .fbin files.",
    )
    return p.parse_args()


def run_pooled(args: argparse.Namespace) -> int:
    os.makedirs(args.output_dir, exist_ok=True)
    paths, ranges = gather_pooled(args.folders)
    if not paths:
        print("no images found in any folder", file=sys.stderr)
        return 1

    print(f"\npooled: {len(paths)} images  ->  {args.output_dir}")

    # Write paths.txt alongside the .fbin files (row i of each .fbin ↔ line i+1 here)
    paths_file = os.path.join(args.output_dir, "paths.txt")
    with open(paths_file, "w") as f:
        for p in paths:
            f.write(p + "\n")

    results = run_extractors(
        paths=paths,
        extractors=args.extractors,
        out_dir=args.output_dir,
        output_root=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )

    manifest = {
        "mode": "pooled",
        "output_dir": os.path.abspath(args.output_dir),
        "device": args.device,
        "batch_size": args.batch_size,
        "extractors": args.extractors,
        "total_images": len(paths),
        "paths_file": os.path.basename(paths_file),
        "folder_ranges": ranges,
        "embeddings": results,
    }
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\npaths    -> {paths_file}")
    print(f"manifest -> {manifest_path}")
    return 0


def run_per_folder(args: argparse.Namespace) -> int:
    if len(args.folders) != 1:
        print(
            f"per-folder mode expects exactly one --folders entry (got "
            f"{len(args.folders)}); call embed.py once per source folder, or "
            f"use --pool.",
            file=sys.stderr,
        )
        return 2

    os.makedirs(args.output_dir, exist_ok=True)
    folder = args.folders[0]
    tag = os.path.basename(os.path.normpath(folder))
    paths = list_images(folder)
    if not paths:
        print(f"[{tag}] no images found", file=sys.stderr)
        return 1

    print(f"\n[{tag}] {len(paths)} images  ->  {args.output_dir}")
    results = run_extractors(
        paths=paths,
        extractors=args.extractors,
        out_dir=args.output_dir,
        output_root=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
        label=tag,
    )

    manifest = {
        "mode": "per_folder",
        "output_dir": os.path.abspath(args.output_dir),
        "device": args.device,
        "batch_size": args.batch_size,
        "extractors": args.extractors,
        "source_folder": os.path.abspath(folder),
        "tag": tag,
        "count": len(paths),
        "embeddings": results,
    }
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nmanifest -> {manifest_path}")
    return 0


def main() -> int:
    args = parse_args()
    if args.pool:
        return run_pooled(args)
    return run_per_folder(args)


if __name__ == "__main__":
    sys.exit(main())

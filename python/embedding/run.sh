#!/bin/bash
set -euo pipefail

# ─── Single folder (existing) ───────────────────────────────────────────
python embed.py \
    --folders /scratch/sim-real/kitti_real_images \
    --output-dir ../../data \
    --extractors inception_v3 clip_vit_b32 resnet50 lpips_vgg segformer pixel \
    --batch-size 512 --device cuda \
    --overwrite

python embed.py \
    --folders /scratch/sim-real/kitti_sim_images \
    --output-dir ../../data \
    --extractors inception_v3 clip_vit_b32 resnet50 lpips_vgg segformer pixel \
    --batch-size 512 --device cuda

# ─── Pooled: combine all images across multiple folders ─────────────────
# Writes ../../data/{inception_v3,clip_vit_b32,...}.fbin   (single .fbin per extractor),
# plus ../../data/paths.txt and ../../data/manifest.json with per-folder row ranges.
python embed.py --pool \
    --folders /scratch/sim-real/kitti_real_images /scratch/sim-real/kitti_sim_images \
    --output-dir ../../data \
    --extractors inception_v3 clip_vit_b32 resnet50 lpips_vgg segformer pixel \
    --batch-size 512 --device cuda

# ─── Generated image data embedding (Gaussian Blur) ───────────────────────────────────────────
python embed.py \
    --folders /scratch/sim-real/gaussian_blur_level_1 \
    --output-dir /scratch/jsu02/sim-real-embedding/kitti_gaussian_blur/level_1 \
    --extractors inception_v3 clip_vit_b32 resnet50 lpips_vgg segformer pixel \
    --batch-size 512 --device cuda 

python embed.py \
    --folders /scratch/sim-real/gaussian_blur_level_2 \
    --output-dir /scratch/jsu02/sim-real-embedding/kitti_gaussian_blur/level_2 \
    --extractors inception_v3 clip_vit_b32 resnet50 lpips_vgg segformer pixel \
    --batch-size 512 --device cuda 

python embed.py \
    --folders /scratch/sim-real/gaussian_blur_level_3 \
    --output-dir /scratch/jsu02/sim-real-embedding/kitti_gaussian_blur/level_3 \
    --extractors inception_v3 clip_vit_b32 resnet50 lpips_vgg segformer pixel \
    --batch-size 512 --device cuda 

# ─── Generated image data embedding (Gaussian Noise) ───────────────────────────────────────────
python embed.py \
    --folders /scratch/sim-real/gaussian_noise_level_1 \
    --output-dir /scratch/jsu02/sim-real-embedding/kitti_gaussian_noise/level_1 \
    --extractors inception_v3 clip_vit_b32 resnet50 lpips_vgg segformer pixel \
    --batch-size 512 --device cuda 

python embed.py \
    --folders /scratch/sim-real/gaussian_noise_level_2 \
    --output-dir /scratch/jsu02/sim-real-embedding/kitti_gaussian_noise/level_2 \
    --extractors inception_v3 clip_vit_b32 resnet50 lpips_vgg segformer pixel \
    --batch-size 512 --device cuda 

python embed.py \
    --folders /scratch/sim-real/gaussian_noise_level_3 \
    --output-dir /scratch/jsu02/sim-real-embedding/kitti_gaussian_noise/level_3 \
    --extractors inception_v3 clip_vit_b32 resnet50 lpips_vgg segformer pixel \
    --batch-size 512 --device cuda 

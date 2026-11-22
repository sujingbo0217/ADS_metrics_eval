# Suitability of Realism Metrics for ADS Testing

This repository evaluates how well image realism metrics reflect the performance
loss caused by controlled degradations in automated driving system (ADS)
perception tasks. The experiments cover 18 degradation types at five severity levels,
four datasets, seven model–dataset evaluations, and 16 metrics. The analysis also
examines the geometry of the representations used by these metrics.

## Setup

```bash
git clone git@github.com:sujingbo0217/ADS_metrics_eval.git
cd ADS_metrics_eval
conda create -n ads python=3.10 -y
conda activate ads
pip install -r requirements.txt
```

## Third-party repositories (`repos/`)

Clone the third-party code at the versions used by this project:

```bash
mkdir -p repos
cd repos
git clone https://github.com/nianticlabs/monodepth2.git
git -C monodepth2 checkout b017a8164fae339b0ce01f3a9e5048104d5abd0e
git clone https://github.com/ldkong1205/RoboDepth.git
git -C RoboDepth checkout 707c72ab7395e87e13fbc8f7c3c52e1e231d4030
git clone -b v1.4.0 https://github.com/open-mmlab/mmdetection3d.git
```

- `monodepth2` provides the networks loaded by `scripts/eval_depth.py`, the Eigen
  test split, and `export_gt_depth.py` for generating depth ground truth.
- `RoboDepth` provides the `dark`, `color_quant`, and `iso_noise` implementations
  in `corruptions/utils.py`, which `scripts/gen_degradation.py` imports directly.
  The other 15 degradations come from `imagecorruptions`.
- `mmdetection3d` provides `tools/create_data.py` for generating KITTI metadata.

## Checkpoints (`ckpts/`)

The seven model–dataset evaluations use six pretrained checkpoints: the same
Cityscapes-trained DeepLabV3+ checkpoint is evaluated on both Cityscapes and
KITTI-200. Download the OpenMMLab configurations and weights from their model
zoos, and the Monodepth2 weights from its release:

```bash
mkdir -p ckpts
mim download mmdet3d --config smoke_dla34_dlaneck_gn-all_4xb8-6x_kitti-mono3d --dest ckpts/
mim download mmdet3d --config pgd_r101-caffe_fpn_head-gn_4xb3-4x_kitti-mono3d --dest ckpts/
mim download mmsegmentation --config deeplabv3plus_r101-d8_4xb2-80k_cityscapes-512x1024 --dest ckpts/
mim download mmsegmentation --config mask2former_swin-l-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024 --dest ckpts/

cd ckpts
M=https://storage.googleapis.com/niantic-lon-static/research/monodepth2
wget -O mono_640x192.zip "$M/mono_640x192.zip"
wget -O mono+stereo_640x192.zip "$M/mono%2Bstereo_640x192.zip"
unzip mono_640x192.zip -d mono_640x192
unzip mono+stereo_640x192.zip -d mono+stereo_640x192
```

## Dataset preparation

Set `ROOT` and `CITYSCAPES` at the top of `scripts/config.py` to your local data
directories. Use the same paths for the shell variables below; these variables
are used by the preparation commands and do not override `scripts/config.py`.

```bash
export KITTI=/path/to/kitti             # matches ROOT in scripts/config.py
export CITYSCAPES=/path/to/cityscapes   # matches CITYSCAPES in scripts/config.py
```

The pipeline writes degraded images to `degraded/` within the detection,
segmentation, and Cityscapes directories. It stores detection condition roots,
cached representations, and task outputs in `$KITTI/cond_roots/`,
`$KITTI/embeddings/`, and `$KITTI/results/`.

### KITTI object detection

Detection is evaluated on the 3,769 images in the validation split.

```bash
mkdir -p "$KITTI/detection"
cd "$KITTI/detection"
B=https://s3.eu-central-1.amazonaws.com/avg-kitti
wget "$B/data_object_image_2.zip" "$B/data_object_label_2.zip" \
     "$B/data_object_calib.zip" "$B/data_object_velodyne.zip"
unzip data_object_image_2.zip -d images     # images/{training,testing}/image_2
unzip data_object_label_2.zip -d labels     # labels/training/label_2
unzip data_object_calib.zip                 # {training,testing}/calib
unzip data_object_velodyne.zip              # {training,testing}/velodyne
S=https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets
mkdir -p ImageSets
for s in train val trainval test; do
    wget -O "ImageSets/$s.txt" "$S/$s.txt"
done
```

The detectors read a standard KITTI layout under `working/`. Create links to the
extracted files:

```bash
cd "$KITTI/detection"
mkdir -p working/training
ln -s "$PWD/images/training/image_2" working/training/image_2
ln -s "$PWD/labels/training/label_2" working/training/label_2
ln -s "$PWD/training/calib"          working/training/calib
ln -s "$PWD/ImageSets"               working/ImageSets
```

Generate `working/kitti_infos_val.pkl` with mmdetection3d's `tools/create_data.py`.
Follow the full directory layout in
`repos/mmdetection3d/docs/en/advanced_guides/datasets/kitti.md`, including the
testing split and Velodyne scans, but omit `--with-plane`. The Velodyne archive
is needed only for this metadata-generation step. You can assemble the required
layout with links to the extracted files above.

Set `KITTI_LAYOUT` to that directory, run the converter once, and copy the
validation metadata to `working/`:

```bash
KITTI_LAYOUT=/path/to/full/kitti/layout
cd "repos/mmdetection3d"
python tools/create_data.py kitti \
    --root-path "$KITTI_LAYOUT" --out-dir "$KITTI_LAYOUT" --extra-tag kitti
cp "$KITTI_LAYOUT/kitti_infos_val.pkl" "$KITTI/detection/working/"
```

For each evaluation condition, `scripts/eval_det.py` creates a directory under
`$KITTI/cond_roots/` with the same layout as `working/`. Its `image_2` link points
to the images for that condition.

### KITTI semantic segmentation

Segmentation is evaluated on the 200 training images (KITTI-200). The `semantic`
label maps use Cityscapes label IDs.

```bash
mkdir -p "$KITTI/segmentation"
cd "$KITTI/segmentation"
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_semantics.zip
unzip data_semantics.zip                    # training/{image_2,semantic,...}, testing/
```

### KITTI raw and KITTI-C for depth estimation

Depth is evaluated on the 697 test images in the Eigen split. Download the raw
drives (approximately 175 GB) to generate depth ground truth; the rest of the
pipeline uses KITTI-C images instead.

```bash
cd "repos/monodepth2"
wget -i splits/kitti_archives_to_download.txt -P "$KITTI/raw_data/"
(cd "$KITTI/raw_data" && unzip "*.zip")
python export_gt_depth.py --data_path "$KITTI/raw_data" --split eigen   # splits/eigen/gt_depths.npz
```

Use RoboDepth's KITTI-C download for both clean and degraded images
(approximately 12 GB). All images are already resized to 640 × 192, so both
conditions use the same resolution. See `repos/RoboDepth/docs/DATA_PREPARE.md`
for details.

```bash
pip install gdown
mkdir -p "$KITTI/depth"
cd "$KITTI/depth"
gdown 1Ohyh8CN0ZS7gc_9l4cIwX4j97rIRwADa -O kitti_c.zip
unzip kitti_c.zip                           # kitti_c/clean/, kitti_c/<degradation>/<severity>/
```

### Cityscapes semantic segmentation

Segmentation is evaluated on the 500 validation images. A Cityscapes account is
required to download the dataset. Use `cityscapesscripts` to download the
archives and generate the train-ID labels read by `scripts/eval_seg.py`:

```bash
pip install cityscapesscripts
mkdir -p "$CITYSCAPES"
cd "$CITYSCAPES"
csDownload leftImg8bit_trainvaltest.zip gtFine_trainvaltest.zip
unzip -o leftImg8bit_trainvaltest.zip && unzip -o gtFine_trainvaltest.zip
CITYSCAPES_DATASET="$CITYSCAPES" csCreateTrainIdLabelImgs   # gtFine/*/*/*_labelTrainIds.png
```

## Build the CUDA extension

Geometry computations use `knn_ext`, a CUDA extension built for the active
Python version and target GPU architecture. The build requires CUDA Toolkit
12.x and CMake 3.18 or later.

```bash
cmake -S . -B build -DPython3_EXECUTABLE="$(which python)" \
      -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build build -j
```

CMake creates `build/` during configuration; no separate `mkdir` is needed.

The build defaults to `Release` and detects the first GPU's architecture with
`nvidia-smi`. To set the architecture explicitly, add `-DKNN_CUDA_ARCH` to
the configuration command. The compiled module is written to `build/` and
symlinked into `python/` for import by `scripts/geometry_lib.py`. Rebuild when
changing the Python version or target GPU architecture.

## Pipeline

Run the stages below in order. `RESULTS` and `EMB_ROOT`
refer to the paths defined in `scripts/config.py`; analysis files are written to
`results/analysis/` unless otherwise noted.

Importing `scripts/config.py` automatically creates `results/analysis/`,
`results/logs/`, `EMB_ROOT`, `RESULTS/task_perf/`, and `RESULTS/logs/`, including
their parent directories. Degradation, embedding-cache, and detection-condition
directories are created as needed. The RQ scripts create their own analysis
subdirectories.

| Stage | Script or command | Main outputs |
| --- | --- | --- |
| Generate degradations | `python scripts/gen_degradation.py --task [det/seg/cityscapes]`; depth uses KITTI-C | Degraded images |
| Evaluate perception models | `bash scripts/run_tasks.sh` (runs `eval_det.py`, `eval_seg.py`, and `eval_depth.py`) | Task performance in `RESULTS/task_perf/` |
| Compute realism metrics | `python scripts/extract_metrics.py --level [distribution/image]` | `distribution_metrics.csv`, `image_metrics_<dataset>.parquet`; embeddings and handcrafted features under `EMB_ROOT` |
| Compute representation geometry | `python scripts/compute_geometry.py` | `geometry.csv`, `geometry_selection.csv`, `image_geometry_<dataset>.parquet` |
<!-- | Analyze RQ1–RQ4 | `scripts/evaluation/rq1.py` through `rq4.py` | `results/analysis/rq_v2/`, `results/analysis/rq/` |
| Run supplementary experiments | `scripts/test/*.py` | `results/analysis/rq_v2/` and `results/analysis/` |
| Generate figures and tables | `scripts/plot/*.py` | Figures in `paper/cvpr/fig/`; generated table rows in `paper/cvpr/sec/` | -->


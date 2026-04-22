"""
Image embedding extractors.

Each extractor has the same signature:

    extract_xxx(image_paths, device='cuda', batch_size=32) -> np.ndarray (N, D)

and returns a float32 array.  Models are loaded lazily on first call and
cached in module-level globals so repeated calls (e.g. over many folders)
don't pay the model-load cost again.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ─── cache ───────────────────────────────────────────────────────────────
_MODEL_CACHE: dict = {}


# ─── dataset ─────────────────────────────────────────────────────────────
class _ImageDataset(Dataset):
    def __init__(self, paths: List[str], transform):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        return self.transform(img)


def _make_loader(paths, transform, batch_size, num_workers=4):
    ds = _ImageDataset(paths, transform)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


@torch.inference_mode()
def _run_feature_model(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    out_dim: int,
    postprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    desc: str = "forward",
    log_every: int = 20,
) -> np.ndarray:
    model.eval().to(device)
    n = len(loader.dataset)
    feats = np.empty((n, out_dim), dtype=np.float32)
    cursor = 0
    n_batches = len(loader)
    t0 = time.time()
    for bi, batch in enumerate(loader):
        batch = batch.to(device, non_blocking=True)
        out = model(batch)
        if postprocess is not None:
            out = postprocess(out)
        out = out.float().cpu().numpy()
        feats[cursor : cursor + out.shape[0]] = out
        cursor += out.shape[0]
        if (bi + 1) % log_every == 0 or (bi + 1) == n_batches:
            done = cursor
            rate = done / max(time.time() - t0, 1e-6)
            eta = (n - done) / max(rate, 1e-6)
            print(
                f"    [{desc}] {done}/{n}  ({rate:.1f} img/s, ETA {eta:.0f}s)",
                file=sys.stderr, flush=True,
            )
    return feats


# ─── Inception v3 (2048) ─────────────────────────────────────────────────
def extract_inception(image_paths, device="cuda", batch_size=32) -> np.ndarray:
    key = "inception_v3"
    if key not in _MODEL_CACHE:
        from torchvision.models import Inception_V3_Weights, inception_v3

        weights = Inception_V3_Weights.IMAGENET1K_V1
        net = inception_v3(weights=weights, aux_logits=True)
        net.fc = nn.Identity()
        _MODEL_CACHE[key] = (net, weights.transforms())
    net, tfm = _MODEL_CACHE[key]
    loader = _make_loader(image_paths, tfm, batch_size)
    return _run_feature_model(
        net, loader, device, out_dim=2048, desc="inception_v3"
    )


# ─── ResNet-50 (2048) ────────────────────────────────────────────────────
def extract_resnet(image_paths, device="cuda", batch_size=32) -> np.ndarray:
    key = "resnet50"
    if key not in _MODEL_CACHE:
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.IMAGENET1K_V2
        net = resnet50(weights=weights)
        net.fc = nn.Identity()
        _MODEL_CACHE[key] = (net, weights.transforms())
    net, tfm = _MODEL_CACHE[key]
    loader = _make_loader(image_paths, tfm, batch_size)
    return _run_feature_model(
        net, loader, device, out_dim=2048, desc="resnet50"
    )


# ─── CLIP ViT-B/32 (512) ─────────────────────────────────────────────────
def extract_clip(image_paths, device="cuda", batch_size=32) -> np.ndarray:
    key = "clip_vit_b32"
    if key not in _MODEL_CACHE:
        try:
            import open_clip
        except ImportError as e:
            raise ImportError(
                "extract_clip needs `open_clip_torch`: pip install open_clip_torch"
            ) from e
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        model.visual.output_tokens = False
        _MODEL_CACHE[key] = (model, preprocess)
    model, preprocess = _MODEL_CACHE[key]
    loader = _make_loader(image_paths, preprocess, batch_size)

    def _encode(x):
        return model.encode_image(x)

    class _Wrap(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m.encode_image(x)

    return _run_feature_model(
        _Wrap(model), loader, device, out_dim=512, desc="clip_vit_b32"
    )


# ─── LPIPS-VGG features (→ PCA to 1024) ──────────────────────────────────
def extract_lpips_embedding(
    image_paths, device="cuda", batch_size=32, pca_dim: int = 1024
) -> np.ndarray:
    """
    Concatenate VGG16 intermediate activations (same 5 layers used by LPIPS),
    global-average-pool each, then PCA down to ``pca_dim``.
    """
    key = "lpips_vgg"
    if key not in _MODEL_CACHE:
        from torchvision.models import VGG16_Weights, vgg16

        weights = VGG16_Weights.IMAGENET1K_V1
        vgg = vgg16(weights=weights).features
        # LPIPS uses outputs of conv layers before pooling at these indices.
        slice_ends = [4, 9, 16, 23, 30]  # same slices as lpips.LPIPS(net='vgg')
        tfm = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        _MODEL_CACHE[key] = (vgg, slice_ends, tfm)
    vgg, slice_ends, tfm = _MODEL_CACHE[key]

    class _VGGFeat(nn.Module):
        def __init__(self, backbone, ends):
            super().__init__()
            self.backbone = backbone
            self.ends = ends

        def forward(self, x):
            feats = []
            prev = 0
            h = x
            for end in self.ends:
                for i in range(prev, end):
                    h = self.backbone[i](h)
                feats.append(h.mean(dim=[2, 3]))  # GAP
                prev = end
            return torch.cat(feats, dim=1)  # (B, 64+128+256+512+512 = 1472)

    model = _VGGFeat(vgg, slice_ends)
    raw_dim = 64 + 128 + 256 + 512 + 512
    loader = _make_loader(image_paths, tfm, batch_size)
    raw = _run_feature_model(
        model, loader, device, out_dim=raw_dim, desc="lpips_vgg/forward"
    )

    target = min(pca_dim, raw.shape[0], raw.shape[1])
    if target >= raw.shape[1]:
        print(
            f"    [lpips_vgg/pca] skipped (target={target} >= raw_dim={raw.shape[1]})",
            file=sys.stderr,
        )
        return raw

    # PCA — do it on GPU with torch.pca_lowrank (randomized SVD) when possible.
    # sklearn's CPU PCA on (N, 1472) with n_components=1024 is single-threaded and
    # has no progress output; on large N it feels like a hang (CPU saturated,
    # GPU idle).  GPU pca_lowrank is ~10-50x faster and we can show progress.
    print(
        f"    [lpips_vgg/pca] {raw.shape} -> ({raw.shape[0]}, {target}) ...",
        file=sys.stderr, flush=True,
    )
    t0 = time.time()
    try:
        dev = device if (device != "cpu" and torch.cuda.is_available()) else "cpu"
        X = torch.from_numpy(raw).to(dev)
        mean = X.mean(dim=0, keepdim=True)
        X = X - mean
        # niter=4 is plenty for convergence when target is well below raw_dim
        _, _, V = torch.pca_lowrank(X, q=target, niter=4)
        reduced = (X @ V).cpu().numpy().astype(np.float32, copy=False)
        print(
            f"    [lpips_vgg/pca] done in {time.time()-t0:.1f}s on {dev} "
            f"(torch.pca_lowrank)",
            file=sys.stderr,
        )
        return reduced
    except Exception as e:
        print(
            f"    [lpips_vgg/pca] torch.pca_lowrank failed ({e}); "
            f"falling back to sklearn PCA (CPU, slow for large N)",
            file=sys.stderr,
        )
        from sklearn.decomposition import PCA

        pca = PCA(n_components=target, random_state=0, svd_solver="randomized")
        reduced = pca.fit_transform(raw).astype(np.float32)
        print(
            f"    [lpips_vgg/pca] done in {time.time()-t0:.1f}s (sklearn)",
            file=sys.stderr,
        )
        return reduced


# ─── SegFormer encoder features ──────────────────────────────────────────
def extract_segformer(image_paths, device="cuda", batch_size=16) -> np.ndarray:
    """
    Use the last-stage encoder output of SegFormer-b0 (dim 256), global-average
    pooled across spatial dims.
    """
    key = "segformer"
    if key not in _MODEL_CACHE:
        try:
            from transformers import (
                SegformerImageProcessor,
                SegformerModel,
            )
        except ImportError as e:
            raise ImportError(
                "extract_segformer needs `transformers`: pip install transformers"
            ) from e
        name = "nvidia/mit-b0"
        processor = SegformerImageProcessor.from_pretrained(name)
        model = SegformerModel.from_pretrained(name)
        _MODEL_CACHE[key] = (model, processor)
    model, processor = _MODEL_CACHE[key]

    def tfm(img):
        enc = processor(images=img, return_tensors="pt")
        return enc["pixel_values"][0]

    loader = _make_loader(image_paths, tfm, batch_size)

    class _Wrap(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            out = self.m(pixel_values=x)
            last = out.last_hidden_state  # (B, C, H, W)
            return last.mean(dim=[2, 3])  # (B, C)

    # Probe output dim on a single forward
    with torch.inference_mode():
        model.to(device).eval()
        sample = next(iter(loader))[:1].to(device)
        probe = _Wrap(model)(sample)
        out_dim = probe.shape[1]

    return _run_feature_model(
        _Wrap(model), loader, device, out_dim=out_dim, desc="segformer"
    )


# ─── Raw pixels ──────────────────────────────────────────────────────────
def extract_pixel(
    image_paths, device="cuda", batch_size=64, size: int = 32
) -> np.ndarray:
    """
    Resize to ``size x size`` RGB and flatten.  Deterministic, model-free
    baseline.  Output dim = 3 * size * size.
    """
    tfm = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ]
    )
    loader = _make_loader(image_paths, tfm, batch_size, num_workers=4)
    out_dim = 3 * size * size
    feats = np.empty((len(loader.dataset), out_dim), dtype=np.float32)
    cursor = 0
    for batch in loader:
        flat = batch.reshape(batch.shape[0], -1).numpy().astype(np.float32)
        feats[cursor : cursor + flat.shape[0]] = flat
        cursor += flat.shape[0]
    return feats

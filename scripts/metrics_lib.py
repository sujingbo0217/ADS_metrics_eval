"""Unify the metric representation as f = M(img1, img2, space)

Embedding backbones also expose
    .pooled(x)  -> one vector per image (net(x) returns the same)
and LpipsBackbone additionally
    .spatial(x) -> the multi-layer feature maps LPIPS consumes
"""
from __future__ import annotations

import functools
import numpy as np
import torch
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ===========================================================================
# phi - embedding spaces
# ===========================================================================
class _Net:
    def __call__(self, x01: torch.Tensor) -> torch.Tensor:
        return self.pooled(x01)


class LpipsBackbone(_Net):
    """The LPIPS library's backbones: net = "vgg" (torchvision VGG16), "alex"
    or "squeeze". All three reproduce lpips.LPIPS exactly; the pipeline uses
    only "vgg".
    """
    def __init__(self, net: str = "vgg"):
        import lpips
        self._m = lpips.LPIPS(net=net, verbose=False).to(DEV).eval()
        self.chns = self._m.chns
        self.pooled_dim = self.chns[-1]

    @torch.no_grad()
    def spatial(self, x01: torch.Tensor) -> list[torch.Tensor]:
        """x01: [N, 3, H, W] in [0, 1]."""
        x = x01.to(DEV) * 2 - 1
        return self._m.net.forward(self._m.scaling_layer(x))

    @torch.no_grad()
    def pooled(self, x: torch.Tensor) -> torch.Tensor:
        """Global-average-pool the deepest map (relu5_3 for VGG) -> [N, chns[-1]]."""
        return self.spatial(x)[-1].mean(dim=(2, 3))

    @property
    def lins(self):
        """The learned per-channel linear weights (1472 params for VGG)."""
        return self._m.lins


@functools.lru_cache(maxsize=None)
def _keras_vgg16_convs(path: str) -> torch.nn.ModuleList:
    import os
    import h5py
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.hub.download_url_to_file(KerasVGG16.URL, path)
    convs, cin = [], 3
    with h5py.File(path, "r") as f:
        for b, (n, w) in enumerate(zip(KerasVGG16.BLOCKS, KerasVGG16.WIDTHS), 1):
            for c in range(1, n + 1):
                g = f[f"block{b}_conv{c}"]
                conv = torch.nn.Conv2d(cin, w, 3, padding=1)
                conv.weight.data = torch.from_numpy(
                    g[f"block{b}_conv{c}_W_1:0"][()].transpose(3, 2, 0, 1).copy())
                conv.bias.data = torch.from_numpy(g[f"block{b}_conv{c}_b_1:0"][()])
                convs.append(conv)
                cin = w
    return torch.nn.ModuleList(convs).to(DEV).eval()


def imread_bgr(x01: torch.Tensor) -> torch.Tensor:
    return (x01.to(DEV) * 255).round()[:, [2, 1, 0]]


def vgg16_preprocess_input(x_bgr: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(KerasVGG16.CAFFE_MEAN, device=x_bgr.device).view(1, 3, 1, 1)
    return x_bgr[:, [2, 1, 0]] - mean


class KerasVGG16:
    """keras.applications VGG16(weights='imagenet', include_top=False), ported
    to PyTorch: the original Caffe weights Keras ships, not torchvision's, so a
    different network from the VGG16 inside LPIPS.

    output      'block5_conv3' or 'block5_pool'
    preprocess  True applies vgg16_preprocess_input;
                False feeds the raw 0-255 BGR array
    """

    URL = ("https://storage.googleapis.com/tensorflow/keras-applications/vgg16/"
           "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
    WEIGHTS = "~/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    BLOCKS = (2, 2, 3, 3, 3)
    WIDTHS = (64, 128, 256, 512, 512)
    CAFFE_MEAN = (103.939, 116.779, 123.68)

    def __init__(self, output="block5_conv3", preprocess=True, weights=None):
        import os
        assert output in ("block5_conv3", "block5_pool"), output
        self.output, self.preprocess = output, preprocess
        self._convs = _keras_vgg16_convs(os.path.expanduser(weights or self.WEIGHTS))

    @torch.no_grad()
    def __call__(self, x01: torch.Tensor) -> torch.Tensor:
        x = imread_bgr(x01)
        if self.preprocess:
            x = vgg16_preprocess_input(x)
        # TF32 off: Keras computes these maps in full float32.
        with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
            k = 0
            for b, n in enumerate(self.BLOCKS):
                for _ in range(n):
                    x = F.relu(self._convs[k](x))
                    k += 1
                if b < 4 or self.output == "block5_pool":
                    x = F.max_pool2d(x, 2)           # Keras 'valid': floor
        return x


class InceptionBackbone(_Net):
    """Inception-v3 pool3 (2048-d) using FID weights.
    """
    def __init__(self):
        from pytorch_fid.inception import InceptionV3
        self.pooled_dim = 2048
        self._m = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]],
                              resize_input=True, normalize_input=True).to(DEV).eval()

    @torch.no_grad()
    def pooled(self, x01):
        return self._m(x01.to(DEV))[0].flatten(1)


class ClipBackbone(_Net):
    MEAN = (0.48145466, 0.4578275, 0.40821073)
    STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, arch="ViT-B-32-quickgelu", pretrained="openai"):
        # "-quickgelu" is REQUIRED with the openai weights: those models were
        # trained with QuickGELU, while open_clip's plain ViT-B-32 config uses
        # standard GELU. Pairing them silently produces slightly wrong features.
        import open_clip
        self.pooled_dim = 512
        self._m = open_clip.create_model(arch, pretrained=pretrained).to(DEV).eval()
        self._mean = torch.tensor(self.MEAN, device=DEV).view(1, 3, 1, 1)
        self._std = torch.tensor(self.STD, device=DEV).view(1, 3, 1, 1)

    @torch.no_grad()
    def pooled(self, x01):
        x = F.interpolate(x01.to(DEV), size=(224, 224), mode="bicubic",
                          align_corners=False).clamp(0, 1)
        return self._m.encode_image((x - self._mean) / self._std).float()


class DinoV2Backbone(_Net):
    """DINOv2 ViT-L/14, configured as in Stein et al. (NeurIPS 2023).

    WHY IT IS HERE. the paper reach the same conclusion from human judgement:
    they name over-reliance on Inception-v3 as a cause of metric failure and 
    report that DINOv2-ViT-L/14 substantially improves agreement. 
    We need self-supervised member, so it could tell whether "this embedding
    is worse".
    """

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    DIMS = {"vits14": 384, "vitb14": 768, "vitl14": 1024, "vitg14": 1536}

    def __init__(self, arch="vitl14"):
        self.pooled_dim = self.DIMS[arch]
        torch.backends.cudnn.allow_tf32 = False
        self._m = torch.hub.load("facebookresearch/dinov2",
                                 f"dinov2_{arch}").to(DEV).eval()
        self._mean = torch.tensor(self.MEAN, device=DEV).view(1, 3, 1, 1)
        self._std = torch.tensor(self.STD, device=DEV).view(1, 3, 1, 1)

    @torch.no_grad()
    def pooled(self, x01):
        x = F.interpolate(x01.to(DEV), size=(224, 224), mode="bicubic",
                          align_corners=False, antialias=True).clamp(0, 1)
        return self._m((x - self._mean) / self._std).float()


class PixelBackbone(_Net):
    def __init__(self, size=(64, 212)):
        self.size = size
        self.pooled_dim = 3 * size[0] * size[1]

    @torch.no_grad()
    def pooled(self, x):
        y = F.interpolate(x.to(DEV), size=self.size, mode="bilinear",
                          align_corners=False)
        return y.flatten(1)


# ===========================================================================
# image-level f: metric(img1, img2, net)
# ===========================================================================
def cosine(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(img1, img2, dim=1)


@torch.no_grad()
def lpips(img1: torch.Tensor, img2: torch.Tensor, net: LpipsBackbone) -> torch.Tensor:
    def _normalize_tensor(x, eps=1e-10):
        return x / (torch.sqrt((x ** 2).sum(dim=1, keepdim=True)) + eps)
    fa, fb = net.spatial(img1), net.spatial(img2)
    total = 0.0
    for l, (a, b) in enumerate(zip(fa, fb)):
        d = (_normalize_tensor(a) - _normalize_tensor(b)) ** 2
        total = total + net.lins[l](d).mean(dim=(2, 3))
    return total.flatten()


# The three VGG16 metrics of Lambertenghi & Stocco (ICST 2024), written as
# their replication code writes them (github.com/ast-fortiss-tum/
# I2I-quality-metrics-study, image_metrics_utils.py). The net is what fixes
# the representation; in their code and in extract_metrics:
#   cossim, cpl : KerasVGG16("block5_conv3", preprocess=True)
#   sss         : KerasVGG16("block5_pool", preprocess=False)
@torch.no_grad()
def cossim(img1: torch.Tensor, img2: torch.Tensor, net) -> torch.Tensor:
    """calculate_cosine_similarity: cosine of the two flattened feature maps.
    Any net works: a VGG16 map gives Cos@VGG, Inception pool3 gives Cos@Inc."""
    return cosine(net(img1).flatten(1), net(img2).flatten(1))


@torch.no_grad()
def cpl(img1: torch.Tensor, img2: torch.Tensor, net) -> torch.Tensor:
    """calculate_perceptual_distances: keras MeanSquaredError of the two
    feature maps, i.e. their mean squared difference."""
    return mse(net(img1), net(img2))


@torch.no_grad()
def sss(img1: torch.Tensor, img2: torch.Tensor, net) -> torch.Tensor:
    """calculate_semantic_segmentation_score: np.mean((s1 - s2) ** 2) of the
    two network outputs. The same functional as cpl; their code differs only
    in the net (the whole notop VGG16, fed unpreprocessed images)."""
    return mse(net(img1), net(img2))


def mse(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    return ((img1 - img2) ** 2).flatten(1).mean(dim=1)


# ---------------------------------------------------------------------------
# pixel metrics - SSIM, TSI, WD, KL, HistI
# ---------------------------------------------------------------------------
def ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    return float(structural_similarity(img1, img2, multichannel=True, channel_axis=2,))


def tsi(img1: np.ndarray, img2: np.ndarray, distances=(1, 2, 3)) -> float:
    import cv2
    from skimage.feature import graycomatrix, graycoprops
    gray_img1 = graycomatrix(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), distances=list(distances), angles=[0], symmetric=True, normed=True,)
    gray_img2 = graycomatrix(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), distances=list(distances), angles=[0], symmetric=True, normed=True,)
    GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")
    s = 0.0
    for d in range(len(distances)):
        for p in GLCM_PROPS:
            s += np.abs(graycoprops(gray_img1, p)[d, 0] - graycoprops(gray_img2, p)[d, 0])  # fixed from 0 to d
    return float(s)


def _density256(image: np.ndarray) -> np.ndarray:
    return np.histogram(image.flatten(), bins=256, range=(0, 256), density=True,)[0]


def wd(img1: np.ndarray, img2: np.ndarray) -> float:
    from scipy.stats import wasserstein_distance
    hist1 = _density256(img1)
    hist2 = _density256(img2)
    return float(wasserstein_distance(hist1, hist2))


def kl(img1: np.ndarray, img2: np.ndarray, eps=1e-10) -> float:
    from scipy.stats import entropy
    hist1, hist2 = _density256(img1), _density256(img2)
    hist1 = hist1 / np.sum(hist1) + eps
    hist2 = hist2 / np.sum(hist2) + eps
    return float(entropy(hist1, hist2))


def histi(img1: np.ndarray, img2: np.ndarray) -> float:
    import cv2
    ch = img1.shape[2]
    assert img1.shape[2] == img2.shape[2]
    hist1 = cv2.calcHist([img1], list(range(ch)), None, [256] * ch, [0, 256] * ch)
    hist2 = cv2.calcHist([img2], list(range(ch)), None, [256] * ch, [0, 256] * ch)
    return float(np.sum(np.minimum(hist1, hist2)) / np.sum(hist1))


# ===========================================================================
# distribution f: metric(imgs1, imgs2, net)
# ===========================================================================
def frechet(images1: np.ndarray, images2: np.ndarray, eps=1e-6) -> float:
    from scipy import linalg
    mu1 = images1.mean(0)
    mu2 = images2.mean(0)
    cov1 = np.cov(images1, rowvar=False) + eps * np.eye(images1.shape[1])
    cov2 = np.cov(images2, rowvar=False) + eps * np.eye(images2.shape[1])
    covmean, _ = linalg.sqrtm(cov1.dot(cov2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(cov1 + cov2 - 2 * covmean))


def kid(images1: np.ndarray, images2: np.ndarray, n_subsets=100, subset_size=1000,
        seed=0) -> tuple[float, float]:
    def mmd2_poly(imgs1: np.ndarray, imgs2: np.ndarray, degree=3, gamma=None, coef0=1.0):
        d = imgs1.shape[1]
        g = 1.0 / d if gamma is None else gamma
        k = lambda x, y: (g * x.dot(y.T) + coef0) ** degree
        n = len(imgs1)
        m = len(imgs2)
        kxx, kyy, kxy = k(imgs1, imgs1), k(imgs2, imgs2), k(imgs1, imgs2)
        np.fill_diagonal(kxx, 0)
        np.fill_diagonal(kyy, 0)
        return float(kxx.sum() / (n * (n - 1)) + kyy.sum() / (m * (m - 1)) - 2 * kxy.mean())
    
    rng = np.random.default_rng(seed)
    s = min(subset_size, len(images1), len(images2))
    vals = [mmd2_poly(images1[rng.choice(len(images1), s, replace=False)],
                      images2[rng.choice(len(images2), s, replace=False)])
            for _ in range(n_subsets)]
    return float(np.mean(vals)), float(np.std(vals))


# ---------------------------------------------------------------------------
# metrics that read images through a net: fn(img1, img2, net)
NET_METRICS = {"lpips": lpips, "cossim": cossim, "cpl": cpl, "sss": sss}
# pixel metrics computed on the GPU from float [0, 1] images
GPU_PIXEL = {"mse": mse}
# pixel metrics computed on the CPU from uint8 BGR images, as L&S compute them
CPU_PIXEL = {"ssim": ssim, "tsi": tsi, "wd": wd, "kl": kl, "histi": histi}


def to_u8(t: torch.Tensor) -> np.ndarray:
    return (t * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()


def cpu_pixel(name: str, a: np.ndarray, b: np.ndarray) -> float:
    a, b = (np.ascontiguousarray(x[..., ::-1]) for x in (a, b))
    return CPU_PIXEL[name](a, b)


def metric(name: str, img1: torch.Tensor, img2: torch.Tensor, net=None):
    """Evaluate a PAIRED metric on a batch of image pairs.

    name       : key in NET_METRICS, GPU_PIXEL or CPU_PIXEL
    img1/2     : float tensors [N, 3, H, W] in [0, 1], reference first
    net        : the representation, for NET_METRICS only

    Returns a length-N numpy array of per-image values.
    """
    if name in NET_METRICS:
        return NET_METRICS[name](img1, img2, net).cpu().numpy()
    if name in GPU_PIXEL:
        return GPU_PIXEL[name](img1.to(DEV), img2.to(DEV)).cpu().numpy()
    if name in CPU_PIXEL:
        return np.array([cpu_pixel(name, to_u8(img1[i]), to_u8(img2[i]))
                         for i in range(len(img1))])
    raise ValueError(name)

"""
Hand-crafted features for Planet Amazon satellite chips.

extract_features(img) returns 25 numbers that describe an image:
  - 18 global features: colour, vegetation index, texture, colour ratios, entropy
  - 7 local features: edges and the most extreme 8x8 patches, which help with
    thin or small objects (roads, rivers) that disappear in whole-image averages
"""

import numpy as np
from PIL import Image

GLOBAL_FEATURES = [
    'R_mean', 'R_std', 'G_mean', 'G_std', 'B_mean', 'B_std',
    'H_mean', 'H_std', 'S_mean', 'S_std', 'V_mean', 'V_std',
    'ExG',
    'texture_mean', 'texture_std',
    'ratio_RG', 'ratio_GB',
    'entropy',
]
LOCAL_FEATURES = [
    'edge_mean', 'edge_std', 'edge_frac',
    'patch_V_min', 'patch_V_max', 'patch_ExG_min', 'patch_ExG_max',
]
FEATURE_NAMES = GLOBAL_FEATURES + LOCAL_FEATURES

SIZE = 64          # images are resized to 64x64 before computing features
PATCH = SIZE // 8  # 8x8 grid of patches
EDGE_THRESHOLD = 0.1


def _patch_means(channel: np.ndarray) -> np.ndarray:
    """Mean of each 8x8 patch, returned as an (8, 8) array."""
    return channel.reshape(SIZE // PATCH, PATCH, SIZE // PATCH, PATCH).mean(axis=(1, 3))


def extract_features(img) -> np.ndarray:
    """img: path to an image file, or an already opened PIL image."""
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    img = img.convert('RGB').resize((SIZE, SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Colour: RGB
    feat = [R.mean(), R.std(), G.mean(), G.std(), B.mean(), B.std()]

    # Colour: HSV (more robust to lighting changes)
    Cmax = arr.max(axis=2)
    Cmin = arr.min(axis=2)
    delta = Cmax - Cmin + 1e-8
    V = Cmax
    with np.errstate(divide='ignore', invalid='ignore'):
        S = np.where(Cmax > 0, delta / Cmax, 0)
    H = np.zeros_like(R)
    mask_r = (Cmax == R)
    mask_g = (Cmax == G)
    mask_b = (Cmax == B)
    H[mask_r] = ((G[mask_r] - B[mask_r]) / delta[mask_r]) % 6
    H[mask_g] = (B[mask_g] - R[mask_g]) / delta[mask_g] + 2
    H[mask_b] = (R[mask_b] - G[mask_b]) / delta[mask_b] + 4
    H = H / 6.0
    feat += [H.mean(), H.std(), S.mean(), S.std(), V.mean(), V.std()]

    # Vegetation: Excess Green, positive where vegetation is dense
    exg = 2 * G - R - B
    feat.append(exg.mean())

    # Texture: spread of brightness inside each patch
    gray = 0.299 * R + 0.587 * G + 0.114 * B
    stds = [
        gray[i:i + PATCH, j:j + PATCH].std()
        for i in range(0, SIZE, PATCH)
        for j in range(0, SIZE, PATCH)
    ]
    feat += [np.mean(stds), np.std(stds)]

    # Colour ratios: green canopy vs blue/brown water
    feat.append(R.mean() / (G.mean() + 1e-8))
    feat.append(G.mean() / (B.mean() + 1e-8))

    # Entropy of the brightness histogram: uniform vs chaotic image
    hist, _ = np.histogram(gray, bins=32, range=(0, 1))
    hist = hist / hist.sum() + 1e-8
    feat.append(-np.sum(hist * np.log2(hist)))

    # Edges (Sobel): roads and river banks are sharp lines
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[1:-1, 1:-1] = (gray[:-2, 2:] + 2 * gray[1:-1, 2:] + gray[2:, 2:]
                      - gray[:-2, :-2] - 2 * gray[1:-1, :-2] - gray[2:, :-2])
    gy[1:-1, 1:-1] = (gray[2:, :-2] + 2 * gray[2:, 1:-1] + gray[2:, 2:]
                      - gray[:-2, :-2] - 2 * gray[:-2, 1:-1] - gray[:-2, 2:])
    edges = np.hypot(gx, gy)[1:-1, 1:-1]
    feat += [edges.mean(), edges.std(), (edges > EDGE_THRESHOLD).mean()]

    # Extreme patches: a single dark river patch or bare-soil patch is
    # visible here even when the rest of the image is forest
    v_patches = _patch_means(V)
    exg_patches = _patch_means(exg)
    feat += [v_patches.min(), v_patches.max(), exg_patches.min(), exg_patches.max()]

    return np.array(feat, dtype=np.float32)

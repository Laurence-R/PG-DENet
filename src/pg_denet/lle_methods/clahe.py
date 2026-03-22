"""CLAHE (Contrast Limited Adaptive Histogram Equalization) for HDR linear data.

GPU-accelerated: luminance extraction and ratio scaling run on CUDA; the CLAHE
histogram equalization itself uses OpenCV on CPU (no CUDA CLAHE in pip opencv).
"""

import cv2
import numpy as np
import torch

from pg_denet.gpu import device, luminance_gpu, to_gpu


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """對 float32 線性空間 BGR 影像的亮度通道套用 CLAHE。

    Args:
        image: 輸入 float32 BGR 影像（線性空間，值可 > 1.0）。
        clip_limit: 對比度限制閾值。
        tile_size:  網格大小。

    Returns:
        增強後的 float32 BGR 影像（線性空間）。
    """
    img_t = to_gpu(image)  # (H, W, 3)

    # ── GPU: luminance ────────────────────────────────────────────────────
    L_t = torch.clamp(luminance_gpu(img_t), min=1e-7)
    L_max = L_t.max()

    # ── CPU: cv2 CLAHE (uint16) ───────────────────────────────────────────
    L_u16 = torch.clamp(L_t / L_max * 65535.0, 0, 65535).to(torch.int32).cpu().numpy().astype(np.uint16)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    L_enhanced_u16 = clahe.apply(L_u16)

    # ── GPU: ratio scaling ────────────────────────────────────────────────
    L_enh_t = torch.from_numpy(L_enhanced_u16.astype(np.float32)).to(device) / 65535.0 * L_max
    ratio = (L_enh_t / L_t).unsqueeze(2)  # (H, W, 1)
    result = img_t * ratio
    return result.cpu().numpy().astype(np.float32)


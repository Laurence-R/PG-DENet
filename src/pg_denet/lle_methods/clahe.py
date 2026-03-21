"""CLAHE (Contrast Limited Adaptive Histogram Equalization) for HDR linear data."""

import cv2
import numpy as np


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """對 float32 線性空間 BGR 影像的亮度通道套用 CLAHE。

    使用 uint16 中間表示（保留 16-bit 精度），在亮度通道上做 CLAHE，
    再還原回原始 HDR 範圍的 float32。

    Args:
        image: 輸入 float32 BGR 影像（線性空間，值可 > 1.0）。
        clip_limit: 對比度限制閾值。
        tile_size:  網格大小。

    Returns:
        增強後的 float32 BGR 影像（線性空間）。
    """
    # 分離亮度與色度：luminance = Rec.709
    L = 0.0722 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.2126 * image[:, :, 2]
    L = np.maximum(L, 1e-7)

    # 將亮度映射至 uint16 範圍以供 CLAHE 使用
    L_max = L.max()
    L_u16 = np.clip(L / L_max * 65535.0, 0, 65535).astype(np.uint16)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    L_enhanced_u16 = clahe.apply(L_u16)

    # 還原回 float32 HDR 範圍
    L_enhanced = L_enhanced_u16.astype(np.float32) / 65535.0 * L_max

    # 以亮度比例調整各通道，保留色度
    ratio = (L_enhanced / L)[:, :, np.newaxis]
    return (image * ratio).astype(np.float32)


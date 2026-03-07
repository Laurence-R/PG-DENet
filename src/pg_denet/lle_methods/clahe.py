"""影像前處理工具（CLAHE、去雜訊等）。"""

import cv2
import numpy as np


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: tuple[int, int] = (8, 8)
) -> np.ndarray:
    """對 BGR 圖片的 L channel 套用 CLAHE 對比度強化。

    Args:
        image: 輸入 BGR 圖片（uint8）。
        clip_limit: 對比度限制閾值。
        tile_size:  網格大小。

    Returns:
        強化後的 BGR 圖片（uint8）。
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    l_enhanced = clahe.apply(l)

    merged = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


"""AGCWD: Adaptive Gamma Correction with Weighting Distribution.

Reference:
    Huang, S.-C., Cheng, F.-C., & Chiu, Y.-S. (2013).
    Efficient Contrast Enhancement Using Adaptive Gamma Correction
    With Weighting Distribution.
    IEEE Transactions on Image Processing, 22(3), 1032-1041.

Algorithm overview:
    1. 從 HSV 的 V channel 取出亮度圖
    2. 計算亮度直方圖的 PDF
    3. 對 PDF 做冪次加權，得到 weighted PDF，再取 CDF
    4. 以 CDF 決定每個強度等級的 Gamma 值：
           T(l) = 255 · (l/255) ^ (1 - CDF(l))
    5. 利用 lookup table 快速映射，並寫回 V channel
"""

import cv2
import numpy as np


def _extract_value_channel(image: np.ndarray) -> np.ndarray:
    """BGR → HSV，回傳 V channel (uint8)。"""
    hsv = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_BGR2HSV)
    return (hsv[:, :, 2] * 255).astype(np.uint8)


def _set_value_channel(image: np.ndarray, v_channel: np.ndarray) -> np.ndarray:
    """將 V channel 寫回 BGR 圖片。"""
    hsv = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = v_channel.astype(np.float32) / 255.0
    return (cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) * 255).astype(np.uint8)


def _build_lut(v_channel: np.ndarray, w: float) -> np.ndarray:
    """建立 256 entry 的 Gamma LUT (論文 Eq. 6-7)。"""
    hist = cv2.calcHist([v_channel], [0], None, [256], [0, 256]).flatten()
    pdf = hist / hist.sum()

    pdf_min, pdf_max = pdf.min(), pdf.max()
    w_pdf = pdf_max * ((pdf - pdf_min) / (pdf_max - pdf_min + 1e-7)) ** w
    cdf = np.cumsum(w_pdf) / (w_pdf.sum() + 1e-7)       # 正規化 CDF

    levels = np.arange(256, dtype=np.float32)
    lut = 255.0 * np.power(levels / 255.0, 1.0 - cdf)   # 向量化，取代雙層 for-loop
    return np.clip(lut, 0, 255).astype(np.uint8)


def apply_agcwd(image: np.ndarray, w: float = 0.8) -> np.ndarray:
    """以 AGCWD 演算法增強低光影像。

    Args:
        image: 輸入 BGR 圖片(uint8)，可為彩色或灰階。
        w:     PDF 加權指數，控制 Gamma 曲線的彎曲程度。
               w → 0 趨近線性，w → 1 接近傳統 AGCWD。

    Returns:
        增強後的 BGR 圖片(uint8)。
    """
    is_color = image.ndim == 3
    v = _extract_value_channel(image) if is_color else image

    lut = _build_lut(v, w)
    enhanced_v = lut[v]                                  # O(1) lookup table 映射

    return _set_value_channel(image, enhanced_v) if is_color else enhanced_v

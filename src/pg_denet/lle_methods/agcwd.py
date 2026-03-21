"""AGCWD: Adaptive Gamma Correction with Weighting Distribution.

Reference:
    Huang, S.-C., Cheng, F.-C., & Chiu, Y.-S. (2013).
    Efficient Contrast Enhancement Using Adaptive Gamma Correction
    With Weighting Distribution.
    IEEE Transactions on Image Processing, 22(3), 1032-1041.

Algorithm overview:
    1. 計算亮度 (Rec.709 luminance)
    2. 量化為 N-bin 直方圖，計算 PDF
    3. 對 PDF 做冪次加權，得到 weighted PDF，再取 CDF
    4. 以 CDF 決定每個 bin 的 Gamma 值
    5. 內插回連續亮度值，以亮度比例調整各通道
"""

import cv2
import numpy as np

N_BINS = 4096  # 高精度直方圖 bin 數量


def apply_agcwd(image: np.ndarray, w: float = 0.8) -> np.ndarray:
    """以 AGCWD 演算法增強 float32 線性空間影像。

    Args:
        image: 輸入 float32 BGR 影像（線性空間，值可 > 1.0）。
        w:     PDF 加權指數，控制 Gamma 曲線的彎曲程度。

    Returns:
        增強後的 float32 BGR 影像（線性空間）。
    """
    # 亮度通道
    L = 0.0722 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.2126 * image[:, :, 2]
    L = np.maximum(L, 1e-7)
    L_max = L.max()

    # 正規化至 [0, 1] 做直方圖
    L_norm = L / L_max

    # 量化成 N_BINS 個 bin
    L_idx = np.clip((L_norm * (N_BINS - 1)).astype(np.int32), 0, N_BINS - 1)
    hist = np.bincount(L_idx.ravel(), minlength=N_BINS).astype(np.float64)
    pdf = hist / hist.sum()

    pdf_min, pdf_max = pdf.min(), pdf.max()
    w_pdf = pdf_max * ((pdf - pdf_min) / (pdf_max - pdf_min + 1e-7)) ** w
    cdf = np.cumsum(w_pdf) / (w_pdf.sum() + 1e-7)

    # 建立 bin 中心值
    levels = np.arange(N_BINS, dtype=np.float64) / (N_BINS - 1)
    # Gamma 校正: T(l) = l ^ (1 - cdf(l))
    mapped = np.power(levels, 1.0 - cdf)

    # 用 LUT 查表，再乘回 L_max
    L_enhanced = mapped[L_idx].astype(np.float32) * L_max

    # 以亮度比例調整各通道
    ratio = (L_enhanced / L)[:, :, np.newaxis]
    return (image * ratio).astype(np.float32)

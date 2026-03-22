"""AGCWD: Adaptive Gamma Correction with Weighting Distribution (GPU-accelerated).

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

import numpy as np
import torch

from pg_denet.gpu import luminance_gpu, to_gpu, to_cpu

N_BINS = 4096  # 高精度直方圖 bin 數量


def apply_agcwd(image: np.ndarray, w: float = 0.8) -> np.ndarray:
    """以 AGCWD 演算法增強 float32 線性空間影像 (GPU)。

    Args:
        image: 輸入 float32 BGR 影像（線性空間，值可 > 1.0）。
        w:     PDF 加權指數，控制 Gamma 曲線的彎曲程度。

    Returns:
        增強後的 float32 BGR 影像（線性空間）。
    """
    img_t = to_gpu(image)  # (H, W, 3)

    # 亮度通道
    L_t = torch.clamp(luminance_gpu(img_t), min=1e-7)
    L_max = L_t.max()
    L_norm = L_t / L_max

    # 量化成 N_BINS 個 bin
    L_idx = torch.clamp((L_norm * (N_BINS - 1)).to(torch.int64), 0, N_BINS - 1)
    hist = torch.bincount(L_idx.flatten(), minlength=N_BINS).float()
    pdf = hist / hist.sum()

    pdf_min = pdf.min()
    pdf_max = pdf.max()
    w_pdf = pdf_max * ((pdf - pdf_min) / (pdf_max - pdf_min + 1e-7)) ** w
    cdf = torch.cumsum(w_pdf, dim=0) / (w_pdf.sum() + 1e-7)

    # 建立 bin 中心值 + Gamma 校正: T(l) = l ^ (1 - cdf(l))
    levels = torch.arange(N_BINS, device=img_t.device, dtype=torch.float32) / (N_BINS - 1)
    mapped = torch.pow(levels, 1.0 - cdf)

    # LUT 查表，乘回 L_max
    L_enhanced = mapped[L_idx] * L_max

    # 以亮度比例調整各通道
    ratio = (L_enhanced / L_t).unsqueeze(2)
    result = img_t * ratio
    return to_cpu(result)

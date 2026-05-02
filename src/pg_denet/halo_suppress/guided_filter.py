import cv2
import numpy as np


def _guided_filter_self(img: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Per-channel self-guided filter (He et al., 2013).

    Uses cv2.boxFilter for box-mean computation; no opencv-contrib required.
    """
    ksize = (2 * r + 1, 2 * r + 1)
    channels = cv2.split(img)
    out_channels = []
    for I in channels:
        mean_I  = cv2.boxFilter(I, ddepth=-1, ksize=ksize, normalize=True)
        mean_I2 = cv2.boxFilter(I * I, ddepth=-1, ksize=ksize, normalize=True)
        var_I   = mean_I2 - mean_I * mean_I

        a = var_I / (var_I + eps)
        b = mean_I * (1.0 - a)

        mean_a = cv2.boxFilter(a, ddepth=-1, ksize=ksize, normalize=True)
        mean_b = cv2.boxFilter(b, ddepth=-1, ksize=ksize, normalize=True)

        out_channels.append(mean_a * I + mean_b)
    return cv2.merge(out_channels)


def highlight_suppress(hdr_float32, r=24, eps=0.05, highlight_gamma=0.4):
    # 基底層：Guided Filter（自身引導）
    base = _guided_filter_self(hdr_float32, r, eps)
    # 細節層：比值（保留邊緣/紋理）
    detail = hdr_float32 / (base + 1e-6)
    
    # 對基底層做 gamma 壓縮（僅壓縮高光，保留暗部）
    # highlight_gamma < 1 壓縮高光；可再加 shadow boost
    base_compressed = np.power(np.clip(base, 0, 1), highlight_gamma)
    
    # 重組
    result = base_compressed * detail
    return np.clip(result, 0, 1)
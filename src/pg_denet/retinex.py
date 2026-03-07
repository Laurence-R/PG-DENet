from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class RetinexResult:
    """Retinex 分解結果。"""
    log_I: np.ndarray   # log domain 光照
    log_R: np.ndarray   # log domain 反射率
    I_display: np.ndarray  # normalized uint8，供顯示用
    R_display: np.ndarray  # normalized uint8，供顯示用

    def recover(self) -> np.ndarray:
        """從 log_I + log_R 還原原圖（uint8）。"""
        recovered = np.exp(self.log_I + self.log_R) - 1
        return np.clip(recovered, 0, 255).astype(np.uint8)


def retinex_decompose(src: np.ndarray, sigma: float = 15.0) -> RetinexResult:
    """對圖片做單尺度 Retinex (SSR) 分解，分離 Illumination 與 Reflectance。

    Args:
        src: 輸入 BGR 圖片（uint8）。
        sigma: 高斯模糊標準差，控制光照估計的平滑程度。

    Returns:
        RetinexResult: (log_I, log_R, I_display, R_display)
    """
    float_img = src.astype(np.float32) + 1.0

    # 估計光照：低頻高斯模糊
    illumination = cv2.GaussianBlur(float_img, (0, 0), sigma)

    log_S = np.log(float_img)
    log_I = np.log(illumination)
    log_R = log_S - log_I        # R = S/I  ==> log(R) = log(S) - log(I)

    I_display = cv2.normalize(log_I, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    R_display: np.ndarray = cv2.normalize(log_R, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return RetinexResult(log_I=log_I, log_R=log_R, I_display=I_display, R_display=R_display)

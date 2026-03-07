"""MSRCR: Multi-Scale Retinex with Color Restoration.

References:
    Rahman, Z., Jobson, D. J., & Woodell, G. A. (1996).
    Multi-scale retinex for color image enhancement.
    IEEE ICIP, 3, 1003-1006.

    Jobson, D. J., Rahman, Z., & Woodell, G. A. (1997).
    A multiscale retinex for bridging the gap between color images and the
    human observation of scenes.
    IEEE Transactions on Image Processing, 6(7), 965-976.

Algorithm:
    1. Multi-Scale SSR (Single-Scale Retinex) per channel:
           R_c^σ(x) = log(I_c(x)) − log(G_σ * I_c(x))

    2. Weighted MSR per channel:
           R_c(x) = Σ_i  w_i · R_c^{σ_i}(x)

    3. Color Restoration Function (CRF):
           C_c(x) = β · [log(α · I_c(x)) − log(Σ_c I_c(x))]

       CRF compensates for the colour distortion introduced by MSR when the
       scene illumination is not neutral.

    4. Final output:
           MSRCR_c(x) = G · C_c(x) · R_c(x) + b

       then normalise each channel to [0, 255].
"""

import cv2
import numpy as np


def _single_scale_retinex(channel: np.ndarray, sigma: float) -> np.ndarray:
    """Compute SSR for one channel (float32 input, log-domain output)."""
    blur = cv2.GaussianBlur(channel, (0, 0), sigma)
    ssr = np.log1p(channel) - np.log1p(blur)   # log1p avoids log(0)
    return ssr

def _multi_scale_retinex(image: np.ndarray, sigmas: list) -> np.ndarray:
    """Compute MSR for whole image"""
    retinex = np.zeros_like(image)
    for sigma in sigmas:
        for c in range(3):
            retinex[:, :, c] += _single_scale_retinex(image[:, :, c], sigma)
    
    msr = retinex/len(sigmas)
    return msr

def _color_restoration(image: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    img_sum = image.sum(axis=2, keepdims=True) + 1e-6     # Σ_c I_c, avoid /0
    crf = beta * (np.log(alpha * image + 1e-6) - np.log(img_sum))
    return crf


def apply_msrcr(
    image: np.ndarray,
    sigmas: list[float] = (15.0, 80.0, 250.0),
    alpha: float = 125.0,
    beta: float = 46.0,
    G: float = 5.0,
    b: float = 25.0,
) -> np.ndarray:
    """Enhance a low-light BGR image using MSRCR.

    Args:
        image:   Input BGR image (uint8, H×W×3).
        sigmas:  Gaussian scales for each SSR pass (paper default: 15, 80, 250).
        alpha:   Nonlinearity strength in the Color Restoration Function.
                 Larger → stronger colour boost.  (paper default: 125)
        beta:    Gain of the Color Restoration Function.  (paper default: 46)
        G:       Global gain applied after CRF × MSR.    (paper default: 192)
        b:       Global offset (bias).                   (paper default: -30)

    Returns:
        Enhanced BGR image (uint8, H×W×3).
    """

    img = image.astype(np.float32) + 1.0    # +1 to keep pixel values > 0

    # ── Step 1 & 2: Weighted Multi-Scale Retinex per channel ─────────────────
    msr = _multi_scale_retinex(img, sigmas)

    # ── Step 3: Color Restoration Function ───────────────────────────────────
    crf = _color_restoration(img, alpha, beta)

    # ── Step 4: MSRCR = G * CRF * MSR + b, then normalise ───────────────────
    msrcr = G * crf * msr + b

    # Per-channel simple scale normalisation to [0, 255]
    out = np.empty_like(msrcr, dtype=np.uint8)
    for c in range(3):
        ch = msrcr[:, :, c]
        ch = cv2.normalize(ch, None, 0, 255, cv2.NORM_MINMAX)
        out[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)

    return out

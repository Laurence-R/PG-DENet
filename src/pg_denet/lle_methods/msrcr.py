"""MSRCR: Multi-Scale Retinex with Color Restoration (GPU-accelerated).

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

    4. Final output:
           MSRCR_c(x) = G · C_c(x) · R_c(x) + b

       then normalise each channel to [0, 1] and scale back to input range.
"""

import numpy as np
import torch

from pg_denet.gpu import gaussian_blur_fft, to_gpu, to_cpu


def apply_msrcr(
    image: np.ndarray,
    sigmas: list[float] = (15.0, 80.0, 250.0),
    alpha: float = 125.0,
    beta: float = 46.0,
    G: float = 5.0,
    b: float = 25.0,
) -> np.ndarray:
    """Enhance a float32 linear-space BGR image using MSRCR.

    All heavy operations (FFT Gaussian blur, log, normalisation) run on GPU.

    Args:
        image:   Input float32 BGR image（線性空間，值可 > 1.0）。
        sigmas:  Gaussian scales for each SSR pass (paper default: 15, 80, 250).
        alpha:   Nonlinearity strength in the Color Restoration Function.
        beta:    Gain of the Color Restoration Function.
        G:       Global gain applied after CRF × MSR.
        b:       Global offset (bias).

    Returns:
        Enhanced float32 BGR image（線性空間）。
    """
    img_t = to_gpu(image) + 1e-6  # (H, W, 3) avoid log(0)
    img_max = img_t.max()

    # ── Step 1 & 2: Multi-Scale Retinex (FFT blur on GPU) ───────────────
    retinex = torch.zeros_like(img_t)
    log_img = torch.log1p(img_t)
    for sigma in sigmas:
        for c in range(3):
            blur_c = gaussian_blur_fft(img_t[:, :, c], sigma)
            retinex[:, :, c] += log_img[:, :, c] - torch.log1p(blur_c)
    msr = retinex / len(sigmas)

    # ── Step 3: Color Restoration Function ───────────────────────────────
    img_sum = img_t.sum(dim=2, keepdim=True) + 1e-6
    crf = beta * (torch.log(alpha * img_t + 1e-6) - torch.log(img_sum))

    # ── Step 4: MSRCR = G * CRF * MSR + b ───────────────────────────────
    msrcr = G * crf * msr + b

    # Per-channel normalisation to [0, 1] then scale to input range
    for c in range(3):
        ch = msrcr[:, :, c]
        c_min = ch.min()
        c_max = ch.max()
        if (c_max - c_min) > 1e-7:
            msrcr[:, :, c] = (ch - c_min) / (c_max - c_min)
        else:
            msrcr[:, :, c] = 0.0

    result = torch.clamp(msrcr * img_max, min=0.0)
    return to_cpu(result)

"""PSNR and SSIM — reference-based full-reference metrics.

Both metrics compare an enhanced image against a clean reference (ground truth).

PSNR (Peak Signal-to-Noise Ratio):
    PSNR = 10 · log10(MAX² / MSE)

    Higher is better. > 30 dB is generally considered good quality.

SSIM (Structural Similarity Index):
    SSIM(x, y) = (2μ_x μ_y + C1)(2σ_xy + C2) / ((μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2))

    Range [−1, 1], higher is better. 1 means identical images.
    Uses a sliding window approach over the image.
"""

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_psnr(enhanced: np.ndarray, reference: np.ndarray) -> float:
    """Compute PSNR between an enhanced image and a clean reference.

    Args:
        enhanced:  Enhanced BGR image (uint8, H×W×3).
        reference: Ground-truth BGR image (uint8, H×W×3).

    Returns:
        PSNR value in dB. Higher is better.
    """
    return float(peak_signal_noise_ratio(reference, enhanced, data_range=255))


def compute_ssim(enhanced: np.ndarray, reference: np.ndarray) -> float:
    """Compute mean SSIM between an enhanced image and a clean reference.

    Args:
        enhanced:  Enhanced BGR image (uint8, H×W×3).
        reference: Ground-truth BGR image (uint8, H×W×3).

    Returns:
        Mean SSIM in [−1, 1]. Higher is better.
    """
    return float(structural_similarity(
        reference, enhanced,
        data_range=255,
        channel_axis=2,
    ))

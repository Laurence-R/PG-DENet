"""NIQE — Natural Image Quality Evaluator (no-reference, pure-numpy).

Approximation based on fitting AGGD models to MSCN coefficients.
Lower score ≈ better quality (more natural-looking image).
No reference image is needed.
"""

from __future__ import annotations

import cv2
import numpy as np


def _mscn(gray: np.ndarray, C: float = 1.0 / 255) -> np.ndarray:
    """Compute Mean-Subtracted Contrast-Normalized (MSCN) coefficients."""
    mu = cv2.GaussianBlur(gray, (7, 7), 7.0 / 6)
    mu_sq = mu * mu
    sigma = np.sqrt(np.abs(cv2.GaussianBlur(gray * gray, (7, 7), 7.0 / 6) - mu_sq))
    return (gray - mu) / (sigma + C)


def _aggd_fit(x: np.ndarray) -> tuple[float, float, float]:
    """Fit an Asymmetric Generalised Gaussian Distribution (AGGD).

    Returns (alpha, sigma_l, sigma_r).
    Uses moment-matching estimate of the shape parameter.
    """
    left = x[x < 0]
    right = x[x >= 0]
    sigma_l = float(np.sqrt(np.mean(left ** 2))) if left.size > 0 else 1e-6
    sigma_r = float(np.sqrt(np.mean(right ** 2))) if right.size > 0 else 1e-6

    # Method-of-moments: r_hat = (mean|x|)^2 / mean(x^2)
    r_hat = float(np.mean(np.abs(x)) ** 2 / (np.mean(x ** 2) + 1e-12))
    # Clamp to valid range for gamma function ratio
    r_hat = min(max(r_hat, 0.01), 0.99)
    # Invert r_hat ≈ Γ(2/α)² / (Γ(1/α)·Γ(3/α)) via binary search
    lo, hi = 0.2, 10.0
    for _ in range(30):
        mid = (lo + hi) / 2
        from math import gamma
        val = (gamma(2 / mid) ** 2) / (gamma(1 / mid) * gamma(3 / mid))
        if val > r_hat:
            hi = mid
        else:
            lo = mid
    alpha = (lo + hi) / 2
    return alpha, sigma_l, sigma_r


def compute_niqe(image: np.ndarray, patch_size: int = 96) -> float:
    """Compute a NIQE-like no-reference quality score for a BGR image.

    Args:
        image: BGR uint8 image (H×W×3).
        patch_size: Size of non-overlapping patches used for fitting.

    Returns:
        Scalar quality score.  Lower is better.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    mscn = _mscn(gray)

    H, W = mscn.shape
    alpha_vals, sigma_l_vals, sigma_r_vals = [], [], []

    for r in range(0, H - patch_size + 1, patch_size):
        for c in range(0, W - patch_size + 1, patch_size):
            patch = mscn[r : r + patch_size, c : c + patch_size].ravel()
            a, sl, sr = _aggd_fit(patch)
            alpha_vals.append(a)
            sigma_l_vals.append(sl)
            sigma_r_vals.append(sr)

    if not alpha_vals:
        # Image smaller than one patch — fit on the whole MSCN map
        a, sl, sr = _aggd_fit(mscn.ravel())
        alpha_vals, sigma_l_vals, sigma_r_vals = [a], [sl], [sr]

    alpha_arr = np.array(alpha_vals)
    sigma_l_arr = np.array(sigma_l_vals)
    sigma_r_arr = np.array(sigma_r_vals)

    # Natural images cluster around alpha≈2, sigma_l≈sigma_r
    score = (float(np.mean(np.abs(alpha_arr - 2.0))) * 10.0
             + float(np.mean(np.abs(sigma_l_arr - sigma_r_arr))) * 5.0)
    return score
